"""Execute the frozen storage comparison on Della."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import time
from pathlib import Path

import torch
from prepare import sha, write
from safetensors.torch import load_file, save_file

from tinymem.memory.quantized_slots import QuantizedSlotMemory
from tinymem.reader.adapter import (
    configure_read_adapter,
    frozen_history_features,
)
from tinymem.reader.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.reader.lora import attach_reader_lora
from tinymem.studies.frontier.baselines import SUPPORTED_CODECS, TextStore
from tinymem.studies.frontier.data import (
    StorageQuestion,
    write_records,
)
from tinymem.studies.frontier.training import (
    answer_loss,
    encode_answer,
    optimizer_step,
    rollout,
    text_vectors,
)

FITTING_STAGES = ('features', 'preflight', 'train')


def checked(study: Path) -> tuple[dict, dict]:
    protocol = json.loads((study / 'protocol.json').read_text())
    for name, expected in protocol['files'].items():
        if sha(study / name) != expected:
            raise ValueError(f'frozen input differs: {name}')
    for function, relative in ((frozen_history_features, 'tinymem/reader/adapter.py'),
                               (QuantizedSlotMemory, 'tinymem/memory/quantized_slots.py')):
        if Path(inspect.getfile(inspect.unwrap(function))).resolve() != (study / 'source/src' / relative).resolve():
            raise ValueError('execution imported a module outside the frozen source')
    return protocol, json.loads((study / 'inputs/dataset.json').read_text())


def sealed_protocol_hashes(study: Path, protocol: dict) -> frozenset[str]:
    """Hashes a pre-evaluation seal may carry: this protocol, or the parent it amends."""
    accepted = {sha(study / 'protocol.json')}
    if 'amends' in protocol:
        accepted.add(protocol['amends']['parent_protocol_sha256'])
    return frozenset(accepted)


def runtime() -> dict:
    return {'packages': {name: importlib.metadata.version(name)
                         for name in ('torch', 'numpy', 'transformers', 'peft', 'safetensors')},
            'cuda': torch.version.cuda, 'threads': torch.get_num_threads(),
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'float32_matmul_precision': torch.get_float32_matmul_precision()}


def questions(data: dict, split: str) -> tuple[StorageQuestion, ...]:
    return tuple(StorageQuestion(**{**q, 'records': tuple(q['records'])}) for q in data[split])


def feature_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def features(study: Path, protocol: dict, data: dict) -> dict[str, torch.Tensor]:
    directory = study / 'features'
    seal = json.loads((directory / 'complete.json').read_text())
    if seal['dataset_sha256'] != sha(study / 'inputs/dataset.json'):
        raise ValueError('feature dataset provenance differs')
    if (seal['protocol_sha256'] not in sealed_protocol_hashes(study, protocol)
            or seal['snapshot_sha256'] != sha(study / 'inputs/model_snapshot.json')
            or seal['encoder_sha256'] != sha(study / 'source/src/tinymem/reader/adapter.py')
            or seal['width'] != 2048 or seal['runtime'] != runtime()):
        raise ValueError('feature model, encoder, width, protocol, or runtime provenance differs')
    path = directory / 'features.safetensors'
    if sha(path) != seal['features_sha256']:
        raise ValueError('feature payload differs from seal')
    tensors = load_file(str(path))
    if set(tensors) != {feature_key(s) for s in data['feature_records']}:
        raise ValueError('feature cache key inventory differs')
    if any(t.ndim != 2 or not 0 < t.shape[0] <= 64 or t.shape[1] != seal['width']
           or t.dtype != torch.float32 for t in tensors.values()):
        raise ValueError('feature tensor shape or dtype differs')
    return {s: tensors[feature_key(s)] for s in data['feature_records']}


def prepare_features(reader, study: Path, data: dict) -> None:
    directory = study / 'features'
    directory.mkdir(exist_ok=False)
    tensors = {}
    started = time.perf_counter()
    for i, text in enumerate(data['feature_records']):
        ids = reader.tokenizer.encode(text + '\n', add_special_tokens=False)
        if not 0 < len(ids) <= 64:
            raise ValueError('current record must fit the declared independent feature window')
        tensors[feature_key(text)] = frozen_history_features(reader, ids).contiguous()
        if i % 100 == 0:
            print(json.dumps({'stage': 'features', 'records': i, 'seconds': time.perf_counter() - started}), flush=True)
    path = directory / 'features.safetensors'
    save_file(tensors, str(path))
    write(directory / 'complete.json', {'dataset_sha256': sha(study / 'inputs/dataset.json'),
                                       'protocol_sha256': sha(study / 'protocol.json'),
                                       'snapshot_sha256': sha(study / 'inputs/model_snapshot.json'),
                                       'encoder_sha256': sha(study / 'source/src/tinymem/reader/adapter.py'),
                                       'runtime': runtime(),
                                       'features_sha256': sha(path), 'records': len(tensors),
                                       'width': reader.model.config.hidden_size,
                                       'encoder': 'frozen current record only, positions reset, no KV cache',
                                       'seconds': time.perf_counter() - started})


def preflight(reader, study: Path, protocol: dict, data: dict) -> None:
    directory = study / 'preflight'
    directory.mkdir(exist_ok=False)
    cache = features(study, protocol, data)
    train = questions(data, 'training')
    ordinary = [next(q for q in train if q.task == task) for task in ('qa1', 'qa2', 'qa3', 'qa4', 'qa5')]
    ordinary = (ordinary * 2)[:8]
    longest = max(train, key=lambda q: len(q.records))
    attach_reader_lora(reader, rank=8, checkpointing=False)
    adapters = configure_read_adapter(reader, trainable=True)
    initial_adapter = [p.detach().clone() for p in adapters]
    rows = []
    for budget in protocol['settings']['budgets']:
        torch.manual_seed(27001)
        with torch.no_grad():
            for parameter, initial in zip(adapters, initial_adapter, strict=True):
                parameter.copy_(initial)
        memory = QuantizedSlotMemory(reader.model.config.hidden_size, slots=budget // 32).to(reader.model.device)
        optimizer = torch.optim.AdamW([*memory.parameters(), *adapters], lr=.0003)
        for label, batch, level in (('ordinary', ordinary, 0), ('noisy', ordinary, 2),
                                    ('longest', [longest, *ordinary[:7]], 2)):
            optimizer.zero_grad(set_to_none=True)
            histories = [write_records(q, tuple(data['noise']['train']), level=level,
                                       seed=protocol['settings']['noise_seed']) for q in batch]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            state = rollout(memory, histories, cache)
            restored = torch.cat([memory.unpack(memory.pack(s[None])) for s in state])
            if not torch.equal(state.detach(), restored):
                raise ValueError('real model write/serialize/reload differs')
            loss = answer_loss(reader, [encode_answer(reader, q) for q in batch], list(memory.memory_vectors(state)))
            result = optimizer_step(reader, memory, adapters, optimizer, loss)
            torch.cuda.synchronize()
            row = {'budget': budget, 'batch': label, 'questions': len(batch),
                   'writes': list(map(len, histories)), 'seconds': time.perf_counter() - started,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(), **result}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del state, restored, loss
        del optimizer, memory
        torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(adapters, lr=.0003)
    for codec in ('full_text', *SUPPORTED_CODECS):
        batch = [longest, *ordinary[:7]] if codec == 'full_text' else ordinary
        store = None if codec == 'full_text' else TextStore(
            1024, codec, tuple(data['dictionary']) if codec == 'dictionary_recent' else ())
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        texts = []
        for q in batch:
            if store is None:
                texts.append('\n'.join(q.records))
            else:
                payload = store.empty()
                for record in write_records(q, tuple(data['noise']['train']), level=2,
                                            seed=protocol['settings']['noise_seed']):
                    payload = store.update(payload, record)
                texts.append(store.decode(payload))
        loss = answer_loss(reader, [encode_answer(reader, q) for q in batch], text_vectors(reader, texts))
        result = optimizer_step(reader, None, adapters, optimizer, loss)
        torch.cuda.synchronize()
        row = {'batch': codec, 'seconds': time.perf_counter() - started,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(), **result}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del loss
    write(directory / 'complete.json', {'correctness_checks_passed': True,
                                       'accuracy_gate_used': False, 'rows': rows,
                                       'runtime': runtime(),
                                       'protocol_sha256': sha(study / 'protocol.json')})


def train(reader, study: Path, protocol: dict, data: dict, cell: int) -> None:
    from tinymem.studies.frontier.fit import fit
    declaration = protocol['cells'][cell]
    if not (study / 'preflight/complete.json').exists():
        raise ValueError('the real-runtime correctness preflight must complete')
    preflight_result = json.loads((study / 'preflight/complete.json').read_text())
    if (preflight_result['protocol_sha256'] != sha(study / 'protocol.json')
            or preflight_result['runtime'] != runtime() or not preflight_result['correctness_checks_passed']):
        raise ValueError('preflight identity, runtime, or result differs')
    torch.manual_seed(declaration['seed'])
    memory = (QuantizedSlotMemory(reader.model.config.hidden_size, slots=declaration['budget'] // 32)
              .to(reader.model.device) if declaration['kind'] == 'learned' else None)
    directory = study / 'training' / str(cell)
    directory.parent.mkdir(exist_ok=True)
    report = fit(reader, memory, features(study, protocol, data) if memory is not None else {},
                 questions(data, 'training'), questions(data, 'validation'),
                 {k: tuple(v) for k, v in data['noise'].items()}, tuple(data['dictionary']),
                 protocol['settings'], directory, declaration['seed'])
    write(directory / 'complete.json', {'protocol_sha256': sha(study / 'protocol.json'), 'cell': cell,
                                       'runtime': runtime(),
                                       'files': {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()},
                                       'report': report})


def load_checkpoint(reader, study: Path, protocol: dict, cell: int):
    directory = study / 'training' / str(cell)
    seal = json.loads((directory / 'complete.json').read_text())
    if seal['protocol_sha256'] not in sealed_protocol_hashes(study, protocol) or seal['cell'] != cell:
        raise ValueError('training provenance differs')
    if seal['runtime'] != runtime():
        raise ValueError('evaluation runtime differs from training')
    for name, expected in seal['files'].items():
        if sha(directory / name) != expected:
            raise ValueError('sealed training output differs')
    declaration = protocol['cells'][cell]
    from tinymem.studies.frontier.fit import _base_parameters, _hash_parameters
    trained = json.loads((directory / 'report.json').read_text())
    if _hash_parameters(_base_parameters(reader)) != trained['base_hash_before']:
        raise ValueError('evaluation base weights differ from training')
    torch.manual_seed(declaration['seed'])
    memory = (QuantizedSlotMemory(reader.model.config.hidden_size, slots=declaration['budget'] // 32)
              .to(reader.model.device) if declaration['kind'] == 'learned' else None)
    attach_reader_lora(reader, rank=8, checkpointing=False)
    configure_read_adapter(reader, trainable=False)
    tensors = load_file(str(directory / 'checkpoint.safetensors'))
    expected = {'adapter.' + n for n, _ in reader.model.named_parameters() if '.lora_A.' in n or '.lora_B.' in n}
    if memory is not None:
        expected.update('writer.' + n for n in memory.state_dict())
    if set(tensors) != expected:
        raise ValueError('checkpoint schema differs')
    if memory is not None:
        memory.load_state_dict({n.removeprefix('writer.'): t for n, t in tensors.items() if n.startswith('writer.')})
        memory.requires_grad_(False).eval()
    with torch.no_grad():
        for name, parameter in reader.model.named_parameters():
            if 'adapter.' + name in tensors:
                value = tensors['adapter.' + name]
                if value.shape != parameter.shape or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError('invalid adapter checkpoint')
                parameter.copy_(value)
    return memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('features', 'preflight', 'train', 'evaluate', 'transfer'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cell', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260918)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    if not torch.cuda.is_available():
        raise RuntimeError('the declared execution device is CUDA')
    protocol, data = checked(args.study)
    if 'amends' in protocol and args.stage in FITTING_STAGES:
        raise ValueError('an amended protocol reuses its parent seals; it does not fit or extract features')
    if args.stage in ('train', 'evaluate', 'transfer') and (args.cell is None or not 0 <= args.cell < len(protocol['cells'])):
        raise ValueError('declare one valid frozen cell')
    if verify_qwen_snapshot(args.model) != json.loads((args.study / 'inputs/model_snapshot.json').read_text()):
        raise ValueError('runtime Qwen snapshot differs from the frozen input manifest')
    reader = load_qwen_reader(args.model, device=torch.device('cuda'), dtype=torch.bfloat16)
    if args.stage == 'features':
        prepare_features(reader, args.study, data)
    elif args.stage == 'preflight':
        preflight(reader, args.study, protocol, data)
    elif args.stage == 'train':
        train(reader, args.study, protocol, data, args.cell)
    elif args.stage == 'evaluate':
        from score import evaluate
        evaluate(reader, load_checkpoint(reader, args.study, protocol, args.cell),
                 args.study, protocol, data, args.cell)
    else:
        from transfer import evaluate_transfer
        evaluate_transfer(reader, load_checkpoint(reader, args.study, protocol, args.cell),
                          args.study, protocol, data, args.cell)


if __name__ == '__main__':
    main()
