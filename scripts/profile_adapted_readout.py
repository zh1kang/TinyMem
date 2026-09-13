"""Disposable paired reader-adaptation profile; no development scoring or resume."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.adapted_readout import configure_read_adapter, frozen_history_features, train_adapted_step
from tinymem.research.readout_experiment import _reader_hash, _source_hashes
from tinymem.research.readout_read import read_state_answer
from tinymem.research.readout_runner import encode_before
from tinymem.research.study_runtime import REPOSITORY, allocation_metrics, execution_record, prepare_device, synchronize, validate_execution
from tinymem.research.update_protocol import file_sha256, load_development_data, load_shared_reader, shared_reader_identity


STEPS = 7
WARMUP = 2
SEED = 1337
LEARNING_RATE = 0.001


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def frozen_hash(reader):
    """Hash frozen parameters and all buffers, including their names and types."""
    mutable = {name for name, p in reader.model.named_parameters() if p.requires_grad}
    digest = hashlib.sha256()
    for name, value in sorted(reader.model.state_dict().items()):
        if name in mutable:
            continue
        header = json.dumps([name, str(value.dtype), list(value.shape)]).encode()
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = prepare_device('cuda')
    data = load_development_data(args.data)
    identity = shared_reader_identity()
    sources = {**_source_hashes(), 'adapted_readout.py': file_sha256(REPOSITORY / 'src/tinymem/research/adapted_readout.py'),
               'profile_adapted_readout.py': file_sha256(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {'kind': 'adapted_readout_profile_v1', 'purpose': 'execution_and_compute_only',
                'input_data_sha256': data.protocol_sha256, 'reader': identity, 'source_sha256': sources,
                'seed': SEED, 'steps_per_arm': STEPS, 'warmup_steps': WARMUP,
                'optimizer': {'kind': 'AdamW', 'lr_all_groups': LEARNING_RATE, 'weight_decay': 0.01,
                              'betas': [0.9, 0.999], 'eps': 1e-8, 'clip_norm': 1.0},
                'checkpoint_selection': 'final_only_disposable', 'reader_mode': 'eval_in_both_arms',
                'checkpointing': False, 'bridge': 'affine', 'persistent_bytes': 66,
                'feature_cache': 'fixed_original_reader_CPU_FP32_training_only_never_used_by_read',
                'selection': 'four_training_histories_at_length_sorted_indices_0_85_170_255',
                'development_scored': False, 'confirmation_opened': False,
                'execution': execution_record(device)}
    write_json(args.output / 'protocol.json', protocol)
    cached, rows, reader_initial = {}, None, None
    summaries = {}
    for arm in ('frozen', 'adapted'):
        reader = load_shared_reader(identity, device)
        if reader.model.get_input_embeddings().weight.dtype != torch.bfloat16:
            raise ValueError('expected BF16 reader embeddings')
        initial_hash = _reader_hash(reader)
        if reader_initial is None:
            reader_initial = initial_hash
            encoded = sorted((encode_before(reader, row) for row in data.train),
                             key=lambda row: (len(row.history_ids), row.history_id))
            if len(encoded) != 256:
                raise ValueError('profile requires the original 256 training histories')
            rows = [encoded[i] for i in (0, 85, 170, 255)]
            write_json(args.output / 'training_encodings.json', [asdict(row) for row in rows])
            for row in rows:
                cached[row.history_id] = frozen_history_features(reader, row.history_ids)
            save_file(cached, str(args.output / 'training_features.safetensors'))
            feature_hash = file_sha256(args.output / 'training_features.safetensors')
        elif initial_hash != reader_initial:
            raise ValueError('paired reader initial values differ')
        arm_dir = args.output / arm
        arm_dir.mkdir()
        adapters = configure_read_adapter(reader, trainable=arm == 'adapted')
        immutable_before = frozen_hash(reader)
        adapter_before = [p.detach().clone() for p in adapters]
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(SEED)
            encoder, bridge = OneShotEncoder(reader.model.config.hidden_size), ReadoutBridge(reader.model.config.hidden_size, 'affine')
        encoder.to(device)
        bridge.to(device)
        parameters = [*encoder.parameters(), *bridge.parameters(), *adapters]
        optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.01)
        save_file({**{'encoder.' + k: v for k, v in encoder.state_dict().items()},
                   **{'bridge.' + k: v for k, v in bridge.state_dict().items()}},
                  str(arm_dir / 'initial.safetensors'))
        if arm == 'adapted' and file_sha256(arm_dir / 'initial.safetensors') != file_sha256(args.output / 'frozen/initial.safetensors'):
            raise ValueError('paired encoder/bridge initial values differ')
        records = []
        with (arm_dir / 'metrics.jsonl').open('x') as handle:
            for step in range(STEPS):
                row = rows[step % len(rows)]
                synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                result = train_adapted_step(reader, encoder, bridge, cached[row.history_id], row.before_ids,
                                           row.queries, optimizer, adapter_parameters=adapters)
                synchronize(device)
                result.update(step=step + 1, warmup=step < WARMUP, history_id=row.history_id,
                              seconds=time.perf_counter() - started, **allocation_metrics(device))
                records.append(result)
                handle.write(json.dumps(result, sort_keys=True) + '\n')
                handle.flush()
                print(json.dumps({'arm': arm, **result}), flush=True)
        if frozen_hash(reader) != immutable_before:
            raise ValueError('frozen reader weights or buffers changed')
        if adapters and not any(not torch.equal(p, old) for p, old in zip(adapters, adapter_before, strict=True)):
            raise ValueError('read adapter did not change')
        save_file({**{'encoder.' + k: v for k, v in encoder.state_dict().items()},
                   **{'bridge.' + k: v for k, v in bridge.state_dict().items()}},
                  str(arm_dir / 'final.safetensors'))
        reader.model.save_pretrained(arm_dir / 'reader_adapter', save_embedding_layers=False)
        reader.model.zero_grad(set_to_none=True)
        reader.model.requires_grad_(False).eval()
        # Reload all saved learned weights on CUDA before checking state reads.
        from peft import set_peft_model_state_dict
        trained_reader_hash = _reader_hash(reader)
        saved_adapter = load_file(str(arm_dir / 'reader_adapter/adapter_model.safetensors'), device=str(device))
        reload_parameters = configure_read_adapter(reader, trainable=True)
        with torch.no_grad():
            for parameter in reload_parameters:
                parameter.zero_()
        set_peft_model_state_dict(reader.model, saved_adapter, adapter_name='default')
        reader.model.requires_grad_(False).eval()
        if _reader_hash(reader) != trained_reader_hash:
            raise ValueError('saved reader adapter failed exact reload')
        saved_modules = load_file(str(arm_dir / 'final.safetensors'), device=str(device))
        restored_encoder = OneShotEncoder(reader.model.config.hidden_size).to(device)
        restored_bridge = ReadoutBridge(reader.model.config.hidden_size, 'affine').to(device)
        restored_encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in saved_modules.items() if k.startswith('encoder.')})
        restored_bridge.load_state_dict({k.removeprefix('bridge.'): v for k, v in saved_modules.items() if k.startswith('bridge.')})
        with torch.inference_mode():
            states = {}
            for row in rows:
                features = cached[row.history_id].to(device).unsqueeze(0)
                state = encoder(features, torch.ones(features.shape[:2], device=device, dtype=torch.bool))
                restored_state = restored_encoder(features, torch.ones(features.shape[:2], device=device, dtype=torch.bool))
                if not torch.equal(state.values, restored_state.values) or not torch.equal(bridge(state), restored_bridge(restored_state)):
                    raise ValueError('saved encoder or bridge failed exact reload')
                states[row.history_id + '.values'] = state.values.cpu()
                states[row.history_id + '.valid'] = state.valid.cpu()
            save_file(states, str(arm_dir / 'states.safetensors'))
            reloaded = load_file(str(arm_dir / 'states.safetensors'), device=str(device))
            checks = []
            for row in rows:
                state = LatentSlotState(reloaded[row.history_id + '.values'], reloaded[row.history_id + '.valid'])
                original = LatentSlotState(states[row.history_id + '.values'].to(device), states[row.history_id + '.valid'].to(device))
                for q in (row.queries[0], row.queries[-1]):
                    options = dict(before_ids=torch.tensor(row.before_ids, device=device),
                                   question_ids=torch.tensor(q.after_ids, device=device), max_new_tokens=8)
                    first = read_state_answer(reader, bridge, original, **options)
                    second = read_state_answer(reader, restored_bridge, state, **options)
                    if first != second or state.nbytes != 66:
                        raise ValueError('serialized-state read differs')
                    checks.append({'history_id': row.history_id, 'case_id': q.case_id,
                                   'serialized_read_equal': True, 'generated_ids': second['generated_ids']})
        write_json(arm_dir / 'serialization_checks.json', checks)
        measured = records[WARMUP:]
        summaries[arm] = {'measured_steps': len(measured), 'seconds_per_step_mean': sum(r['seconds'] for r in measured) / len(measured),
                          'reader_initial_sha256': initial_hash, 'frozen_parameters_unchanged': True,
                          'trainable_encoder_parameters': sum(p.numel() for p in encoder.parameters()),
                          'trainable_bridge_parameters': sum(p.numel() for p in bridge.parameters()),
                          'trainable_adapter_parameters': sum(p.numel() for p in adapters),
                          'serialized_read_checks': len(checks), 'persistent_bytes': 66,
                          'cuda_checkpoint_reload_exact': True,
                          'reader_feature_cache_sha256': feature_hash}
        del reader, encoder, bridge, optimizer, parameters, adapters, adapter_before, state, original, reloaded
        del restored_encoder, restored_bridge, saved_modules, saved_adapter, reload_parameters, restored_state, parameter
        gc.collect()
        torch.cuda.empty_cache()
    if sources != {**_source_hashes(), 'adapted_readout.py': file_sha256(REPOSITORY / 'src/tinymem/research/adapted_readout.py'),
                   'profile_adapted_readout.py': file_sha256(Path(__file__))}:
        raise ValueError('execution sources changed')
    validate_execution(execution_record(device), expected=protocol['execution'])
    write_json(args.output / 'report.json', {'status': 'complete', 'quality_evaluated': False, 'arms': summaries})
    files = {str(p.relative_to(args.output)): file_sha256(p) for p in args.output.rglob('*') if p.is_file()}
    write_json(args.output / 'complete.json', {'kind': 'adapted_readout_profile_complete_v1', 'files': files})


if __name__ == '__main__':
    main()
