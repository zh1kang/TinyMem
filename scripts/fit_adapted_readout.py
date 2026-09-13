"""Fixed eight-history paired fit; no development evaluation or training extension."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from scripts.profile_adapted_readout import frozen_hash, write_json
from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.adapted_readout import configure_read_adapter, frozen_history_features, train_adapted_step
from tinymem.research.adapted_readout_evaluation import evaluate_states, summarize_readiness
from tinymem.research.readout_evaluation import evaluate_full_text
from tinymem.research.readout_experiment import _reader_hash, _source_hashes
from tinymem.research.readout_runner import encode_before
from tinymem.research.study_runtime import REPOSITORY, allocation_metrics, execution_record, prepare_device, synchronize, validate_execution
from tinymem.research.update_protocol import file_sha256, load_development_data, load_shared_reader, shared_reader_identity


SETTINGS = {'seed': 1337, 'steps_per_arm': 200, 'histories': 8,
            'selection_salt': 'tinymem-adapted-readout-readiness-v1:',
            'bridge': 'affine', 'lr': 0.001, 'weight_decay': 0.01,
            'optimizer': 'AdamW', 'betas': [0.9, 0.999], 'eps': 1e-8, 'clip_norm': 1.0,
            'persistent_bytes': 66, 'max_new_tokens': 8, 'checkpoint_selection': 'final_only',
            'known_gate': 0.95, 'missing_gate': 0.95, 'minimum_control_gap': 0.5,
            'full_text_known_gate': 0.95, 'full_text_missing_gate': 0.95,
            'conditions': ['normal', 'zero', 'no_memory', 'shuffled', 'full_text'],
            'reader_mode': 'eval', 'gradient_checkpointing': False}
EXTRA_SOURCES = ('src/tinymem/research/adapted_readout.py',
                 'src/tinymem/research/adapted_readout_evaluation.py',
                 'scripts/profile_adapted_readout.py', 'scripts/fit_adapted_readout.py')


def sources():
    return {**{f'src/tinymem/{name}': digest for name, digest in _source_hashes().items()},
            **{name: file_sha256(REPOSITORY / name) for name in EXTRA_SOURCES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--profile-report', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = prepare_device('cuda')
    data, identity = load_development_data(args.data), shared_reader_identity()
    source_hashes, runtime = sources(), execution_record(device)
    declaration = json.loads(args.declaration.read_text())
    profile = json.loads(args.profile_report.read_text())
    if (file_sha256(args.profile_report) != declaration['profile_report_sha256']
            or profile['status'] != 'complete'
            or any(not arm['cuda_checkpoint_reload_exact'] or not arm['frozen_parameters_unchanged']
                   for arm in profile['arms'].values())):
        raise ValueError('completed profile evidence differs from declaration')
    if (declaration['settings'] != SETTINGS or declaration['source_sha256'] != source_hashes
            or declaration['portable_source_sha256'] != runtime['source_sha256']
            or declaration['data_protocol_sha256'] != data.protocol_sha256 or declaration['reader'] != identity):
        raise ValueError('launch declaration differs from execution inputs')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', {'kind': 'adapted_readout_readiness_v1', 'settings': SETTINGS,
               'declaration_sha256': file_sha256(args.declaration), 'reader': identity,
               'profile_report_sha256': file_sha256(args.profile_report),
               'data_protocol_sha256': data.protocol_sha256, 'source_sha256': source_hashes,
               'execution': runtime, 'development_scored': False, 'confirmation_opened': False,
               'evidence_kind': 'consumed_training_history_fit_only'})
    cached, rows, initial_reader = {}, None, None
    summaries = {}
    for arm in ('frozen', 'adapted'):
        reader = load_shared_reader(identity, device)
        current_reader = _reader_hash(reader)
        if initial_reader is None:
            initial_reader = current_reader
            encoded = [encode_before(reader, row) for row in data.train]
            order = lambda row: (hashlib.sha256((SETTINGS['selection_salt'] + row.history_id).encode()).hexdigest(), row.history_id)
            rows = sorted(encoded, key=order)[:SETTINGS['histories']]
            if [row.history_id for row in rows] != declaration['history_ids']:
                raise ValueError('training history selection differs from declaration')
            write_json(args.output / 'training_encodings.json', [asdict(row) for row in rows])
            for row in rows:
                cached[row.history_id] = frozen_history_features(reader, row.history_ids)
            save_file(cached, str(args.output / 'training_features.safetensors'))
        elif initial_reader != current_reader:
            raise ValueError('paired reader initial values differ')
        directory = args.output / arm
        directory.mkdir()
        adapters = configure_read_adapter(reader, trainable=arm == 'adapted')
        frozen_before = frozen_hash(reader)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(SETTINGS['seed'])
            encoder, bridge = OneShotEncoder(2048), ReadoutBridge(2048, 'affine')
        encoder.to(device)
        bridge.to(device)
        parameters = [*encoder.parameters(), *bridge.parameters(), *adapters]
        optimizer = torch.optim.AdamW(parameters, lr=SETTINGS['lr'], weight_decay=SETTINGS['weight_decay'],
                                      betas=tuple(SETTINGS['betas']), eps=SETTINGS['eps'])
        save_file({**{'encoder.' + k: v for k, v in encoder.state_dict().items()},
                   **{'bridge.' + k: v for k, v in bridge.state_dict().items()}},
                  str(directory / 'initial.safetensors'))
        if arm == 'adapted' and file_sha256(directory / 'initial.safetensors') != file_sha256(args.output / 'frozen/initial.safetensors'):
            raise ValueError('paired encoder or bridge initialization differs')
        records = []
        with (directory / 'metrics.jsonl').open('x') as handle:
            for step in range(SETTINGS['steps_per_arm']):
                row = rows[step % len(rows)]
                synchronize(device)
                started = time.perf_counter()
                metric = train_adapted_step(reader, encoder, bridge, cached[row.history_id], row.before_ids,
                                           row.queries, optimizer, adapter_parameters=adapters)
                synchronize(device)
                metric.update(step=step + 1, history_id=row.history_id, seconds=time.perf_counter() - started,
                              **allocation_metrics(device))
                records.append(metric)
                handle.write(json.dumps(metric, sort_keys=True) + '\n')
                handle.flush()
                if step == 0 or (step + 1) % 20 == 0:
                    print(json.dumps({'arm': arm, **metric}), flush=True)
        if frozen_hash(reader) != frozen_before:
            raise ValueError('frozen parameters or buffers changed')
        save_file({**{'encoder.' + k: v for k, v in encoder.state_dict().items()},
                   **{'bridge.' + k: v for k, v in bridge.state_dict().items()}},
                  str(directory / 'final.safetensors'))
        reader.model.save_pretrained(directory / 'reader_adapter', save_embedding_layers=False)
        trained_reader = _reader_hash(reader)
        reader.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for parameter in configure_read_adapter(reader, trainable=True):
                parameter.zero_()
        set_peft_model_state_dict(reader.model, load_file(str(directory / 'reader_adapter/adapter_model.safetensors'), device=str(device)), adapter_name='default')
        reader.model.requires_grad_(False).eval()
        if _reader_hash(reader) != trained_reader:
            raise ValueError('reader checkpoint reload differs')
        saved = load_file(str(directory / 'final.safetensors'), device=str(device))
        restored_encoder, restored_bridge = OneShotEncoder(2048).to(device), ReadoutBridge(2048, 'affine').to(device)
        restored_encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in saved.items() if k.startswith('encoder.')})
        restored_bridge.load_state_dict({k.removeprefix('bridge.'): v for k, v in saved.items() if k.startswith('bridge.')})
        with torch.inference_mode():
            values = {}
            for row in rows:
                hidden = cached[row.history_id].to(device).unsqueeze(0)
                mask = torch.ones(hidden.shape[:2], device=device, dtype=torch.bool)
                original, restored = encoder(hidden, mask), restored_encoder(hidden, mask)
                if not torch.equal(original.values, restored.values) or not torch.equal(bridge(original), restored_bridge(restored)):
                    raise ValueError('encoder or bridge checkpoint reload differs')
                values[row.history_id + '.values'] = restored.values.cpu()
                values[row.history_id + '.valid'] = restored.valid.cpu()
            save_file(values, str(directory / 'states.safetensors'))
            payload = load_file(str(directory / 'states.safetensors'), device=str(device))
            states = {row.history_id: LatentSlotState(payload[row.history_id + '.values'], payload[row.history_id + '.valid']) for row in rows}
            predictions = evaluate_states(reader, restored_bridge, states, rows)
            predictions.extend(evaluate_full_text(reader, rows))
        with (directory / 'predictions.jsonl').open('x') as handle:
            for row in predictions:
                handle.write(json.dumps(row, sort_keys=True) + '\n')
        summaries[arm] = {**summarize_readiness(predictions), 'reader_initial_sha256': initial_reader,
                          'reader_final_sha256': trained_reader, 'frozen_parameters_unchanged': True,
                          'checkpoint_reload_exact': True, 'completed_steps': len(records),
                          'training_seconds': sum(row['seconds'] for row in records),
                          'shared_parameters': {'encoder': sum(p.numel() for p in encoder.parameters()),
                                                'bridge': sum(p.numel() for p in bridge.parameters()),
                                                'trainable_adapter': sum(p.numel() for p in adapters)}}
        write_json(directory / 'summary.json', summaries[arm])
        print(json.dumps({'arm': arm, 'summary': summaries[arm]}), flush=True)
        del reader, encoder, bridge, optimizer, parameters, adapters, saved, restored_encoder, restored_bridge, states, payload, original, restored, parameter
        gc.collect()
        torch.cuda.empty_cache()
    if sources() != source_hashes or file_sha256(args.declaration) != json.loads((args.output / 'protocol.json').read_text())['declaration_sha256']:
        raise ValueError('execution sources or declaration changed')
    validate_execution(execution_record(device), expected=runtime)
    passed = summaries['adapted']['readiness_passed'] and summaries['frozen']['full_text_passed']
    write_json(args.output / 'report.json', {'status': 'complete', 'arms': summaries, 'readiness_passed': passed,
               'decision': 'ready_for_separate_development_protocol' if passed else 'stop_this_readiness_configuration',
               'evidence_kind': 'consumed_training_history_fit_only', 'generalization_tested': False})
    files = {str(p.relative_to(args.output)): file_sha256(p) for p in args.output.rglob('*') if p.is_file()}
    write_json(args.output / 'complete.json', {'kind': 'adapted_readout_readiness_complete_v1', 'files': files})


if __name__ == '__main__':
    main()
