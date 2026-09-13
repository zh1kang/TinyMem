"""Fixed conflicting-history fit gated by independently verified full-text qualification."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from scripts.profile_adapted_readout import frozen_hash, write_json
from scripts.fit_adapted_readout import SETTINGS as ORIGINAL_SETTINGS
from scripts.qualify_paired_readout import require_qualification, sources as qualification_sources
from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.adapted_readout import configure_read_adapter, frozen_history_features, train_adapted_step
from tinymem.research.paired_readout_data import build_pairs, encode_pairs
from tinymem.research.paired_readout_evaluation import evaluate_pairs, summarize_pairs, verify_state_payload
from tinymem.research.readout_evaluation import evaluate_full_text
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.study_runtime import REPOSITORY, allocation_metrics, execution_record, prepare_device, synchronize, validate_execution
from tinymem.research.update_protocol import file_sha256, load_development_data, load_shared_reader, shared_reader_identity


SETTINGS = {**ORIGINAL_SETTINGS, 'minimum_control_gap': 0.4,
            'swap_rule': 'within_source_pair', 'swapped_donor_known_gate': 0.95}


def sources():
    names = ('src/tinymem/research/paired_readout_evaluation.py', 'scripts/fit_paired_readout.py')
    return {**qualification_sources(), **{name: file_sha256(REPOSITORY / name) for name in names}}


def require_verified_qualification(directory, verification_path, declaration):
    report = require_qualification(directory, report_sha256=declaration['qualification_report_sha256'])
    verification = json.loads(verification_path.read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    if (file_sha256(verification_path) != declaration['qualification_verification_sha256']
            or file_sha256(directory / 'complete.json') != declaration['qualification_complete_sha256']
            or verification.get('verified') is not True or verification.get('qualified') is not True
            or verification.get('report_sha256') != declaration['qualification_report_sha256']
            or verification.get('declaration_sha256') != declaration['qualification_declaration_sha256']
            or protocol['declaration_sha256'] != declaration['qualification_declaration_sha256']
            or protocol['reader'] != declaration['reader']
            or protocol['data_protocol_sha256'] != declaration['data_protocol_sha256']):
        raise ValueError('independent positive qualification evidence differs from declaration')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--qualification-verification', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = prepare_device('cuda')
    data, identity = load_development_data(args.data), shared_reader_identity()
    source_hashes, runtime = sources(), execution_record(device)
    declaration = json.loads(args.declaration.read_text())
    qualification = require_verified_qualification(args.qualification, args.qualification_verification, declaration)
    if (declaration['settings'] != SETTINGS or declaration['source_sha256'] != source_hashes
            or declaration['portable_source_sha256'] != runtime['source_sha256']
            or declaration['data_protocol_sha256'] != data.protocol_sha256 or declaration['reader'] != identity
            or declaration['runtime'] != runtime['runtime']):
        raise ValueError('launch declaration differs from execution inputs')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', {'kind': 'paired_readout_fit_v1', 'settings': SETTINGS,
               'declaration_sha256': file_sha256(args.declaration), 'reader': identity,
               'qualification_report_sha256': file_sha256(args.qualification / 'report.json'),
               'qualification_verification_sha256': file_sha256(args.qualification_verification),
               'qualification_complete_sha256': file_sha256(args.qualification / 'complete.json'),
               'qualification_declaration_sha256': declaration['qualification_declaration_sha256'],
               'data_protocol_sha256': data.protocol_sha256, 'source_sha256': source_hashes,
               'execution': runtime, 'development_scored': False, 'confirmation_opened': False,
               'evidence_kind': 'paired_training_history_fit_only'})
    cached, rows, initial_reader = {}, None, None
    summaries = {}
    for arm in ('frozen', 'adapted'):
        reader = load_shared_reader(identity, device)
        current_reader = _reader_hash(reader)
        if initial_reader is None:
            initial_reader = current_reader
            if initial_reader != qualification['reader_sha256']:
                raise ValueError('fit reader differs from independently qualified reader')
            rows = encode_pairs(reader, build_pairs(data.train))
            if (json.loads(json.dumps([asdict(row) for row in rows]))
                    != json.loads((args.qualification / 'encodings.json').read_text())):
                raise ValueError('paired fit inputs differ from qualified native encodings')
            if [row.history_id for row in rows] != declaration['history_ids']:
                raise ValueError('paired history order differs from declaration')
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
            state_payload_hashes = verify_state_payload(values, payload)
            states = {row.history_id: LatentSlotState(payload[row.history_id + '.values'], payload[row.history_id + '.valid']) for row in rows}
            predictions = evaluate_pairs(reader, restored_bridge, states, rows)
            predictions.extend(evaluate_full_text(reader, rows))
        with (directory / 'predictions.jsonl').open('x') as handle:
            for row in predictions:
                handle.write(json.dumps(row, sort_keys=True) + '\n')
        summaries[arm] = {**summarize_pairs(predictions, rows), 'reader_initial_sha256': initial_reader,
                          'state_payload_sha256': state_payload_hashes, 'state_serialization_exact': True,
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
    passed = summaries['adapted']['binding_passed'] and summaries['frozen']['full_text_passed']
    write_json(args.output / 'report.json', {'status': 'complete', 'arms': summaries, 'binding_passed': passed,
               'decision': 'ready_for_separate_unseen_history_protocol' if passed else 'stop_this_paired_fit_configuration',
               'evidence_kind': 'paired_training_history_fit_only', 'generalization_tested': False})
    files = {str(p.relative_to(args.output)): file_sha256(p) for p in args.output.rglob('*') if p.is_file()}
    write_json(args.output / 'complete.json', {'kind': 'paired_readout_fit_complete_v1', 'files': files})


if __name__ == '__main__':
    main()
