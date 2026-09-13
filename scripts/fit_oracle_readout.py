"""Read a fixed known variant bit without any learned writer or history features."""
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
from scripts.fit_paired_readout import SETTINGS as PAIRED_SETTINGS, require_verified_qualification
from scripts.fit_paired_readout import sources as paired_sources
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.oracle_readout import oracle_variant_state, train_oracle_step, summarize_oracle
from tinymem.research.paired_readout_data import build_pairs, encode_pairs
from tinymem.research.paired_readout_evaluation import evaluate_pairs, verify_state_payload
from tinymem.research.readout_evaluation import evaluate_full_text
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.study_runtime import REPOSITORY, allocation_metrics, execution_record, prepare_device, synchronize, validate_execution
from tinymem.research.update_protocol import file_sha256, load_development_data, load_shared_reader, shared_reader_identity


SETTINGS = {**PAIRED_SETTINGS, 'writer': 'none', 'state_coordinate': [0, 0, 0],
            'variant_values': {'a': -1.0, 'b': 1.0}, 'other_values': 0.0,
            'state_trainable': False, 'normal_validity': [True, True],
            'initial_bridge': 'paired_fit_untrained_bridge',
            'primary_gate': 'known_bit_readout', 'full_text_gate_separate': True}


def sources():
    names = ('src/tinymem/research/oracle_readout.py', 'scripts/fit_oracle_readout.py')
    return {**paired_sources(), **{name: file_sha256(REPOSITORY / name) for name in names}}


def final_decision(summaries):
    adapted = summaries['adapted']
    if not adapted['known_bit_readout_passed']:
        return 'stop_this_oracle_readout_configuration'
    if not adapted['binary_qa_passed']:
        return 'known_bit_readable_absent_qa_failed'
    if not adapted['full_text_passed'] or not summaries['frozen']['full_text_passed']:
        return 'known_bit_readable_full_text_retention_failed'
    return 'oracle_read_path_ready_for_separate_design'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--qualification-verification', type=Path, required=True)
    parser.add_argument('--initial-bridge', type=Path, required=True)
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
    if file_sha256(args.initial_bridge) != declaration['initial_bridge_sha256']:
        raise ValueError('untrained bridge source differs from declaration')
    initial = load_file(str(args.initial_bridge))
    initial_bridge = {k.removeprefix('bridge.'): v for k, v in initial.items() if k.startswith('bridge.')}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', {'kind': 'oracle_readout_fit_v1', 'settings': SETTINGS,
               'declaration_sha256': file_sha256(args.declaration), 'reader': identity,
               'qualification_report_sha256': file_sha256(args.qualification / 'report.json'),
               'qualification_verification_sha256': file_sha256(args.qualification_verification),
               'qualification_complete_sha256': file_sha256(args.qualification / 'complete.json'),
               'qualification_declaration_sha256': declaration['qualification_declaration_sha256'],
               'initial_bridge_sha256': file_sha256(args.initial_bridge),
               'data_protocol_sha256': data.protocol_sha256, 'source_sha256': source_hashes,
               'execution': runtime, 'development_scored': False, 'confirmation_opened': False,
               'evidence_kind': 'privileged_variant_bit_training_fit_only'})
    rows, initial_reader, oracle_payload = None, None, None
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
            oracle_payload = {}
            for row in rows:
                state = oracle_variant_state(row.history_id.rsplit(':', 1)[1], torch.device('cpu'))
                oracle_payload[row.history_id + '.values'] = state.values
                oracle_payload[row.history_id + '.valid'] = state.valid
            save_file(oracle_payload, str(args.output / 'oracle_states.safetensors'))
        elif initial_reader != current_reader:
            raise ValueError('paired reader initial values differ')
        directory = args.output / arm
        directory.mkdir()
        adapters = configure_read_adapter(reader, trainable=arm == 'adapted')
        frozen_before = frozen_hash(reader)
        bridge = ReadoutBridge(2048, 'affine')
        bridge.load_state_dict(initial_bridge)
        for name, tensor in bridge.state_dict().items():
            if not torch.equal(tensor, initial_bridge[name]):
                raise ValueError('loaded bridge initial values differ')
        bridge.to(device)
        payload = load_file(str(args.output / 'oracle_states.safetensors'), device=str(device))
        verify_state_payload(oracle_payload, payload)
        states = {row.history_id: LatentSlotState(payload[row.history_id + '.values'], payload[row.history_id + '.valid']) for row in rows}
        parameters = [*bridge.parameters(), *adapters]
        optimizer = torch.optim.AdamW(parameters, lr=SETTINGS['lr'], weight_decay=SETTINGS['weight_decay'],
                                      betas=tuple(SETTINGS['betas']), eps=SETTINGS['eps'])
        save_file({'bridge.' + k: v for k, v in bridge.state_dict().items()}, str(directory / 'initial.safetensors'))
        if arm == 'adapted' and file_sha256(directory / 'initial.safetensors') != file_sha256(args.output / 'frozen/initial.safetensors'):
            raise ValueError('paired bridge initialization differs')
        records = []
        with (directory / 'metrics.jsonl').open('x') as handle:
            for step in range(SETTINGS['steps_per_arm']):
                row = rows[step % len(rows)]
                synchronize(device)
                started = time.perf_counter()
                metric = train_oracle_step(reader, bridge, states[row.history_id], row.before_ids,
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
        save_file({'bridge.' + k: v for k, v in bridge.state_dict().items()}, str(directory / 'final.safetensors'))
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
        restored_bridge = ReadoutBridge(2048, 'affine').to(device)
        restored_bridge.load_state_dict({k.removeprefix('bridge.'): v for k, v in saved.items()})
        with torch.inference_mode():
            values = {}
            for row in rows:
                state = states[row.history_id]
                if not torch.equal(bridge(state), restored_bridge(state)):
                    raise ValueError('bridge checkpoint reload differs')
                values[row.history_id + '.values'] = state.values.cpu()
                values[row.history_id + '.valid'] = state.valid.cpu()
            verify_state_payload(oracle_payload, values)
            save_file(values, str(directory / 'states.safetensors'))
            payload = load_file(str(directory / 'states.safetensors'), device=str(device))
            state_payload_hashes = verify_state_payload(oracle_payload, payload)
            states = {row.history_id: LatentSlotState(payload[row.history_id + '.values'], payload[row.history_id + '.valid']) for row in rows}
            predictions = evaluate_pairs(reader, restored_bridge, states, rows)
            predictions.extend(evaluate_full_text(reader, rows))
        with (directory / 'predictions.jsonl').open('x') as handle:
            for row in predictions:
                handle.write(json.dumps(row, sort_keys=True) + '\n')
        summaries[arm] = {**summarize_oracle(predictions, rows), 'reader_initial_sha256': initial_reader,
                          'state_payload_sha256': state_payload_hashes, 'state_serialization_exact': True,
                          'reader_final_sha256': trained_reader, 'frozen_parameters_unchanged': True,
                          'checkpoint_reload_exact': True, 'completed_steps': len(records),
                          'training_seconds': sum(row['seconds'] for row in records),
                          'shared_parameters': {'encoder': 0,
                                                'bridge': sum(p.numel() for p in bridge.parameters()),
                                                'trainable_adapter': sum(p.numel() for p in adapters)}}
        write_json(directory / 'summary.json', summaries[arm])
        print(json.dumps({'arm': arm, 'summary': summaries[arm]}), flush=True)
        del reader, bridge, optimizer, parameters, adapters, saved, restored_bridge, states, payload, state, parameter
        gc.collect()
        torch.cuda.empty_cache()
    if sources() != source_hashes or file_sha256(args.declaration) != json.loads((args.output / 'protocol.json').read_text())['declaration_sha256']:
        raise ValueError('execution sources or declaration changed')
    if file_sha256(args.initial_bridge) != declaration['initial_bridge_sha256']:
        raise ValueError('initial bridge source changed during execution')
    validate_execution(execution_record(device), expected=runtime)
    decision = final_decision(summaries)
    passed = decision == 'oracle_read_path_ready_for_separate_design'
    write_json(args.output / 'report.json', {'status': 'complete', 'arms': summaries, 'readiness_passed': passed,
               'known_bit_readout_passed': summaries['adapted']['known_bit_readout_passed'],
               'binary_qa_passed': summaries['adapted']['binary_qa_passed'],
               'decision': decision,
               'evidence_kind': 'privileged_variant_bit_training_fit_only', 'generalization_tested': False})
    files = {str(p.relative_to(args.output)): file_sha256(p) for p in args.output.rglob('*') if p.is_file()}
    write_json(args.output / 'complete.json', {'kind': 'oracle_readout_fit_complete_v1', 'files': files})


if __name__ == '__main__':
    main()
