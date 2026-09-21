"""Freeze the multi-task storage study without reading test answers."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from transformers import AutoTokenizer

from tinymem.data.wikitext import load_wikitext_parquet
from tinymem.reader.pretrained import verify_qwen_snapshot
from tinymem.studies.frontier.data import (
    dictionary_from_training,
    load_questions,
    select_noise,
    split_questions,
)


def sha(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def snapshot_source(study: Path, repo: Path) -> None:
    """Copy the package source and the frontier scripts the study will execute.

    Every stage runs from the study directory with ``source/src`` on the path,
    so the sealed hashes bind the code that ran, not the working checkout.
    """
    for path in sorted((repo / 'src').rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        target = study / 'source' / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for path in sorted(Path(__file__).resolve().parent.glob('frontier_*')):
        if path.suffix in ('.py', '.slurm'):
            shutil.copyfile(path, study / path.name)


def prepare(study: Path, repo: Path) -> None:
    snapshot_source(study, repo)
    target = study / 'inputs'
    target.mkdir(exist_ok=False)
    write(target / 'model_snapshot.json', verify_qwen_snapshot(repo / 'data/raw/pretrained/qwen3-1.7b'))
    source = repo / 'data/raw/tasks_1-20_v1-2/en-10k'
    train, validation = split_questions(load_questions(source, 'train'), seed=2026091802)
    tokenizer = AutoTokenizer.from_pretrained(repo / 'data/raw/pretrained/qwen3-1.7b',
                                             local_files_only=True, trust_remote_code=False)
    noise, excluded, sources = {}, frozenset(), {}
    for split, count in (('train', 1024), ('validation', 256), ('test', 256)):
        path = repo / f'data/raw/wikitext2/{split}.parquet'
        pool = select_noise(load_wikitext_parquet(path, split=split).rows, tokenizer,
                            count=count, seed=2026091803, excluded=excluded)
        noise[split] = pool
        excluded = excluded | frozenset(pool)
        sources[f'wikitext2/{split}.parquet'] = sha(path)
    for task in range(1, 6):
        for split in ('train', 'test'):
            path = next(source.glob(f'qa{task}_*_{split}.txt'))
            # Copy test files and bind their bytes, but do not parse their answers here.
            shutil.copyfile(path, target / path.name)
            sources[path.name] = sha(path)
        path = repo / f'data/raw/babilong/qa{task}/1k.json'
        name = f'babilong-qa{task}-1k.json'
        shutil.copyfile(path, target / name)
        sources[name] = sha(path)
    dictionary = tuple(sorted(set(dictionary_from_training(train)) | set(noise['train'])))
    records = sorted({s for q in (*train, *validation) for s in q.records}
                     | set(noise['train']) | set(noise['validation']))
    record_lengths = [len(tokenizer.encode(s + '\n', add_special_tokens=False)) for s in records]
    if max(record_lengths) > 64:
        raise ValueError('a source record exceeds the declared feature chunk length')
    dataset = {'training': [asdict(q) for q in train], 'validation': [asdict(q) for q in validation],
               'noise': noise, 'dictionary': dictionary, 'feature_records': records,
               'sources': sources,
               'counts': {'training': dict(Counter(q.task for q in train)),
                          'validation': dict(Counter(q.task for q in validation)),
                          'feature_records': len(records), 'max_feature_tokens': max(record_lengths)},
               'context_overlap_train_validation': len({q.records for q in train} & {q.records for q in validation}),
               'test_answers_parsed': False}
    write(target / 'dataset.json', dataset)
    print(json.dumps(dataset['counts']), flush=True)


def frozen_files(study: Path) -> dict[str, str]:
    files = {str(p.relative_to(study)): sha(p)
             for root in (study / 'source/src', study / 'inputs')
             for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    files.update({p.name: sha(p) for p in sorted(study.glob('frontier_*.py'))})
    files.update({p.name: sha(p) for p in sorted(study.glob('frontier_*.slurm'))})
    return files


def freeze(study: Path) -> None:
    files = frozen_files(study)
    cells = [{'kind': 'learned', 'budget': b, 'seed': s}
             for s in (27001, 27002, 27003) for b in (64, 256, 1024)]
    cells += [{'kind': 'text', 'budget': None, 'seed': s} for s in (27001, 27002, 27003)]
    protocol = {
        'version': 1, 'files': files, 'cells': cells,
        'question': 'At equal serialized per-history storage, can answer-trained recurrent memory retain and update useful facts as well as strong explicit text stores?',
        'architecture': {'reader': 'Qwen/Qwen3-1.7B', 'revision': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
                         'feature_encoder': 'unadapted frozen base, current record only, reset positions, no cache',
                         'writer': 'token cross-attention, slot self-attention, gated residual, fixed int8 scale',
                         'memory_width': 32, 'hidden_width': 128, 'quantize': 'every write, int8/127',
                         'supervision': 'native answer tokens only; fresh writer, bridge and Q/V LoRA',
                         'history_state': 'packed slots only, no flags, positions, scales, buffers or retained KV'},
        'settings': {'epochs': 3, 'batch_size': 8, 'writer_lr': .001, 'reader_lr': .0003,
                     'weight_decay': .01, 'schedule_seed': 2026091804, 'noise_seed': 2026091805,
                     'budgets': [64, 256, 1024], 'probe_per_task': 20, 'lora_rank': 8,
                     'curriculum': 'epoch1 clean; epoch2 alternating clean/light; epoch3 clean/light/heavy',
                     'text_condition_schedule': 'independent fixed RNG, codec and budget decoupled from noise'},
        'evaluation': {'official_tasks': ['qa1', 'qa2', 'qa3', 'qa4', 'qa5'],
                       'levels': [0, 2], 'max_new_tokens': 8, 'batch_size': 8,
                       'primary': 'all official test questions, clean and declared distractor variants, macro task accuracy by bytes',
                       'controls': 'zero and other-history memory on fixed first 100 questions per task; full text clean reference',
                       'test_policy': 'fixed final checkpoint, all cells retained, no accuracy gate or test-based selection',
                       'transfer': 'BABILong qa1-qa5 1k full raw text, after fixed training, separate previously exposed benchmark result',
                       'transfer_chunking': 'all characters retained, whitespace boundaries, at most64 native tokens including delimiter',
                       'transfer_read_batch': 'one history: one learned condition or all nine explicit conditions'},
        'limits': ['Compact symbolic state remains a privileged reference; a text-store win is not a universal explicit-storage win.',
                   'Public benchmark examples have prior project exposure and possible pretraining exposure.',
                   'The text reader is fitted jointly across codec and budget conditions; each learned budget has a separate reader.',
                   'Only bytes retained between writes and reads are matched. Shared weights/dictionaries and compute are reported separately.'],
        'no_test_answers_in_preparation': True,
    }
    path = study / 'protocol.json'
    if path.exists():
        raise FileExistsError('do not overwrite a frozen protocol')
    write(path, protocol)
    print('protocol_sha256=' + sha(path), flush=True)


# Files an amendment may change. Fitting code (studies/frontier/fit and training,
# quantized_slots, reader/adapter) and every input stay bound to the parent
# seals. This file may change because the data it produced is hashed under
# inputs/ and cannot be altered by editing the generator afterwards.
AMENDABLE = frozenset({'frontier_prepare.py', 'frontier_run.py', 'frontier_score.py',
                       'frontier_transfer.py', 'frontier_report.py',
                       'source/src/tinymem/studies/frontier/eval.py'})


# Frozen files whose change invalidates a stage's seal. Fitting stages depend on
# every file outside AMENDABLE, which an amendment can never change, so their
# seals stay valid through any chain. frontier_run.py checks only the immediate
# parent; under a chained amendment its evaluate and transfer stages fail closed.
STAGE_DEPENDENCIES = {
    'training': frozenset(),
    'evaluation': frozenset({'frontier_run.py', 'frontier_score.py',
                             'source/src/tinymem/studies/frontier/eval.py'}),
    'transfer': frozenset({'frontier_run.py', 'frontier_transfer.py',
                           'source/src/tinymem/studies/frontier/eval.py'}),
}


def lineage(study: Path, protocol: dict) -> list[dict]:
    """Return the amendment chain, newest first, verifying each archived parent."""
    chain, current = [], protocol
    while 'amends' in current:
        record = current['amends']
        archived = study / record['parent_protocol_file']
        if sha(archived) != record['parent_protocol_sha256']:
            raise ValueError(f'archived parent protocol differs: {archived.name}')
        chain.append(record)
        current = json.loads(archived.read_text())
    return chain


def accepted_seal_hashes(study: Path, protocol: dict, stage: str, cell: int) -> frozenset[str]:
    """Protocol hashes a sealed ``stage/cell`` may carry under ``protocol``.

    An ancestor hash is accepted when no file the stage depends on changed in
    any later amendment, or when a later amendment records an explicit waiver
    for this stage and cell under that ancestor.
    """
    dependencies = STAGE_DEPENDENCIES[stage]
    chain = lineage(study, protocol)
    waived = {w['protocol_sha256'] for record in chain for w in record.get('unaffected_seals', ())
              if w['stage'] == stage and cell in w['cells']}
    accepted, changed = {sha(study / 'protocol.json')}, set()
    for record in chain:
        changed |= set(record['changed_files'])
        parent = record['parent_protocol_sha256']
        if not (changed & dependencies) or parent in waived:
            accepted.add(parent)
    return frozenset(accepted)


def amend(study: Path, reason: str, unaffected: tuple[dict, ...] = ()) -> None:
    """Re-freeze evaluation code after fitting, without touching any fitted seal.

    The parent protocol is renamed, not deleted. The new protocol records what
    changed. Any file that fitting depended on must be byte-identical.
    ``unaffected`` records seals that stay valid although a file their stage
    depends on changed, with the reason; the report discloses them.
    """
    path = study / 'protocol.json'
    parent = json.loads(path.read_text())
    for stage in ('features', 'preflight'):
        if not (study / stage / 'complete.json').exists():
            raise ValueError(f'{stage} must be complete before an evaluation-only amendment')
    for cell in range(len(parent['cells'])):
        if not (study / 'training' / str(cell) / 'complete.json').exists():
            raise ValueError('all training cells must be complete before an evaluation-only amendment')
    files = frozen_files(study)
    changed = {name for name in set(files) | set(parent['files'])
               if files.get(name) != parent['files'].get(name)}
    if not changed:
        raise ValueError('nothing changed; do not amend')
    if changed - AMENDABLE:
        raise ValueError(f'amendment touches fitting-bound files: {sorted(changed - AMENDABLE)}')
    parent_sha = sha(path)
    known = {parent_sha} | {r['parent_protocol_sha256'] for r in lineage(study, parent)}
    for waiver in unaffected:
        if waiver['stage'] not in STAGE_DEPENDENCIES or waiver['protocol_sha256'] not in known:
            raise ValueError('waiver must name a known stage and an ancestor protocol hash')
        if not waiver['cells'] or not waiver['reason']:
            raise ValueError('waiver must name cells and a reason')
        for cell in waiver['cells']:
            seal = study / waiver['stage'] / str(cell) / 'complete.json'
            if not seal.exists() or json.loads(seal.read_text())['protocol_sha256'] != waiver['protocol_sha256']:
                raise ValueError(f'waiver names {waiver["stage"]}/{cell}, which is not sealed under that hash')
    archived = study / f'protocol.parent.{parent_sha[:12]}.json'
    if archived.exists():
        raise FileExistsError('parent protocol archive already exists')
    record = {'parent_protocol_sha256': parent_sha, 'parent_protocol_file': archived.name,
              'changed_files': sorted(changed), 'reason': reason,
              'fitting_reused': 'features, preflight, and all training seals carry an ancestor hash'}
    if unaffected:
        record['unaffected_seals'] = [dict(w) for w in unaffected]
    protocol = {**parent, 'files': files, 'amends': record}
    path.rename(archived)
    write(path, protocol)
    print(json.dumps({'protocol_sha256': sha(path), 'parent_protocol_sha256': parent_sha,
                      'changed_files': sorted(changed)}), flush=True)


def parse_waiver(text: str) -> dict:
    """``stage:cells:protocol_sha256:reason`` with cells as ``1,2`` or ``9-11``."""
    stage, cells, digest, reason = text.split(':', 3)
    parsed = []
    for part in cells.split(','):
        first, _, last = part.partition('-')
        parsed.extend(range(int(first), int(last or first) + 1))
    return {'stage': stage, 'cells': parsed, 'protocol_sha256': digest, 'reason': reason}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'freeze', 'amend'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--reason')
    parser.add_argument('--unaffected', action='append', default=[], type=parse_waiver,
                        help='stage:cells:protocol_sha256:reason for a seal that stays valid')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.study, args.repo)
    elif args.stage == 'freeze':
        freeze(args.study)
    else:
        if not args.reason:
            parser.error('amend requires --reason')
        amend(args.study, args.reason, tuple(args.unaffected))
