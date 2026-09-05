# Reliable memory updates: single-budget design

Status: data construction and streaming tokenization implemented and verified. New training, evaluation, metrics, and reporting runners are still pending. No new model training or scientific accuracy result is reported here.

## Architecture and question

Use the existing query-independent recurrent writer and shared frozen Qwen3-1.7B reader. The persistent state remains two width-eight FP32 vectors and two validity bytes: **66 bytes**. Query pooling and contextual mean/FIFO are independently trained methods, not interventions on the same checkpoint.

Ask whether a new fact or correction damages other stored facts. Keep full recurrent gradients and answer several questions from the same state. Do not retain raw history or KV state alongside the memory.

## Paired data

`src/tinymem/data/memory_updates.py` reuses `make_opaque_qa1_world` for a four-chunk, eight-binding starting history. It branches into exactly one event of each kind:

| State | Event | Known query keys | Absent query keys |
|---|---|---:|---:|
| Before | None | 8 | 2 |
| Addition | Introduce the ninth entity | 9 | 1 |
| Repetition | Repeat an exact current fact | 8 | 2 |
| Correction | Change one existing binding's room | 8 | 2 |

The same ten entities are queried at every state. The tenth is always absent. Repetition and correction target the same randomly selected existing entity; addition and correction use the same new room. Corrections cannot preserve the previous value. All other bindings remain unchanged.

The query index, event label, source identity, and gold answer are metadata, never writer input. Reads receive one question at a time, not the other questions or their ordinal positions. The newcomer is absent in three of four states; report state-specific absent and known accuracy rather than hiding that prevalence in one overall score.

This minimal study measures **one additional write**, not robustness to arbitrarily long update sequences. Additions increase live information, unlike repetitions and corrections. Their losses cannot be attributed solely to preservation failure.

## Independent validation and grouping

Generation uses the existing opaque-world labels and explicit dictionary transitions. A separate strict regular-expression replay reconstructs every before/after answer from visible facts, checking the categories, queries, context identities, event semantics, and complete question coverage. Tests additionally replay using a word-based implementation.

All branches and questions from a world share a source-group assignment. The split validator rejects repeated groups, source episodes, exact source-context hashes, and episode identities both within and across splits. The builder checks each selected representative against the raw source before renaming. Connected-group metadata contains membership sets, not index-aligned episode/context lists.

The source selection preserves the old **256 train / 32 development** world pairings and their consumed-data roles. The new **64 confirmation** worlds use 128 of the remaining **178 eligible native-study-unused groups**, selected by deterministic input-only ranking. They exclude all 832 old association-study groups, the original native holdout, and the prior consumed-source exclusions. They are not claimed to be project-wide untouched or pretraining-clean.

The old confirmation dataset and predictions are never opened. The builder verifies the frozen study trust anchor, reference/reserve manifests, selected representatives, raw training source, and **31 exclusion files**. Conflicting hash declarations fail rather than overwriting one another. Invalid labels or insufficient fresh groups abort the build; there is no answer-dependent replacement or silent filtering.

## Streaming prefix ownership

Every history chunk owns a trailing `\n\n`, including the last chunk before an event. Thus the old state is formed from exactly the same token sequence in the before measurement and every after branch.

`src/tinymem/research/update_encoding.py` accepts only history chunks, not queries or answers. It checks concatenated native-token equality and rejects cross-boundary merges, embedded ambiguous separators, empty chunks, and overlength writes. The existing frozen `encode_history_chunks` uses a different final-separator rule and is deliberately unchanged.

For **all 288 training/development worlds**, the pinned local Qwen tokenizer passes **11,520 native prompt checks**. The largest write is **133 tokens**, below the 512-token write limit. These checks load the tokenizer and model configuration, **not model weights**. Confirmation tokenization must be validated by the future frozen evaluation runner without filtering or changing examples.

## Frozen design versus launch protocol

`configs/memory_update_study.json` fixes the event design, one state budget, comparator family, seeds, outcome families, and interpretation gates. Its status explicitly says it is **not a training-launch protocol**. Before training, the runner must bind an exact objective/schedule, source/data hashes, and one qualified reader. No checkpoint selection from confirmation is allowed.

The shared reader must pass full-context known and missing development accuracy of at least 95% at **each** state, including the nine-live-entity addition state. Text qualification remains distinct from compressed-memory competence.

A competent-preservation claim requires each learned seed to reach at least 95% known and missing accuracy on the before state in development. Otherwise still report all update results, labeled **competence-limited**. Do not silently enlarge memory or the model after failure.

Report conditional forgetting with explicit denominators alongside unconditional accuracy and all correctness transitions. Stale correction answers, absent false answers, and known false abstentions are distinct. Use paired-history uncertainty and separate optimization-seed variability. The study is descriptive, not an omnibus superiority test with undisclosed multiple comparisons.

## Reproduction

From the repository root, with existing source/exclusion artifacts staged:

```bash
.venv/bin/python -m scripts.prepare_memory_updates \
  --output artifacts/predictions/memory_update_data_20260905_v2
```

The output must not already exist. Use a fresh path to reproduce; output location does not change the dataset identity. Generated histories and predictions remain outside Git.

Current verified artifact:

- directory: `artifacts/predictions/memory_update_data_20260905_v2`;
- design SHA-256: `5e24629a1bf5d4f0172260b5b72188725845369bc5b862a96d5fcf521c79d168`;
- data protocol SHA-256: `5e0879c6891aa180a75364f01de738bed9784102cad35cae2b34a2770c5b39e6`.

The earlier unlaunched `memory_update_data_20260905` directory is retained only as a development artifact. Its data hashes match the verified build, but its older provenance contract must not be used for training.

Run focused data tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts='' -q -p no:cacheprovider \
  tests/test_memory_updates.py tests/test_prepare_memory_updates.py tests/test_update_encoding.py \
  tests/test_opaque_qa1.py tests/test_memory_prompt.py
```

Combined with the existing native-memory regression suites: **255 tests pass**. Independent review prompted the explicit trust anchor and conflict checks. The 24 frozen sources, six old checkpoints, original protocol, and unrelated decoder patch remain unchanged. Nothing was submitted to Della or pushed.
