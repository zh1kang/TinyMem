# TinyMem

TinyMem studies **reliable updates to query-independent memory under a fixed persistent-byte budget**.

> Can a learned memory incorporate a new or corrected fact without damaging other stored facts, and how does it compare with compact explicit storage?

The broader motivation is the accuracy–storage trade-off of learned compression. A positive learned-memory result is not required: a reproducible explanation of where updating, preservation, or readout fails is a useful outcome.

## Active scope

```text
old state + current history chunk → existing recurrent writer → bounded state
                                                               ↓
                                         several later queries + shared reader
```

- Keep the existing Qwen3-1.7B native-token writer/reader path.
- Start the update study at 66 persistent bytes: two width-eight FP32 slots and two validity bytes.
- Compare learned query pooling, independently trained contextual mean/FIFO, and strong compact fact-retention references.
- Measure updated-fact accuracy, unchanged-fact preservation, forgetting, stale answers, and absent-entity errors before and after controlled events.
- Keep full recurrent gradients, history-level splits, query-blind writes, and one shared state for several questions.
- Add another budget or change the architecture only when a specific diagnostic justifies it.

See [the specification](PROJECT_SPEC.md), [active plan](docs/research_plan.md), [paired-update data design](docs/memory_update_study.md), and [handoff](docs/cursor_handoff.md). Older implemented models and their tests are historical references, not additional active research requirements.

## Measured evidence—not the new study's results

The completed opaque-association study has six training runs: two writers, three seeds, 1,000 updates each, and 66-byte states. Development means on the same 32 worlds are:

| Method | Known answers | Absent answers |
|---|---:|---:|
| Query pooling | 14.32% | 17.71% |
| Contextual mean/FIFO | 12.76% | 9.38% |

The paired known-answer gap changes sign across seeds. The qualified full-text reader scores 253/256 known and 32/32 absent answers on that development format. Text competence does not establish compressed-state competence.

The existing confirmation run was interrupted. Its full report, final generated training-fit measurements, and current-task fixed-readout oracle remain pending in the evidence inspected at the start of the update study. Do not tune on partial confirmation outputs or retrain the six completed checkpoints.

The first new step is to establish immediate binding competence before interpreting later errors as forgetting. The training-fit and fixed-readout diagnostic tools already exist; their execution is separate from implementation validation.

## Setup and verification

Python 3.11 or newer is required. In the existing uv-managed environment:

```bash
uv pip install --python .venv/bin/python -e '.[dev,research]'
.venv/bin/python -m pytest
.venv/bin/python -m scripts.opaque.smoke --check-only
```

The input preflight requires the locally staged, hash-bound study artifacts and pinned Qwen snapshot. It checks inputs without loading the model or inspecting confirmation answers.

Full-size training and evaluation run through [the Della runbook](docs/della.md). Do not run Qwen on a login node. Scripts use explicit devices and fresh output directories, with no silent fallback or partial-result merging.

## Evidence and boundaries

- `src/tinymem/research/`: native reader, recurrent writer integration, losses, and fixed-readout diagnostics.
- `src/tinymem/memory/`: bounded states and retention implementations.
- `src/tinymem/data/`: source grouping, controlled histories, and symbolic replay.
- `src/tinymem/evaluation/`: metrics and paired-history comparison.
- `scripts/opaque/`: the completed study's portable execution tools.
- `docs/decision_log.md`: chronological decisions and verification.
- `data/` and `artifacts/`: local data and experimental evidence; weights and generated results are not committed.

Persistent state, shared parameters, temporary computation, and training memory are separate costs. Full-context reading is a capability reference, not a byte-matched competitor. Fingerprint lookup is a separately labeled handcrafted association reference, not a Qwen reader.

All questions from a history stay together in data splits and statistical analysis. Previously consumed data cannot become untouched through renaming. Existing reserved and external evaluation answers remain outside development.

The working-tree educational decoder edit is unrelated to the native study and must not be overwritten or accidentally included in a research commit. Use a clean committed export for cluster execution.
