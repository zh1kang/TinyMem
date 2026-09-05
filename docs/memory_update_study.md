# Reliable memory updates: single-budget design

Status: data construction, streaming tokenization, paired reliability metrics, training/evaluation, reader qualification, launch/provenance integration, and command-line runners are implemented and verified. Count-aware multi-seed aggregation/reporting and Della orchestration are also implemented and locally verified. Actual cluster execution and full-size scientific results remain pending. No full-size update-study training or scientific accuracy result is reported here. The prior association study's newly supplied negative outcomes are recorded separately in [association confirmation results](association_confirmation_results.md), explicitly labeled user-reported until artifact verification.

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

## Paired measurement contract

`src/tinymem/evaluation/memory_updates.py` scores one validated history and one fixed checkpoint. Pass exactly 40 `UpdatePrediction(case_id, prediction)` records: ten for the before state and ten for each branch. Records may arrive in any order; missing, duplicate, foreign-history, and wrong-stage IDs fail before scoring. Gold labels come from the independently replayed dataset, not prediction records.

Use `score_update_episode(episode, records).to_dict()` for JSON. Each rate retains its integer numerator/denominator and a derived value. `0/0` becomes JSON `null`, not zero or NaN. The result's mappings are read-only, and serialization returns detached copies.

State-level metrics keep overall, known, and absent accuracy separate. Their denominators are 10, 8/9, and 2/1 respectively. The before state is counted once; its reuse in event contrasts does not create extra observations.

For each branch, the target has before/after accuracy. **Update accuracy** applies only to addition and correction; repetition is evaluated with target after-accuracy, not called a successful change. **Stale-answer rate** applies only to corrected targets and compares the output with the immediately superseded gold value. It is not restricted to targets answered correctly before. Unknown-before additions and unchanged repetitions have undefined stale rates, not measured zero rates.

There are two paired known-fact cohorts:

- `unchanged_known`: every previously known binding whose gold value did not change;
- `untouched_known`: the same cohort excluding the event target. This removes the directly refreshed repetition target when measuring collateral effects.

These cohorts are identical for addition and correction. They are overlapping views, not independent samples. Absent entities never enter known-fact preservation denominators.

For either cohort, let `CC`, `CW`, `WC`, `WW` be counts of correct/correct, correct/wrong, wrong/correct, and wrong/wrong before/after answers, with total `n`:

| Measurement | Formula |
|---|---|
| Before accuracy | `(CC + CW) / n` |
| After accuracy | `(CC + WC) / n` |
| Joint preservation | `CC / n` |
| Conditional preservation | `CC / (CC + CW)` |
| Conditional forgetting | `CW / (CC + CW)` |
| Unconditional forgetting | `CW / n` |

`WC` is recovery, not preservation. `WW` does not require the same wrong text twice. Net accuracy change is `(WC - CW) / n`, which is not the forgetting rate. These are observed answer transitions, not proof of irreversible information loss inside a latent state.

For absent keys, `absent_error` counts every output other than normalized exact `unknown`. Its two disjoint components are `absent_false_answer` (one of the six exact normalized room names) and `absent_invalid_output` (everything else, including empty output). The former is **not a general semantic hallucination rate**: `the office` and multi-answer strings are invalid, not canonical room answers. Known false abstention counts exact `unknown` on known keys. Normalization reuses the existing lowercase alphanumeric scorer; free-form abstention aliases are not newly introduced.

Pool counts across histories **within each optimization seed**, keeping each metric/event/cohort key separate. Do not average nonempty per-history conditional rates or pool overlapping cohorts/branches. For example, forgetting counts `1/1` and `0/7` pool to `1/8`, not `1/2`. Conditional subsets are method-specific: paired histories do not make each method's initially correct facts identical. Interpret them alongside before competence, unconditional accuracy, and full transition counts. Seed variation and paired-history intervals remain the next aggregation boundary.

The hand-counted test fixture has `CC=2, CW=2, WC=1, WW=2` on seven unchanged correction queries: before `4/7`, after `3/7`, joint preservation `2/7`, conditional forgetting `2/4`, and unconditional forgetting `2/7`. It is saved at `artifacts/smoke/memory_update_metrics_20260905/fixture.json`, explicitly labeled synthetic with **no model result**. The combined regression command now passes **293 tests**; this does not claim a green full working-tree suite.

## Core runner verification

`src/tinymem/research/update_runner.py` uses the existing writers and cache-free native reader. A training step writes the four common chunks once, then forks three independent events from the same before state. It minimizes the mean of four state losses, each the mean of ten token-mean answer cross-entropies including the end-of-turn token. This defines the objective, not an authorized training schedule: launch parameters and provenance still need to be frozen.

Every neural write owns exactly 66 bytes of finite, bounded FP32 state plus bool validity. The optimizer owns exactly the writer and projection parameters; reader parameters remain frozen and gradient-free. All four earlier states remain attached to after-event losses. Evaluation projects a state once for its ten independent questions, uses no persistent read cache, and saves byte-counted state snapshots as measurement output, not as additional memory available to a stream.

All five compact references and both controls use the same paired episodes. Dictionaries are derived from training write tokens only, never questions, labels, development, or confirmation. Fingerprint lookup is explicitly marked handcrafted, not a Qwen reader result. Full context is an unbounded capability control and no-memory owns zero history bytes.

Competence gates require exactly the authoritative episode set and rescore all raw predictions against independently replayed gold. Cached metrics and caller-provided pass flags cannot grant qualification. Text qualification checks known and absent accuracy separately at all four stages; learned competence checks only before-state known and absent accuracy. Both use the declared 95% threshold.

Nineteen tests exercise actual tiny random Qwen forwards, generation, and optimization for both learned writers, plus every compact reference/control. Shared-fork losses and parameter gradients match independent replay, with and without gradient checkpointing. An after-only loss reaches all four common writes and its chosen event, but not the other two events. Reader weights remain bit-identical after optimization. Repeated evaluations match, state ownership is checked, and malformed gate coverage fails. These are engineering checks, not model-quality evidence.

The expanded focused regression passes **312 tests**. A read-only independent review found no confirmed core correctness bug; its conditional concern about event splitting was resolved by inspecting `encode_update_chunks`, which returns exactly one token tuple per input chunk and rejects overlength events rather than splitting them. The nonfinite-loss error path was tightened. High-level provenance, CLI, reporting, and Della orchestration were verified in the subsequent milestones below; actual CUDA execution remains unverified.

## Development artifact boundary

`research/update_protocol.py::load_development_data` verifies the design, old-study trust anchor, complete data-generator source set, required transitive input hashes, exclusion manifests, source membership, and exact train/development representative pairings. Cross-split checks use selection metadata; the loader never opens or hashes confirmation histories. The real verified build loads all 256 training and 32 development histories under an explicit filesystem guard forbidding `confirmation.json` access.

`shared_reader_identity` resolves the frozen old reader qualification and adapter hashes without loading Qwen weights. `load_shared_reader` rechecks that identity and verifies the pinned full snapshot before production loading. This does not replace the new per-state development reader gate. Eleven artifact-boundary tests cover omitted hashes, changed sources/data, mismatched pairings, path escape, and adapter corruption; the expanded focused regression passes **323 tests**. Launch freezing, qualification persistence, checkpoint completeness, and confirmation authorization were implemented in the next milestone below.

## Qualification, training, and evaluation commands

`research/update_experiment.py` and `scripts/run_memory_updates.py` now implement the artifact lifecycle. Run from the repository root. Every output is fresh; interrupted directories remain evidence and cannot qualify as complete runs. There is deliberately no automatic resume or checkpoint selection. A `complete.json` seal is written last and binds exact required output hashes. Partial seals fail JSON/hash/coverage validation rather than permitting confirmation.

```bash
PY=.venv/bin/python
DATA=artifacts/predictions/memory_update_data_20260905_v2
# No weights or confirmation histories opened:
$PY -m scripts.run_memory_updates --data "$DATA" check

# GPU allocation only; choose fresh output directories:
$PY -m scripts.run_memory_updates --data "$DATA" --device cuda profile \
  --method query_pool --output artifacts/predictions/update_profile_query
$PY -m scripts.run_memory_updates --data "$DATA" --device cuda qualify \
  --output artifacts/predictions/update_reader_qualification

# STEPS must be an explicit predeclared positive integer, informed by training-only
# profiling. No full-size update launch or training schedule has been frozen here.
$PY -m scripts.run_memory_updates --data "$DATA" --device cuda freeze \
  --qualification artifacts/predictions/update_reader_qualification \
  --output artifacts/predictions/update_launch --steps "$STEPS"
$PY -m scripts.run_memory_updates --data "$DATA" --device cuda train \
  --launch artifacts/predictions/update_launch --method query_pool --seed 1337
# Repeat train for both methods and all three declared seeds.
$PY -m scripts.run_memory_updates --data "$DATA" --device cuda evaluate \
  --launch artifacts/predictions/update_launch --method query_pool --seed 1337 \
  --split confirmation
# References/controls omit --seed. --split is always explicit.
```

The ten-step profile uses training histories only, verifies all seven recurrent state gradients, records logical token counts/timing/allocation samples, and saves **no reusable checkpoint**. Profile completion is never training completion. New full-context qualification uses all new development states; failure is recorded and blocks freezing. A frozen launch binds source/runtime/reader/data identity, exact per-seed schedules, training encodings, and training-only dictionary before any of its six new runs begins. Shared model/dictionary costs are recorded separately from per-stream bytes.

Each train command initializes independently; it never overwrites or retrains an old association checkpoint. AdamW uses learning rate 0.001, weight decay 0.01, clip norm 1.0, full recurrent gradients, and non-reentrant gradient checkpointing. Initial and every-100/final writer weights are saved, with optimizer state at checkpoints. Development is evaluated only from the final checkpoint; weak before-state competence is recorded, not hidden or treated as launch success. Confirmation authorization requires all six completed new runs, checks exact schedule coverage and checkpoint hashes, then opens the new confirmation histories. Reader/runtime identity is strict, including GPU model and numerical settings: use matching allocations; do not silently mix hardware or packages.

Eight integration tests exercise the actual synthetic data builder through qualification, freeze, six tiny-Qwen training runs, final checkpoint loading, and new-confirmation baseline/control evaluation. A scripted perfect text reader is used **only** to exercise the synthetic gate-pass branch; actual random-Qwen gate failure is separately tested and blocks launch. Confirmation access is refused both with zero and five completed runs. Tests also cover mixed-launch checkpoint rejection, tampered/partial artifacts, explicit CLI split selection, a frozen PEFT adapter with live input gradients, and intermediate checkpoint persistence (the latter uses explicitly stubbed optimizer steps, not a learning claim). The ten-step profile test uses real tiny optimization.

The focused regression now passes **331 tests**. Both new and old input-only preflights pass. The production CLI rejects synthetic datasets. No real reader qualification, update launch, full-size training, confirmation evaluation, or cluster submission was performed. Independent reporting validation and Della orchestration were completed in the subsequent milestones below; actual scientific execution remains pending.

## Count-aware reports

```bash
.venv/bin/python -m scripts.report_memory_updates \
  --launch artifacts/predictions/update_launch \
  --split confirmation --output artifacts/predictions/update_report
```

Reporting loads no model weights. It requires all six completed training runs and all thirteen evaluations (six learned checkpoints, five compact references, two controls). It verifies completion files, launch/checkpoint identities, expected histories, reader labels, serialized state bytes, and authoritative rescoring of raw predictions. Altering a cached metric does not change the reported result; a discrepancy fails validation. Outputs are fresh `report.json` and `report.md`, explicitly labeled by evidence type and split.

The estimator pools numerators and denominators across histories **within each seed**, then averages seed-specific ratios for a learned family. It never pools seeds as extra histories. Complete CC/CW/WC/WW counts remain per run. Seed values, sample SD, and range are separate from evaluation-history uncertainty.

The default 10,000-draw paired percentile bootstrap resamples complete histories with one shared count matrix for every method, seed, event, and outcome. Intervals condition on the observed training seeds. Undefined denominator draws are counted; if any occur, the interval is withheld, not recomputed after discarding them. These are descriptive 95% intervals, not multiple-testing-adjusted significance tests. Reported contrasts are query-pool minus mean/FIFO and each learned method minus latest vocabulary/templates, with direction explicit for accuracy versus forgetting.

Markdown surfaces initial competence, correction/staleness, each event's preservation/forgetting, seed variation, paired intervals, and per-seed correction denominators. JSON retains all 80 outcomes, six transition tables, intervals, counts, shared costs, and artifact identities. A strong preservation claim still requires before-state competence; a zero forgetting rate with no initially correct facts is undefined, not success.

Fourteen new focused tests verify count pooling (`1/1 + 0/7 = 1/8`), fixed-seed means versus cross-seed pooling, scalar recomputation of bootstrap draws, pairing, undefined intervals, cluster/family validation, metric tampering, byte payloads, and mixed evaluation identities. The complete synthetic pipeline now produces all thirteen evaluation records and a report. Its durable engineering fixture is `artifacts/smoke/update_report_fixture_20260905_v2/test_six_actual_tiny_training_0/inputs/report/`; it uses one held-out synthetic world and is **not scientific evidence**. Its single-history intervals are degenerate and used only to test reporting. Independent mathematical review found no confirmed error. The focused regression passes **345 tests**. No full-size model experiment or new confirmation examples were run.

## Della pipeline

Use [the separate update runbook](della_updates.md), `scripts/della_updates.slurm`, and `scripts/della_updates.sh`. Transfer committed source, not the unrelated local decoder patch. `scripts/update_transfer_manifest.py` provides a NUL-delimited list of the new data and required provenance/model/checkpoint inputs; the older rsync filter alone does not include the new exclusion manifests. Local rsync dry-run passes.

Profile first, choose an explicit fixed step count and walltime from training-only compute evidence, then run qualification → freeze → six new trainings → thirteen evaluations → report in one allocation. Reader qualification failure stops the pipeline. The historical six checkpoints and old diagnostic scripts remain untouched; the new pipeline does not rerun their completed confirmation. Collect the user-reported existing results before considering any old diagnostic rerun.

Ten shell/transfer tests verify ordering, failure propagation, strict arguments, explicit confirmation flags, source transfer coverage, and preserved old checkpoint/oracle linkage. Combined relevant suites now pass **377 tests**, including the existing portable-runtime/oracle-runner checks. These are local engineering checks, not proof of available Della modules, compatible CUDA wheels, sufficient walltime, or actual GPU execution. No job has been submitted.

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
  tests/test_memory_update_metrics.py tests/test_opaque_qa1.py tests/test_memory_prompt.py
```

The data milestone passed 255 tests with the existing native-memory suites; adding the paired metric tests brings the verified total to **293**. Independent data review prompted the explicit trust anchor and conflict checks. The 24 frozen sources, six old checkpoints, original protocol, and unrelated decoder patch remain unchanged. Nothing was submitted to Della or pushed.
