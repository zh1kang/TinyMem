# TinyMem specification

status updated 2026-09-07: the experiment specified below is closed with a competence-limited negative result.
this document preserves its design contract; it does not authorize another run.
see [closure](docs/paired_update_closure.md) and [follow-up research options](docs/readout_options_2026-09-07.md).
no replacement architecture or schedule is frozen.
implemented code, checkpoints, and evidence remain reproducible references, not a feature backlog.

## Research question

> How reliably can a query-independent learned memory incorporate new or corrected facts without destroying previously stored facts, at a fixed byte budget?

Compare this behavior with compact explicit fact storage. The larger motivation remains the accuracy–storage trade-off of learned compression. Neither implementation completion nor one successful fit establishes a compression advantage.

## Minimal system

Use the existing native-token Qwen3-1.7B path and qualified frozen reader first. The writer processes current history and old state without the future query, answer, gold write address, or supporting-fact labels. One resulting state answers several possible later questions without being changed by those reads.

The initial state budget is 66 bytes:

- values: `[1, 2, 8]` float32, 64 bytes;
- validity: `[1, 2]` boolean, two bytes.

All history-dependent persistent storage must be accounted for. Wide temporary read vectors are allowed but reported separately. No old raw history, reusable hidden-feature cache, KV cache, or extra precision shadow may bypass the state boundary. Shared weights and grammar tables are reported separately from per-stream bytes.

Keep all relevant recurrent transitions attached during training. Frozen reader parameters still permit differentiation with respect to memory inputs. Inference discards the training graph and retains only the declared state.

## Controlled events

Construct paired event branches from the same initial history:

- an addition unrelated to existing bindings;
- a repetition of an existing fact;
- a correction that changes one binding;
- the unchanged starting state as the before-event reference.

Independently replay the before/after answers. A correction must really change the value, a repetition must not, and unaffected bindings must retain the same labels. Distinguish additions that increase live information from events that keep live information constant; do not attribute their capacity difference solely to forgetting.

Record actual visible event/chunk boundaries. They are provided to all methods and must not become uncounted history information. The shared prefix must have the same writes in every branch. No query or label may select what is retained.

## Measurements

Report counts and denominators, not only percentages:

1. Accuracy on newly introduced or corrected facts.
2. Accuracy on unchanged facts before and after the event.
3. Correct-to-incorrect transitions among unchanged facts, including the conditional forgetting rate.
4. Unconditional accuracy and the complete correctness-transition counts, so an initially poor model cannot look reliable simply because it knew little.
5. Stale answers equal to the superseded value, distinguished from other wrong answers.
6. False answers for absent entities, separate from false abstentions on known entities.
7. Persistent bytes, shared parameters, training work, and measured execution cost.

An undefined denominator is explicitly undefined, never silently a zero error rate. Aggregate at the history level, retain pairing, and report optimization-seed variation separately. Use at least three optimization seeds for final learned-method comparisons, not for every implementation smoke test.

## Comparators

Reuse learned query pooling and independently trained contextual mean/FIFO. Keep strong compact raw/latest-fact references, including exact template coding, plus the separately labeled approximate fingerprint association reference. Full history and no memory are capability controls.

Every Qwen-based method shares the qualified reader, visible inputs, query envelope, and storage contract. Train learned baselines under their own policy. Same-checkpoint corruptions are interventions, not independently trained competitors. Do not promote a win over a limited comparator family into superiority over all bounded memory.

## Competence and decision gates

Preserve the existing six-run association comparison without changing its frozen sources, data, seeds, or decision rule. the negative association comparison and paired-update report were subsequently checked against downloaded artifacts.
additional training-fit and source-audit reports have the separate provenance limits recorded in the closure.
do not rerun completed stages or use their confirmation answers to tune a follow-up.

The preserved training-fit tooling uses predetermined final checkpoints. The fixed-readout diagnostic fits one code per training world with the reader and read projection fixed; it is privileged feasibility evidence, not a compressor. Its reported failure remains conditional on the tested interface and optimization, not a fundamental storage limit.

Only interpret update-induced loss once immediate binding competence is demonstrated. If competence is weak, report that limitation and keep the update measurements descriptive. A failed bounded fit does not prove an information-theoretic capacity limit.

Use one budget first. Further budgets or a single architecture intervention need a separately declared, evidence-based question. Do not implement a collection of speculative alternatives.

## Data and execution discipline

- Split by semantic history/source group before generating related event branches or queries.
- Keep all variants, prefixes, and queries from one history in one split.
- Preserve original consumed-data and reserve exclusions; fresh entity names do not create independent source histories.
- Validate answers independently and reject malformed inputs before training.
- Freeze protocols, input/source hashes, seed schedules, and final-checkpoint selection rules before a scientific run.
- Run small tests and synthetic end-to-end checks locally. Supply reproducible Della scripts for full-size work, with fresh outputs and explicit numerical/device settings.
- Do not retrain or overwrite the six completed checkpoints, mix partial executions, or treat a script/fixture as a measured result.

## Engineering and documentation

Prefer existing functions and narrow modules over a new framework. Preserve unrelated working changes. Add tests at actual behavioral boundaries, then independently review important changes.

Keep `docs/decision_log.md` and the TinyMem Obsidian folder updated with decisions, equations, invariants, verification, and findings. Distinguish planned, implemented, validated, running, and measured status. Commit important verified steps locally; generated data, weights, and predictions stay outside commits. Do not push without authorization.

Implementation completion means the data, metrics, runners, reports, scripts, and tests form a reproducible end-to-end path. Scientific completion additionally requires actual full-size outputs and their declared analysis. Neither requires a learned method to win.
