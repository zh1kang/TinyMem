# TinyMem: active research plan

Updated 2026-09-05. Read [the specification](../PROJECT_SPEC.md) for the complete contract. This is a focused study, not a list of future architectures.

## Question

> Can a query-independent learned memory incorporate new or corrected facts without damaging other stored facts, at a fixed byte budget?

Compare with compact explicit storage. A reproducible negative result with a localized failure is useful; obtaining a positive learned-compression score is not the completion criterion.

## What stays fixed initially

- Existing Qwen3-1.7B native reader and recurrent writer interfaces.
- 66-byte state: two width-eight FP32 slots and validity.
- Query-blind writes, multiple questions per state, no retained history/KV bypass.
- Full short-episode recurrent gradients and a shared qualified reader across Qwen-based methods.
- Learned query pooling, independently trained contextual mean/FIFO, and strong compact fact references.
- Source/history grouping and separate training, development, and confirmation roles.

Do not change the architecture while implementing the new measurement path. A later change must answer a specific observed failure; it is not an automatic extra experiment.

## Existing evidence and prerequisite

The six 1,000-update association runs are complete. Development means are 14.32% known / 17.71% absent for query pooling and 12.76% / 9.38% for mean/FIFO. The known-answer advantage changes sign across seeds. Full text scores 253/256 known and 32/32 absent on the qualified development format.

The original confirmation execution was interrupted. No complete confirmation report, final training-fit result, or current-task fixed-readout oracle result was available at the start of this work. Preserve partial outputs without reading them for tuning.

Finish the frozen comparison through the existing `scripts.opaque` tools and [Della execution](della.md), without retraining its checkpoints. Then evaluate the predetermined final seed-1337 checkpoints on all 256 training worlds. Use the current-task code-only oracle to distinguish fitted readout feasibility from writer learning where needed. The first four training worlds, frozen projection/reader, 200 updates, and all nine queries are already declared.

Good training with weak development motivates a generalization diagnosis. Poor training means generalization is not the only problem. Oracle success establishes accessible states for those fitted examples, not a useful writer. Oracle failure remains conditional on its initialization, interface, optimizer, constraints, and budget.

## Implementation milestones

| Milestone | Deliverable | Status |
|---|---|---|
| Scope | Dependency audit, preserve existing work, focused docs and notes | Verified; see decision log |
| Data | Paired event sequences, independent answer replay, source-disjoint manifests | Verified; see [data design](memory_update_study.md) |
| Metrics | Before/after accuracy, preservation, forgetting transitions, stale/absent errors | Planned |
| Runners | Training, reader qualification, and evaluation using existing components | Planned |
| Reporting | History-level pairing, seed variation, uncertainty, tables and figures | Planned |
| Execution | Della scripts, input preflight, small-model end-to-end checks | Planned |
| Handoff | Independent review, final notes, verified local commits, explicit pending results | Planned |

## Event comparison

Start from one shared initial history and branch into unrelated additions, repetitions, and genuine corrections. Preserve the before-state and compare every event with the same initial answers. Labels come from independent symbolic replay; the writer receives only visible history.

Separate the changed binding from unchanged bindings and absent entities. An addition can increase live information; report that separately from constant-live-information corrections and repetitions. Preserve exact streaming-prefix ownership so measuring the before state cannot change the writes used in an after branch.

Measure conditional forgetting among previously correct unchanged answers **and** unconditional before/after accuracy with transition counts. Report empty denominators explicitly. This prevents initially inaccurate methods from appearing artificially resistant to forgetting.

## Scientific protocol

Before full-size training, freeze data identities, source exclusions, event construction, budget, methods, seeds, training objective/schedule, checkpoints, and analysis. Profile only training data to choose a feasible compute budget. Profile updates do not initialize the real study.

Use three optimization seeds for final learned-method comparisons. Resample whole paired histories; questions, branches, and repeated seed labels are not independent histories. Report seed variability separately from evaluation-sampling intervals. Keep full-context and fingerprint references distinct from shared-reader, byte-matched comparisons.

The new study's confirmation must remain outside development. Do not use the old confirmation outcomes to choose the new method, event construction, budget, or checkpoint. Existing source restrictions remain in force.

First require immediate binding competence before making a preservation claim. If that gate fails, still retain the measured results and label them competence-limited. The tooling can be implemented and validated before that scientific decision is available.

## Bounded next decisions

1. Complete the implementation and synthetic validation without waiting for full-size runs.
2. Run the existing diagnosis and new study on Della with recorded, fresh outputs.
3. Inspect actual results and choose at most one targeted follow-up if justified.
4. Consider a small byte-budget curve only after a useful interface is established.

Do not replace a completed negative study with a rescued run or expand the project merely because a method fails. The deliverable is a reproducible account of updating and preservation costs, including where compact explicit memory is preferable.
