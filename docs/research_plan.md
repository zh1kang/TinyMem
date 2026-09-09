# TinyMem: current research decision

updated 2026-09-07.
the paired-update experiment is [closed](paired_update_closure.md), with a competence-limited negative result.
its implementation and protocols remain reproducible; its training and confirmation are not a pending queue.

## what we learned

- both learned writers have poor initial recall on development and, according to the supplied diagnostic, their consumed training histories.
- all six before-state competence gates failed.
- strong explicit-storage references outperform the learned systems.
- the writer-free v3 diagnostic did not establish reliable readout under either projection condition.
- the supplied source audit identifies no remaining concrete implementation defect in v3.
- the formal v3 result remains `inconclusive_full_text_control_failed`, because full text scored 31/32 known rather than the predeclared exact control.

these findings do not prove that 66 bytes cannot represent the task.
they prevent a clean claim about preservation of already-competent learned memory.

## the next question

> which interface makes a compact query-independent state usable for factual recall, before we ask it to survive updates?

[the detailed research review](readout_options_2026-09-07.md) compares nonlinear decompression, reader adaptation, and learned extraction into explicit records against primary literature and the actual code path.
it includes limitations, costs, diagnostic criteria, and stop conditions.
this is a decision document, not a list of architectures to implement.

## recommended choice

for continuous-memory research, the first candidate is a small shared nonlinear decompressor with the same stored codes and frozen reader.
compare linear and nonlinear readout with the same one-shot query-blind encoder, before restoring recurrence.
predeclare shared parameter counts and compute; use an appropriate capacity control before attributing a difference solely to nonlinearity.
a writer-free tiny-set fit is only a readiness check: shared parameters can memorize the set, and a learned linear map already has enough freedom to represent eight arbitrary vectors for four histories.
require held-out-history recall and memory-use controls before recurrent training.

for a reliable working system, learned extraction into explicit bounded records is the more direct route.
it preserves exact updates and avoids opaque-vector readout, but changes the contribution from continuous compression to reliable memory construction.

reader-side LoRA is another justified alternative, not an automatic added stage.
it changes the frozen-reader contract and requires separate parameter accounting and comparison labels.

## what remains fixed

- preserve closed protocols, sources, checkpoints, partial outputs, and negative findings.
- no old confirmation or external answers for development.
- query-blind writing and multiple questions per unchanged state.
- honest persistent bytes, with shared parameters and temporary computation reported separately.
- independent answer validation, history/source-group splits, paired uncertainty, and separate seed variation.
- generated factual recall as a competence gate, not low loss or nonzero gradients alone.
- no new full-size training until a separate bounded protocol and compute schedule are approved.

## next deliverable

the user approved the continuous-readout direction on 2026-09-07, requesting research and discussion with claude before coding.
three consultation rounds and refreshed primary-source readings produced [the lean interface design](readout_interface_design.md).
it uses a dedicated one-shot encoder, paired equally sized affine/GELU bridges, the existing 66-byte state, frozen Qwen, and files rather than a database.
no full-size schedule is selected.

the bounded candidate is now implemented and passes 244 local regression tests, including independent tiny-reader answer-loss and gradient references for both arms.
the before-state runner, controls, paired reports, production CLI, and disposable profiling script are ready for user-run Della checks.
see [the local handoff](readout_handoff.md) for commands, review findings, and explicit verification limits.
freeze initialization, seeds, objectives, total updates, final-only scoring, and success/stop rules before full-size execution.
audit remaining unconsumed source groups before promising a new confirmation set.
no CUDA execution, cluster submission, or new confirmation use occurred in this local milestone.
