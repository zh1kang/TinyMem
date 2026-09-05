# Paired-update implementation verification

## Completed implementation

The lean66-byte study is implemented end to end using existing Qwen, query-pool/mean-FIFO writers, and compact references. No new architecture family or extra budget was added.

| Boundary | Commit | Evidence |
|---|---|---|
| Scope and preserved history | `f913a41` | Focused contract; no unrelated decoder modification |
| Paired data and stable streaming prefix | `7bf5641` | Independent replay, source exclusions,11,520 native tokenizer checks |
| Paired metrics | `cf0ffe1` | Explicit counts, transitions, undefined denominators |
| Core execution | `d1fa317` | Actual tiny-Qwen gradients, generation, shared-state forks |
| Data/reader provenance | `3ba32b8` |256train/32dev real data verified without confirmation access |
| Qualified launch/train/evaluate lifecycle | `c88db89` | Six tiny runs, partial/mixed rejection, PEFT gradients |
| Aggregation/reporting | `042b02a` | Paired count bootstrap, separate seed variation,13evaluation fixture |
| Della orchestration | `814c2ab` | Shell order/failure tests, transfer manifest, local rsync dry-run |

## Final checks

- **1,779 tests pass in38.37seconds** in an isolated clone of committed code through `814c2ab`, with required datasets staged. This is not a claim about the local unrelated decoder edit.
- **377 focused tests pass** in the working checkout, including all new study components and relevant historical runtime/oracle tests.
- Both input-only preflights pass without model loading or confirmation-answer inspection.
- All24 old frozen sources, all15 portable sources, the original protocol, and all six final checkpoint hashes remain identical to the initial baseline.
- The unrelated `continuous_decoder.py` file and its Git patch remain byte-identical and unstaged.
- The new data protocol remains `5e0879c6891aa180a75364f01de738bed9784102cad35cae2b34a2770c5b39e6`.
- The original association protocol remains `e74cf9fa0027da9ad43bbe7914c56d3601c9e4428517912a42663f733ec8e566`.
- Independent read-only reviews examined core recurrence/gates, lifecycle integrity, and aggregation mathematics; concrete coverage/provenance suggestions were addressed.
- Documentation/index notes now distinguish development evidence, user-reported final association results, synthetic validation, implemented tooling, and unrun scientific work.

The first clean-clone test attempts lacked staged inputs (new provenance artifacts, then an unrelated historical test's bAbI validation file). Staging those required inputs resolved both failures; no test/source behavior was changed to obtain the full pass. The update transfer manifest contains required update inputs, not every historical test dataset; stage full bAbI if running the entire historical suite on Della.

Logs: `/tmp/tinymem-update-study/final-clean-suite.txt`, `final-identities.json`, `final-old-preflight.json`, `final-new-preflight.json`, `della-regression.txt`. The isolated checkout path is in `final-clean-checkout.txt`. Temporary logs are not the sole specification; test sources, this record, and the commit history are durable.

## Synthetic artifact

`artifacts/smoke/update_report_fixture_20260905_v2/test_six_actual_tiny_training_0/inputs/report/` holds a generated JSON/Markdown report. It is explicitly synthetic, uses tiny random Qwen and a scripted qualification-pass fixture, and contains one confirmation history. Its degenerate intervals and accuracy are **not scientific findings**. No fixture, dataset, model weight, or generated prediction is committed.

## What is still not established

1. Actual Della/CUDA execution, installed module/package compatibility, throughput, and walltime sufficiency.
2. New-task full-text reader qualification, a chosen/frozen full-size training schedule, and actual update-study results.
3. A useful learned memory or compression advantage. The implementation does not guarantee either.
4. Independent verification of the user's supplied completed old confirmation/fit/oracle aggregates. Local artifact discovery still finds only the original partial confirmation baseline output, not the final cluster report. Preserve the supplied findings as user-reported and collect the completed artifacts; do not rerun by default.

No jobs were submitted, full-size model training started, remote transfers performed, or changes pushed. To execute next, follow `docs/della_updates.md`: clean transfer, input check, training-only profile, explicit compute declaration, then qualification-gated execution. Scientific completion is separate from this implementation completion.
