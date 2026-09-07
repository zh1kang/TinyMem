# paired-update experiment closure

closed on 2026-09-07 at the user's request.
this closes the experimental campaign, not every possible TinyMem research question.
no checkpoint, protocol, threshold, source-bound implementation, or prediction was changed.
no additional training or confirmation evaluation is authorized by this closure.

## conclusion

under the tested 66-byte budget, the learned writers did not establish useful initial factual recall or an advantage over strong explicit-storage references.
poor training fit and unsuccessful writer-free code fitting prevent isolation of reliable updating and forgetting from already-competent memory.
this is a negative result for the tested systems and optimization schedules, not a proof that 66 bytes cannot store the facts.

## completed paired-update comparison

six runs completed exactly 1,000 updates, followed by all thirteen confirmation evaluations.
all six development before-state competence gates failed.

| method | confirmation known accuracy | confirmation absent accuracy |
|---|---:|---:|
| query pooling, three-seed mean | 5.08% | 66.67% |
| contextual mean/FIFO, three-seed mean | 6.32% | 58.85% |
| latest vocabulary | 25.20% | 100% |
| latest template | 74.41% | 100% |
| full context, unbounded | 97.85% | 100% |
| fingerprint, handcrafted reader | 100% | 100% |

these known/absent aggregates summarize the report's state outcomes, not just before-state accuracy.
the fingerprint result is not a Qwen readout result or a general collision-free guarantee.

correction-target accuracy was 7.81% for query pooling and 6.77% for mean/FIFO.
all bounded explicit references reached 100% on correction targets.
the query-minus-mean contrast was +1.04 percentage points, with descriptive 95% interval [-3.12, +5.21].
this does not establish an advantage.
conditional forgetting has very small method-specific initial-correct cohorts; some intervals are undefined and withheld.
low conditional forgetting cannot be reported as successful preservation here.

## training fit

the subsequent consumed-training-history diagnostic reports poor generated recall on all six final checkpoints.
training known accuracy ranges from 0.63% to 16.02%; training-minus-development known gaps range from -1.03 to +1.56 percentage points.
several seeds mostly abstain on known facts.
therefore a development-only generalization gap is not a sufficient explanation.
low token-mean teacher-forced loss does not establish generated recall.

this diagnostic's execution and source audit were supplied by the user, including cluster commit `7ce16d3` and job `13551628`.
its full new diagnostic source and result bundle have not been independently imported into this local checkout.
the reported 4-test and 31-test runs passed; the broader suite was externally killed with exit 137 and is not a full-suite pass.

## writer-free readout feasibility v3

four training histories share one projection; each history owns two width-eight FP32 codes and two validity bytes.
three initialization seeds receive 200 optimization steps per condition.
fixed projection optimizes codes only; learned projection also optimizes one shared 2048-by-8 projection.
no writer is used and the reader remains frozen.

| condition | seed 1337 known / absent | seed 2027 known / absent | seed 4099 known / absent |
|---|---|---|---|
| fixed projection | 0/32, 8/8 | 0/32, 8/8 | 0/32, 8/8 |
| learned projection | 13/32, 4/8 | 13/32, 7/8 | 12/32, 7/8 |

full text generated 31/32 known and 8/8 absent answers.
the formal decision remains `inconclusive_full_text_control_failed`: the declared control required exact 32/32 and 8/8.
independently of this failed control gate, neither fitted condition had a successful seed.
projection learning improved fitted recall, but this small shared-parameter fit is not evidence of generalization or a useful encoder.

### source audit provenance

the user supplied an audit identifying executed HEAD `cf60768e344154f0738a10a83672291b634dc229` and no remaining concrete implementation defect.

- diagnostic source SHA-256: `c7a1ff4bac7deb6c0748d856c96e43c525c853b6a008e1b8137d173e6e0084e1`.
- Slurm source SHA-256: `3e39af920ec81f8767daa95dbc7a09066f91bee95a8cc5ef6b446237c42b591b`.
- commit-bound test SHA-256: `859beb7c66581072670dc4e73bdd7a1de52ffddf0c9b2418bea00285bb42674f`.

v1 failed before optimization because a view retained the four-history backing storage.
v2 cloned each history slice while preserving gradients; its job was canceled before allocation.
v3 added stronger gradient, parameter-change, initialization, and token-identity checks.
according to the supplied audit, scientific settings did not change across these versions.
this statement concerns v1-v3; earlier conversational suggestions were not identical to the executed protocol.

the local v3 download contains outputs, not the separate source-audit bundle described by the user.
the audit conclusion is attributed to that supplied review, not presented as a new independent local source audit.

## evidence locations and verification scope

- main downloaded evidence: `/Users/caleb/Downloads/artifacts/predictions/update_study_della_20260905_1000_v1/`.
- paired data: `/Users/caleb/Downloads/artifacts/predictions/memory_update_data_20260905_v2/`.
- readout evidence: `/Users/caleb/Downloads/update_readout_feasibility_20260907_v3/`.
- separate report copy: `/Users/caleb/Downloads/report/`.

prior local review reran the production no-model report in an isolated committed checkout and matched aggregates, development gates, shared costs, launch identity, and bootstrap intervals against both downloaded reports.
its output is `/tmp/tinymem-import-check.XqAKXO/repo/artifacts/independent_update_verification/`; this temporary location is a verification record, not the durable experiment archive.
old association outputs were also independently aggregated there.
prior readout review verified 59 hash references, rescored 520 predictions, and checked paired initial tensors, changed fitted codes, and fixed-projection immutability.
these are artifact checks, not a fresh GPU execution.

keep the original sealed directories and the cluster source/audit commits intact.
closure does not require rewriting their stored paths or importing cluster history over the unrelated local decoder edit.

## next boundary

[the follow-up research review](readout_options_2026-09-07.md) distinguishes evidence, hypotheses, and candidate methods.
no proposed follow-up is a continuation or rescue of this frozen experiment.
new scientific claims need a separately declared protocol and evaluation data whose source groups have not already been consumed for development or confirmation.
