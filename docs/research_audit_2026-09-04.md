# research audit: 2026-09-04

## conclusion

TinyMem has useful engineering and causal-memory results.
the project has drifted from testing learned compression toward increasingly supervised memory-routing tasks.
keep those tasks as diagnostics, but do not use their success to close the original compression question.
the user clarified that the project should pursue that question at a suitable scale, not reduce its goal to a small supervised demonstration.
the active scope and remaining work are in [research_plan.md](research_plan.md).

this audit checked the saved milestone summaries, relevant training and evaluation paths, current replacement checkpoints, and the numbered Obsidian notes.
an independent read-only review checked milestones 5 through 12.
this was not a line-by-line verification of every source file or a rerun of every experiment.
the source revision was `a0735de93e2c4ff0398c79836884d51d2f94ed9e`.
no training, checkpoint selection, or new external evaluation was performed.

## evidence that remains useful

| study | saved result | supported interpretation |
| --- | --- | --- |
| fixed memory, three seeds | outside-window local 12.93%, importance 21.55%, oracle 25.77% | selecting retained tokens helps within the existing storage schema |
| continuous BABILong 256k | normal 34/100; drop 14, zero 15, shuffle 16 | one frozen seed uses supervised selective memory on controlled-vocabulary input |
| discrete memory | delayed 14.0% versus continuous 23.2%; 11 active codes | first discrete comparison is negative, with partial code collapse |
| MTP4 with fixed memory | 406/1200 versus 265/1200 | a repeatable controlled development gain, not universal improvement |
| broad byte QA | mean-pool 35.5 ± 3.2% across three fine-tuning seeds | improved controlled QA after window and budget changes; query-time memory removal has little effect |
| memory-required byte qa1 | 100% across three seeds; absent 11%, mismatched 0% | the writer-reader path can learn a causal delayed binding |
| two-slot replacement | 99%, 99%, 97%; slot selection 100% | strong supervised state-update development result, with the split limits below |
| LongMemEval | local normalized exact match 0/500 | local string-scoring failure; official semantic accuracy and abstention accuracy are unmeasured |

the old 15.95% byte-QA diagnosis is historical.
it must not remain the current project summary.
the later 35.5% result also uses a repeatedly inspected controlled split, so it is development evidence rather than a new untouched test.

## 1. replacement train and validation content overlaps

`generate_replacement_qa_examples` changes the random seed by split, but draws from a small finite set of episodes.
different seeds do not imply different content.
the audit reconstructed each saved run with its recorded seed, counts, capacity, and segment length.
it compared the tuple `(initial_fact_ids, correction_ids, query_ids)`, without source IDs or split labels.
the answer is deterministic for that tuple.

| capacity | seed | steps | unique training episodes | validation size | validation episodes also in training | exact on the remaining validation episodes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1337 | 2000 | 917/1000 | 100 | 13 | 86/87 |
| 2 | 2027 | 2000 | 915/1000 | 100 | 17 | 82/83 |
| 2 | 4099 | 2000 | 915/1000 | 100 | 8 | 89/92 |
| 3 | 1337 | 2000 | 996/999 | 99 | 2 | 76/97 |
| 3 | 1337 | 3000 | 996/999 | 99 | 2 | 97/97 |

removing exact duplicates does not explain away the high replacement scores.
however, filtering an already inspected validation set does not create a confirmatory test.
surface templates, entity names, and location names also remain shared.
the three seeds change both the sampled training data and optimization, while sharing one pretrained byte checkpoint.
these are not three independently pretrained models.

validation has informed loss weighting, capacity, pretraining behavior, and training budget.
use the term "development validation" for all these results.
future runs need content-disjoint, versioned split manifests and a separate generalization split based on held-out combinations.
do not silently change the legacy generator and relabel old checkpoint evaluations as new tests.

## 2. the successful update policy is directly supervised

the current path is:

```text
K separately encoded facts -> K continuous summaries
one separately encoded correction + bank -> pairwise slot classifier
hard selected slot is replaced -> later query reads the corrected bank
```

each fact is summarized without access to the earlier bank.
the harness fills the initial slots and calls replacement exactly once, at a known correction boundary.
the task uses explicit `Fact:` and `Correction:` prefixes.
the query is not available to the writer, which is a useful causal constraint.
but event boundaries and when to invoke replacement are provided by the harness, not discovered by the model.

the first 250 steps train the controller against the gold replacement slot.
later steps optimize answer loss plus that same slot-label loss.
the first two answer bytes receive weight 8.
therefore, "answer-only loss" describes the token mask, not the whole training objective.
task loss reaches the summaries and controller, but a nonzero gradient does not establish that task loss alone learned the policy.

the next diagnostic can use the existing options `--slot-pretrain-steps 0 --replacement-loss-weight 0` after split repair.
that tests supervision removal without introducing another controller.

## 3. distinguish three meanings of compression

| meaning | what is reduced | current replacement evidence |
| --- | --- | --- |
| activation compression | many token hidden states become one vector | yes, by construction |
| state reduction | obsolete history is replaced by current bindings | learned with direct slot supervision in the small task |
| storage efficiency | fewer total retained bytes at matched task quality | not established by this experiment |

the replacement decoder has 116,416 parameters, its mean compressor has 4,160, and its controller has 16,513.
the full system has 137,089 unique parameters, not the original 5 to 20 million target.
two float32, width-64 slots require 512 value bytes, 2 validity bytes, and 16 int64 position bytes: 530 retained tensor bytes per example.
the corresponding three input facts, including the correction, occupy only 103 to 112 UTF-8 bytes before framing.
this is not an equal-cost comparison of complete inference systems, but it rules out treating "two vectors" as proof of smaller storage than the raw history.

the effective local read boundary in this diagnostic is a fresh 64-byte segment.
`max_local_tokens=512` is the model's configured capacity, not a 512-byte sliding window preserved across these calls.
all comparisons must state the actual visibility mask as well as the configuration limit.

## 4. interventions are not trained baselines

`oracle_slot` and `fifo_slot` reuse the learned-replacement checkpoint.
the former forces the correct address; the latter always replaces initial slot zero.
they are useful counterfactual address interventions.
they are not an independently trained oracle model or a separately trained FIFO system.
normal and oracle-slot scores match, so "stronger than oracle" is incorrect.
the FIFO intervention also keeps the corrected value at the old slot index instead of shifting and appending a full FIFO bank.
a fair method comparison must train each baseline under its own inference policy.

## 5. byte comparisons need a common contract

the fixed raw-token schema retains both token IDs and reconstructible embeddings.
12 such slots use 3,324 bytes, so that result is not a frontier against an efficient raw-ID cache.
`StreamingDecoder.memory_bytes` does not include its raw-token staging window and attention tracker.
WikiText reports continuous values only, while another evaluator also counts validity and positions.
discrete serialized IDs are smaller than the dense vectors materialized to read them.

report separately:

- serialized retained state, including required IDs, masks, positions, and timing state;
- persistent inference tensors, including local KV, token staging, scores, and recurrent state;
- peak temporary inference memory and recomputation cost;
- shared model/codebook parameters and training activation memory.

do not combine these different quantities into one "actual bytes" axis.

## 6. external scoring and tokenizer limits

both installed LongMemEval oracle and cleaned-S files contain 30 `_abs` question IDs.
the official format marks these as abstention questions even when their reference string contains an explanation.
the current adapter does not expose this label, and the evaluator applies ordinary string EM/F1 to all 500 examples.
the official scorer uses a semantic judgment of insufficient information for this subset.
correct the adapter and scoring contract before any future external claim; preserve the existing outputs and 0/500 local EM result.
this audit does not assign a new semantic score. ([official format](https://github.com/xiaowu0162/LongMemEval#-dataset-format), [official evaluator](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py))

the older 256k BABILong positive used a controlled vocabulary that maps unseen prose words to `<unk>`, plus exact evidence supervision for its gate.
it remains useful, but it is not a demonstration of arbitrary natural-language compression.
LongMemEval and consumed BABILong slices must not be reused for model selection.

## 7. optimization explanations need controlled evidence

isolating controller pretraining is a sensible phase-boundary repair and the updated runs perform better.
it is not proof that this was the only cause of seed variation.
the saved three-slot, seed-1337 runs at 2000 and 3000 steps differ before step 2000.
the longer run reaches 100%, but training duration alone has not been isolated as the cause.
both used MPS; backend nondeterminism is a hypothesis, not a diagnosed cause.
there is no saved three-seed 3000-step result in the inspected artifact tree.

## evidence locations

- fixed baselines: `artifacts/predictions/baseline_comparison_multiseed/20260901T203635.755516Z-f64ba578/results.json`
- 256k: `artifacts/predictions/continuous_memory_heldout_256k/56434a58ee88-4ac29f8afa30-20260902T032520.167674Z-dd96756d/results.json`
- discrete: `artifacts/predictions/discrete_memory_comparison/seed_1337.json`
- byte bridge: `artifacts/figures/conversational_qa/summary.md`
- replacement: `artifacts/predictions/replacement_qa/capacity_*/seed_*/a283*/results.json`
- external: `artifacts/predictions/longmemeval_aggregate/20260903T155831.482051Z-7da5a1bf/`
- generator: `src/tinymem/data/replacement_qa.py`
- training and address interventions: `src/tinymem/training/replacement_qa.py`, `src/tinymem/evaluation/replacement_qa.py`
- byte definitions: `src/tinymem/memory/state.py`, `src/tinymem/model/streaming.py`, `src/tinymem/evaluation/codebook.py`, `scripts/run_wikitext_language_model.py`

## validation and unresolved work

the eight replacement tests and full repository test suite passed during this audit.
split overlap and nonduplicate accuracy were reconstructed from saved run settings and per-example predictions, not by retraining.
the split generator, LongMemEval scoring, and byte-reporting issues are recorded defects, not implemented repairs in this documentation revision.
passing tests do not establish dataset independence or validity of a research claim.
