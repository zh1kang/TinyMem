# lean readout-interface study

status: candidate design after research and three read-only consultation rounds with claude, 2026-09-07.
the user approves this direction and asks for research and discussion before coding.
no model implementation or training schedule is frozen by this document.
this is a new development study, not a continuation of the closed update experiment.

## question and allowed claim

> how does the readout interface affect factual recall from a fixed-size, query-independent learned state?

the controlled contrast is an activation in the decompressor of a jointly trained encoder/decompressor system.
the encoder architecture and initial parameters match across arms, but the trained codes can differ.
therefore a difference is not a causal estimate of readout performance on identical fixed codes.
we do not claim architectural novelty, a storage-capacity theorem, or that success identifies the cause of the earlier recurrent failure.

## minimal architecture

```text
complete visible history
        |
        v
frozen Qwen features, no query and no previous state
        |
        v
two learned attention queries
        |
shared Linear(2048,64) -> GELU -> Linear(64,8) -> tanh
        |
        v
2 x 8 FP32 values + 2 validity bytes = 66 persistent bytes
        |
        +--> Linear(8,32) -> identity -> Linear(32,2048)
        |                         or
        +--> Linear(8,32) -> GELU     -> Linear(32,2048)
                                      |
                             frozen Qwen + one question
                                      |
                                generated answer
```

only one arm exists in each run.
the diagram's branches are alternatives, not two stored copies or an ensemble.
attention pooling uses scaled dot products against all visible history-token features, followed by masked softmax.
projection weights are shared between slots; the two learned queries distinguish the slots.
use a dedicated one-shot encoder instead of carrying unused recurrent projections and gates from the old writer.
reuse `LatentSlotState` and frozen reader/prompt/loss utilities without editing launch-bound sources.

both bridge layers have biases in both conditions.
at reader width 2048, each bridge has 67,872 shared parameters, or 271,488 FP32 parameter bytes.
the encoder has 135,752 shared parameters.
these are separate from persistent per-history storage; training optimizer state is also separate.

pair initial tensors, history order, optimizer settings, training budget, and seed across the two arms.
initial output functions differ because GELU differs from identity; record initial vector norms and accuracy rather than silently rescaling one arm.
this is a factorized affine control, not the original direct bias-free projection.
hidden width 32 is one declared candidate, not a width sweep.

## keep storage and execution simple

no database, vector index, retrieval service, remote experiment tracker, or persistent feature-cache service.
the inspected code and dependency manifest do not use a database for this path.

use the existing artifact directory convention:

```text
artifacts/predictions/readout_interface_<run-id>/
  protocol.json
  <arm>_seed_<seed>/
    initial.safetensors
    final.safetensors
    metrics.jsonl
    predictions.jsonl
    complete.json
  report.json
  report.md
```

bind model/adapter identity, data manifests, exact tokenization, sources, initial weights, schedules, and final artifacts with hashes.
a fresh output directory and a completion seal written last are enough; do not add a new registry or workflow engine.

training may reuse frozen features in a bounded CPU-memory cache within the run process.
key entries by the full history/token identity and reader identity, not an individual source-group identifier: each history owns two groups.
treat this as training acceleration, report its memory, and verify cached/fresh feature equivalence under identical tokenization, dtype, positions, and batching.
start with the uncached implementation; add the in-process cache only if profiling makes repeated feature extraction material.

end-user evaluation must encode the visible history once, release text/features from the read interface, and answer all questions from only the detached state and shared parameters.
no cached history features, original tokens, other questions, or gold labels may enter the read call.
training retains the gradient path through the encoder and decompressor; do not confuse that graph with forbidden inference storage.
a Python `del` statement is not proof of isolation: test the narrow function boundary and reload state in a fresh read context.

66 bytes counts tensor payload/backing storage, not a safetensors file's headers.
FP32 state serialization must round-trip exactly; there is no quantization codec or straight-through estimator to add.
batched slices must own only their declared storage at the inference boundary.

## data scope

reuse only the existing 256 training and 32 development histories, taking each episode's before state as serialized.
this keeps eight known and two absent questions per shared state.
the original history may contain prior movements; do not replace it with a new no-rebinding dataset without declaring that change.

input-only loader verification in this design phase established:

| split | histories | independent source groups | known questions | absent questions |
|---|---:|---:|---:|---:|
| training | 256 | 512 | 2,048 | 512 |
| development | 32 | 64 | 256 | 64 |

use exact existing native history token IDs, including separators, with explicit context-length rejection rather than truncation.
only history text reaches the encoder; question/answer IDs are separate loss inputs.
no new dataset builder is needed for this stage.

these are consumed development data, not fresh confirmation.
the original update construction had 178 eligible unused source groups and used 128 for its now-consumed confirmation set.
50 groups is only the arithmetic remainder, not a verified new reserve; at two groups per world it is at most 25 additional worlds before checking later exclusions.
no new confirmation set is promised or opened here.
shared room labels across splits are expected; source/history leakage, not ordinary answer-vocabulary overlap, is the issue.

## measurements and controls

primary development contrast: GELU minus affine generated known-answer accuracy, with absent accuracy reported alongside it.
keep all outcomes even if competence fails.

for every arm and seed report:

- generated known accuracy, correct abstention, known false abstention, and invalid outputs;
- all-eight-known-bindings-correct and all-ten-questions-correct rates per history;
- training and development performance, with their original roles explicit;
- initial/final generated performance and answer-token loss, separating first-answer and stopping-token terms;
- persistent state bytes, shared parameters, temporary vectors, runtime, and training memory;
- normal state, zero values with validity retained, no memory with slots removed, and a fixed derangement of complete states across histories;
- a training-derived marginal-answer reference.

use a fixed input-independent derangement with no self-match, common across arms/seeds.
state shuffling is an intervention, not an independent-history observation.
it does not have to reach exactly the marginal baseline: accidental answer overlap and reader priors can remain.
report its actual gap and donor mapping, and do not choose the permutation from answers.

bootstrap entire histories with one shared resample across all arms and seeds.
pool question counts within each seed and average the seed-specific estimates.
report descriptive history intervals conditional on observed seeds, with seed SD/range and paired seed differences separately.
do not resample individual questions or treat three seeds as extra independent histories.
32 development histories do not imply 32 independent answer trials, and a generic binomial power calculation cannot determine the detectable paired effect.
all three differences having the same sign is not sufficient evidence of an advantage.

## gates and stop rules

engineering readiness, before full-size training:

1. paired arms have identical parameter shapes/counts and initial tensors; activation behavior differs as specified.
2. independent scalar reference matches loss and gradients for a tiny reader.
3. encoder and bridge get gradients, frozen reader gets none, and query inputs cannot affect the written state.
4. inference state owns exactly 66 bytes, round-trips exactly, and yields matching fresh-context reads.
5. actual tiny-model training, checkpoint load, evaluation controls, and reporting run end to end.
6. input loader never opens confirmation histories or predictions.

scientific readiness for any later recurrent-update claim:

- candidate threshold: at least 95% generated known and absent accuracy separately on training and development for every declared seed;
- full-text capability control must pass a separately declared qualification rule, without filtering failures;
- memory-use controls must show that accuracy depends on the supplied state rather than question priors; report paired gaps and uncertainty;
- generated loss reduction alone cannot pass competence.

these thresholds are proposed design criteria, not retroactive changes to old gates or a frozen launch protocol.
finite development evidence does not guarantee 95% population accuracy.
all arm comparisons remain reportable when readiness fails, labeled competence-limited where appropriate.
no automatic budget increase, added adapter, extra seed, checkpoint selection, or training extension follows failure.

## research and claude discussion

primary-source review was refreshed before coding:

- [xRAG](https://arxiv.org/html/2405.13792v1) supports a two-layer learned bridge between frozen components and alignment objectives, but does not establish GELU superiority, eight-dimensional code sufficiency, or byte-matched results.
- [simple context compression](https://arxiv.org/html/2510.20797v1) supports strong contextual pooling and encoder/decoder adaptation in richer representations, not impossibility for a frozen tiny-state setup.
- [the broader review](readout_options_2026-09-07.md) records ICAE, COCOM, information-preservation, and geometry evidence with limits.

three claude consultation rounds converged on paired bridge shapes, a dedicated one-shot encoder, no database, and a development-only first study.
the lead corrected several suggestions instead of accepting them by agreement:

- a failed linear probe is not an encoder-information ceiling;
- equal probe scores do not isolate reader causality;
- a successful new linear arm cannot identify old recurrence as the cause;
- 32 histories are not 32 independent questions;
- FP32 storage does not imply quantization;
- known-answer exact accuracy already measures values, not just entity presence;
- answer-vocabulary overlap is not source leakage;
- deleting a tensor variable alone does not prove no read bypass.

## verified module milestone

implemented the one-shot encoder and paired readout bridges in `src/tinymem/memory/readout_interface.py`.
`src/tinymem/research/readout_interface.py` extracts frozen reader features from history tokens only, then trains the encoder outside the reader's no-gradient scope.
the state owns exactly 66 persistent bytes; both bridge conditions have matched parameter counts and matched initial tensors.

`tests/test_readout_interface.py` passed all 12 tests, including explicit attention and gradient references, two-step training through a tiny frozen Qwen reader, state ownership, and serialized-state readback.
the focused regression selection passed all 99 tests.
a read-only independent Claude review found no blocking defect.

the readback test uses a fresh in-process reader copy, not a separate process.
production Qwen execution, the before-state runner, and full-size training are not yet verified.
these are implementation checks, not measured evidence of factual recall or a compression advantage.

## implementation queue

1. verified: implement the small one-shot encoder and paired bridge module, with ownership, parameter-pairing, gradient, and query-blindness tests.
2. add before-state-only training/evaluation using existing data/reader helpers, fresh-context reads, and the declared controls.
3. add a small runner/report path and a Della script, reusing artifact and statistics conventions without a database or new framework.
4. profile on training data, then explicitly select and freeze steps, optimizer schedule, seed list, walltime, and qualification rules before GPU training.

three paired optimization seeds are the intended final comparison, not permission to launch six full-size jobs now.
no step count, learning rate change, or empirical success is invented in this design phase.
