# after the negative result: readout competence before memory updates

research review: 2026-09-07.
status: analysis and proposed decisions, not an implemented or authorized training schedule.
[the completed experiment is closed](paired_update_closure.md).

## recommendation in brief

keep the broad problem: reliable language-model memory without retaining full conversation history.
do not make beating explicit storage on arbitrary exact bindings a required outcome.

if learned continuous memory remains the research priority, ask:

> which readout interface makes a small query-independent state usable for factual recall, before we ask it to survive updates?

start with one bounded interface experiment, not another recurrent training campaign.
the first candidate is a shared nonlinear decompressor in place of the linear read projection, with the reader frozen and the stored state unchanged.
its purpose is to test interface learning and held-out recall, not claim a new architecture or a compression advantage.
a four-history free-code fit is only an optimization check, not a test of linear-versus-nonlinear representational capacity.
reader adaptation is a separate alternative, not an additional component to add automatically.

if a reliable working memory system is the priority, prefer learned extraction into explicit bounded records, with deterministic correction and lookup.
that direction addresses a real problem while avoiding an artificial soft-vector language barrier.
it is a different contribution from learned continuous compression.

## what failed, and what did not

```text
visible chunk + old state
          |
          v
frozen Qwen features -> pooling/gated writer -> two 8D codes
          ^                                    |
          |                                    v
          +--------------------------- shared linear projection
                                               |
                                               v
                               two 2048D temporary vectors
                                               |
                                      frozen Qwen + question
                                               |
                                          generated answer

observed: poor training recall and poor development recall
bypass: directly fit codes; reliable readout still not established
```

### 1. the system did not establish initial memory competence

this is the primary observed failure.
all learned seeds failed the before-state gate, and the consumed-training-history diagnostic also reports poor fit.
there is no basis for treating the later errors mainly as forgetting of facts that were successfully stored and readable.
full recurrent gradients, frozen-reader ownership, and exact state accounting were checked.
missing cross-chunk backpropagation is not an established cause.
finite nonzero gradients prove a computation path exists, not that its updates solve the task.

### 2. the interface is more restricted than its byte count suggests

`NativeRecurrentMemory.memory_vectors` applies `nn.Linear(8, 2048, bias=False)` independently to the two slots.
for a fixed projection, each temporary vector lies in the same subspace of dimension at most eight, before the reader's nonlinear computation.
training the projection changes that subspace; it does not remove this restriction.
there are 16 stored FP32 values, not 4,096 independent values.

this is not a rank-eight bound on the reader's outputs or a mathematical impossibility proof.
a nonlinear reader can recover many discrete states from low-dimensional inputs.
but choosing a 66-byte budget did not logically require this particular linear parameterization.

at Qwen's width of 2,048, two full-width BF16 vectors alone would occupy 8,192 bytes, before validity or metadata.
that is about 124 times the entire current 66-byte state.
papers that advertise a few dense memory tokens usually do not test anything close to our stored dimensionality.

### 3. the same interface participates in writing and reading

`NativeRecurrentMemory.write` projects the old state, prefixes it to the new chunk, and passes both through frozen Qwen before pooling the current chunk's hidden states.
therefore a poorly interpreted old state can impair later feature extraction as well as final answering.
the pooling writer also receives the old code directly; the Qwen path is not the only recurrence path.

this coupling makes an end-to-end failure hard to locate.
a writer-free readout experiment removes it; a later one-shot encoder test should remove recurrence before restoring it.

### 4. answer cross-entropy was not a sufficient progress measure

`prefix_answer_loss` computes the correct causal answer-token loss, including the end-of-turn token.
`update_answer_loss` equally averages ten question losses within each of four states.
there is no identified target-shift bug in this path.

however, loss averaged over answer tokens can improve through predictable suffixes and stopping behavior without reliable first-answer selection.
teacher forcing supplies the correct earlier answer tokens during training, whereas generation must produce them.
this is a possible contributor to the observed low-loss/poor-recall gap, not a proven sole cause.
the first wrong generated token, first-answer-token loss, and known-versus-absent losses were not separately established by the aggregate CE.

future fit checks should report those quantities alongside free generation.
for this finite-answer task, scoring all complete candidate answer strings is a useful diagnostic of answer ranking, but is not a replacement for the declared free-generation metric.
do not remove absent questions to hide false abstention or change the frozen loss retrospectively.

### 5. the experiment combined too many learning problems too early

randomly initialized compression and read projection had to learn extraction, binding, recurrent retention, correction, and communication with a frozen decoder under one short schedule.
one update consumes one history; 1,000 updates over 256 histories is about 3.9 passes through histories, although each pass supervises forty questions.
this is not evidence that more steps would work, especially after the bounded code-fit failure.
it explains why numerical correctness alone was a weak readiness gate.

### 6. the task favors exact coding, not semantic approximation

opaque entity identifiers and arbitrary room bindings cannot be recovered from general world knowledge.
explicit storage can exploit the known grammar and store exact associations.
the handcrafted fingerprint reference achieved 100% on this finite evaluation, leaving no accuracy headroom against that reference on those examples.
this is not a universal collision-free guarantee, and its reader differs from Qwen.

learned memory could still offer a useful trade-off on less structured inputs, paraphrases, or a measured storage/compute constraint.
but it cannot guarantee exact answers about arbitrary information that a fixed state cannot distinguish.
new tasks should reflect a real application, not be selected because a baseline is weak.

## an additional artifact-only observation: geometry, not a discovered fix

this review loaded the downloaded initial and final v3 safetensors without running Qwen.
for each history and slot, it computed `codes @ projection.T` in FP32.

| final condition | seed 1337 mean vector norm | seed 2027 | seed 4099 |
|---|---:|---:|---:|
| fixed projection | 2.8264 | 2.8850 | 2.4758 |
| learned projection | 2.7680 | 2.8326 | 2.4570 |

across all final tensors, maximum absolute code values remain below 0.292.
the fixed projection's singular values range from approximately 8.7780 to 10.0405.
thus it is not numerically rank-collapsed or badly conditioned as a linear map.
that says nothing about its semantic alignment with Qwen.
final codes do not hit the box boundary; intermediate clipping behavior has not been checked here.

one recent paper studies high-norm vectors with Qwen3-1.7B and reports mean norms of 299.08 versus 1.54 for vocabulary embeddings [5].
our final projected norms are much smaller, but this comparison is across different models/adapters, prompts, tasks, and training procedures.
it is a hypothesis about geometry, not a target norm to copy.
the full paper is more cautious than its landing-page abstract: norm interventions support an enabling factor, not a fully isolated attention mechanism.
its sentence reconstruction also supplies half the target sentence as a prefix, unlike our blind binding recall.

important counterevidence: learned projection increased known recall from 0/32 to 12-13/32 while mean norms remained similar, but absent accuracy fell from 8/8 to 4/8, 7/8, and 7/8.
therefore v3 does not show that simple amplitude growth caused that change or that the resulting answers reliably used history-specific memory.
no complete-answer marginal-guess or shuffled-code comparison was established by this calculation.
do not automatically multiply vectors, remove bounds, or add normalization on the strength of one external paper.
reader normalization and residual computation make raw-norm reasoning insufficient.

reproduce the artifact-only calculation without model weights:

```python
from pathlib import Path
from safetensors.torch import load_file

root = Path('/Users/caleb/Downloads/update_readout_feasibility_20260907_v3')
for directory in sorted(root.glob('*projection_seed_*')):
    tensors = load_file(str(directory / 'step_000200.safetensors'))
    vectors = tensors['codes'] @ tensors['projection'].T
    print(directory.name, vectors.norm(dim=-1).mean().item())
```

## what the literature supports

these are primary-source readings, not independent reproductions.
compression ratios usually count input positions, not total persistent bytes.
none of these papers establishes reliable eight-binding updates in our 66-byte configuration.

| method / evidence | useful lesson | limit for TinyMem |
|---|---|---|
| ICAE [1] | train a compression encoder with reconstruction and continuation before instruction answering; the decoder can remain frozen | default 128 contextualized memory slots, extensive training, not sixteen stored scalars |
| AutoCompressor [2] | train the language model to produce and consume summary vectors across segments | model adaptation and language-modeling utility are not exact-binding preservation or constant-byte proof |
| xRAG [3] | a two-layer nonlinear bridge can connect frozen embeddings to a frozen reader; reconstruction and answer distillation provide alignment training | one full-width retrieval embedding, top-one retrieved documents, no tiny-state recurrent-update guarantee |
| COCOM [4] | joint compression/decoder adaptation improves readout; a frozen-decoder ablation loses accuracy | LoRA adaptation and full-width context vectors; a new reader comparison is not the old frozen-reader experiment |
| high-norm semantic vectors [5] | geometry of the injected vectors is experimentally relevant, including for Qwen3-1.7B | different tasks and much richer vectors; half-text reconstruction prefix; mixed QA results and unmeasured training-seed variance |
| simple context compression [6] | contextual mean pooling can be strong with trained encoder/decoder and distillation; complex pooling is not automatically better | full-width vectors and mostly one-shot documents, not our narrow recurrent FIFO comparator |
| information-preservation audit [7] | retained topics can conceal lost names, dates, and entity details; staged training improves some reconstruction results | better reconstruction did not consistently fix downstream grounding; do not infer exact binding from semantic similarity |
| LLMLingua-2 [8] | learned task-agnostic extraction keeps a text-compatible interface | token deletion alone does not implement reliable corrections or equal-byte fact records |
| Gated DeltaNet [9] | key-addressed delta updates provide a relevant bias for replacing associations | a recurrent matrix per layer/head is not a free 66-byte module, and a different update rule does not fix failed readout |

### specific quantitative comparisons that matter

COCOM's frozen-decoder ablation at compression rate 128 lowers COCOM NQ exact match from 0.519 to 0.421 and ASQA from 0.585 to 0.521 [4, table 6].
this supports testing decoder adaptation, not declaring it necessary in every system.
ICAE and xRAG demonstrate why that stronger declaration would be wrong.

simple context compression trains separate encoder and decoder LoRA adapters and includes Qwen3-1.7B among its model scales [6].
its mean-pooling results show that our next improvement need not be a more elaborate attention pooler.
its frozen-decoder ablation uses Gemma2-2B, so that ablation should not be reported as a Qwen-specific result.

ICAE reports 200,000 pretraining updates and 30,000 instruction-tuning updates at batch size 256 [1, appendix A].
this is context for the difference in training recipe, not a recommendation to copy that compute bill.

## ranked options, not an implementation backlog

### option 1: a learned nonlinear decompressor with frozen Qwen

replace only the linear read projection with a small shared two-layer MLP.
keep the same 66-byte stored codes; larger temporary vectors and shared weights are allowed only with explicit accounting.
a nonlinear mapping can curve the reachable input family rather than restricting every slot to one linear subspace.
it still has only the stored code's intrinsic degrees of freedom and cannot recreate arbitrary lost information.

compare it with a linear decompressor using the same one-shot query-blind encoder, data, questions, seed schedule, and final-step scoring.
remove recurrence from both arms.
predeclare and report shared parameter counts and training compute; a larger MLP is a system-design comparison, not an isolated nonlinearity intervention.
use a parameter-matched linear factorization control if the claim specifically concerns nonlinearity rather than additional parameters or optimization.

an optional writer-free tiny-set check can reject a candidate that cannot fit, but cannot establish useful capacity.
with four histories and two slots there are only eight code vectors: an invertible 8-by-8 code matrix and a learned 2048-by-8 linear projection can already represent eight arbitrary target vectors, even with code coordinates bounded to [-1,1].
therefore an MLP win on that set would concern optimization or parameterization, not removal of a representational impossibility.
a shared MLP can also store facts in its weights while codes serve as history indices.

**why consider it:** nonlinear bridges have precedent in xRAG and the geometry study, and can be evaluated at the existing state budget without reader adaptation.
**risk:** it may fail, memorize, or improve only because of added shared parameters.
**gate:** require fitted competence and unseen-history generalization with memory-use controls before recurrent training.
no successful tiny-set fit alone supports a compression or general-purpose readout claim.

### option 2: teach the reader to interpret memory through a small adapter

keep Qwen's base weights frozen but train a shared read-side LoRA adapter jointly with the code interface.
this is parameter-efficient reader adaptation, not a fully frozen reader.
start without recurrence so extraction and update errors do not obscure readout.

**why plausible:** COCOM and simple context compression support compression-aware decoder training.
**risk:** an adapter can memorize a small training set or damage text/abstention behavior.
**fairness:** preserve a frozen-reader control; qualify the adapted reader on text and compressed inputs.
for a later writer-only comparison, freeze one common trained reader before comparing writers.
for method-specific readers, label the result an end-to-end system comparison and report adaptation data, shared parameters, and compute.

this is an alternative to option 1, not an automatic second attempt after every negative result.

### option 3: learned extraction with explicit bounded storage

```text
new text -> learned entity/value extraction -> bounded record upsert
                                               |
                                   deterministic lookup / text rendering
                                               |
                                           Qwen answer
```

use learning where natural language is ambiguous; use explicit record replacement where exact corrections are required.
first measure extraction correctness separately from storage and rendering.
charge entity identifiers, values, validity, timestamps if used, and any per-stream dictionary to the budget.
global dictionaries may be shared only when genuinely fixed across streams.

**why practical:** it preserves what the current explicit references already do well and removes the need to teach a frozen reader an opaque continuous code.
**meaningful question:** how reliably can natural-language changes update a small exact memory without retaining the full conversation?
**risk:** the controlled grammar is already solved by a parser; a useful extension needs naturally varied language with independent gold, not a new synthetic accuracy claim on the same templates.
**contribution:** reliable memory construction and integration, not a novel continuous-compression architecture.

### options to defer

- full-width memory at a separately declared larger byte budget: useful as a readout capacity control, but not a fair 66-byte improvement.
- lower precision with more coordinates: potentially more usable dimensions per byte, but introduces quantization and state-format questions before readout competence is established.
- key-value or delta-rule recurrence: relevant only after a readable state exists; include matrix, key, metadata, and occupancy costs.
- reconstruction or teacher-distribution distillation: useful staged supervision, not a guarantee; canonical current bindings avoid spending capacity on stale narrative details, but constitute additional training supervision.
- query-conditioned compression: solves a different problem and cannot serve as the query-independent writer in this study.
- retained full history, external retrieval index, or growing KV cache: not a solution under the current persistent-state contract.

## a bounded next-study decision path

1. choose the priority: continuous-memory research or a reliable explicit-memory system.
2. if choosing continuous research, declare one interface change and a matched control before running it.
3. use consumed training data for development diagnostics only; never use old confirmation to select design or hyperparameters.
4. state model/adapter identity, initialization, objectives, optimizer, total steps, seeds, final checkpoint, and maximum number of candidate conditions before execution.
5. require a small-set fit gate on generated known and absent answers separately, with drop/shuffle controls to check memory use.
6. require held-out-history competence from a query-blind encoder before claiming reusable compression.
7. only after that, restore the existing recurrent writes and paired update measurement.

proposed, not frozen: require exact tiny-set known and absent fit for all three seeds for repeatable fitted competence.
report one or two successful seeds as seed-sensitive feasibility, not aggregate success; never use that result to select the final seed.
for an exact-control tiny-set diagnostic, predeclare that any full-text failure stops the diagnostic claim as inconclusive, retaining every item without filtering or replacement.
a full-text generation error does not mathematically make a memory-conditioned correct answer impossible, but it weakens this diagnostic control.
set a separate development qualification threshold on an adequately sized history set; retain known/absent counts and uncertainty.

before freezing any future confirmation set, audit remaining source groups.
old train, development, and both consumed confirmation sets cannot be relabeled untouched.
if independent groups are insufficient, a new data source or a development-only claim is required.

### minimum measurements for a useful quantitative outcome

- generated known/absent accuracy and false abstention, per seed;
- all-bindings-correct rate per history, not just average per-question accuracy;
- answer-token versus stopping-token loss and first-error position;
- complete-answer candidate ranking, clearly labeled diagnostic;
- normal versus zeroed/shuffled state accuracy and a training-derived marginal-answer baseline;
- fitted-history versus held-out-history accuracy;
- persistent bytes, shared parameter bytes, temporary read size, and read/write runtime;
- correction and preservation outcomes only after initial competence.

no predicted accuracy is supplied.
MLP alignment, reader adaptation, and explicit records make different testable bets; none is a promised rescue.

## source list

1. [Ge et al., in-context autoencoder](https://arxiv.org/html/2307.06945v3), sections 2-3 and appendix A.
2. [Chevalier et al., adapting language models to compress contexts](https://arxiv.org/abs/2305.14788).
3. [Cheng et al., xRAG](https://arxiv.org/html/2405.13792v1), sections 3-4 and appendices.
4. [Rau et al., context embeddings for efficient answer generation in RAG](https://arxiv.org/html/2407.09252v1), sections 3-5 and table 6.
5. [Zeng and Tan, frozen LLMs are native decoders for high-norm semantic vectors](https://aclanthology.org/2026.acl-long.1717.pdf), sections 2-4 and table 4.
   the full PDF gives narrower claims than the [landing-page abstract](https://aclanthology.org/2026.acl-long.1717/); this review follows the PDF.
6. [simple context compression: mean-pooling and multi-ratio training](https://arxiv.org/html/2510.20797v1), sections 4-5.
   search indexes also use the title "No Mean Feat"; the fetched primary page uses the title above.
7. [understanding and improving information preservation in prompt compression for LLMs](https://arxiv.org/html/2503.19114v1), sections 4-6 and appendix E.
8. [Pan et al., LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/); [author project description](https://www.microsoft.com/en-us/research/project/llmlingua/llmlingua-2/).
9. [Yang et al., gated delta networks](https://arxiv.org/abs/2412.06464).
