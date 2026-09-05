# Decision Log

## 2026-07-24 - Use three distinct data layers

**Decision:** Use bAbI/BABILong as the primary controlled benchmark, WikiText-2 raw for ordinary language modeling, and cleaned LongMemEval as a held-out external evaluation.

**Why:** This avoids spending the project on a broad synthetic generator while preserving controlled world-state validation, ordinary language-model evidence, and external tests of knowledge updates, temporal reasoning, multi-session memory, and abstention.

**Constraints:**

- Reconstruct and validate the supported bAbI/BABILong task state with a symbolic interpreter.
- Implement only a narrow bAbI-style correction-and-deletion extension.
- Never train or tune on LongMemEval.
- Report the three layers separately.

## 2026-07-24 - Keep datasets reproducible and outside Git

**Decision:** Pin upstream revisions in `data/manifest.json`, install raw files under ignored `data/raw/`, and record downloaded checksums in `data/installed.lock.json`.

**Why:** BABILong and LongMemEval have multi-gigabyte variants. A manifest and lock preserve provenance without bloating repository history.

**Initial selection:**

- full official bAbI archive;
- BABILong `qa1`–`qa5` at 1k, 2k, 4k, and 8k;
- full WikiText-2 raw;
- LongMemEval oracle and cleaned-S;
- defer full BABILong and LongMemEval-M.

## 2026-09-01 - Start learned memory with a standalone masked compressor

**Decision:** Implement the first learned boundary as a differentiable masked mean-pool compressor before connecting recurrent state or segmented training.

**Why:** The existing `StreamingDecoder` is an inference path that uses `torch.no_grad()` and detached expired candidates.
Connecting a learned compressor there would silently prevent future-token loss from training it.

**Contract:**

- accept expired hidden states with shape `[batch, expired_tokens, model_width]`;
- accept a boolean validity mask with shape `[batch, expired_tokens]`;
- return one summary slot with shape `[batch, 1, model_width]`;
- return one validity bit with shape `[batch, 1]`;
- exclude invalid hidden states from both the mean and the gradient;
- keep an all-invalid row at zero after the learned projection.

**Next boundary:** Add a fixed-capacity recurrent bank with functional tensor updates, then build a differentiable segmented training path around both modules.

## 2026-09-01 - Keep continuous recurrence explicit and functional

**Decision:** Represent the recurrent memory bank as a stateless transition from old tensors and a new summary to new tensors.

**Why:** Explicit state makes batch-specific keep or write decisions visible, avoids hidden mutation between examples, and preserves the autograd graph across segment updates.

**Transition:**

- shift left and append the new summary to form a candidate bank;
- use the summary validity bit to select the candidate or the unchanged bank for each batch row;
- apply a three-dimensional broadcast mask to values and a two-dimensional mask to validity;
- keep capacity and model width fixed for every update.

**Next boundary:** Connect the compressor and bank to a differentiable segmented decoder without using the inference-only `StreamingDecoder` path.

## 2026-09-01 - Use CPU for token-by-token fixed-memory evaluation

**Decision:** Run the full fixed-memory comparison on CPU with batches aligned to the four BABILong context groups.

**Why:** The model is too small and sequential for MPS launch overhead to pay off.
On the same six-example, six-policy probe, CPU completed in 16.7 seconds while MPS had not completed after 194 seconds and was stopped.

**Constraint:** This device decision changes only evaluation speed.
Each seed still uses the same model, checkpoint step, examples, memory capacity, and exact streaming code path.

## 2026-09-01 - Select continuous-memory training examples by causal need

**Decision:** Include a bAbI qa1 example in the continuous-memory curriculum only when its answer-bearing supporting fact ends before the query segment begins.

**Why:** Total prompt length does not prove that the answer requires memory.
The first 128-token run selected 55 long prompts, but only 12 placed the supporting fact in an earlier segment.
The corrected 64-token boundary selects 1,388 genuine cross-segment examples.

**Development result:** After 500 steps, normal memory reached 59/364 outside-segment answers, while dropped and zeroed memory each reached 78/364.
Shuffled memory reached 58/364.
The implementation is complete, but the mean-pooled FIFO memory does not meet the Milestone 6 exit criteria.

**Constraint:** Keep this negative result.
Do not tune repeatedly on BABILong, which remains controlled evaluation data.

## 2026-09-01 - Accept the full fixed-memory multi-seed result

**Decision:** Treat the 400-example qa1 comparison across seeds 1337, 2027, and 4099 as the completed Milestone 5 benchmark.

**Matched conditions:** Each checkpoint uses 2,000 bAbI optimization steps, a 128-token local window, the same model configuration, and the same BABILong examples.
Each non-local policy uses 12 raw-token slots and 3,324 bytes per example.

**Outside-window result:** Local reaches 12.9% ± 1.7%, recent reaches 12.8% ± 1.9%, reservoir reaches 12.8% ± 1.9%, attention importance reaches 21.6% ± 2.4%, heavy hitter reaches 20.9% ± 2.7%, and oracle reaches 25.8% ± 0.6%.

**Interpretation:** Importance-based selection helps on every seed.
Recent and random storage do not improve over local truncation.
Oracle is best on every seed, so the retained evidence can help and the selection problem remains open.

**Artifact:** `artifacts/predictions/baseline_comparison_multiseed/20260901T203635.755516Z-f64ba578/` contains the aggregate JSON and checked figure.

## 2026-09-01 - Use multi-slot compression and gate long-delay writes

**Decision:** Replace one-vector pooling with four learned attention summaries per segment, then add a straight-through learned gate around the fixed-capacity recurrent update.

**Why:** Single-query attention did not make the correct memory more useful than shuffled memory.
Four summaries passed the official bAbI validation gate, but FIFO replacement filled the 12-slot bank after three 64-token segments and failed the one BABILong transfer check.

**Training protocol:** Learn ordinary cross-segment bAbI recall for 2,000 steps before continuing for 1,000 steps with up to 512 WikiText-2 training tokens inserted between context and question.
Use WikiText-2 validation text only for delayed model selection.

**Committed validation result:** Normal memory reaches 80/153 on official bAbI validation and 74/153 with 512 validation-filler tokens.
Dropped, zeroed, and shuffled memory reach at most 39/153 on standard validation and 34/153 on delayed validation.

**Constraint:** Keep the BABILong result as a negative transfer result.
Do not tune again on those 400 observed answers.
A new held-out BABILong test set is required before claiming long-context transfer.

**Artifact:** `artifacts/predictions/continuous_memory_validation/b84dde0540da-09595de243a0-20260901T211537.864508Z-9abfee26/` records the passing run from commit `046cc4f`.

## 2026-09-01 - Freeze the negative BABILong 16k transfer result

**Decision:** Use the unobserved official qa1 16k file as the one-time held-out transfer test for the frozen gated checkpoint.

**Provenance:** The file comes from pinned BABILong revision `ee0d588794c7ac098062ee0d247c733d62e94fe2` and has SHA-256 `2687478021246215c84060fa7d1f2be0b7058a09d774c169d9d75a56fd6994c0`.
Its 100 examples have zero exact overlap with the previously observed 1k through 8k files.

**Result:** Normal, dropped, zeroed, and shuffled memory each reach 17/100 overall and 16/98 outside the 64-token local window.
All counterfactual utilities are zero.

**Interpretation:** The checkpoint uses memory at the 512-token development delay but does not extrapolate to the 16k distribution.
The median evidence delay is 8,420 controlled-vocabulary tokens, and 94/100 examples exceed the 512-token training delay.

**Constraint:** Do not tune on these 100 answers or report Milestone 6 as passing held-out long-context transfer.

**Artifact:** `artifacts/predictions/continuous_memory_heldout/c854573275f3-09595de243a0-20260901T212206.676331Z-41e3554d/` records the evaluation from commit `d048641`.

## 2026-09-01 - Separate memory writing from memory contents

**Problem:** The original recurrent gate decided whether to write from the current memory and the proposed summary.
At very long contexts, the decision changed when the memory intervention changed.
This made the write policy part of the intervention instead of a fixed causal mechanism.

**Decision:** Add a memory-independent token gate that sees detached embeddings from the current segment only.
Supervise it with exact evidence-write targets, and calibrate its threshold under a false-positive-rate budget.

**Why:** A fixed 12-slot bank cannot survive even a small false-positive rate over thousands of segments.
If a non-evidence segment has false-positive probability `p`, then about `N * p` incorrect writes accumulate over `N` segments.
Selective memory therefore needs both useful recall and extremely high precision.

**Implementation detail:** The gate input is detached before the auxiliary write loss.
This lets the write gate learn without changing the shared token embeddings that the decoder uses for answer prediction.

**Commits:** `df15be6`, `e1e3e53`, `0d92908`, and `f60a5c1` add the token gate, threshold calibration, gradient isolation, and exact evidence targets.

## 2026-09-01 - Record failed long-context hypotheses before the final fix

**64k result:** The first selective gate reached 27/100 normally and 14/100 with dropped memory.
Only the drop comparison was significant, so the result was incomplete.

**Virtual-position hypothesis:** Bounded virtual positions did not improve the delayed validation result.
The hypothesis remains implemented as an option, but it is not the selected solution.

**Data-split bug:** The distributed training path still used an old cross-segment filter and selected 1,388 records instead of the complete 9,000-example training split.
Commit `1c381e4` removes the stale filter.

**MPS failure:** Variable-length `Conv1d` batches triggered an Apple MPS input-channel assertion.
Commit `a5e3d10` computes the same convolution as explicit linear taps, which preserves the parameters and mathematical operation without the failing backend path.

**Gradient-interference failure:** Joint write-gate loss changed the shared embeddings and reduced answer accuracy to chance.
Commit `0d92908` detaches the gate input and restores a clean optimization boundary.

**Controlled result after isolation:** On 1,000 delayed validation examples, normal memory reached 417 correct answers.
Dropped, zeroed, and shuffled memory reached 157, 158, and 175.
All three paired tests were significant, and the gate had zero false positives with 94.7% recall.

## 2026-09-01 - Use a wider write pattern and compose specialized checkpoints

**128k failure:** The untouched 128k test scored 9/100 normally and 14/100 with dropped memory.
The three-token gate produced 59,730 false positives.

**Root cause:** Most BABILong book prose becomes `<unk>` under the controlled bAbI vocabulary, while common words such as `to`, person names, and places remain visible.
A three-token receptive field can match a short background fragment without recognizing the complete movement fact.

**Decision:** Widen the token gate to 11 tokens so it can identify the full evidence pattern.
Train the decoder and gate as separate specialists, then compose the strongest decoder checkpoint with the strongest wide-gate checkpoint.
Calibrate the composed gate once on the already consumed 1k file.

**Why composition is valid:** The gate consumes detached token embeddings and only controls whether a proposed summary enters memory.
Its checkpoint can therefore be replaced without changing the decoder or compressor computation.
The composition script checks model configuration, vocabulary, position mode, and provenance before it saves the combined checkpoint.

**Consumed-data check:** Across 2k through 8k, the composed model reaches 97/300 normally and 49/300, 50/300, and 41/300 under drop, zero, and shuffle.
The gate produces two false positives across 29,901 negative segments.

**Commits:** `959c3f3`, `548cb1d`, and `0d12c21` add wider token patterns, preserve failed-run artifacts, and implement reproducible checkpoint composition.

## 2026-09-01 - Accept the untouched BABILong 256k single-seed result

**Decision:** Evaluate the composed checkpoint once on the previously untouched official qa1 256k file.

**Provenance:** The file comes from BABILong revision `ee0d588794c7ac098062ee0d247c733d62e94fe2` and has SHA-256 `de96fc69da2ca08478508f04cb4aa5dab8e96f3252a744c7dd036a1c3e1ed6ac`.
Its 100 examples have zero exact overlap with the 1k through 128k records used in earlier development or evaluation.
The composed checkpoint has SHA-256 `0323282f2d03adc6a13c0dbf359bba660f0767e736b4fcf30e02f64ade6c92fb`.

**Result:** Normal memory reaches 34/100.
Dropped, zeroed, and shuffled memory reach 14/100, 15/100, and 16/100.
The respective normal-memory gains are 20, 19, and 18 percentage points.
All paired one-sided tests pass the corrected `alpha = 0.0167` threshold.

**Write behavior:** The gate records 515 true positives, 40 false positives, and 156 false negatives across 606,478 segment decisions.
Precision is 92.8%, recall is 76.8%, and the write rate is 0.0915%.

**Interpretation:** The bounded memory contains causally useful information at 256k because every content-removing or content-breaking intervention significantly reduces accuracy.
This resolves the long-context transfer issue for one frozen seed.
It does not establish a publishable result until the protocol is repeated across independent training seeds.

**Artifact:** `artifacts/predictions/continuous_memory_heldout_256k/56434a58ee88-4ac29f8afa30-20260902T032520.167674Z-dd96756d/` records the evaluation from commit `0d12c21`.

## 2026-09-02 - Accept the first discrete-memory result as negative

**Decision:** Store integer code indices during evaluation, reconstruct their shared codebook vectors only for attention reads, and retain straight-through assignments only during training.

**Why:** Persisting decoded floating-point vectors would not be a true discrete evaluation state.
The separate training representation preserves the gradient path from later answer loss to earlier code decisions.

**Budget:** Twelve continuous slots use 3,180 realized bytes per example.
One hundred eighty-seven int64 discrete slots use 3,179 bytes.
The shared 65,536-byte codebook is reported as model parameters instead of per-example recurrent state.

**MPS issue:** Boolean selection in the usage loss called a nondeterministic `index_put_with_accumulate` backward kernel.
Mask multiplication and reduction preserve the same loss while keeping deterministic MPS training.

**Regularization correction:** Noisy Gumbel probabilities made aggregate soft usage look broad while deterministic selections collapsed.
The corrected loss uses hard straight-through emitted assignments in the forward pass and relaxed gradients in the backward pass.

**Result:** Continuous and discrete memory both reach 25/153 on standard validation.
After a 4,096-token distributed delay, continuous reaches 232/1000 and discrete reaches 140/1000.
The discrete model activates 11 of 256 codes, has hard perplexity 4.17, and assigns 45.3% of proposals to its most common code.

**Interpretation:** Milestone 7 is implemented and measured, but the first Gumbel codebook is a negative research result.
It is partially collapsed and does not yet provide causal memory utility.
Do not present it as an accuracy improvement.

**Artifacts:** `artifacts/predictions/discrete_memory_validation/24fe287a4218-0c4d88212f4d-20260902T140145.270317Z-a481a951/` stores the checkpoint and result.
`artifacts/predictions/discrete_memory_comparison/seed_1337.json` stores the matched comparison.

## 2026-09-02 - Keep hard codes as the primary evaluation state

**Decision:** Add soft codebook evaluation only as an explicit ablation and keep hard integer codes as the default.

**Why:** Soft evaluation carries a probability mixture over codebook rows.
It can isolate the cost of `argmax`, but it is continuous state and cannot support a discrete-memory claim.

**Result:** On the consumed BABILong 1k split, hard codes reach 21/100 overall and 8/81 outside the local window.
Soft mixtures reach 19/100 overall and 6/81 outside the local window.

**Interpretation:** The hard decision is not the only cause of the weak result.
The learned summaries and codebook geometry also fail to retain enough useful information.

**Artifacts:** `artifacts/predictions/discrete_memory_hard_ablation/260502ab09e5-0c4d88212f4d-20260902T141142.926658Z-cf860a71/` and `artifacts/predictions/discrete_memory_soft_ablation/260502ab09e5-0c4d88212f4d-20260902T141152.653667Z-7335bdb2/` store the frozen comparison.

**Commit:** `23e63b0` adds the ablation path.

## 2026-09-02 - Complete the adaptive write-controller implementation

**Decision:** Use a binary keep/write controller with pooled segment state, pooled memory state, and current prediction entropy.

**Training:** Use a hard straight-through Gumbel-Softmax action for the recurrent update and penalize the soft write probability over valid segments.

**Controls:** Compare periodic timing, matched-frequency random timing, a matched surprise threshold, learned timing, and symbolic oracle timing through the same frozen decoder.

**Inspection:** Save per-segment surprise, write probability, hard action, relevance, and update-operation labels.

**Sweep:** Require `lambda_write` values `0`, `1e-4`, `1e-3`, `1e-2`, and `1e-1`, reject protocol mismatches, and plot delayed accuracy against writes per 1,000 tokens.

**Smoke result:** A one-step MPS delayed run completed all five policy evaluations and reloaded its checkpoint.

**Expected failure:** The controller wrote on every valid segment, so learned and matched-random accuracy were both `277/1000`, while relevant and background write rates were both `1.0`.

**Interpretation:** The implementation path is complete, but the research exit criteria are not met until trained multi-seed runs beat matched random and respond more strongly to corrections and relevant facts than to background text.

**Commits:** `c940b34`, `c5ff873`, `561ca5d`, `922a6f3`, `42f397e`, and `600f0f1` implement the controller, integration, controls, checkpoints, traces, and Pareto aggregation.

## 2026-09-02 - Complete the Milestone 8 training sweep

**Protocol:** Sweep `lambda_write` over `0`, `1e-4`, `1e-3`, `1e-2`, and `1e-1` on seed 1337.

**Matched conditions:** Every point uses a 2,000-step standard-memory warmup, 1,000 delayed-training steps, 64-token segments, 12 continuous slots, one summary per write, 512-token distributed training delays, and 4,096-token validation delays.

**Selection:** Choose `lambda_write = 1e-4` because it has the highest delayed accuracy, is Pareto optimal, and significantly beats matched random on the development seed.

**Development result:** Learned timing reaches `247/1000`, compared with `187/1000` for matched random and `197/1000` for periodic writes.

**Replication:** At the selected weight, learned timing reaches `247/1000`, `172/1000`, and `156/1000` for seeds 1337, 2027, and 4099.

**Matched random:** The matching results are `187/1000`, `139/1000`, and `165/1000`.

**Aggregate:** Learned timing averages `19.17% ± 4.86%`, while matched random averages `16.37% ± 2.40%`.

**Robustness:** Learned timing wins and is individually significant on two of three seeds.

**Correction response:** Correction write rates are `92.60%`, `93.61%`, and `44.92%`.

**Background response:** Background write rates are `0.17%`, `0.23%`, and `0.02%`.

**Interpretation:** The controller consistently recognizes facts and corrections, but its answer-quality advantage is seed-sensitive.

**Status:** Milestone 8 implementation and required experiments are complete.

**Status:** The research result is promising but does not meet the stronger requirement that learned timing beat matched random on every independent seed.

**Artifacts:** `artifacts/predictions/adaptive_controller_sweep/seed_1337/aggregate/`, `artifacts/predictions/adaptive_controller_multiseed/results.json`, and `artifacts/predictions/adaptive_controller_events/` contain the frontier, replications, and correction traces.

**Commits:** `8a139d7` adds strict cross-seed aggregation, and `1c9e52a` adds correction-aware event evaluation.

## 2026-09-02 - Complete Milestone 9 multi-token prediction

**Decision:** Keep the ordinary next-token head as the primary task and add independent auxiliary vocabulary heads for offsets 2, 3, and 4.

**Alignment:** For horizon `h`, logits at positions `0:T-h` predict token IDs at positions `h:T`.

**Padding:** A loss term is valid only when both its source position and future target position are valid.

**Objective:** Average the valid cross-entropy losses across enabled horizons, then add the result to answer loss with weight `0.2`.

**Inference:** Do not use auxiliary heads during answer prediction.

**MPS issue:** Boolean selection before cross-entropy called a nondeterministic MPS scatter backward operation.

**Resolution:** Keep the full aligned tensor, mask invalid targets with `-100`, and use cross-entropy `ignore_index=-100`.

**Base sweep:** On seed 1337, no MTP, MTP2, and MTP4 reach `76/400`, `95/400`, and `73/400` on BABILong.

**Fixed-memory development result:** At a 4,096-token validation delay, no MTP reaches `168/1000` and MTP4 reaches `283/1000` on seed 1337.

**Fixed-memory replication:** No MTP reaches `168/1000`, `192/1000`, and `170/1000` for seeds 1337, 2027, and 4099.

**Fixed-memory replication:** MTP4 reaches `283/1000`, `350/1000`, and `332/1000` for the same seeds.

**Full BABILong:** Across three seeds and 1,200 evaluations, fixed no-MTP reaches `265/1200` and fixed MTP4 reaches `406/1200`.

**Delay interaction:** MTP4 changes pooled accuracy by `-6.7`, `+21.2`, `+35.4`, `+40.7`, `+22.2`, and `+5.6` percentage points across the six increasing delay buckets.

**Adaptive interaction:** On seed 1337, adaptive no-MTP, MTP2, and MTP4 reach `122/400`, `61/400`, and `91/400` on BABILong.

**Interpretation:** MTP4 gives a repeatable gain for fixed FIFO memory, but the benefit is not monotonic with delay and does not transfer to the current adaptive controller.

**Status:** Milestone 9 implementation and required experiment matrix are complete.

**Artifacts:** `artifacts/predictions/mtp_sweep/20260902T211044.063567Z/` stores the full development matrix.

**Artifacts:** `artifacts/predictions/mtp_comparison/20260902T211037.806860Z/` stores the three-seed fixed-memory delay comparison.

**Commits:** `00d072d`, `488adb8`, `3b0020b`, `f0e5289`, `30663f3`, `f174b78`, `712e3e2`, `4b2acfe`, `a4d4b6e`, `32ac122`, and `e40bcdc` implement and validate Milestone 9.

## 2026-09-04 - Diagnose the byte-level conversational bridge at class prior

**Problem:** The conversational checkpoint reached 15.95% exact on the 6,000-example controlled holdout, and every task sat at its answer class prior (about one in six for location tasks).

**Diagnosis:** The WikiText byte base used a 64-byte segment with no cross-segment attention and eight mean-pooled FIFO slots.
A bAbI fact line is about 36 bytes and the question tail is about 52 bytes, so the answer position never saw a fact directly.
The fine-tune budget of 1,000 steps at batch 8 and learning rate 3e-4 was also too small: the answer loss was still falling at the end and predictions collapsed to `bedroom`.

**Decision:** Record the compressor and bank family in byte checkpoints through `ByteMemorySpec`, reconstruct it in `load_wikitext_checkpoint`, and expose `--compressor`, `--summaries-per-segment`, `--memory-update`, and `--write-threshold` on the WikiText script.
Legacy checkpoints load unchanged as mean-pool FIFO.

**Protocol:** Seed 1337, d_model 64, 2 layers, 8 slots, 512-byte segment, 1,024-byte WikiText sequences, identical bAbI qa1 to qa5 plus update data, identical holdout.

**Holdout results (6,000 examples, exact):**

| run | window | memory | fine-tune | overall | qa1 | qa2 | qa3 | qa4 | qa5 | updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 64 B | mean FIFO | 1,000 x 8, 3e-4 | 16.0% | 15.1 | 13.7 | 17.7 | 19.7 | 14.8 | 14.7 |
| A | 512 B | mean FIFO | 1,000 x 8, 3e-4 | 16.0% | 17.1 | 16.7 | 15.2 | 15.6 | 18.6 | 12.9 |
| C | 512 B | mean FIFO | 4,000 x 16, 1e-3 | 34.1% | 46.6 | 20.4 | 21.0 | 44.8 | 36.0 | 35.7 |
| D | 512 B | multislot(2) gated | 4,000 x 16, 1e-3 | 37.0% | 43.3 | 21.1 | 17.9 | 45.8 | 32.9 | 61.1 |

**Interpretation:** Neither the window alone (run A) nor the memory family alone (run B, 14.4% validation) moved the result at the old budget.
The window plus a four-times larger fine-tune budget is what lifted the model off the class prior.
The gated multislot memory helps most on the update test split, which has three to four corrections per example.
qa2 and qa3 remain weak because their 1 to 3 KB prompts still overflow the local window.
Both budget runs were still improving at step 4,000 and are single-seed development results.

**Operational note:** Two concurrent CPU torch processes and the one-minute idle sleep timer distorted wall-clock times badly.
Pin thread counts and attach `caffeinate -i -w <pid>` to long runs.

**Artifacts:** `artifacts/predictions/wikitext_language_model/e8b31f5c5c3f-ea5c94151693-*` and `e8b31f5c5c3f-3024324307a6-*` hold the 512-byte bases.
`artifacts/predictions/conversational_qa_budget/` holds runs C and D.
`artifacts/predictions/conversational_qa_holdout/e8b31f5c5c3f-8606ba78542b-*` and `e8b31f5c5c3f-a0e6538acfe7-*` hold their holdouts.

## 2026-09-04 - Replicate the byte-level bridge across seeds and test memory utility

**Question:** After the window and budget fix, does the byte-level model use its bounded memory, and does the result replicate across seeds?

**Protocol:** Fixed 512-byte bases (mean-pool FIFO and multislot attention with gated updates, 8 slots each), fine-tune seeds 1337, 2027, and 4099 with 4,000 steps at batch 16 and learning rate 1e-3, the 6,000-example controlled holdout, a dropped-memory-at-query ablation, and a qa1 delay sweep with 0 to 2,048 bytes of WikiText-2 validation filler inserted before the question.
A delayed-recall curriculum variant trained seed 1337 of both families with up to 2,048 bytes of WikiText-2 train filler in half of the training examples.

**Holdout (three seeds, exact %, mean ± sd):**

| family | qa1 | qa2 | qa3 | qa4 | qa5 | updates | overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean-pool FIFO | 45.8 ± 3.7 | 20.5 ± 5.1 | 20.7 ± 0.4 | 46.9 ± 3.5 | 36.1 ± 4.2 | 43.0 ± 6.7 | 35.5 ± 3.2 |
| multislot + gated | 39.8 ± 6.7 | 19.2 ± 2.9 | 19.4 ± 1.6 | 45.1 ± 3.2 | 31.6 ± 2.2 | 45.5 ± 13.5 | 33.4 ± 4.0 |

**Memory ablation:** Hiding memory at the query changes overall exact accuracy by +0.1 points for both families across seeds.
On a 891-byte prompt the next-token logits move by less than 0.01 with memory dropped.

**Evidence split (seed 1337):** Accuracy is 38.5% and 42.3% when every supporting fact lies in the answer's 512-byte segment, and 19.3% and 19.5% when any fact lies beyond it, against a 16.7% class prior.

**Delay sweep:** Both families fall from 44% and 38% at zero filler to 15 to 17% at 64 bytes of filler, although 80% of prompts still fit one segment at that delay.
The fine-tuned in-window heuristic breaks under an unfamiliar filler session, so the sweep measures format brittleness rather than memory.

**Curriculum:** Training with filler removed the format shift but cost in-window accuracy at the same budget: holdout 20.4% (mean-pool) and 21.9% (multislot), delay curves at 23 to 25% for zero filler and 14 to 20% beyond, and no change with memory dropped.

**Interpretation:** The window plus budget fix is a robust positive result across three seeds.
The bounded memory is a clean negative result in the byte path: neither family reads from memory, the gated multislot bank does not beat mean-pool FIFO on average, and one seed's 61% update score is seed variance (sd 13.5 points).
Making memory matter needs either a much longer curriculum budget or a training signal that only memory can satisfy.

**Tooling:** `scripts/evaluate_conversational_qa.py --memory-condition`, `scripts/evaluate_conversational_delay.py`, `--training-delay-max-bytes` on the fine-tune script, fine-tune progress logging, and `scripts/aggregate_conversational_qa.py` for tables and figures under `artifacts/figures/conversational_qa/`.

**Artifacts:** `artifacts/predictions/conversational_qa_multiseed/`, `conversational_qa_delayed_curriculum/`, `conversational_qa_holdout/`, `conversational_qa_delay/`, and `artifacts/figures/conversational_qa/summary.md`.

## 2026-09-04 - audit the claims and retain the original compression question

objective: review the code, saved results, and Obsidian history before choosing the next architecture.
the user clarified that the goal remains learned compression versus bounded baselines, and that research scale should not be limited to a small supervised task.

evidence: the latest two-slot replacement scores are 99%, 99%, and 97%, but use direct slot supervision.
reconstructed validation overlaps are 13/100, 17/100, and 8/100; nonduplicate exact scores remain 86/87, 82/83, and 89/92.
these are useful development results, not fresh held-out confirmation.
two slots use 530 retained tensor bytes versus 103 to 112 UTF-8 history bytes before framing, so storage superiority is unproved.

additional findings: raw-token baselines retain redundant embeddings; byte definitions differ across experiments; LongMemEval has 30 `_abs` cases that the current evaluator does not score with gold abstention semantics.
the 0/500 remains local string EM, not official semantic accuracy.
the corrected three-slot run reaches 100% at 3000 steps on seed 1337, but its earlier trajectory differs from the 2000-step run and there is no completed three-seed confirmation.

decision: keep the original question and measure a query-independent, fixed-total-state accuracy frontier.
retain the current decoder as a mechanistic reference and plan a separate capable pretrained-reader path.
prioritize query-independent recurrent memory and a controlled accuracy–storage comparison; the current scope is specified in the active research plan.
do not copy growing-cache or query-aware literature results into a fixed-state claim.

validation: eight replacement tests and the full repository suite passed; artifact settings and per-example predictions were reconstructed without retraining.
an independent read-only audit checked older milestones, and an independent review found no blocking issue in the revised research documents.
Glimpse provided the known replacement function's context, while rg and direct file reads supplied exact text and source evidence.

scope: documentation and study notes changed; model code and artifacts did not.
the generator, cost-accounting, and external-scoring fixes remain queued, not completed.
no new GPU allocation, large download, training run, commit, or push was performed.

records: [audit](research_audit_2026-09-04.md), [active plan](research_plan.md), and [literature review](literature_review_2026-09-04.md).
Obsidian notes 12, 14, and 21 were corrected; note 22 and the index describe the current direction.
next: repair measurement, profile candidate readers, and select a training scale after compute constraints are known.

## 2026-09-04 - repair replacement splits

completed: `history_disjoint_v2` partitions semantic histories before sampling and keeps alternate queries in one split.
optimizer seeds no longer change v2 training data by default.
exact manifests are saved before training; independent symbolic replay checks answers and slot metadata.
legacy replay remains available as `legacy_seed_v1`.

validation: 18 focused tests and all 1,115 tests passed; independent review found no blocking issue and compared 3,600 legacy examples.
the two-step CPU command completed and verified zero overlap and manifest-before-checkpoint ordering.
this is a pipeline smoke test, not a positive learning result.
artifact: `artifacts/smoke/replacement_split_v2/capacity_2/seed_1337/a28373c9c859-759c920c45c1-20260904T210410.801267Z-31954e67`.
commit: `0b0acd9`; Obsidian note 23 records the rationale and tests.
next: recover LongMemEval abstention labels from saved IDs, then repair byte accounting and implement the recurrent writer.

### replacement v2 development replication declared before training

run the supervised reference and the answer-loss-only diagnostic on the repaired data.
both use capacity 2, 64-byte segments, 1,000 training examples, 100 validation examples, data seed 0, validation seed 10,000, and optimizer seeds 1337, 2027, 4099.
both use 2,000 steps, batch 64, learning rate 0.001, weight decay 0.01, and the existing semantic-byte weighting.
the reference uses 250 controller-only warmup steps and slot CE weight 1; the answer-loss-only diagnostic uses neither.
both share the frozen-start WikiText parent `e8b31f5c5c3f-ea5c94151693-20260903T220041.343793Z-44fa4522/checkpoint.pt`.
use CPU serial execution to avoid concurrent accelerator contention.
these are development controls, not a storage-frontier comparison or a final holdout.
do not change budgets in response to validation scores within this comparison.

## 2026-09-04 - recover local LongMemEval abstention diagnostics

completed: gold labels are derived from the official `_abs` suffix and never enter prompts.
direct and shard evaluation share a local confusion-matrix calculation with undefined ratios represented by null.
old records with IDs remain supported; contradictory explicit labels fail closed.
all original EM, F1, coverage, and selective accuracy values remain unchanged.

validation: 16 focused tests, all 1,122 repository tests, and independent review passed.
the four original 125-example shards were re-aggregated without model generation or teacher calls.
result: `0/500` normalized EM, 30 gold-unanswerable cases, zero detected abstentions, recall 0/30, undefined precision.
official semantic accuracy remains unmeasured.
artifact: `artifacts/predictions/longmemeval_reaggregation_v2/20260904T210702.937377Z-2662b77e`.
commit: `f2b21ab`; Obsidian notes 12 and 23 and the index are updated.

compute inspection: this machine has an Apple M3 Pro, 18 GiB unified memory, MPS available, and no CUDA.
the project environment has torch 2.13.0 but no transformers or peft, and no pretrained model weights were found in the Hugging Face hub cache.
no large download or paid allocation has been started; the compute choice is pending.

## 2026-09-04 - fix the grouped-query shuffle intervention

issue found during the first v2 run: adjacent alternate queries can share a history, so rolling by one row is no longer a fully mismatched-memory control.
fix: construct a history-group derangement, reject impossible groupings, and record the new shuffle policy in results.
normal scores and the seven non-shuffle conditions are unchanged in independent runtime comparison.
validation: 19 focused tests and independent checks of 3,069 group/order cases passed.
commit: `817bd1e`; Obsidian note 23 records the issue and repair.
the supervised checkpoints will be re-scored after training finishes, with original results preserved.

## 2026-09-04 - add compact raw state and actual allocation accounting

completed: CompactTokenRetention stores only narrow integer IDs and per-row lengths, with functional fixed-capacity updates.
ordinary byte histories can use uint8, costing capacity + 8 bytes per row.
tensor_storage_bytes counts unique backing allocations and does not undercount a small view into a large tensor.
independent review found and verified a fix for foreign capacity/vocabulary state acceptance.
validation: nine focused tests and 360 independent randomized suffix/allocation cases passed; full worktree suite passed at 1,137 tests including the in-progress writer tests.
commit: `16ce309`; Obsidian note 24 records the storage contract and limits.
remaining: raw-reader benchmark integration, serialized format costs, and peak temporary-memory profiling are not complete.

## 2026-09-04 - implement narrow recurrent memory and raw recomputation

completed: a small research adapter jointly writes old latent slots and current segment features, with fixed slot count and independent memory width.
training concatenates history before fixed-byte chunking, supplies no future query or slot labels to the writer, and uses unweighted answer CE.
all short-episode transitions remain attached; frozen-reader tests still deliver delayed gradients to earlier states.
query generation uses only question tokens and generated prefixes.

the raw adapter now recomputes from retained IDs with no padding gap or persistent KV.
its query-aligned logits and loss match explicit retained-prefix decoding.
validation: 10 focused adapter/runner tests, all 1,142 repository tests, and independent review passed.
final two-step smoke: `artifacts/smoke/recurrent_slots_final/526ea70338f4-a3cde778de4f-20260904T211722.169856Z-b87fb18f`.

issues resolved before commit: saved parent config fields were stale, generation-limit rejection occurred too late, and an attempted code_dim override conflicted with the shared reader-width invariant.
the corrected protocol explicitly separates narrow persistent shape from projected reader interface, preserves ExperimentConfig invariants, and validates generation before run creation.
checkpoint loading with weights_only and prediction replay were verified.

commits: `60e7547` for recurrent training and `f00776f` for raw recomputation.
Obsidian note 24 covers both mechanisms and cost limits.
no pretrained model, distillation run, complete byte frontier, or external transfer result is claimed.

### accelerator smoke and next diagnostic

the same two-step recurrent command also passed on MPS at `artifacts/smoke/recurrent_slots_mps/c3763c5e66f5-a3cde778de4f-20260904T212015.339049Z-a4e69600`.
this checks device execution, not throughput or training quality.
after the three answer-loss-only replacement runs finish, run one declared narrow-writer development diagnostic at seed 1337, two width-8 float32 slots (66 retained history bytes), fixed 64-byte history chunks, 2,000 steps, batch 64, learning rate 0.001, full reader adaptation, ordinary unweighted answer CE, and the same v2 data seeds/counts.
do not rescue the run or change its budget based on validation.
the purpose is to test the end-to-end learning path before the pretrained-reader stage, not declare superiority over raw retention.

### pretrained-reader compute decision

the active plan requires a capable pretrained comparison, not only more byte-model sweeps.
no suitable cached backbone or transformers installation is present.
a practical local screening candidate is Qwen/Qwen3-1.7B, with standard causal attention and a 4.08 GB published repository; this is a candidate, not a passed competence gate.
the newer Qwen3.5 family includes hybrid gated-delta blocks, so it introduces more reader-state behavior to audit.
model weights, tokenizer revision, dependency versions, device compatibility, and visible-evidence competence must be pinned and checked before selecting a research backbone.
sources: [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B), [published file sizes](https://huggingface.co/Qwen/Qwen3-1.7B/tree/main), [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B).
the user's approval for a large local download or a remote GPU budget is still pending.

## 2026-09-04 - complete repaired replacement replication

all six declared runs completed with the same exact manifest hash `b202ed38eb2716a31e0e2f68e6b817480da8dcd186f321287974f4189e2e5a52`.
supervised reference: 100%, 100%, 98%, mean 99.3 ± 1.2% sample sd.
answer-loss-only: 62%, 67%, 67%, mean 65.3 ± 2.9% sample sd.
the latter reaches 94% corrected-query accuracy but only 36.7% unchanged-query accuracy.
both depend on memory: drop and different-history controls average about 18 to 19%.

the three supervised checkpoints were replayed after the shuffle repair, with exact equality required for every non-shuffle condition.
original records were preserved; replay results sit beside them.
interpretation: the signal survives the split repair, but direct slot supervision is doing important work for preservation of other facts at this budget.
this is not a storage advantage, a complete compression frontier, or a final held-out result.
report: `docs/replacement_v2_results.md`; summary and inspected figure: `artifacts/figures/replacement_v2_development/`.
Obsidian note 23 records the protocol, outcome, and limitations.

## 2026-09-04 - complete the first narrow-writer diagnostic

the declared seed-1337 run completed 2,000 steps in 128.2 training seconds on CPU, without a budget rescue.
it stores two width-8 float32 slots plus validity, totaling 66 retained history-state bytes per stream at segment boundaries.
development exact accuracy: normal 49%, drop 17%, zero 10%, different-history shuffle 20%, history without the correction 26%.
this is evidence of useful information passing through the narrow state, but low absolute quality and only one seed.
it is not a win over matched raw retention or pooling; those training comparisons have not run.
the new diagnostic also differs from the replacement controller in objective weighting and history chunking, so its 49% must not be treated as an isolated architecture ablation against 65.3%.
artifact: `artifacts/predictions/recurrent_slot_v1_development/c3763c5e66f5-78c9f6b62401-20260904T212240.172225Z-f8233cb8`.

local batch complete: six code-only commits, all 1,142 tests passing, CPU/MPS smoke tests, checkpoint replay, six replacement training runs, one narrow-writer run, and updated Obsidian notes 23/24.
tracked worktree is clean; pre-existing Markdown and generated local artifacts remain outside commits.
no push, paid compute, large model download, or LongMemEval retraining occurred.
next authority-dependent step: approve a pinned pretrained reader download and dependencies locally, or provide remote GPU access and a spending limit.
then complete reader competence screening, total-state cost accounting, matched independently trained baselines, distillation, and reserved generalization evaluation.
the overall research project is not complete.

## 2026-09-04 - authorize the pretrained-reader gate

the user approved the local Qwen3-1.7B download and research dependencies.
this does not authorize paid compute or additional large backbones.
Qwen is the shared reader for later memory methods; full-context Qwen is a capability reference, not a storage-matched competitor.
pin model revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` and optional Transformers `5.16.1`.
the existing environment was created by uv and has no pip module, so use uv to add the optional dependency and install it without changing torch.

predeclare the first gate: 100 development examples each from bAbI qa1, qa2, and qa3 training files; 200 history-disjoint-v2 correction queries; 64 exact-copy controls; and 64 randomized binding controls.
independent review raised the correction sample from 100 to 200 before inference so both corrected and unchanged groups have 100 questions.
sample at most one question per official bAbI episode and validate each selected answer through the existing symbolic interpreter.
use seed 10000, a single fixed chat prompt, non-thinking greedy decoding, and a 16-native-token generation limit.
require at least 95% exact accuracy separately for qa1, corrected queries, and unchanged queries.
the pass criterion requires at least 100 cases in each required category; smaller smoke tests cannot pass the gate.
copy and randomized binding scores preserve exact case and punctuation after trimming outer whitespace.
copying, randomized bindings, qa2, and qa3 are additional diagnostics, not silently substituted gate criteria.
run full-context and question-only conditions with the same question and output instructions.
record exact inputs and fingerprints before inference; do not inspect reserved bAbI test files or LongMemEval.
do not alter the prompt or decoding rule in response to gate results without a separately labeled development protocol.

implementation complete: `324c652` adds the optional pinned native reader; `21e8f76` adds the fixed competence gate.
all 1,153 repository tests pass, plus independent reproduction of the 11 focused tests and smoke artifacts.
all ten local model files match the pinned upstream hashes, totaling 4,079,434,577 bytes.
the Mac GPU smoke completed 28 predictions and failed the formal minimum-sample criterion as intended.
the smoke contains real wrong answers on visible long stories, so the full reader gate is necessary.
the full fixed run is now started with defaults and no training or prompt changes.
Obsidian note 25 explains the reader's comparison role, native token costs, controls, scoring, and exact commits.

the first complete qa1 group scored 84/100, below the predeclared threshold.
independent inspection found stale and wrong-entity answers, including short histories, rather than formatting failures.
before changing prompts or reader adaptation, replay the same 100 qa1 prompts singly in bfloat16 on MPS, then in float32 on MPS.
use unchanged prompts, labels, greedy decoding, and output limit.
these are numerical/batching diagnostics on consumed development data, not new confirmatory tests.
retain all original outputs and compare per-case prediction identity as well as accuracy.
also run one predeclared oracle-current-fact control on the same 100 qa1 cases.
choose the latest movement for the queried person directly from the context, never from the answer label.
this diagnoses fact selection and update handling, not learned compression.
the one-off reproducible diagnostic source is retained with its artifacts at `artifacts/predictions/reader_gate_numerics/diagnose.py` and fingerprints itself before inference.

full gate complete at `artifacts/predictions/reader_gate/20260904T213639.047632Z-bffa79b4`.
full-context scores: qa1 84/100, qa2 41/100, qa3 26/100, corrected 100/100, unchanged 88/100, copying 64/64, random bindings 57/64.
all 628 question-only controls scored zero against factual answer labels.
the predeclared reader gate fails; no memory training started.
inference time was 294.5 seconds; end-of-run MPS allocated memory 3,441,150,464 bytes and driver memory 6,383,452,160 bytes are not peak or bounded-state measurements.
Obsidian note 25 preserves the full result and the 13 stale/3 wrong-entity qa1 error diagnosis.

numerical replay: bfloat16 single-example inference reproduces all 100 original qa1 predictions and scores 84/100.
float32 single-example inference also scores 84/100 and matches 99 raw predictions; it does not fix the gate failure.
do not attribute the observed failure to batching or bfloat16 based on these results.

next local stage: fit a small rank-8 query/value LoRA reader adapter with ordinary visible-evidence answer CE.
the base Qwen weights stay frozen and the resulting adapter will be shared by every subsequent memory method.
PEFT 0.20.0 is added as an optional research dependency through uv.
prefer this bounded adaptation over adding a larger backbone or a second memory mechanism before reader competence is established.
exclude all gate bAbI episodes and entire episodes containing an exact gate history, across tasks, before creating internal train/development groups.
use replacement v2 train only and keep paired questions in one internal split.
profile full-size backward memory and throughput before fixing a training budget.
verify adapter-only updates, answer alignment, save/reload equality, and checkpointed gradients before training.

## 2026-09-04 - qualify and adapt the shared reader

fresh-model numerical checks confirm 84/100 qa1 accuracy with float32 and 84/100 with single-example bfloat16 inference.
batching and dtype do not explain this gate failure.
the first diagnostic reused a model after globally changing its dtype, which also converted normally-float32 rotary frequency buffers.
its oracle score is not a clean fresh-reader comparison.
preserve that artifact, and use separate fresh model loads for the repaired controls.
the fresh oracle-current-fact control scores 71/100, with a verified float32 RoPE buffer; many failures answer unknown.
the production loader was not affected because it selects weight dtype during loading rather than globally casting the loaded module.
these controls diagnose reader behavior on consumed development cases, not compression quality.

the full-size low-rank training profile passed on MPS: 87, 114, and 1,409 forward tokens took 1.98, 0.95, and 6.70 seconds respectively.
all three backward passes had finite, nonzero gradient norms.
maximum observed driver allocation at the recorded step boundaries was 5,123,112,960 bytes, not a measured peak.
the adapter has 1,605,632 trainable parameters; the base model remains frozen.
profile artifact: `artifacts/predictions/reader_adaptation_profile/20260904T215302.556334Z-561e7953`.
profile updates are discarded and do not initialize the real experiment.

predeclare one shared-reader development run: optimizer seed 1337, data seed 0, 400 optimizer steps, four microbatches of one example per step, learning rate 0.0001, AdamW weight decay 0.01, and gradient clipping at norm 1.
use rank-8 query/value LoRA with alpha 16 and zero dropout, bfloat16 frozen weights, and non-reentrant activation checkpointing.
sample a category uniformly, then an example uniformly within it.
use 400 training and 50 internal-development bAbI episodes per task, plus 1,000 paired replacement-v2 train questions partitioned internally by semantic history.
exclude connected components of duplicate contexts across all three bAbI tasks, including gate-related components, before question selection.
save checkpoints at steps 100, 200, 300, and 400; select only by the lowest internal macro-category answer CE.
evaluate that one selected checkpoint on the unchanged consumed reader gate.
do not select a checkpoint or extend the budget from the gate answers.
this is reader qualification, not a multiseed memory result.

independent review found a possible split-integrity error in the first runner draft: an adapter could be evaluated against a different gate from the one excluded during training.
the repaired runner checks original bAbI source hashes before data selection and binds each adapter to the exact excluded gate manifest hash.
adapter evaluation rejects any different reconstructed manifest before generation.
regression tests cover changed source files, changed gate manifests, incorrect adapter-gate pairing, checkpoint selection, adapter save/reload, and protocol recording before backward.

stage committed as `3454066 feat: add disjoint low-rank reader adaptation` after all 1,164 tests passed and independent review cleared the split-integrity fix.
the declared main run starts from fresh pinned weights at `artifacts/predictions/reader_adaptation/20260904T215534.309647Z-9ef7a86d`.
Obsidian note 26 explains LoRA, answer alignment, frozen-reader gradients, checkpointing, split grouping, and the review issue.

the fixed seed-0 adapter manifest contains 1,980 training questions (400 per bAbI task and 390 per correction subgroup) and 370 internal-development questions (50 per bAbI task and 110 per correction subgroup).
the groups contain 312 excluded bAbI episodes, and train/development semantic history IDs are disjoint.
the earlier seed-1337 data-construction check had different replacement subgroup counts; it is not the main run manifest.

while reader training runs, implement only the independent native-prefix read foundation, not a memory writer training run.
the API consumes explicit native prompt fragments around temporary memory vectors and does not store inputs or history on a module.
raw IDs and projected latents must later occupy the same native Facts-field location; the caller owns this envelope and state accounting.
the byte decoder's per-layer external-memory injection is not equivalent to Qwen's causal soft-token prefix and is not reused.
source inspection found that Transformers 5.16.1 forces `use_cache=True` for decoder-only `inputs_embeds` generation.
the research prefix generator therefore recomputes the supplied prompt plus generated suffix with an explicit `use_cache=False` on every forward call.
the ordinary full-context gate remains unchanged and may use a temporary generation cache as already declared.

Glimpse retrieved the exact pretrained generation symbol efficiently, but its call graph resolved the dynamic tokenizer decode call to the unrelated byte Vocabulary.decode method.
use Glimpse for the symbol body and broad context; verify dynamic edges with source inspection and rg rather than treating them as exact.

the prefix read foundation is committed as `63a07a7 feat: add cache-free native memory reads`.
all 1,172 tests pass; independent review and eight focused prefix tests confirm the final numeric and length checks.
review reproduced an overflow from finite float64 memory into a lower-precision reader dtype; validating the converted tensor fixes it without detaching gradients.
Obsidian note 27 records the interface, different soft-prefix semantics, actual state versus temporary allocation, and unresolved integration checks.
the native full-size parity/gradient smoke source is prepared at `artifacts/predictions/native_prefix_smoke.py`; run it only after the shared-reader training process releases the GPU.

## 2026-09-04 - preserve the native history/query token boundary

the tokenizer audit found that naively encoding opening, history, and suffix separately changes every one of the 1,256 saved gate prompt ID sequences.
the problem is the history/suffix seam: Qwen often represents a final period plus newlines as one token.
the repaired ownership rule keeps the opening through `Facts:\n`, encodes history with its following two-newline separator, and begins the query-only suffix at `Question:`.
the new encoder reproduces all 1,256 stored native sequences exactly, with one common opening and 90 unique question-only suffixes.
history and answer changes cannot change either uncounted prompt fragment; question changes cannot change the history IDs.
boundary-bearing tokens are part of counted history, not reconstructed from uncounted punctuation or old text.
unsupported inputs, such as a leading-newline history that merges with the opening, are rejected after exact native-token equality checking.
the encoder can tokenize a stream longer than the reader window, but the query envelope itself must fit; later memory reads enforce their actual combined length.
no memory writer was trained during this interface work.

the native boundary contract is committed as `db50789 feat: preserve native memory prompt boundaries` after all 1,179 tests passed and independent review cleared the implementation.
the environment dependency check also passes: all 51 installed packages are compatible.
Obsidian note 27 now includes the token-boundary problem, its exact ownership rule, verification counts, and commit link.

## 2026-09-04 - complete the declared shared-reader training run

the fixed 400-step run completed in 1,448.55 seconds after setup, including internal evaluation and checkpoint writes.
it processed 280,444 logical training forward tokens and 4,512 supervised answer tokens in 1,600 microbatches.
these token counts exclude internal evaluation and activation-checkpoint recomputation.
recorded update time totals 1,171.57 seconds.
the largest step-boundary MPS allocated and driver measurements are 3,551,346,688 and 6,378,471,424 bytes; these are not peak measurements.

internal macro answer CE at steps 100/200/300/400 is 0.1831605 / 0.1549859 / 0.1825689 / 0.1146269.
step 400 therefore wins the predeclared internal-selection rule.
the model did not automatically choose the last checkpoint; it happened to have the best measured internal loss.
checkpoint: `artifacts/predictions/reader_adaptation/20260904T215534.309647Z-9ef7a86d/step_000400`.
the unchanged gate replay is now running at `artifacts/predictions/reader_gate_adapted/20260904T222016.495967Z-4da340a2`.
no other checkpoint will be selected using the gate answers.

independent replay confirms every training count and checkpoint decision, plus 231,084 logical internal-evaluation forward tokens.
the run therefore processes 511,528 logical forward tokens including internal evaluation, before counting activation-checkpoint recomputation.
all adapter sidecars and source/data hashes match, and the adapted gate manifest is byte-identical to the original.

storage caveat for the next frontier: int32 is the current native-ID tensor layout, not the information-theoretic minimum.
the 151,936-entry vocabulary fits 18 bits per ID.
before claiming a compression advantage over strong raw retention, include a losslessly bit-packed raw-ID control or explicitly limit the claim to the int32 layout.
for example, 25 packed IDs plus an eight-byte length need 65 bytes, versus 14 int32 IDs plus length at 64 bytes under the same 66-byte cap.
no packed implementation or comparative result is claimed yet.

## 2026-09-04 - pass the shared-reader gate and full-size prefix checks

the selected step-400 adapter passes the unchanged development gate.
full-context scores are qa1 100/100, qa2 77/100, qa3 81/100, corrected 100/100, unchanged 100/100, exact copy 64/64, and randomized bindings 64/64.
all 628 question-only controls still return exactly `unknown`.
all 1,256 predictions are complete, and the gate manifest is byte-identical to the original.
the three required categories each exceed the predeclared 95% threshold.
independent review reproduced the scores, data exclusions, hashes, and internal checkpoint selection.
the adapter loses eight previously correct qa2 answers and five qa3 answers despite improving both aggregate scores.
multi-fact reasoning is therefore still imperfect; do not attribute every later multi-fact failure to compression.
this single-seed qualification uses consumed development cases, not a fresh confirmatory test.

freeze `reader_adaptation/20260904T215534.309647Z-9ef7a86d/step_000400` as the shared reader for the next memory stage.
the adapted gate artifact is `artifacts/predictions/reader_gate_adapted/20260904T222016.495967Z-4da340a2`.
inference took 277.32 seconds.
no Qwen memory writer has been trained yet, and no learned-compression gain is claimed.

the full-size native-prefix check also passes on MPS with that frozen adapter.
all 14 cases, two per category, have exactly equal answer loss and generated token IDs through native token input and the cache-free raw-embedding prefix.
forward hooks verify `use_cache=False` and absent returned KV state on the prefix path.
input memory remains unchanged, and checkpointed backward produces a finite prefix gradient norm of 0.0000039751653 with no reader parameter gradients.
artifact: `artifacts/predictions/native_prefix_smoke_20260904`.
the check took 71.29 seconds and verifies the interface, not compression or streaming quality.

next: implement and test lossless packed native-ID retention, then integrate the existing query-blind recurrent slots with this frozen reader.
keep the first training run a bounded development diagnostic with declared data, budget, cross-segment gradients, and memory interventions.
do not expand the architecture or compute scope at this boundary.

## 2026-09-04 - complete lossless packed raw retention

commit `6189f17` adds a uint8 payload containing both fixed-width native IDs and their length header.
the initial draft still spent eight bytes on a length counter; packing that field avoids weakening the raw comparator.
at the Qwen vocabulary size, 29 IDs plus a five-bit length fit in exactly 66 tensor-storage bytes.
the state retains no decoded IDs, masks, embeddings, or KV; append returns a new allocation.
shared capacity and vocabulary metadata must accompany saved payloads.
this is fixed-width packing, not optimal entropy coding, and excludes temporary decoding workspace, framework object/allocator overhead, and checkpoint-file overhead from the tensor-storage figure.
all 1,229 tests pass, including CPU/MPS round trips and native-prefix read parity.
independent review verifies 168 formats, 1,008 append transitions, 3,024 decoded rows, and 3,072 malformed-state cases.
Obsidian note 28 explains the format, cost, fairness issue, and remaining limits.
the full-size prefix audit also confirms one shared qa3 error: parity is 14/14, correctness is 13/14, and the full-size backward check covers one case.

next integration uses the existing recurrent slot writer and a trainable narrow-to-reader projection.
each write supplies only projected old valid slots and the current native history chunk to the frozen Qwen decoder, then uses its current-chunk hidden states to update the narrow bank.
no query, gold answer, full-history KV, raw staging cache, or lifetime position counter enters the writer.
positions restart for each bounded write/read call.
training keeps the short stream's recurrent graph intact; inference retains only the final values and validity tensors.
the reader stays frozen, including its selected LoRA adapter, while activation checkpointing remains available for gradients into memory inputs.

predeclare the full-size integration profile: three optimizer steps on the shortest, median, and longest native histories among qa1/correction examples in the shared-reader training manifest.
use seed 1337, two width-8 float32 slots, eight-native-token chunks, AdamW learning rate 0.001, weight decay 0.01, batch one, and clipping at norm one.
retain the complete short-stream graph and verify finite nonzero answer gradients at every earlier state.
all updates are profile-only and will not initialize a research training run.
the exact inputs, code hash, shared reader hashes, initial writer weights, and protocol are saved before backward.
the one-off source is `artifacts/predictions/profile_native_recurrent_memory.py`.

the first profile attempt stopped before backward because its diagnostic hook rejected an explicitly supplied `past_key_values=None` from the normal Qwen LM wrapper.
this is an overly strict verifier, not retained KV state.
the hook now requires the supplied value to be None as well as `use_cache=False` and a None output cache.
the original protocol and exact script are preserved in `artifacts/predictions/native_recurrent_profile_20260904`.
independent review also found an unnecessary first-chunk graph through an empty projected bank.
returning a non-gradient empty vector tensor avoids that work while preserving gradients through populated states.
rerun the unchanged three-case profile in a new artifact directory after focused regression checks.

the second full-size profile exposed a real deterministic-MPS incompatibility in boolean slot selection during backward.
an isolated reproduction confirms that both boolean indexing and index_select use unsupported `index_put_with_accumulate_mps` when deterministic algorithms are required.
basic slices followed by concatenation preserve slot order and gradients without that operation.
the native read projection now uses this form and still excludes invalid slots before projection.
a real MPS regression covers a partially valid bank and a complete two-write answer-loss backward pass with deterministic algorithms enabled.
the original failed profile and source remain at `artifacts/predictions/native_recurrent_profile_20260904_v2`.
no training result is taken from either failed attempt, and determinism is not disabled.

the repaired full-size profile passes at `artifacts/predictions/native_recurrent_profile_20260904_v3`.
histories of 12, 24, and 71 native tokens require 2, 3, and 9 writes.
answer gradients reach every earlier state; first-state norms are 3.95837, 1.58112, and 0.02385.
reader gradients are absent, all calls are cache-free, and each state uses 66 tensor-storage bytes.
the writer/projection has 33,216 shared trainable parameters.
update times are 3.70, 1.24, and 2.32 seconds including warm-up effects; maximum recorded step-boundary driver allocation is 4,057,776,128 bytes, not peak memory.
all profile updates are discarded.
the full suite passes at 1,239 tests, and independent review verifies the deterministic-MPS repair and active-LoRA recurrent gradient path.
Obsidian note 29 records the mechanism, both failed attempts, the repair, and the profile result.
the completed integration is committed as `9913f88`.

predeclare a first quality pilot: use the 400 qa1 training examples and 50 qa1 internal-development examples from the existing disjoint reader-adaptation manifest.
initialize a fresh two-slot width-8 writer at seed 1337; freeze the selected reader throughout.
train 200 AdamW updates at learning rate 0.001, weight decay 0.01, clip norm one, batch one, and eight-token chunks with full short-stream BPTT.
sample uniformly from the 400 training examples, use answer-only CE, and evaluate the final checkpoint only.
compare normal, dropped, zeroed, and different-history memory, plus packed recent raw IDs at the same 66-byte payload cap and a full-history reader reference.
all evidence is absent from the query envelope; some histories fit the raw budget, so this pilot tests the bottleneck without claiming a raw-retention advantage regime.
use the same native prompt fragments and eight-token greedy answer limit for all conditions.
this is a single-seed consumed-development diagnostic, not a confirmatory frontier or an independently adapted set of baselines.
do not extend the budget or select a different checkpoint from its development answers.

independent prelaunch review clears the pilot's history-only writes, frozen reader, checkpoint rule, input controls, and protocol-before-training ordering.
the 400 training contexts belong to 393 connected history groups; the 50 development contexts belong to 48 groups.
all contexts are distinct within each selected split, and both context and connected-group overlap across splits are zero.
the history permutation excludes both identical context and identical history groups without using questions or answers.
the pilot is launched at `artifacts/predictions/native_memory_pilot_20260904` from a fresh writer, not either profile checkpoint.

## 2026-09-04 - preserve the negative native-memory pilot

all 200 updates and 300 fixed prediction records are complete.
learned memory scores 8/50, different-history memory 8/50, dropped and zeroed memory 0/50, packed recent raw IDs 47/50, and full history 50/50.
normal and different-history conditions both answer `kitchen` on every example; drop and zero answer `unknown`.
correct history has no useful greedy-retrieval effect at this budget.
answer CE is not exactly invariant: paired normal/different-history changes average 0.03426 in absolute value, with maximum 0.13212.
normal mean CE is 1.09159 versus 1.08839 for mismatched memory, so the probability changes do not establish a useful correct-history advantage either.
do not describe 16% versus 0% dropped accuracy as a successful memory result.

training uses 151 distinct sampled cases, 25,837 logical forward tokens, and 572 supervised answer tokens.
it takes 223.96 seconds; evaluation takes 140.48 seconds.
the maximum recorded step-boundary driver allocation is 4,049,485,824 bytes, not a peak.
the final checkpoint hash is `f3b0f9328ac73704efb401d3feae9b80c447d8a9d82b29a8303a214c4d586b1f`.
independent review reproduces the data, source, checkpoint, native token, sampling, and scoring evidence.

post-hoc symbolic replay identifies complete support inside the raw buffer in 47 cases and outside it in three.
packed raw retention is correct on all 47 retained-support cases and wrong on the other three; full context fixes all three.
only 22 complete histories fit the raw buffer, which demonstrates why whole-history overflow is not the same as relevant-evidence overflow.
the analysis source and per-case native/character spans are saved with the run.
this small single-seed development pilot is not the final delayed-relevance frontier.
Obsidian note 30 records the negative result and its interpretation.

next diagnostic: distinguish narrow readout alignment from recurrent writer optimization before adding another architecture or repeating a large sweep.
an oracle fit of independent bounded memory vectors on a tiny balanced training-only subset can test readout capacity without claiming query-blind compression.
if it fits, focus on training the writer, first with complete facts per chunk and then with boundary/delay stress.
do not silently extend or replace the completed pilot.

## 2026-09-04 - isolate the narrow memory readout

predeclare a privileged training-only fit with no recurrent writer.
select the shortest native qa1 history per location class from the existing 400 training cases, breaking ties by case ID.
the six selected contexts and history groups are distinct, but their questions include only three unique native query envelopes.
fit independent two-slot width-8 float32 codes through tanh and a shared bias-free 8-to-2048 projection, using the same frozen selected Qwen reader.
use fresh seed 1337 after reader loading, 200 AdamW steps, learning rate 0.001, weight decay 0.01, clipping at one, and six accumulated microbatches per step.
each update averages all six answer-only losses, including end-of-turn tokens.
evaluate the final checkpoint only on the fitted training cases, all 36 code/query pairings, and drop, zero, and full-history controls.
also report the 18 unique model inputs to avoid overweighting repeated questions.
predeclare strict success as six correct fitted answers, six unknown responses with dropped memory, and at least 15 donor-answer matches among the 18 unique code/query inputs.
the code is label-supervised and selected by known case identity; this cannot establish query-blind compression, generalization, or a fixed-byte frontier.
the prior pilot is unchanged, and no development, official test, or external evaluation is run for this diagnostic.

the oracle code, runner, and behavioral tests are committed as `2dfd3eb`.
all 1,249 tests pass, including deterministic MPS, payload ownership, shared readout parity, and a tiny-Qwen command-line run.
independent review verifies accumulation equivalence with frozen active LoRA and finds no prelaunch blocker.
the fixed full-size run is launched at `artifacts/predictions/native_memory_oracle_20260904`.
its exact script, implementation sources, data, hashes, environment, and initial weights are saved before optimizer updates.
the planned training budget is 1,200 microbatches, 93,400 logical forward tokens, and 3,400 supervised tokens.

the oracle run completes and passes its fixed rule: 6/6 fitted answers, 18/18 unique donor-code matches, 36/36 full-grid matches, and 6/6 unknown outputs under both drop and zero controls.
full history also scores 6/6.
normal mean answer CE is 0.000240861; off-diagonal CE 5.21243 is against the original query-case answer, not the donor answer.
training takes 425.72 seconds and evaluation 24.43 seconds.
independent review reproduces every prediction, score, token count, hash, and final-only checkpoint rule.
checkpoint SHA256: `deef6350977ccf3c6bb93d3f6bb348d0d30001ee6215e3fd4ffe815551157496`.
the narrow readout can express distinct answers under privileged fitting; the negative writer pilot is unchanged and remains unexplained by any one controlled factor.
Obsidian note 31 records the mechanism, result, costs, and limits.

predeclare the next training-only writer gate on the same six fitted histories.
use fresh writer seed 1337, copying only the oracle's saved untrained initial read projection, never trained oracle weights or code targets.
use segment length 32, which gives one write for each 12-token history, and the same 200 balanced AdamW updates with six accumulated microbatches.
evaluate only the final checkpoint; fit success requires 6/6 fitted answers and 6/6 unknown responses under dropped memory.
retain mismatched, zero, and full-history controls.
also evaluate all six histories against the three unique questions using independently replayed world-state answers: ten answerable cells and eight unknown cells.
this probe includes the six fitted questions and twelve unfitted questions on already fitted histories, so it is not held-out-history generalization.
all histories fit packed raw retention; neither fitting success nor the binding probe can establish a storage advantage.

prelaunch review confirms the writer gate's history-only input, resets, untrained-projection transfer, fixed 200-by-six loop, and cache-free controls.
the probe now labels each fitted pair explicitly and reports the twelve unfitted pairs separately from the full eighteen-cell grid.
89 focused native-memory/oracle/prefix/symbolic tests pass.
the full-size run is launched at `artifacts/predictions/native_writer_gate_20260904` from `artifacts/predictions/fit_native_writer_gate.py`.
this is an experiment using the already committed model, not another architecture change.
Glimpse returned the focused prefix-loss call path correctly in this stage; rg remained the direct tool for file discovery and exact consumer/control searches.
this supports keeping the current tool split, but no search-latency benchmark was run.

## 2026-09-04 - writer overfit succeeds, binding transfer does not

the fixed writer gate completes all 200 updates and 120 prediction records.
fitted answers score 6/6; mismatched histories score 0/6; drop and zero both return unknown on all six fitted questions.
all three query variants for each history produce its original fitted answer.
on the twelve unfitted pairs, memory scores 0/4 for newly queried facts and 0/8 for missing entities.
full history scores 4/4 for newly queried facts but only 2/8 for missing entities, so the latter also exposes a reader-capability gap.
the six incorrect full-history missing-entity outputs name rooms, not alternative abstention phrases.
do not collapse these two issues into one claimed cause of the original pilot failure.

normal fitted mean CE is 0.000186574.
training takes 564.80 seconds for 1,200 microbatches, 107,800 logical forward tokens, and 3,400 supervised tokens.
evaluation takes 46.01 seconds; maximum recorded step-boundary driver allocation is 4,007,428,096 bytes, not peak memory.
checkpoint SHA256: `95bcf17bf3fdd289444ce80b176ec056a953c952f297d8d6b9c3689e132c996f`.
Obsidian note 32 records the exact world-state grid, controls, costs, and interpretation.

revise the immediate queue: use multiple answerable questions per history with the current reader frozen before adding recurrence difficulty.
separately expand visible-evidence reader qualification to partial missing-entity cases; empty-facts abstention is insufficient.
if reader adaptation is required, keep it separate from writer comparisons and requalify one shared frozen version across all methods.
the small binding probe is consumed training-history diagnostic material, not a new holdout.
the original fixed-byte learned-compression question remains open; first establish usable multi-question bindings.
the final independent audit verifies all 120 writer predictions and aggregates, exact initial/final weights and hashes, and constant fitted-answer outputs on all eighteen normal grid cells.
all 1,249 tests pass after both runs; all 51 installed packages are compatible.
tracked Git state is clean after the code-only commit `2dfd3eb`; notes and experiment artifacts remain local and uncommitted as requested.

## 2026-09-04 - continue to final research results

the user explicitly requests persistent work through final results, including investigation and literature-guided changes when methods fail.
the active objective is a reproducible fixed-byte comparison with multiple seeds, predeclared held-out histories, causal controls, and a bounded positive or negative conclusion, not another six-example stopping point.
use approved local resources; new paid compute and external publication still need separate authority.

predeclare the multi-question known-binding fit: same six histories, frozen reader, segment length 32, and exact untrained initial writer from the previous gate.
the known-query counts are 2, 2, 2, 1, 1, and 2, totaling ten answers.
write each history once per optimizer step, average its known-question answer losses, then average the six history losses.
train 200 AdamW steps with learning rate 0.001, weight decay 0.01, and clipping at one; evaluate only the final checkpoint.
the fit criterion is ten correct known answers and ten unknown outputs for their dropped-memory controls.
the eight missing-entity cases remain diagnostic-only, because the full-history reader is not qualified on them.
the four previously new known pairs now become training targets and must not be described as held out.
the budget is 1,200 history writes, 2,000 answer-loss reads, 170,200 logical forward tokens, and 5,800 supervised tokens, more compute than the previous single-question fit.
different-history accuracy need not be zero because some donor histories share a correct binding.
independent review and tiny CPU/MPS checks verify shared-state gradient equivalence, no reader gradients, strict initialization loading, token accounting, and no future-query input to the writer.

the implementation is committed locally as `cbfbf8f`, after 22 focused and 1,252 full-suite tests pass.
the fixed run starts at `artifacts/predictions/native_multiquery_gate_20260904`.
Obsidian note 33 records the objective, gradient branching, training-pair roles, fixed budget, and pending result.

the original shared-reader training manifest contains 1,980 training and 370 internal-development answers, with zero `unknown` targets in either split.
expanding qa1 questions for Mary, John, Daniel, and Sandra, without changing their histories or partitions, yields 1,207 known and 393 missing-entity training pairs, and 145 known and 55 missing-entity development pairs.
this motivates a separate reader-abstention adaptation stage, not a change to the active writer experiment.

## 2026-09-04 - partial-evidence reader continuation protocol

the derived qa1 query helper is committed as `1b4b51a`, with 83 lead-run focused tests, 91 independent focused tests, and 1,262 full-suite tests passing.
independent regex replay validates all 450 training/internal-development histories and the unchanged split identities.
the excluded 100-history gate expands to 292 known and 108 missing-entity pairs; 100 duplicate the original qa1 prompts and 300 are new pairs.
keep the fixed training-derived name roster Mary, John, Daniel, and Sandra.

predeclare a separate continuation of the selected step-400 reader adapter: 200 steps, four accumulated examples, learning rate 0.00005, AdamW weight decay 0.01, clip one, optimizer reset, and seed 1337.
uniformly sample six categories, then one query within the category; only the existing LoRA parameters train.
use final checkpoint only, with no development checkpoint selection.
record all original and derived native evaluation inputs before training, and bind the source adapter to its original training protocol and manifest hashes.
accept only at least 95% in full-context qa1, both correction groups, exact copy, and randomized bindings, plus at least 95% in both partial-context known and missing groups, and 628/628 empty-context unknown responses.
qa2/qa3 remain separate capability-limited panels.
this reuses a consumed development gate and does not touch external or final compression tests.
independent review reproduces all 1,256 original gate prompts and predicts 800 microbatches, 690 distinct cases, 138,758 forward tokens, and 2,144 supervised tokens.
the one-off runner is `artifacts/predictions/adapt_reader_abstention.py`; the proposed output is `artifacts/predictions/reader_partial_evidence_20260904`.
Obsidian note 34 records the gap, symbolic labels, protocol, and pending result.

the active writer run still uses its original unmodified source snapshot and old frozen reader.
small repository tests run during training, so elapsed timings are descriptive local measurements, not isolated hardware benchmarks.
the next performance check will compare serial and batched reads only after the active full-model process exits.

## 2026-09-04 - do not confuse a narrow interface with general feasibility

the primary-source review supports retaining 66 bytes as one frontier point, not the sole feasible architecture.
our two width-8 states constrain every expanded Qwen input to one shared eight-dimensional subspace; the writer also projects token features to width eight before aggregation.
two full-width Qwen bf16 vectors would use 8,192 value bytes, or 8,194 with two validity bytes, compared with our 66 bytes.
the 12-token fit histories need only 28 bytes with a capacity-12 packed raw format, so fitting them is not a raw-storage compression result.
512 Qwen IDs need 1,154 packed bytes including their capacity-dependent length header.
these are calculated tensor-format payloads, not full-process memory measurements.

[No Mean Feat](https://arxiv.org/html/2510.20797v2) pools full-width hidden states from a trained bidirectional encoder.
its Gemma2-2B ablation loses 13.3 average macro-F1 points when the encoder is frozen, compared with 3.6 when the decoder is frozen.
this does not isolate our writer failure, but argues against treating frozen narrow feature pooling as a universal test of learned compression.
[ICAE](https://arxiv.org/html/2307.06945v4) also retains full-width slots and trains an encoder for its fixed reader.
their token-count ratios do not establish an advantage over packed raw IDs with recomputation.

if narrow multi-question or broader development training fails, compare wider temporary aggregation at unchanged storage and a full-width persistent oracle/readout control before declaring a general negative.
use one shared state per history for every oracle question, not one code per answer.
retain the narrow result and report larger-state results as a different budget, with raw retention at each budget.
do not run the 33/66/132-byte sweep merely to repeat a diagnosed interface failure.
after a mechanism works on development data, freeze methods and budgets before confirmatory evaluation.

## 2026-09-04 - multi-question fit completes at 8/10

the declared ten-answer gate fails: known memory 8/10, different history 1/10, dropped/zero unknown 10/10, and full history 10/10.
all four added known targets are correct, but two original targets regress: Daniel/bedroom becomes hallway and John/garden becomes bathroom.
histories 0 and 5 answer both known people correctly; histories 1 and 2 still collapse to one answer.
missing-entity memory remains 0/8 versus full history 2/8.
known-query mean CE is 0.0906651; exact answers remain the decision metric.
training takes 882.39 seconds for 170,200 logical tokens and 5,800 targets; evaluation takes 46.83 seconds.
checkpoint SHA256: `d950be7ac5e83df927a15bd777540e7d5cfaf4d764d78e95e5e4ea0e5adf4ad4`.
independent audit verifies all 120 predictions, 24 native encodings, exact initial weights, every aggregate, and token accounting.
all 72 full/drop/zero control rows replay the preceding writer experiment exactly.
Obsidian note 33 is updated with the result, regressions, and limits.

the next memory diagnostic fits one oracle code per history jointly to all its known questions, using the old frozen reader and exact untrained original oracle initialization.
keep 200 updates, six history losses, learning rate 0.001, weight decay 0.01, clip one, final-only evaluation, and serial prefix reads.
the read projection is verified identical to the initial writer projection; no trained memory weights are transferred.
this uses 155,800 logical forward tokens and 5,800 supervised tokens, with no history encoding forward pass.
all eighteen world-state cells receive five evaluation conditions, for 90 prediction rows.
the output is planned at `artifacts/predictions/shared_history_oracle_20260904`.

## 2026-09-04 - batched-read profile rejected for current experiments

the five-round synchronized full-model profile completes at `artifacts/predictions/batched_prefix_profile_20260904`.
serial median is 3.44190 seconds for ten weighted read/backward queries, versus 2.95358 seconds for batch ten, a 1.1653x read-path speedup only.
all batch sizes fail the predeclared 0.01 maximum CE error, with a maximum difference of 0.243862.
batch-ten gradient relative error is 0.01225 and cosine 0.999928, within their separate limits.
do not promote the optimization or loosen the threshold; investigate the full-size difference while retaining serial reads for the next scientific run.
tiny CPU/MPS tests, including bfloat16 and frozen LoRA, passed, so the full-size check adds a material qualification absent from unit tests.
the source and metrics are preserved; the new helper remains uncommitted pending diagnosis.
the separate 200-step reader-abstention continuation is now running and does not use that helper.

the numerical audit finds a specific dispatch hypothesis: nine queries have 78 input positions and three answer tokens; office has 77 positions and two answer tokens.
mixed padding makes the installed Transformers SDPA path use an explicit causal-padding mask and repeated KV heads instead of the mask-free GQA path.
a tiny CPU reproduction confirms that dispatch change but retains matching CE, so the full-size discrepancy is not yet attributed to it.
the next isolated diagnostic must save per-query and per-token values, compare office with keep2 versus keep3, then compare explicit-mask and mixed-batch hidden states and output logits.
do not label this a target-alignment bug or a confirmed bf16 cause before that check.

## 2026-09-04 - reader qualification passes and padding is isolated

the fixed reader continuation passes: partial known 291/292, missing entity 108/108, and empty-context unknown 628/628.
all required original full-context groups remain perfect; qa2 is 80/100 and qa3 is 85/100, still below the separate competence threshold.
training uses the declared 138,758 forward tokens and 2,144 targets in 650.06 seconds; gate generation takes 382.50 seconds, excluding internal-development CE evaluation.
freeze the final adapter after independent audit: `artifacts/predictions/reader_partial_evidence_20260904/step_000200`.
adapter SHA256 is `1c4402c1c808474f7f964e4e492215f98aac3998ea9c0b40fe2a81d6b26ad9f0`.
the shared-history oracle still uses the old adapter for a controlled comparison with the previous fit.

the isolated numerical diagnostic completes at `artifacts/predictions/batched_prefix_diagnostic_20260904`.
the office query with one left pad reproduces the full 0.243862 CE discrepancy even when run alone; hidden-state maximum error is 1.5 and relative L2 error is 0.015797.
an explicit causal mask without padding and keeping three rather than two final logits both match serial exactly.
the nine equal-length queries and a duplicated unpadded office pair also match serial hidden states and CE exactly.
actual API losses match independent CE recomputation, excluding a reduction or target-cropping explanation for this run.
this isolates padding as the reproducing condition, not a specific low-level kernel defect or dtype-only cause.
replace padded batching with exact input-length and answer-length groups, restore original result order, and retain the same numerical thresholds.
18 focused tests pass; the same five-round full-model profile is rerunning before promotion.
scientific oracle training remains serial as predeclared.

the grouped profile passes all unchanged thresholds: zero CE error, maximum gradient relative L2 0.009606, and minimum cosine 0.999953 across the batching modes.
batch-ten median read/backward time is 3.16009 seconds versus serial 3.57008 seconds, a 1.12974x speedup for these ten training queries only.
this is not a full-training or held-out quality result.
independent code review passes, followed by all 1,268 tests; the code-only commit is `d1186e5`.
the shared-history oracle starts at `artifacts/predictions/shared_history_oracle_20260904` with its original serial protocol.

## 2026-09-04 - reserve native-study held-out history groups

freeze 512 qa1, 256 qa2, and 256 qa3 connected groups before further model development.
the explicit 31 native manifests/protocols exclude whole components by context, episode ID, or connected-history ID, including declared but unsampled prior data.
the five en-10k training files contain 50,000 question rows, 18,000 episodes, and 9,700 connected components; 1,639 components are excluded by prior native use.
selection uses fixed SHA256 ranks with seed 20260904, then a deterministic representative episode and its final question, without filtering by answers or performance.
the reserve contains 1,024 groups, 1,050 member episodes, and 5,224 distinct member contexts.
independent review checks closure, no prior overlap, representative selection, and invariance under reversed traversal.
the frozen manifest is `artifacts/predictions/native_holdout_reserve_20260904/data_manifest.json`.
its SHA256 is `968a639f8cd0617120aab161ca44244d08e06278c85401db974a7ce6bf5db5f4`, exactly matching the independent prelaunch calculation.
source, parser, schema, runner, and every exclusion-file hash are saved.
no model has evaluated the reserve, and no official test or LongMemEval data is opened for this stage.
exclude every member of the selected groups from all later reader and memory training/development, not just the representative questions.
this is native-path unused material, not certified project-wide untouched or pretraining-clean data; older byte-model source pools overlap, and uncaptured ad hoc use is not ruled out.
methods, budgets, seeds, delays, and derived query rules still need final preregistration after development; this freezes data identities only.

## 2026-09-04 - prepare wider computation at unchanged stored bytes

add optional `QueryPoolSlotWriter`: learned full-width attention over current hidden states and lifted old slots, followed by a 2,048-to-64-to-8 GELU compression head and an elementwise gated update.
retain the old `RecurrentSlotWriter` as the default; all default initial weights and checkpoint keys match the previous constructor sequence.
both variants retain two width-8 float32 slots and two validity bytes, totaling 66 bytes.
the wider variant has 185,040 shared memory-module parameters versus 33,216, so this is not a parameter-matched comparison.
the full native reader, query-blind write boundary, cache-free reads, and attached recurrent gradients are unchanged.
49 focused tests, independent CPU review, and all 1,298 repository tests pass, including empty-segment identity gradients and deterministic MPS backward.
commit `ea06729` contains only the completed implementation and tests.
Obsidian note 36 explains the attention queries, equations, shapes, storage distinction, tests, and limits.

the conditional fit runner is `artifacts/predictions/fit_native_query_pool_gate.py`.
it refuses to launch until the shared-history oracle passes its final ten-answer gate.
use the same old frozen reader, exact untrained read projection, six histories, ten known questions, 200 balanced updates, and serial reads.
the fit uses 170,200 logical forward tokens and 5,800 targets, with final-only selection and 90 world-state predictions under five conditions.
the frozen holdout-reserve manifest is verified, and all training history memberships and contexts are checked for nonoverlap.
prelaunch review passes; the accuracy experiment has not started.

## 2026-09-04 - shared-history oracle stops at nine answers

the fixed oracle fits 9/10 known answers, versus 1/10 with a different history and 10/10 with full history.
dropped and zero memory return `unknown` on all ten known questions.
John's garden binding still produces Daniel's bathroom answer in history two.
all 90 predictions, symbolic labels, native encodings, checkpoint hashes, and token budgets pass independent audit.
training takes 671.51 seconds and evaluation 33.11 seconds, using 155,800 forward tokens and 5,800 targets.
the final checkpoint SHA256 is `2748766c25b6d0967ee11d3187155fa082d63dc5e022e34d1f8bb29edd0eb569`.
text encoding is not the only remaining issue, but declining late training loss does not support a capacity-limit claim.

declare one final warm-start refinement: 200 additional balanced steps at learning rate 0.0003, unchanged data, old frozen reader, state, objective, and final-only evaluation.
reset AdamW explicitly because the prior run did not save optimizer state; this is not an exact optimizer resume.
the cumulative budget is 400 updates, 311,600 forward tokens, and 11,600 targets.
retain the failed run and the original untrained oracle separately from the refinement's actual starting checkpoint.
the wider writer can start only if this refinement passes the unchanged ten-answer and ten-drop-abstention gate.
if it still fails, stop this narrow optimization branch and test a declared full-width readout control instead of repeating learning-rate searches.
neither outcome is a learned-compression or held-out result.

## 2026-09-04 - independently trainable contextual mean/FIFO baseline

add `MeanPoolSlotWriter` and `writer_kind="mean_pool"` to the native path.
pool current full-width hidden features by a masked mean, apply a 2,048-to-64-to-8 GELU/tanh head, then append one summary and evict the oldest slot.
the frozen encoder still sees old slots and the current chunk, so name this contextual mean-pool + FIFO, not independent-chunk pooling.
there is no learned pooling attention, gate, or replacement rule.
two slots retain 66 bytes, with 148,040 shared module parameters versus query pooling's 185,040.
one valid slot after the first write is a genuine policy and compute difference, not a comparison to hide.
train the baseline independently under its own policy with the same reader, data, initial read projection, and answer-loss opportunity.
77 focused tests, independent CPU review, and all 1,332 tests pass.
the new three-write regression verifies gradients after direct FIFO eviction through the contextual encoder; old default initialization, checkpoint keys, and RNG state are unchanged.
code-only commit: `b2843da`; Obsidian note 37 documents the mechanism and limits.
accuracy training has not started, and no result is inferred from the tests.

## 2026-09-04 - comparison design and raw-coding fairness

prepare two independently trained methods, full-width query pooling and contextual mean/FIFO, at two slots with widths 4, 8, and 32.
their persistent states use 34, 66, and 258 bytes; full-vocabulary packed raw capacities are 14, 29, and 114 IDs, using 32, 66, and 258 bytes respectively.
use optimization seeds 1337, 2027, and 4099 with a shared frozen qualified reader and copied initial read projections within each seed/budget pair.
these are provisional comparison choices, not a frozen confirmatory protocol.
complete visible sentences are the simplest common streaming boundary for every method, including a raw latest-fact policy without hidden partial-sentence staging.
this changes the prior fixed-token chunk regime and does not establish arbitrary-boundary robustness.

the raw latest-fact reference reconstructs its temporary subject map only from retained packed text and the current sentence, with no question or supporting-fact label.
it preserves actual sentence text and uses the fixed qa1 movement grammar to ignore unrelated prose.
it is domain-aware and must not be described as a general learned semantic memory.
four current bAbI bindings may already fit the 66-byte native-ID budget, so delay-only improvements over recent truncation are insufficient.
an additional fairness risk is that a shared training-vocabulary remap can use fewer than 18 bits per Qwen token ID.
audit that coding opportunity before finalizing budgets or launching eighteen fresh runs; shared dictionary cost belongs beside shared model-parameter cost.
do not select a weak lossless representation merely to create a learned-compression advantage.

## 2026-09-04 - fixed oracle refinement passes

the final step-400 oracle answers all ten known questions correctly, versus one with a different history.
dropped and zeroed memory return `unknown` on all ten; missing-entity normal accuracy remains 0/8 with the old reader.
known mean answer CE is 0.0024330489.
the additional 200 updates take 665.82 seconds and final evaluation takes 33.34 seconds.
the cumulative 400-update budget is 311,600 forward tokens and 11,600 supervised tokens.
final checkpoint SHA256: `1e5d70ddd54e5acd3694bec6045471f0cbb59bdcfbe6883167a6b8bb47719884`.
the unchanged ten-answer gate passes, so the proposed wider writer fit is eligible for launch after the result audit.
this is privileged shared-history readout feasibility, not learned compression or a same-compute win over the 200-step writer.

paired-history uncertainty is committed separately as `c03304d` after 25 focused tests, independent review, and all 1,357 tests at that validation point.
resample history groups after averaging paired seed differences; do not count correlated questions or repeated seed scores as independent histories.
the interval is conditional on the trained checkpoints, with per-seed variability reported separately.
Obsidian note 38 records the method and the Python seed-key validation issue found during review.

## 2026-09-04 - latest-fact raw baseline and a smaller lossless ceiling

commit `8a4955b` adds the query-independent latest-movement raw-sentence policy, with no stored subject map or gold evidence flags.
70 focused tests, independent CPU review, and all 1,404 tests pass at this validation point.
review found and fixed stale-state creation from inline multiple facts and from line separators recognized by `splitlines()` but not by the original validation.
the local Qwen tokenizer confirms exact text reconstruction across all six consumed fit histories and policy recreation between updates.
the input contract is complete visible sentences, so arbitrary token-boundary robustness is not claimed.

the train-only vocabulary audit finds 21 native IDs across all 400 saved qa1 training histories and all 2,354 latest-fact update states.
the largest selected state has 29 IDs; five bits per code plus a five-bit length header occupy 19 bytes.
the shared uint32 dictionary needs 84 serialized bytes, with its Python lookup-table runtime cost reported separately.
this is evidence coverage, not yet reader accuracy.
it makes the proposed 34/66/258-byte qa1 sweep unsuitable as a strong-raw capacity frontier.
implement a training-frozen short-code format with native-ID escapes for unseen tokens before revising the comparison.
retain qa1 for causal learned-writing generalization; a higher-entropy capacity extension is a separately declared experiment, not a hidden change to this benchmark.
Obsidian note 39 explains the coding issue, parser repairs, baseline scope, and effect on the research plan.

## 2026-09-04 - lossless vocabulary storage is complete

commit `1966af0` adds training-frozen vocabulary codes, native-ID escapes, and integration with latest-fact retention.
99 focused tests, independent review, and all 1,407 tests pass.
review found a length-header boundary bug: enough payload bits did not guarantee that the number of entries fit the count field.
both fit checks and eviction now enforce the payload-bit limit and representable entry count.
the state remains one fixed byte tensor, with no per-stream dictionary.

the training audit also finds all 120 complete sentences from four people, five movement forms, and six rooms.
a shared exact-sentence table can encode each in seven bits; four latest sentences and a three-bit count require only 31 bits, or four bytes.
the shared sentence text uses 3,484 UTF-8 bytes before offsets.
this is an analytical training-coverage bound, not a measured reader result or a general-purpose codec.
the 19-byte token remap is therefore not the strongest possible lossless raw reference.

prepare one explicitly labeled bAbI-derived capacity stress instead of the proposed eighteen-run low-entropy sweep.
combine two independent four-person stories, replace names with eight fresh 80-bit IDs, permute the six room names, and interleave four complete-sentence chunks while preserving each source's order.
ask every known entity and one absent entity from the same query-blind state.
keep paired short-name and changed-location controls.
eight exact fresh identities exceed a 66-byte payload, but this is not a lower bound on answer accuracy: queries supply identities, so partial IDs or lossy fingerprints may suffice.
strong raw and domain-specific fingerprint controls must remain visible.
first qualify the frozen reader on the exact new format; do not infer a compression failure from a reader failure.
freeze the final training, baseline, compute, and confirmation protocol after that qualification and a ten-update local profile.

the wider writer fit is running on the earlier six-history development gate with its predeclared 200-step budget.
no new full-size model process will run beside it.

## 2026-09-04 - wider writer fit passes its fixed gate

the wider writer answers 10/10 known questions after 200 updates, versus 1/10 under different-history memory and 10/10 unknown responses when memory is dropped or zeroed.
known mean answer CE is 0.02983870483; missing-entity memory remains 0/8 with the old reader, versus 2/8 with full history.
all 90 predictions, 18 symbolic/native encodings, 54 unchanged controls, source hashes, and initial weights pass independent audit.
training takes 855.295 seconds and evaluation 35.457 seconds; the budget is 170,200 forward tokens and 5,800 targets.
checkpoint SHA256: `18ed90affe445a1ec7510a74e6ea1a4096a8c7490191bcda01ee205de18b066e`.
the state remains 66 bytes, with 185,040 shared parameters versus the earlier narrow writer's 33,216.
these 12-token histories use one write, so this is not learned multi-write recurrence, held-out compression, or a win over raw retention.

## 2026-09-04 - freeze the derived association data and exact chunk interface

commit `88fa86c` adds the source-preserving opaque-world transformation after 19 tests and independent review.
commit `fb86bd1` adds exact native chunk boundaries, shared-history answer loss with explicit chunks, and whole-history preflight validation.
31 focused chunk/training tests and all 1,463 repository tests pass.
Obsidian note 40 explains the task, controls, provenance, gradients, and issues.
the source-disjoint 256/32/128-world dataset is frozen at `artifacts/predictions/opaque_qa1_data_20260905`.
all 832 source groups are unique across and within splits, and all original reserved groups are excluded.
the new confirmation sources exclude all prior native-consumed groups; exact input-based selection fingerprints match independent reconstruction.

review fixed a renaming error caused by parser-normalized whitespace and substring offsets.
the initial builder dry run also parsed reserved source answers before filtering; no model evaluated them and they did not affect selection.
the corrected fact-only first pass fixes all representative selections before parsing only selected nonreserved answers.
preserve this access-boundary issue in the technical write-up rather than describing the first dry run as answer-blind.

the exact-format reader qualification is running on 32 development worlds, with full opaque, full short, and no-facts conditions, totaling 864 greedy unpadded predictions.
thresholds are at least 95% known and missing accuracy for each full-context variant and 100% no-facts abstention.
the confirmation set is not evaluated in this qualification.

## 2026-09-04 - stronger raw and approximate association references

commit `be5dec2` adds exact template-coded latest facts: 87 bits per opaque-name sentence, six-bit count, six complete sentences in 66 bytes, and 308 shared grammar JSON bytes.
20 focused tests and independent layout/update review pass; all 1,463 tests pass at that integration point.
commit `6a7a2f4` adds the separate handcrafted fingerprint reference: 16-bit SHA prefix plus three-bit room, direct lookup without Qwen, and explicit last-write collision behavior.
16 focused tests, independent real-collision and layout checks, and all 1,479 tests pass.
this is approximate semantic storage, not lossless raw text or learned compression.
Obsidian note 41 explains both mechanisms and comparison limits.

the durable training-only audit is `artifacts/predictions/raw_capacity_audit_20260905`.
it verifies exact reconstruction of all 2,354 original qa1 update states in 19 bytes.
the 256 opaque training worlds use 215 native token IDs, with an 860-byte shared vocabulary table.
at 66 bytes, vocabulary-coded latest facts retain 2.01171875 facts per world on average and the template method retains exactly six.
fingerprint lookup answers 2,048/2,048 known and 256/256 absent training queries correctly, with no observed within-world known or absent-name collisions.
no pretrained model, development set, or confirmation set is used in this audit.
these are coding and symbolic coverage results, not Qwen answer accuracy or held-out evidence.

supplemental native-run exclusion evidence is saved separately at `opaque_qa1_data_20260905/supplemental_exclusion_audit.json` without rewriting the frozen data protocol.
it binds the later oracle, wider writer, grouped profile, and numerical diagnostic artifacts, rechecks all 31 original exclusion files, and confirms no additional source groups beyond the existing six training histories.

## 2026-09-04 - exact-format reader misses the absent-identity gate

the opaque reader gate completes in 317.353 seconds and fails the unchanged acceptance rule.
full opaque histories score 246/256 known (96.09375%) and 28/32 absent (87.5%).
full short histories score 254/256 known and 32/32 absent; no-facts inputs return unknown on all 288 queries.
independent audit verifies all 864 predictions, 576 native encodings, source hashes, and scores.
prediction SHA256: `d6e970f4cd7682a52206e49c25f8bd1e25323c8df5062ed8c6d45bbc9d7e1a8f`.
the four failed absent queries produce room names, not malformed answers.
the complete IDs are absent and differ in at least 14 to 17 hexadecimal positions from every present ID.
some share short prefixes with entities whose rooms are predicted, suggesting weak identity matching without proving an attention mechanism.

declare one fixed continuation from the accepted partial-evidence adapter: 100 updates, fresh AdamW at 0.00002, weight decay 0.01, clipping at one, and four serial microbatches per update.
each update samples one known opaque, one absent opaque, one known short, and one absent short query from the 256 training worlds.
train only the existing rank-eight query/value LoRA; do not add a compressor, change names, shorten identities, lower thresholds, or select intermediate checkpoints.
evaluate only the final adapter on the exact saved 864 development prompts with the same unpadded greedy generation.
the memory-training runner must reject this failed reader until a later qualified result is available.
the confirmation worlds remain unevaluated.

## 2026-09-04 - validate the final comparison boundary

both final evaluators pass an independent executable preflight review.
valid six-run fixtures are accepted; duplicate paths, duplicate writer/seed pairs, missing steps, wrong checkpoint or reader hashes, and wrong token totals are rejected.
confirmation files are opened only after the reader, data protocol, completed run matrix, and final checkpoint checks.
the scripts preserve source snapshots and distinguish generation-only timing from complete read cost.
the aggregation plan resamples worlds, not their correlated questions, and reports optimization-seed variation separately.
this verifies the tools, not the as-yet-unrun final comparison.

## 2026-09-04 - exact-format reader repair passes

the fixed 100-update continuation passes all unchanged development requirements.
full opaque histories score 253/256 known and 32/32 absent; full short histories score 256/256 known and 32/32 absent.
all 288 no-facts queries return unknown.
the independent static audit verifies all 100 updates, 400 microbatches, 360 distinct queries, 156,703 logical forward tokens, and 965 targets, plus native-tokenizer replay for all training and evaluation inputs.
training takes 681.662352875 seconds and evaluation takes 341.348422250 seconds.
the final adapter weights have SHA256 `9f55aab03ccbf91a29765ca4d5083080be7747badf5e072814600ca781fdbb2c`.
the prediction SHA256 is `9e3add28609c068c3c79efc3ccb4037aefe8e286011fbb84772234db6f8b3b5c`.
this is reader qualification on development data, not confirmation or a compression success.
preserve the original 28/32 absent failure; it motivated a training-only repair, not a changed threshold or shortened identifier.
profile each writer for ten fresh updates before freezing the complete six-run budget.

## 2026-09-04 - freeze and start the six-run association study

independent review verifies all 864 final reader predictions, their decoded token IDs, and the unchanged qualification thresholds.
both full-size ten-update writer profiles pass, with finite positive gradients through all four states on every update.
query pooling uses 13,523 forward tokens and 247 targets, with a 5.799951625-second median update.
mean/FIFO uses 13,513 forward tokens and 247 targets, with a 5.698896125-second median update.
all initial weights, shared initial read projections, schedules, native encodings, source hashes, and profile checkpoints are independently verified.
freeze two methods, seeds 1337/2027/4099, 1,000 updates per run, 66 bytes, and final-only evaluation.
the projected training time is 9.582373125 hours, excluding evaluation.
protocol: `artifacts/predictions/opaque_memory_study_20260905/protocol.json`.
SHA256: `e74cf9fa0027da9ad43bbe7914c56d3601c9e4428517912a42663f733ec8e566`.
the first fresh query-pool seed-1337 run starts at 03:04 UTC on September 5, or 23:04 local time on September 4.
all six runs must finish before confirmation opens; profile weights are not reused.

the confirmatory family compares known-answer accuracy with trained mean/FIFO and all four raw methods using five 99% paired-world intervals and 100,000 bootstrap resamples.
absent-answer noninferiority is a separate gated family with a five-point margin and an observed 122/128 per-seed threshold.
different-history results are descriptive because cyclic donors couple worlds.
do not call an overlapping interval equivalence or general impossibility.

commit `ee6c9c5` adds prediction-independent association timing and change-count labels for the already-declared diagnostics.
16 focused tests, independent replay of all 7,776 train/development variant labels, and all 1,495 repository tests pass.
the 24 frozen study source files are unchanged by this separate reporting module.
Obsidian notes 38, 40, and 42 record the reader repair, statistical rules, training protocol, diagnostic meanings, and current limits.

## 2026-09-04 - implement the predeclared result assessment

commit `773a467` implements the five-baseline result rules without changing the frozen study.
reader limitation, known superiority/inferiority/inconclusiveness, practical point gain, and gated absent-answer noninferiority remain distinct.
the expected world, seed, and bootstrap design must come from the frozen protocol rather than from the result being checked.
review's malformed fixtures exposed acceptance of consistently wrong metadata and boolean-valued gains; both are now rejected.
these were fixture defects, not evidence that real experiment records were corrupted.
49 focused tests, independent review, and all 1,544 repository tests pass.
Obsidian note 38 records the mechanism, strict boundaries, limitations, review fixes, and commit.
the eventual report integration and actual final assessment remain pending while training runs.

## 2026-09-04 - verify the final report before confirmation

commit `afe013e` adds the report CLI, explicit result assessment, and two static scientific figures.
known answers, absent answers, optimization-seed variation, and paired-world bootstrap intervals remain separate.
deterministic raw methods do not acquire artificial optimization replication from copied seed labels.
review found two report defects: ignored analysis-source hashes and omitted seed standard deviations.
the final version checks the required analysis sources against the frozen protocol, resolves relative/absolute path aliases, and retains both method and paired-gain sample standard deviations.
tests use self-contained synthetic fixtures and do not depend on ignored experiment artifacts.
21 focused report tests and all 1,565 repository tests pass; independent review approves the source and statistics checks.
both figure layouts were inspected with explicit synthetic, non-research labels.
Obsidian notes 38 and 42 record interpretation, limits, review fixes, and the commit.
the six-run training protocol and its 24 frozen source files are unchanged; confirmation remains closed.
at 03:26 UTC on September 5, queue the other five training runs behind the existing first process with a direct fail-on-error shell sequence.
the order remains query/1337, mean/1337, mean/2027, query/2027, query/4099, mean/4099.
the queue waits for the first process to exit and requires its final result file before starting the next model.
each later command must exit successfully before the following one starts; no second full-size Qwen is loaded in parallel.
confirmation evaluation is not part of this training queue and still requires the complete independently checked run matrix.

## 2026-09-04 - prepare a conditional representation diagnostic

independent architecture review and primary-source checks support a small adaptation-location ablation if the fixed study fails.
the proposed three 200-update arms use the predetermined query-pool seed-1337 checkpoint: writer-only continuation, write-side LoRA only, or answer-read LoRA only.
only train/development data may be used; no confirmation-based checkpoint selection or retrospective change to the six-run study is allowed.
the exact proposal and limits are recorded in `docs/research_plan.md` and Obsidian note 42.
no experimental adapter path is implemented or trained yet.

installed PEFT changes active routing and `requires_grad` in `set_adapter`.
checkpoint recomputation must keep the original route; equal forward losses alone cannot establish correct gradients.
the present frozen-reader guard remains correct for the running study.

Glimpse initially aborted with an implausible allocation request while reading its index.
one `glimpse index build --force /Users/caleb/TinyMem` completed in 0.33 seconds, and the targeted `write` call graph then succeeded.
literal searches still use `rg`; no experiment source or model state changed during index repair.

## 2026-09-04 - tiny fixed-route adapter probe

the CPU-only prototype uses separate encoder and decoder module trees with shared frozen base parameters and distinct LoRA parameters, configurations, buffers, and routing state.
copying occurs before checkpoint hooks are installed; no adapter switch occurs during a graph's lifetime.
the prototype preserves the production frozen-reader guard and copies the write mathematics only for trusted tiny float32 inputs.
four writes and nine answer losses are checked for writer-only, encoder-only, and decoder-only gradient ownership, with checkpointing and base-weight sharing each enabled or disabled.

the initial 12-case probe passes with identical adapter values.
the stronger 12-case probe deliberately changes encoder LoRA values: encoder logits move by 0.18588504195213318 while decoder logits stay bit-identical.
all trainable gradients match their non-checkpointed, unshared references exactly; the maximum observed difference is zero.
independent in-memory replay reproduces every reported value and checks that optimizer steps change only the intended component.
artifacts: `fixed_adapter_routes_cpu_20260905` and `fixed_adapter_routes_cpu_divergent_20260905` under `artifacts/predictions`.
the divergent probe source SHA256 is `1901c74e14a132da15ab28a6fe8bb8cb96daf06ff638527e6744e19a773f13b6`.
these are tiny synthetic numerical checks, not compression accuracy, a production adapter implementation, or full-size MPS validation.
adapter-only save/reload, repeated optimizer steps, dtype/device aliasing, and rejection of unsafe hooks or base-weight mutation remain open checks.
all 24 frozen study source hashes still match, and the original six-run study continues without changes.

## 2026-09-04 - queue the frozen final evaluation sequence

at 03:59 UTC on September 5, start a separate fail-on-error shell sequence that waits for the entire training queue to exit.
it requires the last training result file, then uses the independently reviewed evaluator preflight to verify all six complete runs before opening confirmation.
the sequence runs the seven raw/reference methods, all six learned checkpoints with their declared interventions, the frozen aggregation, and the committed report CLI.
outputs are under `opaque_memory_study_20260905/confirmation`, `analysis`, and `report`.
one full-size model runs at a time; no checkpoint is selected from development or confirmation outcomes.
do not change any of the 24 frozen source files until this complete evaluation and aggregation sequence has finished.
the queue's existence is not a completed evaluation or a successful research result.

## 2026-09-05 - verify within-world answer diagnostics

commit `443212e` measures answer correctness, prediction diversity, reference diversity, and the hindsight constant-answer ceiling for one world's known questions.
exact case coverage and a shared history are required; absent queries are analyzed separately.
an identical prediction can be correct for multiple people in the same room, so identical answers alone do not imply an error.
low accuracy alone does not establish query blindness.
32 focused diagnostic tests, independent checks of all 864 train/development variant groups, and all 1,581 repository tests pass.
Obsidian note 42 records the mechanism, interpretation, limits, and commit.
this analysis module does not change any frozen study source or training choice.
at 04:09 UTC the first run has completed 709 of 1,000 updates; no actual final answer-pattern result is available yet.
reference-only train/development checks put the hindsight constant-answer ceiling at 743/2,048 (36.28%) and 88/256 (34.38%) respectively.
all train/development worlds have at least three distinct correct rooms; no confirmation data is read for this check.
these bounds are not model scores or trained baselines.

## 2026-09-05 - check repeated updates and adapter tensor reload

the CPU continuation probe passes 24 cases across writer/encoder/decoder ownership, checkpointing, shared frozen weights, and tied embeddings.
four uninterrupted updates match two updates, save/reload, and two further updates exactly in losses, gradients, and parameters.
only the intended component changes on every optimizer step, and restored Adam step counters advance correctly.
artifacts: `artifacts/predictions/fixed_adapter_continuation_cpu_20260905`.
source SHA256: `d7341f46600ec4e621fc530075ec71952d78da41782ce65285e9af2b3405bdb9`.
independent review verifies the 24-case grid, source hash, and assertions without rerunning inference.
the probe rebuilds the same tiny random base and configuration from fixed code; it injects saved adapter tensors rather than testing every PEFT reload API.
it is not a standalone checkpoint contract or MPS/bfloat16 validation.
production continuation must bind base and adapter configurations, reject same-shape scaling changes, preserve optimizer parameter ordering, and forbid adapter merging or other shared-base mutation.
Obsidian note 42 records these limits; no frozen study source changes.

## 2026-09-05 - review newer compression training evidence

primary-source follow-up covers Chimera and LatentPress v2, with an independent code review of LatentPress at `cae57c4f168a66aa3a66ead2e18c3c9bbff5e0a3`.
Obsidian note 22 records the methods, source links, storage/evaluation differences, and limited implications for our next diagnosis.
the original audit remains explicitly labeled as a historical snapshot rather than a current unresolved-defect list.
no external benchmark answers, model weights, or third-party training data are imported.
the fixed six-run comparison and conditional adaptation-location diagnosis remain unchanged.

## 2026-09-05 - first final development result

the query-pool seed-1337 run completes 1,000 updates and all 288 development predictions.
known accuracy is 22/256 (8.59%); absent-query accuracy is 14/32 (43.75%).
zero of 32 worlds have a constant known-query prediction; each has two to five distinct predictions.
the output includes 77 false known-query abstentions and 109 known queries predicted as `kitchen`; 14 of those kitchen predictions are correct.
last-mention-chunk counts and correct answers are 4/24, 2/25, 9/104, and 7/103.
the one-off descriptive analysis validates exact case coverage, symbolic history labels, scores, and source hashes without opening confirmation.
this is not a successful binding result or a final multi-seed comparison, and it does not justify changing the fixed protocol.
training uses 1,345,703 logical forward tokens and 24,662 targets in 5,537.695340583101 seconds; final development evaluation takes 104.10204074997455 seconds.
final checkpoint SHA256: `362e91197c8a6f0acb0da5e287cc4a50a3f16b52c99d1cc6afad5eb9351f5402`.
metrics SHA256: `b03e501aafe6e1b881de08d1fc232462e57e94f01adaea7733fe66e766ed3e3a`.
independent audit verifies all scheduled updates, token totals, checkpoint shapes and finite values, optimizer state at step 1,000, and every prediction's source case, native input, decoded tokens, and score.
it finds no mismatch; it does not replay the full model or open confirmation.
development prediction SHA256: `0d9e1f55138e022e94cd775e12789b27d0215bd2ddde6a6d218e37f89afc09f3`.
the training result does not itself bind that prediction hash; the separate development-diagnostics result binds both files and its analysis sources.
the mean-pool seed-1337 run starts automatically from fresh weights; all remaining training and final evaluation stay queued.
Obsidian note 42 and the project index record the poor development result and its limits.
independent replay of the descriptive report also verifies all 32 world patterns, timing/update cohorts, four input hashes, five analysis-source hashes, and saved source copies.
descriptive report SHA256: `4446dff00e6714a81beff400058fa0e59ee72e469f01affcd7716fe06460ca10`.

## 2026-09-05 - test the complete reporting path with synthetic records

`opaque_reporting_pipeline_fixture_20260905_v2` exercises the unchanged frozen aggregator and committed report CLI on synthetic prediction files for all methods, seeds, variants, and 128 fixture worlds.
the synthetic outcome is positive by construction and is not a research result.
the first fixture version used an unsupported protocol name and was correctly rejected; the corrected fixture keeps the expected schema name and explicit synthetic labels.
the complete aggregation, 100,000-resample intervals, decision assessment, and both figures pass; the figure labels and layout are inspected.
the fixture does not simulate model training, validate real state payloads, or open confirmation data.
no production or frozen study source changes.

## 2026-09-05 - separate training fit from interface feasibility

review of the old oracle confirms one or two known questions per history, a trainable shared projection, and no absent-answer training.
its ten fitted answers do not qualify the current eight-known-plus-one-absent task.
before the proposed adapter ablations, measure final training generation for all 256 worlds using the predetermined query-pool and mean-pool seed-1337 checkpoints.
then use a separately declared current-task oracle with the reader and final read projection fixed; fitting shared weights would be a distinct, more privileged condition.
these diagnostics do not change the ongoing frozen study.

the first query-pool run's last 100 online losses average 0.780477.
training answer lengths including the stop token are 599 two-token and 1,705 three-token sequences.
a hypothetical uniform distribution over the seven complete answers has token-mean loss 0.732954 on these lengths, without using any history.
this is a mathematical reference, not an evaluated model, and online CE is not final training fit.
the one-off `evaluate_opaque_training_fit.py` checks all 256 worlds and 2,304 queries in a source- and checkpoint-bound dry run without loading Qwen.
actual inference has not started and must wait until the existing full-size queues finish.
the research plan and Obsidian note 42 record the revised diagnostic order and the limits of the oracle claim.

## 2026-09-05 - implement the fixed-readout oracle component

commit `9fe5811` adds direct bounded history codes with an independently cloned fixed projection buffer.
one materialized two-by-eight float32 state owns exactly 66 bytes, while its clone preserves gradients to only the selected history code.
the caller must apply the declared post-step box projection; nonfinite values are rejected rather than silently repaired.
the old jointly trained oracle remains unchanged.
32 focused tests and all 1,603 repository tests pass, including exact CPU checkpoint loss/gradient agreement across four states and nine queries per state.
independent review finds no blocker and checks additional dtype, noncontiguous-input, readout-equivalence, and lifetime cases.
no full-size oracle fit or confirmation read occurs during these checks.
all 24 frozen study sources remain unchanged.
Obsidian note 43 records the mechanism, limits, proposed fixed four-world protocol, and local commit link.
the reviewed training-fit runner SHA256 is `f084d0cafd3b4160e28b779b188aa9c45e2250b197f729a8d44ba750e0178f3b`.
its deterministic settings now match the study, and its protocol records numerical settings and package versions.

## 2026-09-05 - bound the precision and width hypothesis

read-only architecture review identifies a possible later equal-byte tradeoff: two width-16 bfloat16 slots instead of two width-eight float32 slots.
this is not a new mandatory sweep and does not replace the training-fit and current fixed-projection checks.
constructor-based CPU counts verify 235,104 query-pool and 202,114 mean/FIFO parameters if current temporary widths are retained.
temporary widths 40 and 74 instead give 185,544 and 185,594 parameters, but introduce another architectural change.
the existing reader cast occurs after projection, so it does not establish that stored float32 precision is unnecessary.
the final query-pool seed-1337 projection has rank eight, with singular values from 8.778308 to 10.101588; simple projection-rank collapse is not observed.
neither matrix rank nor learned-query norms identify the actual state information or attention selectivity.
Obsidian note 43 and the research plan record the hypothesis and its required dtype, state-ownership, gradient, and baseline controls.
no full model, confirmation data, frozen source, or wider-state training is involved in this check.

## 2026-09-05 - second final development result

mean-pool seed 1337 completes 1,000 updates and all 288 development predictions.
known accuracy is 33/256 (12.890625%); absent accuracy is 1/32 (3.125%).
it predicts kitchen for 192 known questions, of which 24 are correct, and 27 absent questions.
there are 20 false known-query abstentions and two constant-answer worlds.
last-mention-chunk accuracy is 5/24, 3/25, 9/104, and 16/103.
these are poor one-seed development results, not a completed comparison or causal diagnosis.

training takes 5,147.64927241602 seconds and evaluation takes 112.42829429195262 seconds.
logical forward tokens total 1,344,703 and targets total 24,662.
the 1,000-token difference from query pooling is exactly one fewer initial-fill token per update.
independent audit verifies the paired schedule and initial projection, all updates, five finite final tensors with 185,066 parameters, optimizer state, all 288 native prompts/generated tokens/scores, and all frozen/source/input hashes.
it also rechecks all descriptive patterns and timing/update cohorts, without full-model or confirmation access.
checkpoint SHA256: `a6e4401f474f6515c61bac075050ef9d6681323049a97a1a3e52054e59939fc1`.
metrics SHA256: `61274d18cfb06633691f34449f4ac16270a428d909c9e0c6e87ecf8c3cdd6d30`.
result SHA256: `67a0863f469734679c6ca857f06f92741170f6062f162c5fc17fb04f2437bcb4`.
prediction SHA256: `c7b2985df1066ab958fa32b3a3a5254fa75aaedae16375f1ce6a4133025ffc58`.
descriptive report SHA256: `89b67ccdbf0f022b33e7faeeff2a691774a1929a97ae0a0eac647e6ce735d3a7`.
mean-pool seed 2027 starts automatically; the four remaining runs and final evaluation stay queued.
both final seed-1337 training-fit dry runs now pass for all 256 worlds and 2,304 queries; actual training-fit inference has not started.
Obsidian notes 42 and 43, the index, and README record the result and its limits.

## 2026-09-05 - check the answer-length bias hypothesis

the pinned tokenizer maps office and unknown to two answer tokens including stop, and the other rooms to three.
training counts are bathroom 348, bedroom 341, garden 315, hallway 372, kitchen 329, office 343, and unknown 256.
the constant answer-sequence distribution that minimizes token-mean CE has probability proportional to count divided by token length and loss 0.7280549963184155.
its largest probability belongs to office, while the unweighted frequency maximum is hallway.
thus neither simple count bias nor this length-normalization optimum alone explains the kitchen-dominated generation.
this mathematical reference is not evaluated inference and does not show that the frozen read interface can realize the optimum.
do not change the frozen objective based on this insufficient explanation; the next diagnostic remains final training fit.

## 2026-09-05 - prepare the code-only fit and queue training-fit evaluation

the one-off `fit_opaque_fixed_oracle.py` implements the declared four-world, nine-query, 200-update diagnostic.
it loads the final seed-1337 query-pool states from verified training-fit outputs and freezes the existing projection and qualified reader.
initial normal generation and loss must exactly match those saved states' training-fit records.
drop, zero, and full-context controls must remain unchanged after code fitting.
the diagnostic records both recipient-answer accuracy and control-appropriate accuracy; cyclic donors require unknown for all queries.
the initial and final evaluations do not select an intermediate checkpoint.

independent review finds one malformed-input issue: indexing batch zero before validation would discard an extra saved batch.
the fix checks complete values and boolean-mask shapes before extraction.
31 CPU tests cover component ownership, fixed weights, gradients, checkpoint recomputation, the exact optimizer objective, control coverage, and malformed state shapes.
the full-size fit and its real input dry run must wait for completed training-fit artifacts; CPU checks are not accuracy evidence.
all 24 frozen study sources still match.

training-fit queue session `79536`, shell PID `94622`, waits on evaluation shell PID `78024`.
it requires all five final report files, then evaluates every training query for query-pool seed 1337 followed by mean-pool seed 1337.
the existing training queue remains live; mean-pool seed 2027 is the third of six runs.
no oracle optimization is queued before inspecting final training fit.
Obsidian note 43 and the index record the plan, implemented checks, validation repair, and pending scientific result.
the second independent review verifies the fix, all 31 CPU tests, and rejection of all 12 malformed input fixtures.
reviewed runner SHA256: `c895adde574a48395fca34fa4f516d4c2a4246c634f246191b7b1041e178f142`.
the runner is ready for its real preflight once the training-fit artifacts exist; it has not run a scientific fit.

## 2026-09-05 - preserve the runnable study code in ordered commits

a handoff audit finds five of the frozen protocol's 24 sources are excluded from Git by the artifact ignore rule.
preserve the exact paths and bytes with an explicit code-file allowlist, rather than moving live scripts or changing the protocol.
four local, code-only commits record 15 Python files in dependency order:

- `c3133bc`: source reservation, opaque-world preparation, and raw-capacity audit.
- `d89bed0`: partial-evidence and exact-format reader qualification.
- `7e4170d`: frozen protocol, writer training, baseline/intervention evaluation, and aggregation.
- `3786a94`: final training-fit and fixed-oracle diagnostics, runner tests, and development analysis.

independent audit matches 12 candidates to existing recorded hashes and every available saved source copy.
the other three match reviewed versions and have dry-run or CPU validation, not completed full-size results.
249 focused tests pass, and the final repository-plus-runner suite passes 1,612 tests in 9.20 seconds.
all 24 frozen sources now match Git `HEAD` and the unchanged study hashes.
all staged diffs were inspected; no Markdown, models, data, generated predictions, or ignore-rule changes entered the commits.
the working tree has no tracked modifications and nothing was pushed.

the code archive is not a complete artifact release.
the final handoff must still preserve frozen protocols, exclusion manifests, data, vocabulary, adapter/checkpoints, and package snapshots outside Git.
the absolute freezer source path requires the preserved path layout or a separately labelled relocation protocol; never silently rewrite the live record.
the training, held-out evaluation, and later training-fit queues remain unchanged.
Obsidian notes 42 and 43 and the index record the new commit links and pending scientific results.

## 2026-09-05 - third final development result

mean-pool seed 2027 completes 1,000 updates and all 288 development predictions.
known accuracy is 32/256 (12.50%); absent accuracy is 8/32 (25.00%).
known-query predictions are unknown 89, office 63, bedroom 50, hallway 37, and kitchen 17.
garden and bathroom are never predicted.
zero worlds have one constant known-query answer; distinct-prediction counts 2, 3, 4, and 5 occur in 1, 8, 18, and 5 worlds.
last-mention-chunk accuracy is 2/24, 2/25, 18/104, and 10/103.
the low known-answer result repeats across the two completed mean-pooling seeds, but their answer biases differ.
this does not isolate a forgetting mechanism or prove comparative inferiority; final confirmation and training-fit checks remain pending.

training takes 5,145.139371583937 seconds and evaluation takes 95.24065145896748 seconds.
logical forward tokens total 1,344,644 and targets total 24,664.
the last 100 online token-mean losses average 0.7949970042705536; this is not final training-set fit.
independent audit verifies all 1,000 scheduled updates, fresh CPU initialization, the paired projection rule, five finite final tensors with 185,066 parameters, optimizer step 1,000, all 24 frozen hashes, nine source copies, and input/data/adapter hashes.
it replays all 288 native encodings, generated-token decoding, stop rules, exact scores, and descriptive patterns without full-model inference or confirmation access.
checkpoint SHA256: `e3b432d091e94ac630377c2800923388ca321dc7bb9458b48a1969e5ef48baea`.
metrics SHA256: `3d5a2d16cc7cd26ccffaa42a8ee8d1303071b8452e860a48a971dfbbea5dab0e`.
result SHA256: `d3e7195c637e06a2479257116b3d2b1018951ea0f05f35b9868d1a63b15a6abe`.
prediction SHA256: `d4c9b9cc45344e3bcd41a97a5fa43726769cafe39a6429773709eb07ca1d44e2`.
descriptive report SHA256: `bd9344f19bf225ed6dd93168defec33dfb359d7f48b6bd812748dae7c7391254`.

query-pool seed 2027 starts automatically as the fourth of six runs.
the confirmation and later final training-fit queues remain live and waiting in the declared order.
Obsidian note 42, the index, and README record the new result and its limits.
only derived artifacts and notes change in this phase; no source changes or new code commit are needed.

## 2026-09-05 - fourth final development result

query-pool seed 2027 completes 1,000 updates and all 288 development predictions.
known accuracy is 56/256 (21.875%); absent accuracy is 1/32 (3.125%).
known-query predictions are office 134, kitchen 99, hallway 20, and unknown 3.
bedroom, bathroom, and garden are never predicted.
one world has a constant known-query answer; distinct-prediction counts 1, 2, 3, and 4 occur in 1, 13, 17, and 1 worlds.
last-mention-chunk accuracy is 6/24, 5/25, 21/104, and 24/103.

the known-answer gap over paired mean/FIFO is +9.375 points for seed 2027 and -4.296875 points for seed 1337.
seed-2027 query pooling also scores 21.875 points below paired mean/FIFO on absent queries.
the better known-answer score in one seed is not a stable-benefit claim, and memory dependence remains untested until the fixed interventions.
keep all final seeds and the predeclared confirmation analysis; do not select the better-looking development seed.
final training-fit and fixed-readout diagnostics remain the next failure checks after the frozen study.

training takes 5,147.772872749949 seconds and evaluation takes 97.58659316599369 seconds.
logical forward tokens total 1,345,644 and targets total 24,664.
the last 100 online token-mean losses average 0.8033079087734223; this is not final training-set fit.
independent audit verifies all 1,000 updates, fresh CPU initialization, byte-identical paired schedules and initial projections, ten finite final tensors with 185,040 parameters, and finite optimizer state at step 1,000.
the query writer uses exactly one additional logical input token per update compared with paired mean/FIFO.
all 24 frozen hashes, nine source copies, input hashes, and descriptive report bindings match.
all 288 native encodings, generated-token decodings, stop rules, exact scores, and descriptive groups pass without repeated full-model inference or confirmation access.
checkpoint SHA256: `ed69ca96f87c4d42ce32eec682d86120c852f8054ab1490d211d3deef8c17450`.
metrics SHA256: `3ac4a2b634e8da588b0900318e33af71719f1ab63fb6a3ee50e310fcac8921aa`.
result SHA256: `a883f4091cfaf5866961b31754fe032e471662fca1c17769952ff9fb4244b2b0`.
prediction SHA256: `8f5d4b950fb06ed8e9160dc9aa782a5a4df8ab69c73d77afc862dfcd1629ad36`.
descriptive report SHA256: `497b4ffb757d477e4ab501b454e274f17ba98d7ce0b91496830d1304554b3d08`.

query-pool seed 4099 starts automatically as the fifth of six runs.
the confirmation and later final training-fit queues remain live and waiting in order.
Obsidian note 42 now separates the four results by seed and method; the index and README record current status.
only derived artifacts and notes change in this phase; the experiment code remains unchanged and already committed.

## 2026-09-05 - fifth final development result

query-pool seed 4099 completes 1,000 updates and all 288 development predictions.
known accuracy is 32/256 (12.50%); absent accuracy is 2/32 (6.25%).
known-query predictions are kitchen 250, unknown five, and bedroom one.
31 kitchen predictions and the bedroom prediction are correct.
absent-query predictions are kitchen 30 and unknown two.
all 26 constant-known-output worlds answer kitchen; the other six worlds each have two distinct outputs.
last-mention-chunk accuracy is 4/24, 2/25, 9/104, and 17/103.
this establishes output-level answer collapse on those 26 worlds, not absence of information in the stored state or a causal writer-versus-reader diagnosis.

all three query-pool development runs are complete.
known accuracy has mean 14.322916666666666% and sample standard deviation 6.82569917816727 percentage points.
absent accuracy has mean 17.708333333333332% and sample standard deviation 22.606806681469486 points.
these summarize optimization-seed variation on the same 32 development worlds; they are not confidence intervals or independent additional examples.
the final mean/FIFO run, confirmation baselines and interventions, and final training-fit checks remain pending.

training takes 5,150.220767208142 seconds and evaluation takes 114.58619287493639 seconds.
logical forward tokens total 1,345,727 and targets total 24,660.
the last 100 online token-mean losses average 0.7962804388999939; this is not final training-set fit.
independent audit verifies all 1,000 scheduled updates, fresh CPU initialization, ten finite final tensors with 185,040 parameters, 66-byte retained state, and finite optimizer state at step 1,000.
all 24 frozen hashes, nine source copies, input bindings, token totals, 288 native encodings, generated-token decodes, stop rules, symbolic answers, exact scores, descriptive cohorts, and the three-seed summary match.
the audit does not repeat full-model inference or access confirmation data or the active mean-pool run.
checkpoint SHA256: `96e7633c6e2a39daeadf4648b60dced67be3686c809785826b315f886dbc9692`.
metrics SHA256: `ca4c5a9c5c74060903e7032e943803f9470e25736c14dfc0e000305aebca7572`.
result SHA256: `3481d7e4cb8258b8161ce1b66864bb7cd0633e9482781ddb923843ffaa73645e`.
prediction SHA256: `d9b2f01583575f36e1d24cd08e83d828ee32e2d10b3cc6fa244c6304cd4b604e`.
descriptive report SHA256: `66491e085d2d5ef66f7eec4aae1470c959da5df25b8aa2ac0230e74ede0c9b04`.

mean-pool seed 4099 starts automatically as the sixth and final training run.
the confirmation and later final training-fit queues remain live and waiting in order.
Obsidian note 42, the index, and README record the new result, observed collapse, and remaining uncertainty.
only derived artifacts and notes change; the frozen experiment code remains unchanged and already committed.

## 2026-09-05 - supplemental source preservation and handoff audit

while the final mean-pool run remains active, independent read-only review finds one additional untracked historical source: `artifacts/predictions/audit_opaque_study_sources.py`.
commit `ca62377` preserves its exact 61 lines at the original path; no frozen source, protocol, output, ignore rule, model, dataset, or Markdown file enters the commit.
the source SHA256 is `dfbaf9ee2819a0513d7d132bac8ee41c0ba9dc0e71ebd6fb697a4ee786a29bd2`, matching the saved supplemental audit.
the saved audit SHA256 is `218fbc5be899faedfd77a2f8bd8c7d3b61ad75a5181385f48147e8b0bb2af4b7`.
all eight supplemental evidence hashes and their study, reserve, source-fit, selection, and source-group bindings match.
the audit relies on prior frozen group membership and exclusion flags, checks batched-profile lineage through protocol hashes, and reports a fixed original-exclusion count.
it is preserved as a historical script, not promoted to a reusable validator or a project-wide untouched-data claim.
do not use Python optimization flags or rerun its exclusive-create write over the original output.

validation: syntax compilation passes, the complete repository plus runner suite passes 1,612 tests in 9.83 seconds, and all 24 frozen sources still match their recorded hashes in both the worktree and Git `HEAD` after the commit.
the commit is local and not pushed; the tracked worktree is clean.
no repeated full-model inference or confirmation-answer access is part of this audit.

the final artifact handoff still needs an explicit file list and `SHA256SUMS`, including final optimizers, development predictions, aggregation tables, and figures not all covered by existing result hashes.
each training protocol records 51 package versions, but dependency installation is not locked.
the older explicitly post-launch environment record matches that package map and supplies Python 3.11.14, macOS 26.5.2, and arm64 evidence; retain it and add a dated final hardware and numerical-settings record.
preserve exact commands, the checkout commit, data and all referenced exclusion evidence, vocabulary, pinned Qwen snapshot, qualified adapter, six final training records, seven confirmation evaluation directories, analysis/report outputs, and both final training-fit outputs.
replay remains repository-root, path, and Apple/MPS dependent; do not alter historical protocols to disguise these limits.
Obsidian note 42 records these requirements and the new commit.
next: finish and audit the sixth training run, then the existing confirmation and training-fit queues; only then run any separately justified fixed-readout diagnostic.

## 2026-09-05 - six-run training completed and confirmation started

the final mean-pool seed-4099 run completes 1,000 updates and all 288 development predictions; the training queue exits successfully at 11:57:40 UTC.
known accuracy is 33/256 (12.890625%); absent accuracy is 0/32.
known predictions are kitchen 210, office 25, unknown 11, hallway four, bathroom three, bedroom two, and garden one.
absent predictions are kitchen 28 and office four.
all seven constant-known-output worlds answer kitchen; distinct-prediction counts 1, 2, 3, and 4 occur in 7, 17, 7, and 1 worlds.
last-mention-chunk accuracy is 3/24, 2/25, 14/104, and 14/103.
these are output-level failures, not a causal writer-versus-reader diagnosis or proof that the stored state contains no information.

all three mean/FIFO development seeds have known-answer mean 12.760416666666666% and sample standard deviation 0.22552744890219756 percentage points.
their absent-answer mean is 9.375%, with sample standard deviation 13.621559198564604 points.
paired query-minus-mean known gaps are -4.296875, +9.375, and -0.390625 points; the mean gap is +1.5625 with sample standard deviation 7.042092334890604 points.
paired absent gaps are +40.625, -21.875, and +6.25 points.
these describe optimization seeds on the same 32 development worlds, not confidence intervals or independent additional examples.
no stable-benefit claim or checkpoint selection is made from these development results.

training takes 5,142.823728374904 seconds and evaluation takes 110.26018337509595 seconds.
logical forward tokens total 1,344,727 and targets total 24,660.
the last 100 online token-mean losses average 0.764880793094635; final training-set fit remains unmeasured.
independent CPU-only audit verifies every scheduled update, fresh initialization, paired schedule and initial projection, five finite final tensors with 185,066 parameters, finite optimizer state at step 1,000, and 66-byte retained state.
query pooling uses exactly one additional logical input token per update.
all 24 frozen source hashes, nine source copies, input/data/reader/adapter bindings, 288 native encodings, decodes, stop rules, symbolic answers, scores, descriptive cohorts, and three-seed statistics match.
the audit does not repeat full-model inference or access confirmation inputs or outputs.
checkpoint SHA256: `ebd5ee94fd30573e95c1fa8b103f0d98e2b3d838760574cc462a25474ae9e9e8`.
optimizer SHA256: `0cbd953cb14eef16f788bce26f99479ac151add57bc678ba6200804c10433e50`.
metrics SHA256: `2d50c0083ced1c00f6efb30d8701560e07000562f98dcc12f3daf01193cfdae7`.
result SHA256: `ff1d6738ff734ff0c76c9e10f20e8d8d77ab33edd30f3e71423721c8253c163a`.
prediction SHA256: `056c8d9cc9448c8b6ac466b41d89dbb7acb51caad58cd30e15297940fe5022b4`.
descriptive report SHA256: `6a32df5ec5befe683533e4e6772738ff713da004bb0aebbf4023387052dfba58`.

the existing confirmation queue starts the baseline evaluation after checking all six final training records; native session 47939 and its Python child are live.
the first completed confirmation world produces 63 baseline/reference prediction records.
the learned-memory interventions, aggregation, report, and queued seed-1337 training-fit checks still remain.
keep all frozen sources unchanged and do not tune on the confirmation results.
Obsidian note 42, its index entry, README, and the active research plan now record all six development results and the active confirmation stage.
only derived analysis and notes change in this phase; the experiment code is already committed, so no new code commit is needed.

## 2026-09-05 - CUDA handoff and explicit failure diagnosis

the local confirmation session disappears before completing baseline evaluation; the last observed whole-world progress contains 1,512 predictions.
the previous evaluation and training-fit process IDs are no longer live, and there is no complete confirmation report or final training-fit result.
the reason for process termination is not established.
no local full-size run is restarted, no confirmation answers are inspected for tuning, and partial outputs are preserved.

the user requests a CUDA option, Della Slurm scripts, a transfer command, and a research-direction handoff for cursor.
commit `20d1410` adds separate portable train, baseline, memory-intervention, training-fit, fixed-oracle, and aggregate/report entry points under `scripts/opaque`.
all 24 frozen sources and historical JSON bytes remain unchanged.
the new runtime explicitly resolves the original Mac root into the current checkout and records device, package, dtype, numerical, hardware, and source identity separately from the frozen study.
CUDA requires native BF16 support, deterministic operation, and an explicit cuBLAS workspace configuration; there is no CPU fallback.
aggregation requires compatible execution records, and the oracle requires same-backend training-fit evidence.

the default Della job requests one public `gpu40` allocation, four CPU cores, 24 GB host memory, and 15 minutes for smoke.
it uses the existing default Slurm account and no explicit QOS.
the `all` mode runs smoke, the complete confirmation comparison, aggregation/report, both seed-1337 training-fit checks, and the predeclared fixed-projection oracle.
it does not retrain the six completed writers or automatically launch further sweeps.
outputs are fresh and exclusive-create; no partial result is skipped or merged.
the GPU model, Linux package installation, Slurm settings, speed, and memory use still require the actual cluster smoke test.
no SSH, remote transfer, or job submission was performed.

independent review identifies and resolves omitted vocabulary/development checks, incomplete runtime metadata validation, and optional-dependency collection failures.
the synthetic end-to-end report test also catches integer-versus-JSON seed keys; the portable report now reads the serialized summary before using the existing report functions.
the supplied rsync recipe is tested with relative paths from the repository root because this Mac's rsync does not trim the absolute `/./` prefix as expected.
no actual large transfer is performed.
the full `data/` tree is intentionally included at the user's request; current completed study inputs are allowlisted, while partial outputs, unrelated experiments, caches, and `.venv` are excluded.

validation: the 20 new tests pass, real-input preflight passes for six completed runs and 24 frozen sources, source hashes and both shell syntax checks pass, and the final full suite passes 1,632 tests in 28.26 seconds in an isolated checkout with existing raw datasets available.
all 14 committed task files are byte-identical to the tested isolated files.
an unrelated working-tree replacement of `continuous_decoder.py` appeared during the task, deleting 483 lines and adding 31; it is preserved and not committed.
the full working-tree run has 83 failures and 1,549 passes due to that edit, while the isolated committed version is green.
the transfer therefore starts from a clean local clone of the new commit, then adds datasets and completed artifacts through rsync.
the user has been asked whether the unrelated edit was intended; no restoration is assumed.

the research question is retained, not reduced to obtaining one positive result.
the current three-seed development means remain 14.32% known / 17.71% absent for query pooling and 12.76% / 9.38% for trained mean/FIFO.
this does not establish a stable learned-memory advantage or identify the causal failure.
the next diagnostic sequence is final generated training fit, then privileged code fitting with the reader and projection fixed.
a pass isolates feasible readout states for four training histories; a failure remains conditional on this projection, initialization, constraints, optimizer, and budget.
writer-side versus reader-side adaptation, teacher distillation, and width-versus-precision remain conditional next experiments, not proven fixes.
keep the current architecture until a concrete diagnostic identifies a necessary change.
Obsidian note 44 records this reasoning and the commit, notes 42 and 43 plus the index record the interrupted status, and `docs/cursor_handoff.md` supplies the project brief.
`docs/della.md` supplies exact transfer, setup, smoke, production, and result-retrieval commands.
Markdown, model files, datasets, results, and the unrelated edit are not included in the commit; nothing is pushed.

## 2026-09-05 - Focus the project on reliable bounded-memory updates

The user authorizes a complete but lean update study and local code/documentation milestone commits. The active question is incorporation of new or corrected facts without damaging other facts, compared with compact explicit storage. Start with the existing native Qwen3-1.7B architecture at 66 bytes; do not build a speculative architecture collection or require a positive learned-memory result.

Scope/dependency review found rejected roadmap prose but no corresponding executable implementation in `src/`, `scripts/`, or `tests/`. Removed those plans from the active specification, research plan, handoff, literature summary, and Obsidian direction/index notes. Removed the old unrelated implementation curriculum from Obsidian note 05. Historical implemented components and their tests remain reproducibility references, not active requirements. The frozen native path imports bounded-state writers, prompt/read helpers, data validators, and reporting utilities; deleting these to simplify the tree would break the existing experiment. An independent read-only review confirmed that documentation removal does not weaken the runtime hash protections.

Baseline: Git `20d141052ca202a253b84c6fee1b9d5684571b2b`. The working decoder patch and untracked documentation/smoke/goal-runtime files predated this milestone. Only intended research documentation is added to the commit; the decoder and runtime/generated files are excluded. Existing research audit, results notes, and Della runbook are retained with the documentation so their links work in a clean checkout.

Preserved identities:
- frozen study protocol SHA-256: `e74cf9fa0027da9ad43bbe7914c56d3601c9e4428517912a42663f733ec8e566`;
- sorted JSON of its 24-entry `source_sha256` map: `8bc97126430a6d7582b0b32cb3dd120a356789f1bcb5b69b4f96ffdbdbc625e4`;
- unrelated working decoder SHA-256: `ea932d482a39e01b455a012f936d0babad5bb9a793de13c3b137a4f56e19cb65`;
- all six final checkpoint hashes match both their original results manifests and the pre-cleanup snapshot.

Validation: the eight-suite command in `tests/README.md` passes **166 tests in 4.48 seconds**. `python -m scripts.opaque.smoke --check-only` verifies six completed runs and all 24 frozen sources with `model_loaded=false` and `confirmation_answers_read=false`. The exact tracked patch matches its before-cleanup copy. Exhaustive rejected-direction searches over source, scripts, tests, repository docs, and the TinyMem vault return no matches. This does not claim a newly green full working-tree suite or a CUDA run. Detailed local snapshots and logs are under `/tmp/tinymem-update-study/`.

Obsidian notes 22 and 45 and the index now agree with the repository contract. This is scope cleanup, not a new model result. Next implement paired event data and independent replay, then metrics, runners, reports, and Della integration. No training, confirmation inspection, cluster submission, or push occurred. Scope cleanup was committed locally as `f913a41`.

## 2026-09-05 - Freeze paired one-event data and stable streaming prefixes

Implemented `memory_updates.py`, `prepare_memory_updates.py`, `update_encoding.py`, and a single-budget design at `configs/memory_update_study.json`. The initial eight-binding, four-chunk history branches into an addition, exact repetition, or genuine correction. All states answer the same ten queries, including a newcomer and an always-absent entity. Addition raises live information to nine bindings; the other branches keep eight. One event per branch is the declared limit, not a long-delay reliability result.

Reused the original 256 training and 32 development source-world pairings without relabeling them untouched. Selected 64 new confirmation worlds from 178 remaining eligible native-study-unused source groups, excluding all old association groups and the original reserve. All 31 exclusion files are reverified. The old confirmation dataset is not opened; representative selection uses input history only, then labels are checked. Independent regex replay validates generated state transitions, and tests use an additional word-based oracle. Incorrect inputs abort without resampling.

Every chunk owns two trailing newlines so a before-state write prefix is token-identical in all after branches. The new encoder takes no questions/answers and fails on merges or overlength inputs; the old frozen prompt helper is unchanged. All 288 real training/development worlds pass 11,520 native prompt/tokenization checks, with at most 133 tokens per write. Only tokenizer/configuration files are loaded, not Qwen weights or confirmation data.

Independent review identified a missing frozen trust anchor and possible conflicting hash overwrites; both are fixed with regression tests. Review's suggested index pairing of source episodes and context hashes was rejected: 9,552 of 9,700 original connected groups have different list lengths because they are membership sets. Exact representatives are already checked against raw source text before generating and hash-binding the data. Nested malformed query records also fail closed. No query ordinal is exposed to the reader, so fixed query order is not a new answer channel.

The design gates full-text reading at each state, including nine live bindings, and labels preservation claims competence-limited unless the before state meets the declared per-seed threshold. It is explicitly not yet a launch protocol: the exact training objective/schedule and shared reader/source/data bindings must be frozen by the later runner. No architecture or budget sweep is added.

Verification: **255 tests pass in 4.91 seconds**, including the real builder on synthetic inputs, zero-old-confirmation-access guards, strict event/replay/split checks, reproducibility, tokenization, and existing native-memory regressions. Final data are `artifacts/predictions/memory_update_data_20260905_v2`; all dataset hashes match the initial unlaunched build. Design SHA-256 is `5e24629a1bf5d4f0172260b5b72188725845369bc5b862a96d5fcf521c79d168`; data-protocol SHA-256 is `5e0879c6891aa180a75364f01de738bed9784102cad35cae2b34a2770c5b39e6`. Logs are under `/tmp/tinymem-update-study/`.

The old input-only preflight still passes for all six checkpoints and 24 frozen sources, and the original decoder patch is byte-for-byte preserved. Obsidian note 46 explains the design and verification; note 45 and the index link the milestone. Next implement paired reliability metrics with explicit denominators. No full-size training, scientific accuracy measurement, cluster submission, or push occurred. The data milestone was committed locally as `7bf5641`.

## 2026-09-05 - Measure paired accuracy transitions without hiding denominators

Added `evaluation/memory_updates.py` and 38 focused tests. Scoring requires one validated episode and exactly 40 distinct case-ID-aligned prediction records for a single checkpoint; wrong-history/stage IDs and partial or duplicated results fail. Source groups remain attached to the scored history for later paired analysis. Gold labels are replayed from the dataset, never accepted from prediction records.

Each rate stores integer counts, with an undefined value for zero denominator. State metrics separate overall/known/absent accuracy. Event-target update accuracy applies to additions and corrections, not repetitions. Staleness compares a corrected target's answer with its immediately superseded gold value regardless of before correctness; it does not treat unknown-before additions as stale.

The two known-fact cohorts distinguish all unchanged bindings from untouched bindings, excluding the directly refreshed repetition target in the latter. Full CC/CW/WC/WW tables support before/after accuracy, joint preservation, conditional preservation/forgetting, and unconditional forgetting. Recovering a previously wrong answer is not preservation, and net accuracy change is not forgetting. Empty initial-correct sets yield null conditional rates rather than appearing to have zero forgetting.

Absent errors partition into canonical room responses and invalid outputs; empty output is an error, not an invented room or a correct abstention. The canonical-room rate is deliberately not a semantic hallucination detector. Existing answer normalization is reused, without adding free-form abstention aliases. Pool history counts within each seed and metric key; overlapping before measures/cohorts cannot be treated as independent samples. Conditional subsets differ between methods, so initial competence and unconditional results remain necessary context.

Independent read-only review re-derived the formulas and found no mathematical error. Its aliasing concern prompted read-only defensive mappings with tests; returned JSON remains an independent mutable copy. Tests include a hand-counted seven-query table (CC=2, CW=2, WC=1, WW=2), all-correct/all-unknown cases, malformed responses, unequal conditional denominators (1/1 and 0/7 pool to 1/8), arbitrary record order, and randomized transition/decomposition conservation.

Verification: **293 relevant tests pass in 5.06 seconds**, including the existing data and native-memory regressions. `artifacts/smoke/memory_update_metrics_20260905/fixture.json` contains explicitly synthetic predictions, their dataset, source hashes, and scored output—not Qwen findings. Logs are in `/tmp/tinymem-update-study/metrics-final-tests.txt`. No data-design or generator changes were made. The input-only preflight passes for all six completed runs and 24 frozen sources; new data/source hashes and the original decoder patch are unchanged. No confirmation records were read, model training started, or jobs submitted. Obsidian note 47 and the repository measurement contract document the definitions and limits. Next: actual training/evaluation/reader-qualification runners, with provenance, byte and recurrent-gradient checks.

## 2026-09-05 - Verify shared-state update execution and preserve new negative evidence

Added the core `research/update_runner.py` and 19 tests using actual tiny random Qwen. Training shares four initial writes and independently forks three events. The objective averages token-mean answer CE across ten queries per state and four states. Optimization owns only the writer/projection; every state is finite, bounded, and exactly 66 bytes. Independent replay matches shared loss and parameter gradients, including checkpointed execution. An after-only loss reaches all four prefix states and only its selected fork. Frozen reader parameters remain unchanged and gradient-free.

Evaluation covers both neural methods, all five compact references, and full/no-context controls. Ten reads share each state; repeated evaluations are identical. Retention forks are functional, dictionaries use training write tokens only, and fingerprint lookup does not call Qwen. Competence gates rescore complete raw predictions against authoritative episodes instead of trusting cached metrics or pass flags. These core functions are not yet a launch/provenance wrapper or CLI.

Independent read-only review found no confirmed bug. The suggested event-splitting concern does not apply: the stable encoder returns one tuple per chunk and rejects overflow. Nonfinite loss is now checked before backward for a clear failure boundary. Regression: **312 tests pass in 8.00 seconds**; logs `/tmp/tinymem-update-study/runner-regression.txt`. The input-only old-study preflight passes (six runs, 24 sources, no model/confirmation reads). The unrelated decoder patch remains byte-identical and unstaged.

The user supplied completed association confirmation and diagnostic aggregates during recovery from harness errors. `docs/association_confirmation_results.md` preserves these as **user-reported, not independently artifact-verified**: query-pool known mean 15.85%, mean/FIFO 14.42%, difference +1.43 points (99% interval −0.29 to +3.13); latest templates 74.41%, full history 97.36%. Absent qualification failed. Training fit and the tested fixed-projection oracle also failed. Thus generalization alone cannot explain the failure, and writer-only improvements are not established as sufficient. No information-theoretic impossibility follows. Individual seed labels remain unverified rather than inferred from presentation order.

Obsidian note 48, note 45, and the index record both the core verification and this evidence update. No full-size training, confirmation-example inspection, cluster submission, or push occurred. The runner task remains open until high-level provenance, qualification, CLI, and checkpoint boundaries are implemented and tested. Core execution was committed locally as `d1fa317`.

## 2026-09-05 - Verify development inputs without opening confirmation histories

Added `research/update_protocol.py` and eleven artifact-boundary tests. The development loader checks exact design identity, the frozen old-study trust anchor, complete generator-source coverage, all required transitive input and exclusion hashes, source metadata disjointness, and exact representative pairings for loaded histories. It proves cross-split source separation from selection metadata without opening confirmation histories. Existing connected-group episode/hash arrays remain membership sets, not zipped pairs.

The real verified data build loads **256 training and 32 development worlds** under a `Path.open` guard that rejects any access to `confirmation.json`. Its protocol remains `5e0879c6891aa180a75364f01de738bed9784102cad35cae2b34a2770c5b39e6`. The existing adapter identity resolves to `opaque_reader_adaptation_20260905/step_000100`, with weight SHA `9f55aab03ccbf91a29765ca4d5083080be7747badf5e072814600ca781fdbb2c`; no weights loaded. Production loading rechecks the reader identity and pinned snapshot. New-task qualification is still required separately.

Tests caught an invalid assumption that the all-three-split validator could accept train/development alone; cross-split metadata plus exact pairing now supplies that boundary without inspecting confirmation. The real-data check also caught overly broad inclusion of historical provenance sources beyond those the builder binds; the loader now mirrors the builder's raw-source and exclusion contract exactly. Both successful real checks followed the fixes.

Verification: **323 tests pass in 8.77 seconds** (`/tmp/tinymem-update-study/protocol-regression.txt`). The original decoder patch is unchanged. Obsidian note 48 records this increment. No training, confirmation examples, cluster submission, or push. Qualification-result persistence, frozen launch schedule, checkpoint completeness, CLI, aggregation, and Della execution remain open. The provenance increment was committed as `3ba32b8`.

## 2026-09-05 - Complete the update experiment runner lifecycle

Added `research/update_experiment.py` and `scripts/run_memory_updates.py`: input-only check, training-only ten-step profile, new development reader qualification, frozen launch, independent per-method/seed training, and explicit-split evaluation. Fresh output directories and exact hash-covered completion seals prevent partial outputs from granting qualification, training completeness, or confirmation access. The launch binds the shared reader, actual dtype/runtime, execution sources, exact training encodings, schedules, dictionary, objective, optimizer, and final-only checkpoint policy. No real schedule has been chosen/frozen; `--steps` is mandatory.

Profile weights are not saved or reused. Qualification is rescored from raw predictions and failure blocks freezing. Training persists initial/every-100/final weights, optimizer state, per-step metrics, final development outputs, and the before-state competence interpretation. Evaluation requires each chosen final checkpoint to verify; new confirmation histories additionally require all six new runs to complete. Compact shared dictionary and learned/shared model costs are distinguished from persistent stream bytes. The production CLI rejects synthetic data and requires explicit development/confirmation selection. Existing six association checkpoints are never used as new-run outputs or retrained.

Independent review found no critical defect but requested five-of-six refusal, mixed-launch rejection, checkpoint-boundary tests, seed-helper provenance, explicit confirmation choice, and clearer preflight behavior. These are addressed. Its gradient concern is already enforced in `train_update_step`; an additional real frozen PEFT test verifies seven nonzero state gradients and unchanged reader parameters. Cwd checks enforce repository-root execution. Strict hardware/runtime matching is documented rather than silently relaxed.

Verification: **331 tests pass in 10.80 seconds** (`/tmp/tinymem-update-study/experiment-regression.txt`). Eight new tests include actual random-Qwen failed qualification; synthetic scripted qualification success followed by six real tiny training runs, checkpoint loading and all selected confirmation paths; refusal before any or all-but-one runs; hash tampering, mixed launches and partial outputs; real ten-step training-only profiling; and explicitly stubbed optimization for the 100/101 persistence boundary. Scripted/stubbed fixtures are not model-quality findings. The new CLI input check verifies 256 train/32 development and the adapter without loading weights/confirmation; the old preflight still verifies six runs/24 frozen sources. Decoder patch unchanged.

Obsidian note 49 documents commands, lifecycle, and limits. No full-size model training, launch freezing, confirmation examples, Della submission, or push occurred. The runner task is implemented; next build independently validated multi-seed aggregation, paired-history uncertainty, reports, then cluster orchestration. Runner lifecycle committed as `c88db89`.

## 2026-09-05 - Report paired update outcomes without pooling optimization seeds

Added `evaluation/update_aggregate.py`, `evaluation/update_report.py`, and `scripts/report_memory_updates.py`. Each metric pools raw counts across histories within a seed; a learned family's estimate is the equal mean of seed-specific pooled ratios. Per-seed transition counts, rates, SD/range, and all outcomes remain available. One multinomial whole-history resample matrix is shared across all runs and metrics. Descriptive percentile intervals condition on the observed optimization seeds rather than treating seeds as independent histories. Undefined conditional-denominator draws are counted; any such draw withholds that interval, with no silent conditioning or zero imputation.

Reporting requires all six complete training runs and thirteen sealed evaluation artifacts, validates launch/checkpoint identities, full history coverage, reader labels and reconstructed persistent state sizes, and rescores raw outputs against authoritative replayed data. Cached-score discrepancies fail. Markdown shows initial known/absent competence, event-specific preservation/forgetting, correction/staleness, seed variation, paired intervals, and per-seed denominators. JSON preserves all80 metrics/six transition views, costs, and source/artifact identities. Contrasts are descriptive, not omnibus significance claims.

Verification: **345 tests pass in 12.52 seconds** (`/tmp/tinymem-update-study/report-regression.txt`). Fourteen focused tests cover independent scalar bootstrap calculations, count-vs-ratio pooling, fixed-seed-vs-pooled estimands, exact paired cancellation, undefined draws, malformed coverage, byte corruption, cached-metric disagreement and mixed provenance. The existing integration test now creates all13 tiny/synthetic evaluations and the actual report. Durable fixture: `artifacts/smoke/update_report_fixture_20260905_v2/test_six_actual_tiny_training_0/inputs/report/`, explicitly synthetic with a single history; its degenerate intervals are not scientific uncertainty. Independent mathematical review found no confirmed bug; within-run metric-key consistency was hardened from its robustness note.

Obsidian note50 records estimator definitions, uncertainty limits, commands, and fixture status. No scientific result is inferred from tiny models or scripted qualification. No real confirmation examples were read, model weights loaded for reporting, cluster job submitted, or changes pushed. Della orchestration and final verification remain. Reporting milestone committed as `042b02a`.

## 2026-09-05 - Supply a separate fail-fast Della update pipeline

Added `della_updates.sh`, `della_updates.slurm`, and a NUL-delimited `update_transfer_manifest.py`. Transfer committed code from a fresh clone, preserving the unrelated Mac decoder patch. The manifest includes new data, all transitive source/exclusion inputs, pinned reader/model, and historical checkpoint dependencies without collecting partial confirmation or local smoke outputs. A local rsync dry-run passes; no transfer to the cluster occurred.

The pipeline profiles training-only computation, then requires an explicitly chosen positive step count. Full execution performs new reader qualification, freezes one launch, trains six new method/seed runs, evaluates all13 methods/checkpoints, and generates the verified report in one allocation. Qualification failure aborts; no silent adaptation, budget increase, resume, or old-study retraining occurs. Strict runtime/GPU identity remains enforced. The runbook leaves walltime/step placeholders because full-size throughput is not measured.

The existing association pipeline and frozen sources are unchanged. Newly user-reported completed old results should be collected, not automatically rerun. Existing training-fit/oracle commands are documented only for genuinely unfinished diagnostics and reuse historical checkpoints with matching execution identity. The new pipeline never runs the old `all` stage.

Verification: **377 tests pass in 33.13 seconds**, combining10 new shell/transfer tests with update, old handoff/runtime, and native oracle-runner suites (`/tmp/tinymem-update-study/della-regression.txt`). Shell tests use a declared executable stub, verify six trainings precede13confirmation calls, failure stops, and invalid arguments execute nothing. `bash -n` passes for new and old wrappers. The prior synthetic Python pipeline exercises actual tiny training/checkpoint/report logic separately. Local transfer dry-run output is `/tmp/tinymem-update-study/transfer-dry-run.txt`.

Obsidian note51 and `docs/della_updates.md` give exact stages, clean export, mandatory schedule declaration, resource caveats, partial-output policy, and result collection. No Slurm submission, remote execution, environment change, push, or scientific accuracy claim. Final project verification remains. Della tooling committed as `814c2ab`.

## 2026-09-05 - Final committed-tree verification and status reconciliation

A fresh clone of committed code through `814c2ab` passes **1,779 tests in38.37seconds** after staging its required input artifacts and bAbI validation dataset. Initial attempts exposed missing staged inputs, not source failures; these were supplied without editing tests. The working checkout's377focused tests already pass; the unrelated decoder edit is deliberately excluded from the full-suite claim. Logs: `/tmp/tinymem-update-study/final-clean-suite.txt` and `final-clean-checkout.txt`.

Final identity verification matches all24 frozen sources,15 portable sources, six historical final checkpoints, both study/data protocols, and the original unrelated decoder file/patch. Both old/new input-only preflights pass. No real confirmation histories were opened by these checks. `docs/update_verification.md` records commits, evidence, fixture limits, missing cluster evidence, and exact next steps.

Reconciled README, specification, active plan, Cursor handoff, implementation guide, and Obsidian current notes with completed tooling and newly supplied old negative outcomes. Removed stale instructions to rerun completed association stages by default. Discovery still finds only the original partial baseline output locally; final reported cluster results remain explicitly user-reported until collected and verified. No full-size update schedule, reader qualification, or scientific results are claimed.

All implementation components are present, locally tested, independently reviewed at their major boundaries, and committed in verified steps. No remote transfer, Slurm submission, full-size training, dataset/prediction commit, or push occurred. Actual scientific execution is the next user-controlled stage, not an implementation blocker.
