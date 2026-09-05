# TinyMem research handoff

handoff code: `20d1410`, local and not pushed.
verification: 1,632 tests pass for the committed project plus the CUDA handoff; real-input preflight passes and all 24 frozen sources match.
no remote job has been submitted and CUDA hardware execution remains unverified.

## current objective

how reliably can query-independent learned memory incorporate new or corrected facts without damaging other stored facts, at a fixed persistent-byte budget?
compare with compact explicit storage; a positive learned-memory result is not required.
use `PROJECT_SPEC.md` and `docs/research_plan.md` for the focused implementation contract.
one derived task at 66 bytes is not the final storage frontier or a general impossibility result.
implement the paired event data, before/after metrics, runners, reports, and Della scripts using existing components first.
the old comparison and its diagnostics remain frozen prerequisites, not a license to tune on confirmation.

## decisions and evidence

the byte-level model remains an educational and mechanistic reference.
its earlier gains largely came from information still inside the local window; memory interventions did not establish useful bounded memory.
we therefore added a separate native-token Qwen3-1.7B research path to reduce the risk that a weak language model hides a working memory mechanism.
Qwen is the shared reader and feature extractor, not just an external comparison model.
the learned and raw-memory methods use the same qualified frozen reader; the handcrafted fingerprint reference is labeled separately.

the reader was adapted on training data and qualified on development data before memory training.
on the opaque development full-context gate, it answered 253/256 known queries and 32/32 absent queries correctly.
that establishes competence with text, not competence with arbitrary compressed vectors.

the controlled task has four history chunks, eight known entities, and one absent entity.
entity IDs are fresh opaque identifiers; the writer must retain associations, not memorize a small global name list.
one state is shared across all nine questions, and the writer does not see future questions or answers.
the state has two width-eight float32 slots plus two boolean validity entries: 66 persistent bytes.
temporary expansion to Qwen's width is not retained memory.

query pooling and trained mean/FIFO each have three optimization seeds and 1,000 updates per seed.
their parameter counts are closely matched, but their logical forward-token counts differ by one token per update, so this is not an exact FLOP match.
the loss averages the nine per-query answer-token losses, including the stop token.
all four recurrent writes remain attached to autograd, with non-reentrant activation checkpointing.
this is already full-episode BPTT for these examples; simply proposing to add cross-segment gradients is not a new fix.

## current result

all six training runs and their development evaluations are complete and audited.
the following are means across three seeds on the same 32 development worlds, not held-out confirmation results:

| method | known answers | absent answers |
|---|---:|---:|
| query pooling | 14.32% | 17.71% |
| trained mean/FIFO | 12.76% | 9.38% |

the query-pool known-answer standard deviation is 6.83 percentage points, versus 0.23 for mean/FIFO.
the paired mean known-answer gap is only 1.56 points, with a 7.04-point standard deviation across seeds.
these development numbers do not establish a stable learned-compression advantage.
some runs repeatedly answer `kitchen` for different entities in the same world, and abstention is poor.

the Mac confirmation evaluation was interrupted during baseline generation.
no complete confirmation report or final training-fit result exists yet.
partial output is preserved but excluded from the CUDA handoff.
do not tune on the confirmation answers or combine partial Mac and CUDA results.

## the unresolved cause

we do not yet know whether the writer fails to encode bindings, the recurrent updates lose them, the fixed read interface cannot use them, or the objective/optimization favors a low-information solution.
we also have not measured final generated training accuracy, so calling this only a generalization failure would be premature.
a low online CE is insufficient: it can improve through predictable output and stop tokens without correct entity-room answers.
the current read matrix has rank eight; this rules out simple rank collapse, not poor use of its directions.
simple label frequency and token-length weighting do not by themselves explain the observed kitchen-heavy outputs.

## next diagnosis and decision rules

1. finish the unchanged confirmation comparison, including raw baselines, full-history, drop, zero, donor, short-name transfer, and counterfactual controls.
   use the declared paired-world statistical rules, not a selected best seed.
2. evaluate the final query-pool and mean-pool seed-1337 checkpoints on all 256 training worlds and all nine questions.
   this separates failure to fit the training task from a training-to-development gap.
3. fit one privileged code per history for the first four training worlds, shared across nine questions.
   freeze the qualified Qwen reader and the final query-pool read projection; optimize only the bounded 66-byte codes for the predeclared 200 updates.
   require exact initial replay against the same-backend training-fit output.

| observation | justified next investigation |
|---|---|
| training good, development poor | investigate generalization to fresh IDs, history variation, and capacity interference |
| training poor, fitted codes succeed | useful readout states exist for these examples; isolate history encoding, recurrent writes, and their optimization |
| training poor, fitted codes fail | investigate the fixed projection/reader interface and bounded optimization; do not call storage fundamentally impossible |
| full-context controls fail | repair or requalify the reader before attributing failure to compression |

32/32 known and 4/4 absent oracle answers demonstrate fitted readout feasibility for four training worlds only.
this would not show that a writer can produce those codes, generalize, or beat raw retention.
failure is conditional on the fixed projection, initialization, state bounds, optimizer, and update budget.

keep the existing architecture while implementing controlled update measurements.
require immediate binding competence before attributing subsequent errors to forgetting.
if the diagnostic identifies a specific failure, predeclare at most one targeted follow-up on training/development material.
any architectural or budget change is a separately declared experiment; retain the present negative evidence.

## instructions for cursor

work in `/scratch/gpfs/JORDANAT/caleb/TinyMem`.
read this file and `docs/della.md`, then inspect Git state and the listed scripts.
the full hand-written decoder is not the current Qwen execution path.
the transfer must contain committed code, the pinned Qwen snapshot, qualified adapter, study protocol, six completed runs, opaque data, and vocabulary.
the Mac's unrelated `continuous_decoder.py` edit must not be silently restored, committed, or transferred into the production checkout.

use `scripts/della.slurm` and the `scripts.opaque` modules, not the archived MPS-only runners under `artifacts/predictions`.
first run package checks, the no-inference input preflight, and tests.
submit only the smoke job, inspect its exit status, numerical checks, and logs, then submit the fixed pipeline.
never run full Qwen inference or training on the login node.
do not invent a Slurm account or request a QOS; use the existing default account.
do not change package versions silently if installation fails.

preserve all 24 frozen source files and historical JSON bytes.
do not replace checkpoint files, alter seeds or step counts, shorten the held-out evaluation, weaken assertions, or select new examples after seeing results.
portable code resolves the old Mac root explicitly; never edit manifests to make their hashes pass.
new runtime identity is recorded separately, and aggregation rejects different backends or numerical configurations.
do not merge partial results or resubmit into an existing output directory.
do not run multiple full models in one allocation.

if any stage fails, report the command, job ID, exit state, final error, and affected output path before changing the research design.
if a CUDA operation is unsupported, reproduce it on a small training-only case and keep the negative diagnostic.
do not solve an execution error by silently enabling CPU fallback or disabling deterministic checks.
after completion, report exact counts by condition and seed, training-fit CE and accuracy, oracle controls, and the declared statistical conclusion.
separate verified results from hypotheses about the cause.
copy evidence back from scratch and update the technical notes before proposing the next experiment.
the user now authorizes local commits of important verified code and documentation milestones.
keep the TinyMem Obsidian notes current as well; they live outside this Git repository.
do not commit datasets, weights, generated predictions, unrelated existing edits, or goal-runtime files.
do not push without authorization. Provide scripts for full-size work; do not label queued runs as findings.
