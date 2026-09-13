# TinyMem

TinyMem studies whether query-independent learned memory can accept new or corrected facts without damaging other stored facts, under a fixed persistent-byte budget.
The repository contains memory implementations, controlled experiments, and tests.

**Phase one is complete.**
The tested systems did not demonstrate an advantage over compact explicit storage.
The controlled follow-up found that repeating a true fact can reduce recall of other facts, even when no correct answer changes.
This occurred across three fresh writer initializations, in both LM answers and a separate fixed linear probe.
Many LM errors remained recoverable by the probe.

## Results

The final task has four independent binary facts and a 66-byte learned state.
Each of three writer seeds produces two paired continuation arms, called uniform and correction weighted.
After eight prefix writes, the evaluation applies eight truthful writes to either one fact or all four facts.
“Unspoken” means the other three facts in the single-fact condition.
Changes are measured from each model's own prefix endpoint.

| Seed | Arm | Balanced probe minus LM | Balanced probe change | Unspoken probe change | Unspoken LM change |
|---|---|---:|---:|---:|---:|
| 2027 | Uniform | +2.34 | +4.69 | -32.55 | -31.90 |
| 2027 | Correction weighted | +1.56 | +2.73 | -32.55 | -34.24 |
| 2028 | Uniform | +11.33 | +8.59 | -24.61 | -19.53 |
| 2028 | Correction weighted | +7.81 | +5.47 | -25.65 | -23.70 |
| 2029 | Uniform | +7.81 | -5.47 | -16.93 | -9.77 |
| 2029 | Correction weighted | +0.78 | -2.73 | -15.36 | -13.93 |

All values are percentage points.
Balanced endpoints contain 256 known-answer cases per arm; unspoken endpoints contain 768 dependent cases across 64 prefixes.
The [results CSV](results/phase1.csv) includes exact before/after counts, paired errors, and denominators.
Initial known LM recall is 60/64, 62/64, and 50/64 for the three seeds; none is excluded.

![Three-seed confirmation results](results/phase1.png)

Averaged equally over seeds, unspoken probe recall falls by 24.70 points for uniform continuation and 24.52 for correction weighting.
The corresponding LM losses are 20.40 and 23.96 points.
Balanced refresh improves probe recall in two seeds and reduces it in the third.
After balanced refresh, the probe recovers 8-54 LM errors per arm and seed, but also misses some answers the LM gets right.
Thus, LM errors and recoverability from the stored state are different measurements.
Reduced accuracy under fixed decoders does not establish irreversible information loss.

Earlier exploratory runs established useful initial storage (63/64 known answers), but incomplete preservation through sixteen updates (72/128 in the answer-only arm).
Correction weighting later improved primitive corrections from 53/64 to 60/64 while reducing long-path recall from 99/128 to 91/128.
These are separate exploratory comparisons, not outcomes independently replicated by the final three-seed study.

The final confirmation holds the reader, features, data order, wording, and previously inspected evaluation panel fixed.
It tests sensitivity to writer initialization on this panel, not generalization to new readers or language.
The paired arms are not six independent replications.
Balanced and single-fact schedules differ in both fact coverage and repetition frequency per fact.
The reader had privileged training on assigned representations.
An explicit parser can store this task's four facts exactly in one byte with a fixed schema; the learned state uses 66 bytes plus shared parameters.
This method has not established bAbI/BABILong performance or a storage advantage.

## Architecture

```text
Initial history -> frozen Qwen features -> learned initializer -> state
Current statement + previous state -> learned updater -> next state
Detached state -> fixed affine bridge + one question -> frozen reader -> answer
```

The initializer and updater use [QueryPoolSlotWriter](src/tinymem/memory/query_pool_slots.py).
Each pools 2,048-wide features into two width-eight slots and applies a coordinate-wise gated update:

```text
next = old + sigmoid(gate(old, candidate)) * (tanh(candidate) - old)
```

The persistent state contains sixteen FP32 values and two Boolean validity flags: 66 bytes of tensor storage.
Shared weights, temporary computation, and file headers are separate costs.
The writer receives no future question, answer, or gold write address.
The [read boundary](src/tinymem/research/readout_read.py) accepts a detached state and one question, without history features or a raw-history bypass.
Reads do not update memory.
The affine bridge projects each slot through widths 8, 32, and 2,048 into the frozen adapted Qwen3-1.7B reader.

Each final lineage trains an initial writer for 400 steps, freezes its initializer, trains a joint answer-plus-state updater for 400 steps, then forks two 400-step continuations.
All four-write training transitions remain attached.
A state-loss coefficient is calibrated from sixteen training-only gradient measurements and then shared by both continuation arms.
The fixed linear probe is trained on old training-path states, with four fact-order validation folds and 99 trajectory-label shuffle controls per fit.

## Setup and tests

Use Python 3.11 or newer.
The lock file records the dependency resolution; the research extra supplies the pretrained-reader dependencies.

```bash
uv sync --locked --python 3.11 --extra dev --extra research
uv run --no-sync python -m pytest
```

Tests use small tensors, synthetic inputs, and tiny randomly initialized readers.
They do not require the research checkpoints and do not measure full-size model accuracy.
A focused check of the final architecture is:

```bash
uv run --no-sync python -m pytest \
  tests/test_query_pool_slots.py \
  tests/test_readout_read.py \
  tests/test_independent_fact_joint_update.py \
  tests/test_independent_fact_content_probe.py \
  tests/test_independent_fact_lineage_training.py \
  tests/test_independent_fact_lineage_evaluation.py
```

## Repository layout

| Path | Contents |
|---|---|
| `src/tinymem/memory/` | Stored-state contracts, writers, and readout bridges |
| `src/tinymem/research/` | Reader integration, training objectives, controlled data, and probes |
| `src/tinymem/evaluation/` | Evaluation, paired comparisons, and report calculations |
| `scripts/` | Experiment entry points and cluster helpers |
| `tests/` | Behavioral and end-to-end tests |
| `results/` | Compact phase-one quantitative results |
| `data/manifest.json` | Dataset source and checksum declarations |

The repository also retains earlier model and memory experiments where they remain implemented or support the research code.
They are not requirements for a second phase.
Datasets, pretrained weights, checkpoints, local installation records, agent logs, and full run outputs are excluded from Git.
Dataset licenses and model terms apply separately from this repository.

## Reproduction scope

The public checkout supports the implementations and tests above.
Historical full-size runners additionally require their original sealed source/input bundles, including plans, parent checkpoints, features, and declarations.
Those bundles are not distributed in this repository.
Some declaration checks name original documentation paths; these are archived experiment inputs, not missing installation steps for the unit tests.
Do not bypass hash checks or substitute this cleaned checkout for a frozen execution snapshot.
Cluster scripts require site-specific resource settings and a staged reader cache.
Submit from the repository root or pass `sbatch --chdir`; `HF_HOME` can select an existing cache and otherwise defaults to `.cache/huggingface` under the working directory.
For the portable legacy path reader, `TINYMEM_ORIGINAL_REPOSITORY` explicitly declares an archived absolute root to relocate; unrelated roots and parent traversal remain rejected.

The final recorded runs completed 4,800 optimizer steps in total.
Their audits checked 78 sealed result files, 48 calibration records, 116,100 reader records, and exact reconstruction of 24,672 saved CPU states.
An independent SVD implementation checked the probes, 594 shuffled-label fits, and paired bootstrap calculations.
These checks did not repeat every optimizer step or GPU generation.
The compact table reports those audited results; it does not replace the full evidence archive.
