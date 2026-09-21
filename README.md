# TinyMem

TinyMem asks whether a language model can keep a small learned memory under a fixed byte budget, update it with new or corrected facts, and still recall facts that were not changed, as well as compact explicit storage can.
The memory is written before the questions arrive, and the writer never sees a question, an answer, or a gold write address.
The repository contains the memory implementations, the frozen study runners, and the tests.

The project ran as a sequence of fixed, pre-declared studies on a frozen Qwen3-1.7B reader.
Each study is sealed before it runs: source, inputs, schedule, and evaluation are hashed, every seed is retained, and no accuracy threshold selects a checkpoint or a seed.
Negative results are reported with the positive ones.

## Findings

1. **Learned writers did not show a storage advantage.**
   On a four-fact task that an explicit parser stores exactly in one byte, gated and delta writers at 66 and 258 bytes did not establish reliable recall.
   Repeating a true fact reduced recall of unmentioned facts by about 25 percentage points across three writer seeds, in both LM answers and a fixed linear probe.
2. **The read interface works when the state is correct.**
   With a parser writing fixed addresses into a 258-byte delta state, fresh bridges and rank-8 Q/V LoRA readers recalled 14,336 of 14,336 cases on the four-fact task and 97.9% to 99.5% on official bAbI QA1 development questions.
   Answers followed donor memory contents, so the reader depends on the state.
3. **Learned writers fail by collapse, and two patches repair most of it.**
   With training-only state supervision, two of three writers drove their write strength or hidden activations to zero and produced almost-empty states.
   A fixed write strength helped two seeds and hurt one.
   Adding affine-free LayerNorm over the 64 hidden coordinates raised correction from 72.86% to 95.88% and retention from 69.46% to 92.60% across 12 fresh writer seeds and three frozen readers, at unchanged memory, parameters, and budget.
   Paired whole-seed 95% intervals are +11.6 to +35.0 and +13.1 to +33.3 points.
   Most of the retention gain came from a better initial state; reserved wording still loses 8 to 15 points.
4. **State-supervised writers reach oracle accuracy on QA1; answer-only writers do not.**
   Three state-supervised writers scored 97.1% to 99.8% on the official QA1 test (mean 98.21%) against 98.28% with exact oracle memory.
   The same architecture trained only from answer tokens scored 14.5% to 44.1% on development questions across 12 fits (mean 29.06%).
   A saved-checkpoint diagnostic attributed the failure to write interference (35.5% disturbance from unrelated writes versus 1.7% in successful controls), not to end-of-sequence loss.
5. **The address function is the bottleneck for answer-only learning.**
   Freezing the key branch from a state-supervised writer and training only a fresh value branch from answers raised development recall from 31.8% to 59.0% (paired 95% interval +26.0 to +28.5 points), with retention across unrelated writes rising from 20.6% to 58.5%.
   Six of twelve pairs still failed, and QA1 keeps a 100%-accurate two-byte explicit reference, so this is a narrow mechanism result.
6. **At equal bytes, answer-trained learned memory lost to every explicit text store, and the reader ignored it.**
   Across bAbI tasks 1 through 5 at 64, 256, and 1,024 retained bytes, jointly trained int8 slot memory scored 20.7% to 21.7% five-task accuracy in every budget and seed.
   Zero-memory and other-story controls scored 16.6% to 17.2%, and QA1 and QA2 accuracy was identical to three decimals across all nine cells: the reader learned the answer prior and nothing from the state.
   Compressed text stores reached 62% to 67% at 64 bytes and 99.3% to 99.7% at 256 bytes; a training-only dictionary store reached 99.7% at 64 bytes.
   The paired whole-story bootstrap difference against the dictionary store is -78 to -79 points at every budget (95% intervals within -80.3 to -76.7).
   Under heavy text distractors every method, including the text stores, fell to 21% to 27%, so bounded explicit storage is not a solved problem either.

This closes the project's question.
Learned writers can be made to write correct states when the state is supervised, and the read interface is sound, but end-to-end answer supervision under a byte budget did not produce a memory the reader used, let alone one that beat compact explicit storage.
Details are under [Storage frontier](#storage-frontier).

## Architecture

```text
Current statement -> frozen Qwen features -> learned writer(old state) -> new state (fixed bytes)
Serialized state -> learned bridge + one question -> Qwen with frozen or LoRA-adapted weights -> answer
```

Three writer families were tested.

| Writer | Update | Bytes tested | Module |
|---|---|---|---|
| Gated slots | `next = old + sigmoid(gate) * (tanh(candidate) - old)` | 66 | [gated_slots.py](src/tinymem/memory/gated_slots.py) |
| Delta rule | `next = state + outer(key, beta * (value - key @ state))` | 66, 258 | [delta_slots.py](src/tinymem/memory/delta_slots.py) |
| Quantized attention slots | token cross-attention, slot self-attention, gated residual, int8 every write | 64, 256, 1,024 | [quantized_slots.py](src/tinymem/memory/quantized_slots.py) |

Bytes count only the state retained between writes and reads.
Shared weights, dictionaries, and temporary computation are reported separately.
Reads never modify the state, and no raw history or KV cache bypasses it.

## Studies

Each study has a runner that prepares a sealed bundle, trains every declared cell, seals, evaluates, and reports.
Stages refuse to run if any hashed input, source file, or runtime differs from the seal.

| Study | Question | Runner | Package | Cells |
|---|---|---|---|---|
| Phase one | Can a gated writer preserve unmentioned facts through truthful repetition? | tag `phase-one-archive` | archived | - |
| Delta comparison | Do residual updates with learned addresses retain better than gating? | [run_delta_study.py](scripts/run_delta_study.py) | `studies/delta` | 3 seeds x 2 widths x 2 writers |
| Correct-state readout | Can the read interface use a correct query-independent state? | [run_oracle_study.py](scripts/run_oracle_study.py) | `studies/oracle` | 3 readers |
| Learned writer | Can a writer learn correct states, and why does it collapse? | [run_distilled_study.py](scripts/run_distilled_study.py) | `studies/distilled` | 3 or 12 writers |
| QA1 readout | Does the read interface transfer to an official benchmark? | [run_qa1_study.py](scripts/run_qa1_study.py) | `studies/qa1` | 3 readers |
| Storage frontier | At equal bytes, does answer-trained memory match strong text stores? | [scripts/frontier/](scripts/frontier/) | `studies/frontier` | 9 learned + 3 text |

The distilled runner takes `--fixed-beta 0.75`, `--normalize-hidden`, and `--replicate` to select the gate patch, the normalization patch, and the 12-seed panel.
Every runner supports a separate `prepare --smoke` bundle whose results never enter a scientific comparison.
[Running a study](#running-a-study) gives the commands.

### Phase one results

The final phase-one task has four independent binary facts and a 66-byte learned state.
After eight prefix writes, eight truthful writes go to one fact or to all four.
Changes are measured from each model's own prefix endpoint, in percentage points.

| Seed | Arm | Balanced probe minus LM | Balanced probe change | Unspoken probe change | Unspoken LM change |
|---|---|---:|---:|---:|---:|
| 2027 | Uniform | +2.34 | +4.69 | -32.55 | -31.90 |
| 2027 | Correction weighted | +1.56 | +2.73 | -32.55 | -34.24 |
| 2028 | Uniform | +11.33 | +8.59 | -24.61 | -19.53 |
| 2028 | Correction weighted | +7.81 | +5.47 | -25.65 | -23.70 |
| 2029 | Uniform | +7.81 | -5.47 | -16.93 | -9.77 |
| 2029 | Correction weighted | +0.78 | -2.73 | -15.36 | -13.93 |

![Phase one: change in recall after eight truthful writes, by writer seed and arm](results/phase1.png)

Balanced refresh moves recall by a few points in either direction; repeating one fact costs 10 to 34 points on the facts that were not mentioned.
The [results CSV](results/phase1.csv) has exact counts and denominators.
The arms share a reader, data order, and evaluation panel, so they are not six independent replications.

### Learned writer and QA1 readout

The distilled study trains the 258-byte delta writer to reproduce parser-written states and then reads the states it produces with three frozen readers.

![Writer collapse: correction and retention per writer seed, with and without hidden LayerNorm](results/writer_collapse.png)

Five of twelve control seeds sit at chance on both metrics; their write strength or hidden activations collapsed to zero.
Affine-free LayerNorm over the 64 hidden coordinates lifts every collapsed seed above 90% and helps most of the others; one seed loses 2.5 points of retention.
Memory, parameter count, and byte budget are unchanged.
Per-seed values are in [writer_collapse.csv](results/writer_collapse.csv).

![QA1 readout: state-supervised, frozen-key, and answer-only writers against oracle and zero-memory bands](results/qa1_readout.png)

The same writer architecture spans the whole range depending on what supervises it.
State supervision reaches the oracle band on the official test.
Answer-only training sits between 14% and 44% on development questions.
Freezing the key branch from a state-supervised writer and training only the value branch from answers reaches 92% or better in six of twelve pairs and stays below 30% in the other six; freezing an answer-only key branch instead does not help.
Per-cell values are in [qa1_readout.csv](results/qa1_readout.csv).

### Storage frontier

The closing study asks: at equal retained bytes per history, can answer-trained recurrent memory retain and update facts as well as strong explicit text stores?
Nine learned cells (budgets of 64, 256, and 1,024 bytes, three seeds) and three text-store cells were trained jointly from answer tokens on official bAbI tasks 1 through 5 with WikiText-2 distractors, for three epochs and 16,875 optimizer steps each.
Each record is encoded by the unadapted frozen Qwen base and written by token cross-attention into learned slots; every write quantizes to int8 and the retained state has no flags, scales, or buffers.
Explicit baselines are compressed recent text, compressed text selected for lexical diversity, and a training-only dictionary log (152,830 shared dictionary bytes, reported separately), each capped at the same budgets.
Evaluation scored all 5,000 official test questions in clean and heavy-distractor variants from the fixed final checkpoint, with zero-memory and other-story controls, then transferred to all 500 installed BABILong 1k examples as raw text.

![Storage frontier: five-task accuracy by retained bytes, clean and with distractors](results/storage_frontier.png)

| Method | 64 B | 256 B | 1,024 B |
|---|---:|---:|---:|
| Learned int8 slots | 21.0 (20.9-21.0) | 20.7 (20.6-20.9) | 21.7 (21.0-22.5) |
| Compressed recent text | 67.0 (66.8-67.3) | 99.7 (99.6-99.8) | 99.7 (99.7-99.8) |
| Compressed diverse text | 62.3 (62.2-62.5) | 99.3 | 99.7 (99.7-99.8) |
| Dictionary recent text | 99.7 (99.6-99.8) | 99.7 (99.7-99.8) | 99.7 (99.7-99.8) |

Five-task mean exact-match accuracy in percent on clean official test questions; parentheses give the range across three seeds.
Per-seed values are in [storage_frontier.csv](results/storage_frontier.csv).
Zero-memory controls averaged 17.2% and other-story controls 16.6% for the learned cells.
Full text scored 99.6% to 99.8%, and the hand-written two-byte QA1 state scored 100%.
Learned-cell training cross-entropy plateaued at 0.65 against 0.18 for the text readers.
On BABILong 1k transfer, learned memory scored 19.7% to 20.6% at every budget, while compressed recent text reached 34.5%, 50.1%, and 88.4% at the three budgets.

Under heavy distractors, where four WikiText records follow each fact and sixteen more precede the question, every method fell to 20.6% to 26.6%.
The text stores lose the facts to distractor text under the byte cap; the learned memory had nothing to lose.

Execution notes.
All twelve training cells completed under the original protocol.
The first evaluation array stopped on the nine learned cells because PEFT re-enabled adapter gradients when its `disable_adapter` context exited and the frozen-reader check in `generate` refused to continue; the fix restores the frozen adapted reader after the base encoder runs.
The first report attempt wrote the analysis and then failed on a figure with error bars a few ulp below zero.
Both fixes were applied as recorded, evaluation-only protocol amendments that reuse the fitted seals and refuse changes to fitting-bound files.
The sealed report lists the amendment chain, the protocol hash each cell's seal carries, and the one recorded waiver (text-cell evaluations that completed under the original protocol through an unchanged code path).

## Setup and tests

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --python 3.11 --extra dev --extra research
uv run --no-sync python -m pytest
```

The suite has 326 tests and runs on CPU in about 90 seconds.
Tests use small tensors, synthetic inputs, and tiny randomly initialized readers.
They check contracts and execution paths; they do not measure full-size model accuracy.

The figures in `results/` are drawn from the CSV tables next to them, which were extracted from the sealed study bundles:

```bash
uv run --no-sync python scripts/make_figures.py
```

## Data and model

No dataset, parsed benchmark file, or model weight is tracked in Git.
Two scripts download and verify everything the studies read; the loaders in `src/tinymem/data/` and `src/tinymem/studies/*/data.py` parse the raw files on demand.

```bash
uv run --no-sync python scripts/download_data.py    # bAbI (en-10k), BABILong qa1-qa5 at 1k-8k, WikiText-2 raw
uv run --no-sync python scripts/download_model.py   # Qwen3-1.7B at the pinned revision, with Hub hashes
```

Both scripts write under `data/raw/`, which is ignored by Git, and check every file against the sizes and hashes recorded in [data/manifest.json](data/manifest.json) or on the Hub.
The model download is about 4 GB.
Dataset licenses and model terms apply separately from this repository.

## Running a study

Each runner works on one study directory and moves through fixed stages.
`prepare` builds the data, copies the source tree, and writes `protocol.json`; every later stage re-hashes the protocol and refuses to run on a changed input, source file, or runtime.
Cells are addressed by `--cell N`; a stage that has already produced a cell directory refuses to overwrite it.

Run from the repository root with `MODEL=data/raw/pretrained/qwen3-1.7b`.

```bash
# Delta comparison (12 cells) and correct-state readout (3 cells)
uv run --no-sync python scripts/run_delta_study.py prepare --output artifacts/delta --snapshot $MODEL --device cuda
uv run --no-sync python scripts/run_delta_study.py train --study artifacts/delta --snapshot $MODEL --cell 0   # ... 11
uv run --no-sync python scripts/run_delta_study.py seal --study artifacts/delta
uv run --no-sync python scripts/run_delta_study.py reference --study artifacts/delta --snapshot $MODEL
uv run --no-sync python scripts/run_delta_study.py evaluate --study artifacts/delta --snapshot $MODEL --cell 0
uv run --no-sync python scripts/run_delta_study.py report --study artifacts/delta
# run_oracle_study.py has the same stages without reference.

# Learned writer: features on GPU, writer fitting on CPU, scoring with the oracle readers on GPU
uv run --no-sync python scripts/run_distilled_study.py prepare --output artifacts/distilled --parent artifacts/oracle \
    --snapshot $MODEL --device cuda --normalize-hidden --replicate
uv run --no-sync python scripts/run_distilled_study.py features --study artifacts/distilled --snapshot $MODEL
uv run --no-sync python scripts/run_distilled_study.py train --study artifacts/distilled --cell 0            # ... 11
uv run --no-sync python scripts/run_distilled_study.py seal --study artifacts/distilled
uv run --no-sync python scripts/run_distilled_study.py evaluate --study artifacts/distilled --snapshot $MODEL --cell 0
uv run --no-sync python scripts/run_distilled_study.py report --study artifacts/distilled

# QA1 readout (3 cells)
uv run --no-sync python scripts/run_qa1_study.py prepare --output artifacts/qa1 --snapshot $MODEL \
    --training-file data/raw/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt
uv run --no-sync python scripts/run_qa1_study.py train --study artifacts/qa1 --snapshot $MODEL --cell 0
uv run --no-sync python scripts/run_qa1_study.py evaluate --study artifacts/qa1 --snapshot $MODEL --cell 0
uv run --no-sync python scripts/run_qa1_study.py report --study artifacts/qa1

# Storage frontier (12 cells); stages after prepare run inside the study directory
uv run --no-sync python scripts/frontier/frontier_prepare.py prepare --study artifacts/frontier
uv run --no-sync python scripts/frontier/frontier_prepare.py freeze --study artifacts/frontier
cd artifacts/frontier && export PYTHONPATH=$PWD/source/src
python frontier_run.py features --study . --model ../../$MODEL
python frontier_run.py preflight --study . --model ../../$MODEL
python frontier_run.py train --study . --model ../../$MODEL --cell 0       # ... 11
python frontier_run.py evaluate --study . --model ../../$MODEL --cell 0
python frontier_run.py transfer --study . --model ../../$MODEL --cell 0
python frontier_report.py --study .
```

Add `--smoke` to the delta, oracle, or distilled `prepare` to get a bundle with one cell and a few steps that exercises the code path end to end.
Full cells were run on a CUDA GPU with 40 GB of memory and BF16 support (Ampere or newer); the distilled writer fits run on CPU.
Runtime and hardware are recorded in each cell's seal, and a cell refuses to score under a different runtime than the one it was trained with.

### Slurm

[scripts/slurm/gpu.slurm](scripts/slurm/gpu.slurm) and [scripts/slurm/cpu.slurm](scripts/slurm/cpu.slurm) wrap any runner stage in one job and pass the array index as `--cell`.
They carry no site names; give partition, account, or GPU constraints on the `sbatch` line, and set `TINYMEM_ENV` to a file that activates your Python environment if it is not the repository `.venv`.

```bash
sbatch scripts/slurm/gpu.slurm scripts/run_delta_study.py prepare --output artifacts/delta --snapshot $MODEL --device cuda
sbatch --array=0-11 scripts/slurm/gpu.slurm scripts/run_delta_study.py train --study artifacts/delta --snapshot $MODEL
sbatch --dependency=afterok:<train job> scripts/slurm/cpu.slurm scripts/run_delta_study.py seal --study artifacts/delta
sbatch --array=0-11 scripts/slurm/cpu.slurm scripts/run_distilled_study.py train --study artifacts/distilled
```

The frontier study copies [frontier_gpu.slurm](scripts/frontier/frontier_gpu.slurm) and [frontier_cpu.slurm](scripts/frontier/frontier_cpu.slurm) into the study directory at `prepare` and hashes them with the protocol.
Submit them from inside the study directory with `TINYMEM_ENV` and `TINYMEM_MODEL` set:

```bash
cd artifacts/frontier
export TINYMEM_ENV=/path/to/.venv/bin/activate TINYMEM_MODEL=/path/to/qwen3-1.7b
sbatch frontier_gpu.slurm features
sbatch --array=0-11 frontier_gpu.slurm train
sbatch frontier_cpu.slurm
```

## Repository layout

| Path | Contents |
|---|---|
| `src/tinymem/memory/` | Stored-state contract and the three writers |
| `src/tinymem/reader/` | Frozen Qwen loader and verifier, prefix reader, LoRA adaptation, fixed prompt, and exact-match scoring |
| `src/tinymem/data/` | bAbI and WikiText parsers and the reader case contract |
| `src/tinymem/studies/` | One package per study (`delta`, `oracle`, `distilled`, `qa1`, `frontier`): data, protocol, fitting, scoring, and report code; shared `runtime.py` and `artifacts.py` |
| `scripts/` | Study runners, `download_data.py`, `download_model.py`, `make_figures.py`, `profile_training.py`, and Slurm templates |
| `tests/` | Behavioral and end-to-end tests |
| `results/` | Compact result tables (CSV) and the figures drawn from them |
| `data/manifest.json` | Dataset source and checksum declarations |

The checkout holds only the code that the studies above execute.
The from-scratch decoder, the explicit token-retention baselines, the readout-bridge studies, and the phase-one independent-fact lineage were removed after phase one closed; they remain in Git history at tag `phase-one-archive`.
Each sealed study bundle also carries its own copy of the source tree at the time it ran, so sealed results verify against that copy and not against this checkout.
Datasets, pretrained weights, checkpoints, and full run outputs are excluded from Git.

## Reproduction scope

The code above reruns every study from public data and the pinned model.
Exact numbers depend on the recorded runtime: PyTorch, Transformers, and PEFT versions from `uv.lock`, BF16 CUDA kernels on an Ampere-class GPU, deterministic algorithms, and TF32 disabled.
A rerun on other hardware is a replication, not a verification of the sealed bundles, and the seals will say so.
Do not bypass hash checks or substitute this checkout for a frozen execution snapshot.
