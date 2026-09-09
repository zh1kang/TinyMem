# readout-interface local handoff

## scope

the local pre-Della implementation is ready for user-run device profiling.
no CUDA test, production Qwen inference, Della job, full-size training, or confirmation evaluation was run during this milestone.
tiny random-reader results establish implementation behavior, not factual recall.

```text
local checks complete -> user runs Della profiles -> approve fixed schedule -> paired training
```

## local evidence

- 242 tests passed in 42.29s with the command below.
- the profile test also exercises a BF16 tiny reader on CPU with FP32 encoder/bridge/state.
- the production `check` command verified 256 training and 32 development histories and the existing adapter, without loading model weights or confirmation histories.
- a two-step affine/GELU smoke run, final checkpoint reload, controlled reads, disposable profile, and the report CLI completed at `artifacts/predictions/readout_local_handoff_20260908/`.
- the smoke report uses one seed and two histories per split; it is not a scientific result.
- `uv pip check --python .venv/bin/python` passed for all 51 installed packages.
- `bash -n scripts/della_readout_profile.slurm` and `git diff --check` passed.
- independent read-only Claude review found no blocking findings; review disposition is below.

the full repository suite was not rerun for this milestone.
CUDA numerical behavior, device memory, cluster modules, queue constraints, and production-model runtime remain unverified.

```bash
.venv/bin/python -m pytest tests/test_readout*.py tests/test_update*.py \
  tests/test_prefix_reader.py tests/test_della_handoff.py \
  tests/test_mean_pool_slots.py tests/test_query_pool_slots.py
.venv/bin/python scripts/run_readout_interface.py check \
  --data artifacts/predictions/memory_update_data_20260905_v2
```

## transfer and preflight

transfer committed code only, with the existing data/model/adapter artifacts described in [the Della setup](della.md).
do not transfer the unrelated local `src/tinymem/model/continuous_decoder.py` rewrite or the local virtual environment.
do not use `scripts/della_updates.sh` to launch this new study.
keep the frozen update sources and all previous output directories unchanged.

on Della, use the existing `tinymem-py311` environment and run from the repository root:

```bash
python -m pip check
python scripts/run_readout_interface.py --help
python scripts/run_readout_interface.py check \
  --data artifacts/predictions/memory_update_data_20260905_v2
mkdir -p logs
```

check the Slurm script's repository path, module, partition, and GPU constraint against the available allocation.
the script uses the existing project's 24G host-memory and 30-minute profile envelope; these are not measured requirements or a full-training allocation.
`TINYMEM_CONDA_MODULE` and `TINYMEM_CONDA_ENV` override the existing module and environment names.

## disposable profiles, run by the user

the following is a proposed small compute probe, not an approved scientific optimization schedule.
it executes eight disposable steps on four length-spaced training histories, including the longest.
all ten queries remain present in every selected history.
profile selection does not depend on answers or accuracy.
the shared data loader verifies train/development provenance, but profile encoding and evaluation use training histories only.
no trained weights or prediction answers are saved by the profile.
use fresh output names if a job fails; there is no resume or automatic retry.

```bash
stamp=$(date -u +%Y%m%dT%H%M%SZ)
for arm in affine gelu; do
  sbatch scripts/della_readout_profile.slurm \
    --data artifacts/predictions/memory_update_data_20260905_v2 \
    --output "artifacts/predictions/readout_profile_${stamp}_${arm}" \
    --arm "$arm" --seed 17 --histories 4 --steps 8 \
    --learning-rate 0.001 --weight-decay 0.01 --max-new-tokens 8
done
```

bring back each directory's `protocol.json`, `schedule.json`, `metrics.jsonl`, `evaluation.json`, and `complete.json`, plus Slurm logs.
`complete.json` is written last and hashes the four profile artifacts.
`metrics.jsonl` records every training-step duration, including the cold first step.
`evaluation.json` separates four-control readout time from full-text time.
CUDA memory is peak allocated tensor memory including the resident reader, not total GPU reservation or host RAM.
CPU/MPS peaks are explicitly unavailable, not zero.
model loading, input encoding, hashing, and file I/O are not included in the per-step timing.
profile outputs are not reusable checkpoints or evidence of scientific qualification.

## decisions required before full-size training

1. inspect both profiles for finite losses/gradients, complete output, longest-history memory, and runtime variation after the cold step.
2. choose a common GPU model, software environment, numeric settings, host memory, and walltime with headroom.
3. explicitly approve and record total steps, learning rate, weight decay, seed list, generation limit, and final-only checkpoint policy for both arms.
4. declare the full-text qualification rule and the final competence/stop criteria before observing full-size results.
5. check cross-process repeatability on the chosen Della hardware before six long jobs.
6. retain uncached history features unless measured cost justifies a separately verified bounded cache.

there are 288 histories across train/development.
each complete arm run evaluates four controls both initially and finally, plus one full-text reference per split: 25,920 generated query evaluations before counting the three teacher-forced loss passes per query.
include this evaluation cost, not only optimizer steps, in the walltime estimate.
a length-spaced four-history profile is a coverage probe, not an unbiased runtime sample of all histories.

## explicit runs and reports after schedule approval

`run` requires every schedule value explicitly and processes all 256 training and 32 development histories.
there is no confirmation subcommand or automatic extension after a failed gate.
use the same approved values for every arm/seed and fresh output directories.

```bash
# set these variables only after schedule approval
python scripts/run_readout_interface.py run \
  --data artifacts/predictions/memory_update_data_20260905_v2 \
  --device cuda --output "$RUN_OUTPUT" --arm "$ARM" --seed "$SEED" \
  --steps "$STEPS" --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" --max-new-tokens "$MAX_NEW_TOKENS"

# provide all completed affine/gelu arm directories for every declared seed
python scripts/run_readout_interface.py report \
  --runs "$AFFINE_RUN" "$GELU_RUN" \
  --output "$REPORT_OUTPUT" --resamples 2000 --bootstrap-seed 0
```

for three seeds, pass all six run paths to `--runs` in one report command.
each run owns its protocol, exact encodings, initial/final weights, metrics, predictions, and completion seal.
the separate report directory owns `report.json` and `report.md`; the JSON embeds every run seal and paired statistics.
this is the implemented lean layout, without a second root launch registry.
intervals use shared whole-history resamples across arms and seeds; seed variability is reported separately.
reports retain competence-limited results and label full-text qualification `rule_not_frozen`; they do not certify a later launch rule automatically.

## review disposition and limits

review found no local blocking defect in the profile, paired statistics, or CLI.
the first bounded review attempt exhausted its turn limit; subsequent read-only review returned findings.
GPU model, CUDA build, reader dtype, and deterministic-mode identity are now recorded and required to match across paired runs.
full-text reference equality remains strict: do not silently accept changed predictions or replace failed references.
if equal-reader jobs disagree, inspect hardware/software and reproducibility before aggregation; do not weaken the check after seeing outcomes.
this is a conservative choice rather than the review's suggested float-tolerant comparison.

the marginal-answer mode includes `unknown` and may therefore be an always-abstain reference.
it is not a category-aware oracle or a strong explicit-memory baseline.
the suspected BF16/FP32 boundary is already explicit (`last_hidden_state.float()`), and now has CPU BF16 profile coverage.
this does not qualify CUDA kernels or production Qwen execution.

the unrelated decoder SHA-256 remains `ea932d482a39e01b455a012f936d0babad5bb9a793de13c3b137a4f56e19cb65`.
no dependency manifest, lock file, frozen update source, or previous experiment result was changed for this milestone.
