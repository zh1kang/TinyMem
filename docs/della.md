# della runbook

code commit: `20d1410` (local, not pushed).
the committed code plus this handoff passes 1,632 tests in an isolated checkout with the existing datasets available.
the separate uncommitted decoder edit causes 83 failures in the working checkout and is deliberately excluded from the clean export below.
actual CUDA execution, cluster package installation, and Slurm submission are not yet verified.

run the current Qwen association study from `/scratch/gpfs/JORDANAT/caleb/TinyMem`.
use the existing cursor terminal on della for cluster commands.
use a terminal on the Mac for transfers.
the default Slurm account is used; no account name or QOS is inferred.

## transfer the committed code and current inputs

the local checkout has an unrelated edit in `src/tinymem/model/continuous_decoder.py`.
the commands below preserve that edit on the Mac but transfer committed code from a temporary clone.
this also transfers the local commits that are not on GitHub.
use a new destination, or inspect existing cluster edits before overwriting files.
do not use `rsync --delete`.

on the Mac:

```bash
cd /Users/caleb/TinyMem
ssh ck2867@della-gpu.princeton.edu 'mkdir -p /scratch/gpfs/JORDANAT/caleb/TinyMem'
TINYMEM_EXPORT=$(mktemp -d /tmp/tinymem-della.XXXXXX)
git clone --no-hardlinks /Users/caleb/TinyMem "$TINYMEM_EXPORT/TinyMem"
rsync -aP "$TINYMEM_EXPORT/TinyMem/" \
  ck2867@della-gpu.princeton.edu:/scratch/gpfs/JORDANAT/caleb/TinyMem/
rsync -aPR --filter='merge /Users/caleb/TinyMem/scripts/della-rsync.filter' \
  data/ artifacts/ docs/della.md docs/cursor_handoff.md \
  ck2867@della-gpu.princeton.edu:/scratch/gpfs/JORDANAT/caleb/TinyMem/
```

add `--dry-run` to either `rsync` command to inspect it first.
the first command copies a real Git checkout, not the macOS environment.
the second copies datasets, the pinned Qwen model, the qualified reader, the six completed training runs, and the runbooks.
it excludes partial confirmation, incomplete training-fit outputs, unrelated experiment outputs, caches, and intermediate local environments.
all existing dataset files are included intentionally; `data/` is about 4.9 GB, mostly the 4.2 GB Qwen snapshot on disk.
the clone's `origin` points to the Mac path; do not fetch or push from it on the cluster.

if the local source edit is later resolved, a direct folder transfer is also possible:

```bash
rsync -aP --filter='merge /Users/caleb/TinyMem/scripts/della-rsync.filter' \
  /Users/caleb/TinyMem/ \
  ck2867@della-gpu.princeton.edu:/scratch/gpfs/JORDANAT/caleb/TinyMem/
```

do not use that shortcut while there are uncommitted source changes you do not want to run.

## create a Linux environment

in cursor's cluster terminal:

```bash
cd /scratch/gpfs/JORDANAT/caleb/TinyMem
mkdir -p logs
module purge
module load anaconda3/2026.7
conda create -n tinymem-py311 python=3.11 pip -y
conda activate tinymem-py311
python -m pip install --only-binary=torch -e '.[dev,research]' \
  'torch==2.13.0' 'numpy==2.4.6' 'safetensors==0.8.0' \
  'accelerate==1.14.0' 'tokenizers==0.23.2' 'huggingface-hub==1.30.0'
python -m pip check
git diff --quiet HEAD -- src scripts tests pyproject.toml
python -m scripts.opaque.smoke --check-only
python -m pytest -q
```

reuse the environment if it already exists; do not recreate it over unrelated work.
these package versions match the local study, including the PEFT and Transformers pins in `pyproject.toml`.
the Linux CUDA wheel, driver compatibility, and actual GPU run are not yet verified.
if a version or module is unavailable, stop and inspect the error before choosing a documented replacement.
do not change versions silently or copy `.venv` from the Mac.

`--check-only` hashes the frozen sources, model snapshot, completed training artifacts, vocabulary, reader adapter, and training/development data without loading Qwen or inspecting confirmation answers.
the GPU jobs use offline model loading; stage inputs before submission.

## submit smoke first

```bash
sbatch scripts/della.slurm smoke
squeue --me
```

use the returned job ID to inspect `logs/tinymem-JOB_ID.out` and `.err`.
the job requests one public 40 GB GPU allocation, four CPU cores, 24 GB host RAM, and 15 minutes.
`gpu40` can be a full GPU or a 40 GB MIG slice; the log records the assigned device.
host RAM and GPU memory are different resources.
the test loads the actual model and adapter, uses the first training world, checks both writers, verifies all four state gradients, performs an optimizer step, and runs generation.
this is an execution check, not a scientific accuracy result.

if needed, set `TINYMEM_CONDA_MODULE` and `TINYMEM_CONDA_ENV` before `sbatch` to select an existing module and environment.
do not override `CUDA_VISIBLE_DEVICES`; Slurm sets it.

## run the fixed evaluation and diagnostics

after smoke succeeds:

```bash
sbatch --time=12:00:00 scripts/della.slurm all
squeue --me
```

12 hours is a walltime limit, not a measured runtime estimate.
the pipeline repeats smoke on its assigned GPU, then runs:

```text
smoke -> all confirmation baselines -> six checkpoint interventions -> report
      -> query and mean training-fit -> fixed-projection code oracle
```

the six trained writers already exist; `all` does not retrain them.
all output goes to `artifacts/predictions/della_JOB_ID/`.
the report is `analysis/report/report.json`, with accuracy and contrast plots beside it.
training-fit results are under `training_fit/`, and the code-only diagnostic is under `fixed_oracle/`.

the original Mac confirmation was interrupted after partial baseline progress.
it has no complete result and is not merged with CUDA outputs.
the new execution records backend, packages, dtypes, numerical settings, hardware, and portable source hashes separately from the unchanged study.
aggregation rejects incompatible execution records.
the oracle requires its training-fit result from the same execution configuration.

individual stages are available as `smoke`, `profile`, `confirmation`, `training-fit`, and `oracle`.
the optional second argument is a repository-relative output root.
an oracle-only job must use the root containing its completed CUDA training-fit results.
outputs are exclusive-create: a failed or partial stage is not resumed, overwritten, or silently skipped.
inspect failures before rerunning; use a fresh output root or the explicit underlying commands for untouched stages.
do not launch duplicate jobs into the same root.

to measure fresh training speed and four-segment gradients before any new training study:

```bash
sbatch --time=00:30:00 scripts/della.slurm profile
```

this runs ten updates for each writer with no confirmation evaluation.
the portable training entry point also accepts `--device cuda`, `--device mps`, or `--device cpu`.
new full training runs need their own declared protocol and output paths; do not replace the six historical runs.

## collect results

on the cluster:

```bash
jobstats JOB_ID
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

on the Mac, after completion:

```bash
rsync -aP \
  ck2867@della-gpu.princeton.edu:/scratch/gpfs/JORDANAT/caleb/TinyMem/artifacts/predictions/della_JOB_ID/ \
  /Users/caleb/TinyMem/artifacts/predictions/della_JOB_ID/
```

replace `JOB_ID` with the actual ID.
also retain both Slurm logs and `jobstats` output for the environment and resource record.
scratch is not a backup; retain completed evidence elsewhere.

cluster settings follow the official [Della](https://researchcomputing.princeton.edu/systems/della), [Slurm](https://researchcomputing.princeton.edu/support/knowledge-base/slurm), and [PyTorch](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch) guidance.
