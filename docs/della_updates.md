# Della: paired-update study

This is a **new** single-budget experiment, separate from the completed association comparison. Its code/CLI/synthetic checks work locally; actual CUDA execution, installed cluster packages, Slurm resource availability, walltime, and scientific reader qualification are **not verified**. No jobs have been submitted by this implementation session.

## Transfer committed code and exact inputs

Preserve the unrelated local decoder edit. Export committed code through a fresh local clone, as in `docs/della.md`; never copy the working source tree over the cluster checkout. Do not push or use `rsync --delete`. Inspect an existing destination for changes first.

```bash
cd /Users/caleb/TinyMem
EXPORT=$(mktemp -d /tmp/tinymem-update-export.XXXXXX)
git clone --no-hardlinks . "$EXPORT/TinyMem"
# Inspect a dry-run before removing --dry-run:
rsync -aPn --exclude=.venv/ --exclude=__pycache__/ "$EXPORT/TinyMem/" \
  ck2867@della-gpu.princeton.edu:/scratch/gpfs/JORDANAT/caleb/TinyMem/
.venv/bin/python -m scripts.update_transfer_manifest > "$EXPORT/inputs.paths0"
rsync -aPnr --from0 --files-from="$EXPORT/inputs.paths0" ./ \
  ck2867@della-gpu.princeton.edu:/scratch/gpfs/JORDANAT/caleb/TinyMem/
```

Remove `n` from `-aPn` / `-aPnr` only after inspecting the transfer plan. The NUL-delimited manifest includes the new dataset, its transitive input/exclusion manifests, pinned Qwen snapshot, qualified reader, old frozen protocol/sources, and six existing checkpoints. Old data are included for preserved diagnostics; copying files is not evaluating confirmation. It excludes current partial confirmation outputs and local smoke artifacts. The old `della-rsync.filter` alone is **insufficient** for new data provenance; use the manifest command above. A local `rsync --dry-run` of this manifest passes.

Use the existing Linux environment instructions in `docs/della.md`. Their module/package versions still require verification on the cluster. Do not silently substitute package versions or copy the Mac `.venv`. Prepare `logs/` before submission.

## Input checks on the login node

```bash
cd /scratch/gpfs/JORDANAT/caleb/TinyMem
mkdir -p logs
# Activate the existing verified environment first.
python -m pip check
python -m scripts.run_memory_updates \
  --data artifacts/predictions/memory_update_data_20260905_v2 check
python -m scripts.opaque.smoke --check-only
```

Both checks avoid loading model weights or reading confirmation answers. The new check verifies all256 training/32 development worlds, source/exclusion hashes, and existing adapter identity. Model work requires a GPU Slurm allocation; never run it on the login node.

## Profile, then choose one fixed schedule

```bash
sbatch scripts/della_updates.slurm profile artifacts/predictions/update_profile_JOBTAG
```

Replace `JOBTAG` with a unique chosen label. The default request is one public40GB GPU, four CPUs,24GB host RAM,30minutes; availability and sufficiency are unverified. Profiling performs ten training-only steps per writer, checks all seven recurrent state gradients, saves timings/token/allocation information, and produces no reusable checkpoint. Estimate runtime from measured profile output before choosing walltime or training steps. There is no fabricated recommended runtime and no automatically expanded budget.

The old negative evidence does not establish that longer training will solve the compressed-reader problem. This new run measures the declared objective and records competence-limited outcomes if necessary; it is not a promise of successful learned compression. Choose `STEPS` explicitly before launching. There is no parameter sweep or confirmation-based schedule selection.

## One allocation for qualification, six new runs, and reporting

```bash
# Set from training-only profiling and a predeclared compute budget, not held-out scores:
STEPS=YOUR_FIXED_POSITIVE_INTEGER
sbatch --time=YOUR_WALLTIME scripts/della_updates.slurm run \
  artifacts/predictions/update_study_JOBTAG "$STEPS"
```

The placeholders are intentional: these commands do not claim a measured full-size runtime or silently choose a scientific schedule. `run` executes:

```text
input check → new full-text development qualification → freeze exact launch
            → query/mean × seeds1337,2027,4099
            → all13 confirmation evaluations → verified report
```

The new qualification must meet95% known **and** absent accuracy separately at each of four states, including the nine-binding addition. Failure is preserved and immediately stops before launch/training. No reader adaptation or architecture fallback is attempted. Every new training run starts fresh; the six historical association checkpoints are not retrained or overwritten.

The entire pipeline uses one allocation so GPU model/runtime/numerical identity stays matched. `gpu40` can select heterogeneous devices; separate-stage execution must match the captured device model/capability and packages exactly. Do not relax identity checks to merge results. Runtime records expose mismatches.

Each stage uses exclusive-create outputs and hash-checked completion. Partial runs are **not resumable** and never satisfy confirmation authorization. A timeout during training requires inspecting logs and planning fresh output with sufficient allocation, not counting an intermediate checkpoint as final. If training completed but evaluation was never started, a matching allocation can use `evaluate OUTPUT_ROOT`; untouched individual methods can also use the Python CLI. Do not rerun `evaluate` over existing partial/successful outputs. Reports alone need no GPU:

```bash
bash scripts/della_updates.sh report artifacts/predictions/update_study_JOBTAG
```

## Existing association results and diagnostics

The user reports the old confirmation, training-fit, and fixed-projection oracle completed with negative results; matching artifacts still need local verification. **Do not run the old `all` pipeline again by default.** Collect the existing completed output and logs first. Keep the old frozen protocol and portable tooling unchanged.

If an old diagnostic truly has not completed, the existing `scripts/della_run.sh training-fit ROOT` loads the two seed1337 checkpoints without retraining; `oracle ROOT` requires the checked query-pool training-fit under that same root/runtime. Never point it at new update checkpoints. Its fixed code oracle and training-fit gates have existing small-model tests. This is diagnosis of the old task, not a success gate for the new update study.

## Verification and collection

Local shell tests prove qualification precedes freezing, six training calls precede all13 explicit confirmation calls, failures stop later stages, invalid paths/steps fail before execution, and no old training command is invoked. Existing old-script checkpoint/diagnostic linkage remains unchanged. Synthetic Python tests exercise actual tiny training, checkpoint loading, confirmation authorization and report generation; shell tests use an explicit command stub rather than pretending to run Slurm/CUDA.

After a real job, retain `logs/tinymem-update-JOB_ID.out/.err`, `sacct`, `jobstats`, the complete output tree, and profile evidence. The result is `OUTPUT_ROOT/report/report.json` and `.md`. Transfer back to a fresh Mac path, preserving partial outputs separately. Scratch is not a backup.
