#!/bin/bash
# Separate paired-update pipeline. Never rerun the completed association study.
set -euo pipefail

task=${1:?usage: della_updates.sh check|profile|qualify|run|evaluate|report OUTPUT_ROOT [STEPS]}
output=${2:?a fresh repository-relative output root is required}
data=${TINYMEM_UPDATE_DATA:-artifacts/predictions/memory_update_data_20260905_v2}
python=${TINYMEM_PYTHON:-python}

case "$task" in check|profile|qualify|run|evaluate|report) ;; *) echo "unknown task: $task" >&2; exit 2;; esac
if [[ "$output" = /* || "/$output/" = *"/../"* || "$output" = . || "$output" = "" ]]; then
  echo "output must be a safe repository-relative path" >&2; exit 2
fi
if [[ "$task" != check && "$task" != report && -z ${SLURM_JOB_ID:-} ]]; then
  echo "model work requires a Slurm GPU allocation" >&2; exit 2
fi
if [[ "$task" = run ]]; then
  steps=${3:?run requires an explicitly chosen fixed number of training steps}
  if [[ ! "$steps" =~ ^[1-9][0-9]*$ ]]; then echo "steps must be positive integer" >&2; exit 2; fi
fi

base=("$python" -m scripts.run_memory_updates --data "$data" --device cuda)
"${base[@]}" check
case "$task" in
  check) ;;
  profile)
    for method in query_pool mean_pool; do
      "${base[@]}" profile --method "$method" --output "$output/profile_$method"
    done
    ;;
  qualify)
    "${base[@]}" qualify --output "$output/qualification"
    ;;
  run)
    # New qualification and all new training share this one allocation/runtime.
    # Any failure stops the pipeline. No architecture/budget fallback is allowed.
    "${base[@]}" qualify --output "$output/qualification"
    "${base[@]}" freeze --qualification "$output/qualification" --output "$output/launch" --steps "$steps"
    for seed in 1337 2027 4099; do
      for method in query_pool mean_pool; do
        "${base[@]}" train --launch "$output/launch" --method "$method" --seed "$seed"
      done
    done
    bash scripts/della_updates.sh evaluate "$output"
    ;;
  evaluate)
    # Every evaluation checks all six completions before opening confirmation.
    for seed in 1337 2027 4099; do
      for method in query_pool mean_pool; do
        "${base[@]}" evaluate --launch "$output/launch" --method "$method" --seed "$seed" --split confirmation
      done
    done
    for method in recent_native recent_vocabulary latest_vocabulary latest_template fingerprint full_context no_memory; do
      "${base[@]}" evaluate --launch "$output/launch" --method "$method" --split confirmation
    done
    bash scripts/della_updates.sh report "$output"
    ;;
  report)
    "$python" -m scripts.report_memory_updates --launch "$output/launch" --split confirmation --output "$output/report"
    ;;
esac
