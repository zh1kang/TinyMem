#!/bin/bash
# Run inside the allocated GPU job, from the repository root.
set -euo pipefail

task=${1:-smoke}
output=${2:-artifacts/predictions/della_${SLURM_JOB_ID:?submit through sbatch}}
study=artifacts/predictions/opaque_memory_study_20260905
data=artifacts/predictions/opaque_qa1_data_20260905
gate=artifacts/predictions/opaque_reader_adaptation_20260905
vocabulary=artifacts/predictions/raw_capacity_audit_20260905/opaque_train_vocabulary.json
runs=(query_pool_seed_1337 mean_pool_seed_1337 query_pool_seed_2027 mean_pool_seed_2027 query_pool_seed_4099 mean_pool_seed_4099)

case "$task" in
  smoke)
    python -m scripts.opaque.smoke --device cuda --output "$output/smoke"
    ;;
  profile)
    for writer in query_pool mean_pool; do
      python -m scripts.opaque.train --device cuda --profile --steps 10 \
        --writer-kind "$writer" --data "$data" --reader-gate "$gate" \
        --output "$output/profile_$writer"
    done
    ;;
  confirmation)
    python -m scripts.opaque.baselines --device cuda --data "$data" \
      --reader-gate "$gate" --vocabulary "$vocabulary" --split confirmation \
      --study-protocol "$study/protocol.json" --output "$output/confirmation/baselines"
    evaluations=()
    for run in "${runs[@]}"; do
      python -m scripts.opaque.memory --device cuda --training-run "$study/$run" \
        --study-protocol "$study/protocol.json" --split confirmation \
        --output "$output/confirmation/$run"
      evaluations+=("$output/confirmation/$run")
    done
    python -m scripts.opaque.aggregate --study-protocol "$study/protocol.json" \
      --baseline-run "$output/confirmation/baselines" --memory-runs "${evaluations[@]}" \
      --output "$output/analysis"
    ;;
  training-fit)
    for writer in query_pool mean_pool; do
      python -m scripts.opaque.training_fit --device cuda \
        --training-run "$study/${writer}_seed_1337" --study-protocol "$study/protocol.json" \
        --output "$output/training_fit/${writer}_seed_1337"
    done
    ;;
  oracle)
    python -m scripts.opaque.oracle --device cuda --training-run "$study/query_pool_seed_1337" \
      --training-fit "$output/training_fit/query_pool_seed_1337" \
      --study-protocol "$study/protocol.json" --output "$output/fixed_oracle"
    ;;
  all)
    for stage in smoke confirmation training-fit oracle; do
      bash scripts/della_run.sh "$stage" "$output"
    done
    ;;
  *)
    echo "usage: sbatch scripts/della.slurm {smoke|profile|confirmation|training-fit|oracle|all} [output-root]" >&2
    exit 2
    ;;
esac
