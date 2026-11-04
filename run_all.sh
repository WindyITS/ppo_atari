#!/usr/bin/env bash
# Train all 4 conditions (fog off/on x enemy mask off/on) across 3 seeds = 12 runs.
# Usage:
#   bash run_all.sh                 # default: runs 4 jobs in parallel at a time (3 waves of 4)
#   MAX_PARALLEL=6 bash run_all.sh  # run 6 jobs in parallel at a time (2 waves of 6)
#   MAX_PARALLEL=12 bash run_all.sh # run all 12 simultaneously

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

MAX_PARALLEL="${MAX_PARALLEL:-6}"
TIMESTEPS="${TIMESTEPS:-5000000}"
SEEDS=(1 2 3)

mkdir -p logs checkpoints

running=0
for seed in "${SEEDS[@]}"; do
  for fog_flag in "" "--use-fog"; do
    for mask_flag in "" "--mask-enemy"; do
      fog_id=0; [[ -n "$fog_flag" ]] && fog_id=1
      mask_id=0; [[ -n "$mask_flag" ]] && mask_id=1
      tag="fog_${fog_id}_mask_${mask_id}_seed_${seed}"

      echo "Launching ${tag} ..."
      python ppo_atari.py \
        --seed "${seed}" \
        --total-timesteps "${TIMESTEPS}" \
        ${fog_flag} ${mask_flag} \
        > "logs/${tag}.log" 2>&1 &

      running=$((running + 1))
      if (( running >= MAX_PARALLEL )); then
        wait -n
        running=$((running - 1))
      fi
    done
  done
done

wait
echo "All 12 runs completed! Checkpoints saved in ./checkpoints/"
