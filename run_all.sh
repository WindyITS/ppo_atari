#!/usr/bin/env bash
# Train all 4 conditions (fog off/on x enemy mask off/on) across each seed in SEEDS (default 2 = 8 runs).
# Usage:
#   bash run_all.sh                 # default: CNN, runs up to 8 jobs in parallel
#   ARCH=lstm bash run_all.sh       # CNN + LSTM (1 frame per step) instead of the frame-stacked CNN
#   MAX_PARALLEL=4 bash run_all.sh  # limit to 4 simultaneous jobs

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ARCH="${ARCH:-cnn}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
TIMESTEPS="${TIMESTEPS:-10000000}"
SEEDS=(1 2)
CONDITIONS=4  # fog off/on x enemy mask off/on
TOTAL_RUNS=$(( ${#SEEDS[@]} * CONDITIONS ))

mkdir -p logs checkpoints

running=0
failed=0
for seed in "${SEEDS[@]}"; do
  for fog_flag in "" "--use-fog"; do
    for mask_flag in "" "--mask-enemy"; do
      fog_id=0; [[ -n "$fog_flag" ]] && fog_id=1
      mask_id=0; [[ -n "$mask_flag" ]] && mask_id=1
      tag="${ARCH}_fog_${fog_id}_mask_${mask_id}_seed_${seed}"

      echo "Launching ${tag} ..."
      uv run pong-rl \
        --arch "${ARCH}" \
        --seed "${seed}" \
        --total-timesteps "${TIMESTEPS}" \
        ${fog_flag} ${mask_flag} \
        > "logs/${tag}.log" 2>&1 &

      running=$((running + 1))
      if (( running >= MAX_PARALLEL )); then
        wait -n || failed=$((failed + 1))
        running=$((running - 1))
      fi
    done
  done
done

while (( running > 0 )); do
  wait -n || failed=$((failed + 1))
  running=$((running - 1))
done

if (( failed > 0 )); then
  echo "${failed} of ${TOTAL_RUNS} runs failed. Check ./logs/ for details."
  exit 1
fi
echo "All ${TOTAL_RUNS} runs completed! Checkpoints saved in ./checkpoints/"
