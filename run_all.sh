#!/usr/bin/env bash
# Two-stage sweep (needs bash >= 4.3 for `wait -n`).
#   Stage 1, scratch:  every architecture x condition x seed, trained from scratch to TIMESTEPS.
#   Stage 2, finetune: each completed clean scratch run's 10M checkpoint, fine-tuned on mask, fog and
#                      fog_mask up to the same TIMESTEPS (the step count continues from the parent).
# Usage:
#   bash run_all.sh                           # both stages; ARCHS="cnn lstm attn_lstm", SEEDS="1 2 3"
#   STAGES=scratch bash run_all.sh            # only stage 1 (or STAGES=finetune)
#   ARCHS=lstm SEEDS="1" bash run_all.sh      # a subset
#   MAX_PARALLEL=6 bash run_all.sh            # at most 6 simultaneous jobs (default 8)
# Runs whose final.pt already exists are skipped, so rerunning the script resumes an interrupted sweep.
# An unfinished run's folders are moved to <folder>.aborted_<time> before it is restarted.
# Layout: checkpoints/<stage>/<arch>/<condition>/seed_<n>/ and the same folders under logs/,
#         where a fine-tune's condition names its parent, e.g. finetune/lstm/clean_10.0M_to_fog/seed_1.

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

read -r -a ARCHS <<< "${ARCHS:-cnn lstm attn_lstm}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3}"
read -r -a STAGES <<< "${STAGES:-scratch finetune}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
TIMESTEPS="${TIMESTEPS:-15000000}"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-step_10.0M.pt}"
PARENT_LABEL="${PARENT_CHECKPOINT#step_}"
PARENT_LABEL="${PARENT_LABEL%.pt}"

for stage in "${STAGES[@]}"; do
  if [[ "${stage}" != scratch && "${stage}" != finetune ]]; then
    echo "Unknown stage '${stage}' (use scratch and/or finetune)" >&2
    exit 2
  fi
done

CONDITIONS=(clean mask fog fog_mask)
declare -A FLAGS=([clean]="" [mask]="--mask-enemy" [fog]="--use-fog" [fog_mask]="--use-fog --mask-enemy")

running=0
launched=0
skipped=0
failed=0

# launch <run folder> <pong-rl arguments...>
launch() {
  local run="$1"
  shift
  if [[ -f "checkpoints/${run}/final.pt" ]]; then
    echo "Skipping ${run} (final.pt exists)"
    skipped=$((skipped + 1))
    return
  fi
  local folder
  for folder in "checkpoints/${run}" "logs/${run}"; do
    if [[ -e "${folder}" ]]; then
      mv "${folder}" "${folder}.aborted_$(date +%Y%m%d_%H%M%S)"
      echo "Moved unfinished ${folder} aside"
    fi
  done
  mkdir -p "logs/${run}"
  echo "Launching ${run} ..."
  uv run pong-rl "$@" --total-timesteps "${TIMESTEPS}" > "logs/${run}/train.log" 2>&1 &
  running=$((running + 1))
  launched=$((launched + 1))
  if (( running >= MAX_PARALLEL )); then
    wait -n || failed=$((failed + 1))
    running=$((running - 1))
  fi
}

finish_stage() {
  while (( running > 0 )); do
    wait -n || failed=$((failed + 1))
    running=$((running - 1))
  done
}

for stage in "${STAGES[@]}"; do
  case "${stage}" in
    scratch)
      for seed in "${SEEDS[@]}"; do
        for arch in "${ARCHS[@]}"; do
          for condition in "${CONDITIONS[@]}"; do
            # shellcheck disable=SC2086  # FLAGS values are intentionally split into separate flags
            launch "scratch/${arch}/${condition}/seed_${seed}" --arch "${arch}" --seed "${seed}" ${FLAGS[${condition}]}
          done
        done
      done
      ;;
    finetune)
      for seed in "${SEEDS[@]}"; do
        for arch in "${ARCHS[@]}"; do
          parent_dir="checkpoints/scratch/${arch}/clean/seed_${seed}"
          parent="${parent_dir}/${PARENT_CHECKPOINT}"
          for condition in mask fog fog_mask; do
            run="finetune/${arch}/clean_${PARENT_LABEL}_to_${condition}/seed_${seed}"
            # Only fine-tune from a finished parent, whose checkpoints can no longer be rewritten.
            if [[ ! -f "checkpoints/${run}/final.pt" && ! ( -f "${parent}" && -f "${parent_dir}/final.pt" ) ]]; then
              echo "Cannot launch ${run}: parent ${parent} is missing or its run is unfinished"
              failed=$((failed + 1))
              continue
            fi
            # shellcheck disable=SC2086
            launch "${run}" --arch "${arch}" --seed "${seed}" --init-from "${parent}" ${FLAGS[${condition}]}
          done
        done
      done
      ;;
  esac
  # Fine-tuning needs the clean scratch checkpoints, so each stage finishes before the next starts.
  finish_stage
done

echo "Launched ${launched}, skipped ${skipped} completed, failed ${failed}."
if (( failed > 0 )); then
  echo "Check the train.log files under ./logs/ for details."
  exit 1
fi
echo "All runs completed! Checkpoints saved in ./checkpoints/"
