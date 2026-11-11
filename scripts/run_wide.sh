#!/usr/bin/env bash
# Size control for the attention agent (docs/research_plan.md, experiment E12): attn_lstm_wide, about as many
# parameters as the lstm agent, trained with exactly the protocol of the attn_lstm runs it is compared with:
#   scratch   clean and fog_mask from random weights, 15M steps each
#   finetune  the clean run's step_10.0M.pt fine-tuned on fog_mask up to 15M (5M steps; fresh Adam, learning
#             rate annealed over the fine-tune), mirroring checkpoints/finetune/attn_lstm/clean_10.0M_to_fog_mask
# At most MAX_PARALLEL runs at a time (the host loses power under heavy load); scratch runs go first, and a
# fine-tune starts once its parent's step_10.0M.pt exists and a slot is free.
# Rerunning the script skips finished runs and resumes interrupted ones from their newest milestone that
# loads; training refuses to start in a folder that already holds final.pt, so no checkpoint is touched.
# Usage (on the training host, from the repository root, when no attn_lstm_wide run is active):
#   nohup bash scripts/run_wide.sh >> logs/wide.log 2>&1 &
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"
ARCH=attn_lstm_wide
SEEDS="${SEEDS:-1 2 3}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
TOTAL=15000000
VIEW_FLAGS_clean=""
VIEW_FLAGS_fog_mask="--use-fog --mask-enemy"
failures=0
# A second copy would resume runs that are still going.
if pgrep -f "[p]ong-rl .*${ARCH}" > /dev/null; then echo "${ARCH} runs are already active; not starting" >&2; exit 1; fi

loads() {  # loads <checkpoint>: true if torch can read it (a power cut may have left a broken file)
  uv run python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" "$1" \
    > /dev/null 2>&1
}

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n || failures=$((failures + 1))
  done
}

launch() {  # launch <run folder> <pong-rl args...>
  local run="$1"; shift
  if [[ -f "checkpoints/${run}/final.pt" ]]; then echo "Skipping ${run} (finished)"; return; fi
  wait_for_slot
  mkdir -p "logs/${run}"
  if [[ -e "checkpoints/${run}" ]]; then
    # Interrupted (e.g. a power cut): --resume keeps the folder, step counter and learning-rate schedule.
    local latest="" candidate
    for candidate in $(ls "checkpoints/${run}"/step_*.pt 2>/dev/null | sort -r || true); do
      if loads "${candidate}"; then latest="${candidate}"; break; fi
      echo "Unreadable ${candidate}; trying an earlier milestone" >&2
    done
    if [[ -z "${latest}" ]]; then echo "No milestone to resume ${run} from; inspect it by hand" >&2; return; fi
    if [[ "$(basename "${latest}")" == "step_15.0M.pt" ]]; then
      echo "${run} stopped between its last milestone and final.pt; finish it by hand" >&2; return
    fi
    echo "Resuming ${run} from ${latest} at $(date -Is)"
    echo "=== resumed from ${latest} at $(date -Is) ===" >> "logs/${run}/train.log"
    uv run pong-rl --resume "${latest}" >> "logs/${run}/train.log" 2>&1 &
    return
  fi
  echo "Launching ${run} at $(date -Is)"
  echo "=== launched at $(date -Is) ===" >> "logs/${run}/train.log"
  uv run pong-rl --arch "${ARCH}" --total-timesteps "${TOTAL}" "$@" >> "logs/${run}/train.log" 2>&1 &
}

for seed in ${SEEDS}; do
  for view in clean fog_mask; do
    flags="VIEW_FLAGS_${view}"
    # shellcheck disable=SC2086  # the view flags are meant to split into words
    launch "scratch/${ARCH}/${view}/seed_${seed}" --seed "${seed}" ${!flags}
  done
done

# Fine-tunes: wait for each parent's 10M milestone (renamed into place only once complete and on disk).
pending="${SEEDS}"
while [[ -n "${pending// /}" ]]; do
  waiting=""
  for seed in ${pending}; do
    parent="checkpoints/scratch/${ARCH}/clean/seed_${seed}/step_10.0M.pt"
    if [[ -f "${parent}" ]]; then
      # shellcheck disable=SC2086
      launch "finetune/${ARCH}/clean_10.0M_to_fog_mask/seed_${seed}" --seed "${seed}" \
        ${VIEW_FLAGS_fog_mask} --init-from "${parent}"
    else
      waiting="${waiting} ${seed}"
    fi
  done
  pending="${waiting}"
  if [[ -n "${pending// /}" ]]; then sleep 120; fi
done
while (( $(jobs -rp | wc -l) > 0 )); do
  wait -n || failures=$((failures + 1))
done
echo "All ${ARCH} runs ended at $(date -Is); ${failures} failed (see logs/<run>/train.log)"
(( failures == 0 ))
