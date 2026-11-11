#!/usr/bin/env bash
# Follow-up training controls for the fine-tune vs scratch question (docs/research_plan.md, experiment E6).
# Every run fine-tunes a clean scratch run's step_10.0M.pt on fog_mask with a 15M-step budget (to 25M), so
# its learning-rate schedule and budget on the new view match a run from scratch step for step:
#   ft15M        cnn and lstm, all weights from the parent
#   reinit_core  lstm, the parent's convolutional encoder with a fresh LSTM, actor and critic
# Runs go to their own tagged folders (e.g. checkpoints/finetune/lstm/clean_10.0M_to_fog_mask__ft15M/seed_1);
# training refuses to start in a folder that already holds final.pt, so no existing checkpoint is touched.
# Rerunning the script skips finished runs and resumes interrupted ones from their latest milestone.
# Usage (on the training host, from the repository root):
#   nohup bash scripts/run_controls.sh > logs/controls.log 2>&1 &
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"
SEEDS="${SEEDS:-1 2 3}"
TOTAL=25000000

launch() {  # launch <arch> <seed> <tag> [extra args...]
  local arch="$1" seed="$2" tag="$3"; shift 3
  local run="finetune/${arch}/clean_10.0M_to_fog_mask__${tag}/seed_${seed}"
  if [[ -f "checkpoints/${run}/final.pt" ]]; then echo "Skipping ${run} (finished)"; return; fi
  if [[ -e "checkpoints/${run}" ]]; then
    # Interrupted (e.g. a power cut): continue from the latest milestone with --resume, which keeps the
    # folder, step counter and learning-rate schedule and never rewrites an existing file.
    local latest
    latest="$(ls "checkpoints/${run}"/step_*.pt 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -z "${latest}" ]]; then echo "No milestone to resume ${run} from; inspect it by hand" >&2; return; fi
    echo "Resuming ${run} from ${latest}"
    echo "=== resumed from ${latest} at $(date -Is) ===" >> "logs/${run}/train.log"
    uv run pong-rl --resume "${latest}" >> "logs/${run}/train.log" 2>&1 &
    return
  fi
  mkdir -p "logs/${run}"
  echo "Launching ${run}"
  uv run pong-rl --arch "${arch}" --seed "${seed}" --use-fog --mask-enemy --total-timesteps "${TOTAL}" \
    --init-from "checkpoints/scratch/${arch}/clean/seed_${seed}/step_10.0M.pt" --run-tag "${tag}" "$@" \
    > "logs/${run}/train.log" 2>&1 &
}

for seed in ${SEEDS}; do
  launch lstm "${seed}" ft15M
  launch cnn "${seed}" ft15M
  launch lstm "${seed}" reinit_core --reinit lstm actor critic
done
wait
echo "All control runs finished"
