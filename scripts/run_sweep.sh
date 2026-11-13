#!/usr/bin/env bash
# Train any set of runs with the protocol of the paper (docs/research_plan.md), in one pool of at most
# MAX_PARALLEL runs (the host loses power under heavy load). RUNS lists one spec per architecture,
#   <arch>:<scratch views, comma-separated>:<fine-tune target views, comma-separated or empty>
# for the seeds in SEEDS:
#   scratch   checkpoints/scratch/<arch>/<view>/seed_<n>, 15M steps from random weights
#   finetune  checkpoints/finetune/<arch>/clean_10.0M_to_<view>/seed_<n>: the clean scratch run's step_10.0M.pt
#             fine-tuned on <view> up to 15M (5M steps; fresh Adam, learning rate annealed over the fine-tune)
# Scratch runs are launched in the order given (list clean first: the clean runs are the fine-tunes' parents);
# a fine-tune starts once its parent's step_10.0M.pt exists and a slot is free. Finished runs are skipped and
# interrupted ones resume from their newest milestone that loads; training refuses to start in a folder that
# already holds final.pt, so no checkpoint is touched. DRY_RUN=1 only lists the runs.
# Example (E15, more seeds for the fog + hidden paddle comparison):
#   SEEDS="4 5 6" RUNS="attn_lstm:clean,fog_mask:fog_mask attn_lstm_wide:clean,fog_mask:fog_mask lstm:fog_mask:" \
#     nohup bash scripts/run_sweep.sh >> logs/sweeps/seeds_4-6.log 2>&1 &
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"
# Recomputing every replay chunk in the backward pass (threshold 0) is exact and needs far less GPU memory;
# the earlier runs used 2048, which changes memory use only, not the maths.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPLAY_THRESHOLD=0
SEEDS="${SEEDS:-1 2 3}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
TOTAL=15000000
RUNS="${RUNS:?set RUNS, e.g. RUNS=\"lstm:fog_mask:\"}"
DRY_RUN="${DRY_RUN:-0}"
VIEW_FLAGS_clean=""
VIEW_FLAGS_mask="--mask-enemy"
VIEW_FLAGS_fog="--use-fog"
VIEW_FLAGS_fog_mask="--use-fog --mask-enemy"
pids=()
# A second launcher would resume runs that are still going.
if [[ "${DRY_RUN}" != 1 ]] && pgrep -f "[p]ong-rl" > /dev/null; then echo "Training runs are active; not starting" >&2; exit 1; fi

loads() {  # loads <checkpoint>: true if torch can read it (a power cut may have left a broken file)
  uv run python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" "$1" \
    > /dev/null 2>&1
}

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n || true  # only frees a slot; statuses are counted from pids at the end
  done
}

launch() {  # launch <arch> <run folder> <pong-rl args...>
  local arch="$1" run="$2"; shift 2
  if [[ -f "checkpoints/${run}/final.pt" ]]; then echo "Skipping ${run} (finished)"; return; fi
  if [[ "${DRY_RUN}" == 1 ]]; then echo "Would run ${run}: --arch ${arch} $*"; return; fi
  wait_for_slot
  mkdir -p "logs/${run}"
  if [[ -e "checkpoints/${run}" ]]; then
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
    pids+=($!)
    return
  fi
  echo "Launching ${run} at $(date -Is)"
  echo "=== launched at $(date -Is) ===" >> "logs/${run}/train.log"
  uv run pong-rl --arch "${arch}" --total-timesteps "${TOTAL}" --replay-checkpoint-threshold "${REPLAY_THRESHOLD}" \
    "$@" >> "logs/${run}/train.log" 2>&1 &
  pids+=($!)
}

# Scratch runs, in the order of RUNS and of each spec's views.
for spec in ${RUNS}; do
  IFS=: read -r arch views _ <<< "${spec}"
  for view in ${views//,/ }; do
    for seed in ${SEEDS}; do
      flags="VIEW_FLAGS_${view}"
      # shellcheck disable=SC2086  # the view flags are meant to split into words
      launch "${arch}" "scratch/${arch}/${view}/seed_${seed}" --seed "${seed}" ${!flags}
    done
  done
done

# Fine-tunes: wait for each parent's 10M milestone (renamed into place only once complete and on disk).
pending=()
for spec in ${RUNS}; do
  IFS=: read -r arch _ targets <<< "${spec}"
  for view in ${targets//,/ }; do
    for seed in ${SEEDS}; do pending+=("${arch}:${view}:${seed}"); done
  done
done
while (( ${#pending[@]} > 0 )); do
  waiting=()
  for item in "${pending[@]}"; do
    IFS=: read -r arch view seed <<< "${item}"
    parent="checkpoints/scratch/${arch}/clean/seed_${seed}/step_10.0M.pt"
    if [[ -f "${parent}" || "${DRY_RUN}" == 1 ]]; then
      flags="VIEW_FLAGS_${view}"
      # shellcheck disable=SC2086
      launch "${arch}" "finetune/${arch}/clean_10.0M_to_${view}/seed_${seed}" --seed "${seed}" \
        ${!flags} --init-from "${parent}"
    else
      waiting+=("${item}")
    fi
  done
  pending=("${waiting[@]+"${waiting[@]}"}")
  if (( ${#pending[@]} > 0 )); then sleep 120; fi
done
failures=0
for pid in "${pids[@]+"${pids[@]}"}"; do
  wait "${pid}" || failures=$((failures + 1))
done
echo "All runs (${RUNS}; seeds ${SEEDS}) ended at $(date -Is); ${failures} failed (see logs/<run>/train.log)"
(( failures == 0 ))
