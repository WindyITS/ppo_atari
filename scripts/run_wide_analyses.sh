#!/usr/bin/env bash
# The attention and memory analyses of the small attention agent, repeated on the size-matched attn_lstm_wide
# (E12c, docs/research_plan.md), with the same scripts and settings and the default output files (rows carry
# the architecture; every step skips work already done, so rerunning resumes and adds new runs):
#   E2   record_rollouts.py      30 frozen-evaluation games per final model, own view
#   E3   probe_memory.py         linear probes of the hidden ball's height from the LSTM state
#   E4   memory_interventions.py memory wipes as the ball enters the fog
#   E10  attention_block.py      block attention to the ball before the fog (fog / fog_mask models only)
#   E13  attention_maps.py       attention on ball / paddles over a flight (summary of the E2 recordings)
# VIEWS chooses the training views whose finished models (scratch and clean_10.0M_to_<view> fine-tunes) are
# analysed. CPU only, WORKERS processes per step, so it can run beside training.
# Usage (on the training host, from the repository root):
#   VIEWS="clean fog_mask" nohup bash scripts/run_wide_analyses.sh >> logs/analysis/wide_analyses.log 2>&1 &
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PATH="$HOME/.local/bin:$PATH"
ARCH=attn_lstm_wide
VIEWS="${VIEWS:-clean fog_mask}"
WORKERS="${WORKERS:-3}"

runs=() checkpoints=() blockable=()
for view in ${VIEWS}; do
  for folder in checkpoints/scratch/${ARCH}/${view}/seed_* checkpoints/finetune/${ARCH}/clean_10.0M_to_${view}/seed_*; do
    [[ -f "${folder}/final.pt" ]] || continue
    runs+=("${folder#checkpoints/}")
    checkpoints+=("${folder}/final.pt")
    if [[ "${view}" == fog || "${view}" == fog_mask ]]; then blockable+=("${folder}/final.pt"); fi
  done
done
if (( ${#runs[@]} == 0 )); then echo "No finished ${ARCH} runs for views: ${VIEWS}" >&2; exit 1; fi
echo "=== $(date -Is) ${#runs[@]} runs: ${runs[*]}"

echo "--- E2 recordings"
uv run python scripts/record_rollouts.py --runs "${runs[@]}" --workers "${WORKERS}"
echo "--- E3 probes"
uv run python scripts/probe_memory.py --match "__${ARCH}__" --workers "${WORKERS}"
echo "--- E4 memory wipes"
uv run python scripts/memory_interventions.py --checkpoints "${checkpoints[@]}" --workers "${WORKERS}"
if (( ${#blockable[@]} > 0 )); then
  echo "--- E10 attention block"
  uv run python scripts/attention_block.py --checkpoints "${blockable[@]}" --workers "${WORKERS}"
fi
echo "--- E13 attention over a flight"
uv run python scripts/attention_maps.py summary --arch "${ARCH}" --out "analysis/attention_maps_${ARCH}.json"
echo "=== $(date -Is) done"
