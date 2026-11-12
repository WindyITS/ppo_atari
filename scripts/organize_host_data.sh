#!/usr/bin/env bash
# One-off reorganisation of the training host's data (2026-11-12); see docs/data.md for the resulting layout.
# Moves and merges only: nothing is deleted. Superseded results, aborted runs and leftovers of the 2026-11-05
# Codex supervisor go to archive/ (gitignored), with archive/README.md saying what each item is and why it was
# set aside. Current results keep one file per experiment under analysis/ and evaluations/ with plain names.
# Safe to rerun: every step checks that its source still exists and never overwrites (mv -n).
# Usage (on the host, from the repository root): bash scripts/organize_host_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
stamp=$(date -Is)
note() { echo "- $*" >> archive/README.md; echo "$*"; }
move() {  # move <source> <destination> <reason>
  local src="$1" dst="$2" why="$3"
  [[ -e "${src}" ]] || return 0
  mkdir -p "$(dirname "${dst}")"
  mv -n "${src}" "${dst}"
  note "\`${dst}\` (was \`${src}\`): ${why}"
}

mkdir -p archive
if [[ ! -f archive/README.md ]]; then
  cat > archive/README.md <<'EOF'
# Archive

Data set aside from the working folders: superseded result versions, aborted training runs and leftovers of
earlier tooling. Nothing here backs a claim in the paper or the report; docs/data.md lists the current files.
Kept, not deleted, so that every earlier number stays traceable.

EOF
fi
echo "## Reorganisation of ${stamp}" >> archive/README.md

# Superseded result versions
move analysis/attention_block.jsonl archive/analysis/attention_block.v1.jsonl \
  "E10 first run (attn_lstm only, no erase_ball branch); superseded by the E10b rerun, now analysis/attention_block.jsonl"
move logs/attention_block.log archive/logs/attention_block.v1.log "log of the E10 first run"
move analysis/decoy_tracking.regression_v1.jsonl archive/analysis/decoy_tracking.regression_v1.jsonl \
  "E5b with a joint regression, replaced by correlations (commit bfdee10)"
move analysis/enemy_shift.confounded_v1.jsonl archive/analysis/enemy_shift.confounded_v1.jsonl \
  "E5c shifting the whole enemy region, ball included (confounded); fixed in commit fd9b8fc"
move analysis/scripted_baselines.score_only_v1.jsonl archive/analysis/scripted_baselines.score_only_v1.jsonl \
  "E11 first version (score only, no return rate); replaced in commit 4cf17e5"
move evaluations/sweep_stochastic.before_wide.jsonl.bak archive/evaluations/sweep_stochastic.before_wide.jsonl \
  "copy taken before the E12 evaluation was appended; an exact prefix of evaluations/sweep_stochastic.jsonl"

# One file per experiment: plain names and merged parts
if [[ -f analysis/attention_block_v2.jsonl && ! -e analysis/attention_block.jsonl ]]; then
  mv analysis/attention_block_v2.jsonl analysis/attention_block.jsonl
  note "\`analysis/attention_block.jsonl\` (was \`analysis/attention_block_v2.jsonl\`): renamed, the only current E10 file"
fi
move logs/attention_block_v2.log logs/analysis/attention_block.log "renamed with its data file"
parts=(analysis/scripted_baselines_v2_*.jsonl)
if [[ -e "${parts[0]}" && ! -e analysis/scripted_baselines.jsonl ]]; then
  cat "${parts[@]}" > analysis/scripted_baselines.jsonl
  for p in "${parts[@]}"; do move "${p}" "archive/analysis/scripted_baselines_parts/$(basename "${p}")" \
    "one of the six E11 runs, merged into analysis/scripted_baselines.jsonl (one row each)"; done
fi
if [[ -f analysis/random_control_2.jsonl ]]; then
  cat analysis/random_control_2.jsonl >> analysis/random_control.jsonl
  move analysis/random_control_2.jsonl archive/analysis/random_control_2.jsonl \
    "E3b twins of the attn_lstm models (second batch, after commit bb65494); its rows were appended to analysis/random_control.jsonl"
fi
if grep -qaP '\x00' analysis/probes.jsonl 2>/dev/null; then
  cp -n analysis/probes.jsonl archive/analysis/probes.with_nul_line.jsonl
  tr -d '\000' < analysis/probes.jsonl | grep -v '^$' > analysis/probes.jsonl.clean
  mv analysis/probes.jsonl.clean analysis/probes.jsonl
  note "\`archive/analysis/probes.with_nul_line.jsonl\`: analysis/probes.jsonl as it was, with one line of NUL bytes left by a power cut; the working file now has that line removed (no row lost)"
fi

# Aborted training runs (restarts of 2026-11-05; run_all.sh set them aside before restarting)
while IFS= read -r folder; do
  move "${folder}" "archive/checkpoints_aborted/${folder#checkpoints/}" "training run aborted and restarted on 2026-11-05"
done < <(find checkpoints -maxdepth 4 -type d -name "*.aborted_*" | sort)
while IFS= read -r folder; do
  move "${folder}" "archive/logs_aborted/${folder#logs/}" "log of an aborted run"
done < <(find logs -maxdepth 4 -type d -name "*.aborted_*" | sort)

# Leftovers of the 2026-11-05 Codex supervisor and other stray files
for f in RUN_CONTEXT.md sweep_finetune.log sweep_scratch.log logs/sweep_handler.py logs/sweep_handler.log \
         logs/sweep_handler.lock logs/sweep_handler.pid logs/sweep_status.json logs/gpu_memory_guard.py \
         logs/gpu_memory_guard.log logs/gpu_memory_guard.lock logs/gpu_memory_guard.pid \
         logs/gpu_memory_status.json logs/eight_job_baseline.json; do
  move "${f}" "archive/codex_supervisor_2026-11-05/$(basename "${f}")" \
    "run supervisor and notes written by Codex for the first sweep (not part of the code base)"
done
move logs/e6_eval.sh archive/misc/e6_eval.sh "one-off launcher of the E6 evaluation (results in evaluations/controls_stochastic.jsonl)"
move scripts/attention_maps.py.host_pre_commit.bak archive/misc/attention_maps.py.host_pre_commit \
  "stale hand-copied draft of scripts/attention_maps.py, replaced by the committed version"
if [[ -d logs/__pycache__ ]]; then move logs/__pycache__ archive/misc/logs__pycache__ "bytecode of the Codex scripts"; fi

# Logs: analysis runs under logs/analysis/, sweep launchers under logs/sweeps/ (training logs stay in
# logs/scratch and logs/finetune, where the training command writes them)
for f in logs/baseline_*.log logs/decoy_tracking.log logs/enemy_shift.log logs/memory_interventions*.log \
         logs/probes_*.log logs/random_control*.log logs/record_rollouts.log logs/scripted_baselines.log \
         logs/view_interventions.log logs/wide_analyses.log; do
  if [[ -e "${f}" ]]; then move "${f}" "logs/analysis/$(basename "${f}")" "analysis log"; fi
done
for f in logs/controls.log logs/e6_eval.log logs/e6_eval_2.log logs/eval_attn_lstm.log logs/eval_wide.log logs/wide.log; do
  # wide.log may still be written by run_wide.sh: mv keeps the open file, which then grows in its new place
  if [[ -e "${f}" ]]; then move "${f}" "logs/sweeps/$(basename "${f}")" "sweep or evaluation launcher log"; fi
done
echo >> archive/README.md
echo "done; see archive/README.md"
