#!/usr/bin/env bash
# Copy the training host's results into this checkout, at the same paths, for the figures, tables and paper
# (docs/data.md). The host is the source of every result; this copy is read-only by convention.
#   analysis/      every result file, except the raw game recordings (analysis/rollouts/, ~4 GB)
#   evaluations/   frozen evaluations
#   archive/       its README.md and the set-aside results (archive/analysis, archive/evaluations), not the
#                  aborted checkpoints or old tooling
# A result produced here (e.g. analysis/attention_strip.npz) is copied to the host by naming it after --push;
# nothing else is ever sent. Usage, from the repository root:
#   bash scripts/sync_results.sh
#   bash scripts/sync_results.sh --push analysis/attention_strip.npz
set -euo pipefail
HOST="${HOST:-computing}"
REMOTE="${REMOTE:-projects/ppo_atari}"
cd "$(dirname "$0")/.."
mkdir -p analysis evaluations archive
rsync -a --exclude rollouts/ "${HOST}:${REMOTE}/analysis/" analysis/
rsync -a "${HOST}:${REMOTE}/evaluations/" evaluations/
rsync -a "${HOST}:${REMOTE}/archive/README.md" archive/README.md
rsync -a "${HOST}:${REMOTE}/archive/analysis" "${HOST}:${REMOTE}/archive/evaluations" archive/
if [[ "${1:-}" == "--push" ]]; then
  shift
  for file in "$@"; do  # only the named files, to the same path on the host, never over an existing one
    rsync -a --ignore-existing --relative "${file}" "${HOST}:${REMOTE}/"
  done
fi
echo "analysis/ and evaluations/ are in sync with ${HOST}:${REMOTE}"
