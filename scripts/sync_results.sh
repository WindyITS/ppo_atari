#!/usr/bin/env bash
# Copy the training host's results into this checkout, at the same paths, for the figures, tables and paper
# (docs/data.md). The host is the source of every result; this copy is read-only by convention.
#   analysis/      every result file, except the raw game recordings (analysis/rollouts/, ~4 GB)
#   evaluations/   frozen evaluations
#   archive/       only its README.md (the list of set-aside data), not the data itself
# Files produced here (e.g. analysis/attention_strip.npz) are not removed by a sync; add --push to copy
# them to the host as well. Usage, from the repository root:
#   bash scripts/sync_results.sh [--push]
set -euo pipefail
HOST="${HOST:-computing}"
REMOTE="${REMOTE:-projects/ppo_atari}"
cd "$(dirname "$0")/.."
mkdir -p analysis evaluations archive
rsync -a --exclude rollouts/ "${HOST}:${REMOTE}/analysis/" analysis/
rsync -a "${HOST}:${REMOTE}/evaluations/" evaluations/
rsync -a "${HOST}:${REMOTE}/archive/README.md" archive/README.md
if [[ "${1:-}" == "--push" ]]; then
  rsync -a --ignore-existing --exclude rollouts/ analysis/ "${HOST}:${REMOTE}/analysis/"
fi
echo "analysis/ and evaluations/ are in sync with ${HOST}:${REMOTE}"
