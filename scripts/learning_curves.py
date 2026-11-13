"""Training curves: mean training-episode return per 0.5M agent steps, for every run (data for the paper's
learning figure).

    uv run python scripts/learning_curves.py            # on the training host, from the repository root

Reads the episode logs that training writes next to each train.log (logs/<scratch|finetune>/<arch>/<label>/
seed_<n>/episodes_<launch>.jsonl, one file per launch). A run resumed after an interruption has several files
that overlap: the steps between the milestone it resumed from and the moment it stopped were played twice. For
each launch after the first, the episodes of earlier launches at or after its first step are dropped, so every
step counts once, from the launch whose weights became the final model. Episodes of a fine-tune are counted
from its parent's 10M milestone (global step 10M = 0 steps on the new view). Control runs (labels with "__")
and runs without a final checkpoint are skipped.
Output: analysis/learning_curves.json, per run {run, arch, kind, view, seed, bin_steps, start, mean_return
per bin, episodes per bin}.
"""

import argparse
import json
from pathlib import Path

import numpy as np

BIN = 500_000


def episodes(folder: Path) -> list[tuple[int, float]]:
    """(global_step, return) of every episode of a run, each step counted once (see the module docstring)."""
    launches = []
    for path in sorted(folder.glob("episodes_*.jsonl"), key=lambda p: int(p.stem.split("_")[1])):
        rows = []
        for line in path.read_text(errors="replace").splitlines():
            try:  # a power cut can leave a torn or NUL-filled line; it is skipped
                rows.append(json.loads(line.strip("\x00")))
            except json.JSONDecodeError:
                continue
        if rows:
            launches.append([(int(r["global_step"]), float(r["episode_return"])) for r in rows])
    kept = []
    for launch in launches:
        first = launch[0][0]
        kept = [e for e in kept if e[0] < first] + launch
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("analysis/learning_curves.json"))
    args = parser.parse_args()
    rows = []
    for folder in sorted(args.logs.glob("*/*/*/seed_*")):
        kind, arch, label, seed = folder.relative_to(args.logs).parts
        if kind not in ("scratch", "finetune") or "__" in label:
            continue
        if not (args.checkpoints / kind / arch / label / seed / "final.pt").exists():
            continue
        found = episodes(folder)
        if not found:
            continue
        steps, returns = np.array([s for s, _ in found]), np.array([r for _, r in found])
        start = 10_000_000 if kind == "finetune" else 0
        bins = ((steps - start) // BIN).astype(int)
        n = int(bins.max()) + 1
        means = [float(returns[bins == b].mean()) if (bins == b).any() else None for b in range(n)]
        counts = [int((bins == b).sum()) for b in range(n)]
        view = label.removeprefix("clean_10.0M_to_")
        rows.append({"run": f"{kind}/{arch}/{label}/{seed}", "arch": arch, "kind": kind, "view": view,
                     "seed": int(seed.removeprefix("seed_")), "bin_steps": BIN, "start": start,
                     "mean_return": means, "episodes": counts})
        print(f"{rows[-1]['run']}: {len(found)} episodes, {n} bins", flush=True)
    args.output.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
