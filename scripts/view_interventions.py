"""Enemy-paddle decoy test (E5 in docs/research_plan.md): do agents read the ball's height off the enemy paddle?

    uv run python scripts/view_interventions.py --workers 10

Each checkpoint plays the frozen evaluation's games on its own view three times, changing only what the model
sees (the game itself is untouched):
    none          the plain view
    mirror_enemy  the enemy paddle shown at its mirrored height (top <-> bottom), a wrong but realistic cue
    erase_enemy   the enemy paddle removed (as the mask view does)
If a policy reads the ball's height from the enemy paddle, mirror_enemy should hurt it more than erase_enemy.
Output: JSON lines (default analysis/view_interventions.jsonl), one row per checkpoint and transform; reruns
skip rows already written.
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pong_rl.analysis import scene  # noqa: E402
from pong_rl.analysis.player import Player  # noqa: E402

TRANSFORMS = {
    "none": None,
    "mirror_enemy": scene.mirror_enemy,
    "erase_enemy": lambda obs: scene.erase(obs, slice(scene.COURT_Y_MIN, scene.COURT_Y_MAX), scene.ENEMY_REGION),
}


def run(path: str, transform: str, episodes: int, seed: int) -> dict:
    torch.set_num_threads(1)
    player = Player(path)
    returns, lengths = [], []
    try:
        for game in range(episodes):
            player.reset(seed + game)
            while not player.done:
                player.step(transform=TRANSFORMS[transform])
            returns.append(player.episode_return)
            lengths.append(player.t)
    finally:
        player.close()
    wins = sum(r > 0 for r in returns)
    return {"checkpoint": path, "arch": player.arch, "train_view": player.train_view, "view": player.view,
            "kind": player.checkpoint.get("kind", "scratch"), "run": player.checkpoint.get("run"),
            "transform": transform, "episodes": episodes, "seed": seed, "mean_return": float(np.mean(returns)),
            "win_rate": wins / episodes, "returns": returns, "mean_length": float(np.mean(lengths))}


def default_checkpoints(root: Path) -> list[Path]:
    found = []
    for arch in ("cnn", "lstm", "attn_lstm"):
        found += root.glob(f"scratch/{arch}/*/seed_*/final.pt")
        found += root.glob(f"finetune/{arch}/clean_10.0M_to_*/seed_*/final.pt")
    return sorted(p for p in found if ".aborted_" not in str(p) and "__" not in p.parent.parent.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=None)
    parser.add_argument("--transforms", nargs="+", choices=list(TRANSFORMS), default=list(TRANSFORMS))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("analysis/view_interventions.jsonl"))
    args = parser.parse_args()
    paths = args.checkpoints or default_checkpoints(args.root)
    done = set()
    if args.output.is_file():
        for line in args.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["checkpoint"], row["transform"], row["episodes"], row["seed"]))
    jobs = [(str(p), t) for p in paths for t in args.transforms if (str(p), t, args.episodes, args.seed) not in done]
    print(f"{len(paths)} checkpoints, {len(jobs)} jobs to run, {args.workers} workers", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    started = time.time()
    with args.output.open("a", encoding="utf-8") as stream, \
            concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run, p, t, args.episodes, args.seed): (p, t) for p, t in jobs}
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            p, t = futures[future]
            try:
                row = future.result()
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"[{count}/{len(jobs)}] {p} {t}: {row['mean_return']:+.1f}, wins {row['win_rate']:.0%} "
                      f"({time.time() - started:.0f} s)", flush=True)
            except Exception as exc:
                failures += 1
                print(f"[{count}/{len(jobs)}] {p} {t}: FAILED {exc!r}", file=sys.stderr, flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
