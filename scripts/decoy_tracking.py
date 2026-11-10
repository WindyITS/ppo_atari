"""Where does an agent put its paddle when the enemy paddle it sees is mirrored? (E5b, docs/research_plan.md)

    uv run python scripts/decoy_tracking.py --workers 6

E5 showed that mirroring or erasing the enemy paddle wrecks every fog-trained agent. This asks what the agent
does with the mirrored cue. On the plain view the enemy paddle follows the ball (under fog the ball's hidden
height and the enemy paddle correlate at r 0.94-0.96), so the two cues cannot be told apart; the mirror
(enemy row r shown at row 90 - r) separates them. For every incoming pass through the agent's half
(pong_rl.analysis.rollouts.incoming_passes) the agent's paddle height when the ball reaches the paddle line is
regressed on two cues, both standardised:
    ball     the true height where the ball meets the paddle line
    enemy    the enemy paddle's height as the model saw it at that moment (mirrored under mirror_enemy)
A policy that reads the ball's height off the enemy paddle follows the enemy cue under the mirror; one that
tracks the ball (seen, or remembered) keeps following the ball. Also reported: the share of passes returned.

Checkpoints: finals trained on fog and on clean (the transform does nothing on masked views). Each plays
--episodes games (env seeds --seed, --seed + 1, ...) under none and mirror_enemy. Output: JSON lines (default
analysis/decoy_tracking.jsonl), one row per checkpoint and transform; reruns skip rows already written.
"""

import argparse
import functools
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pong_rl.analysis import scene  # noqa: E402
from pong_rl.analysis.batch import run_all  # noqa: E402
from pong_rl.analysis.player import Player  # noqa: E402
from pong_rl.analysis.rollouts import incoming_passes  # noqa: E402

TRANSFORMS = {"none": None, "mirror_enemy": scene.mirror_enemy}
MIRROR = scene.COURT_Y_MIN + scene.COURT_Y_MAX - 1  # row r is shown at MIRROR - r (90 on the 84x84 frame)


def play(path: str, transform: str, episodes: int, seed: int) -> tuple[dict, Player]:
    player = Player(path)
    rows = {k: [] for k in ("game", "ball_x", "ball_y", "enemy_y", "agent_y", "visible", "reward")}
    returns = []
    try:
        for game in range(episodes):
            player.reset(seed + game)
            while not player.done:
                step = player.step(transform=TRANSFORMS[transform])
                rows["game"].append(game)
                for k in ("ball_x", "ball_y", "enemy_y", "agent_y", "visible"):
                    rows[k].append(step.scene[k])
                rows["reward"].append(step.reward)
            returns.append(player.episode_return)
    finally:
        player.close()
    arrays = {k: np.asarray(v, dtype=bool if k == "visible" else float) for k, v in rows.items()}
    arrays["game"] = arrays["game"].astype(int)
    return {"arrays": arrays, "returns": returns}, player


def regress(agent, cues: dict) -> dict:
    """Standardised least-squares weights of the agent's paddle height on the cues, plus R^2."""
    X = np.column_stack([(c - c.mean()) / (c.std() + 1e-9) for c in cues.values()])
    y = agent - agent.mean()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / (y ** 2).sum() if (y ** 2).sum() > 0 else float("nan")
    return {**{name: float(b) for name, b in zip(cues, beta)}, "r2": float(r2)}


def analyse(job: str, episodes: int, seed: int) -> dict:
    path, transform = job.split("::")
    torch.set_num_threads(1)
    played, player = play(path, transform, episodes, seed)
    arrays = played["arrays"]
    passes = incoming_passes(arrays)
    agent, ball, enemy, returned = [], [], [], []
    for p in passes:
        last = p.rows[-1]
        if not (np.isfinite(arrays["agent_y"][last]) and np.isfinite(arrays["enemy_y"][last])):
            continue
        agent.append(arrays["agent_y"][last])
        ball.append(p.exit_y)
        seen = arrays["enemy_y"][last]
        enemy.append(MIRROR - seen if transform == "mirror_enemy" else seen)
        # returned: the agent wins the ball back (next reward in this game is not a loss right after the pass)
        after = np.flatnonzero((arrays["game"] == p.game) & (np.arange(len(arrays["game"])) > last)
                               & (arrays["reward"] != 0))
        returned.append(not (len(after) and arrays["reward"][after[0]] < 0 and after[0] - last <= 20))
    agent, ball, enemy = map(np.asarray, (agent, ball, enemy))
    row = {"job": job, "checkpoint": path, "transform": transform, "arch": player.arch,
           "train_view": player.train_view, "kind": player.checkpoint.get("kind", "scratch"),
           "run": player.checkpoint.get("run"), "episodes": episodes, "seed": seed,
           "mean_return": float(np.mean(played["returns"])), "passes": int(len(agent)),
           "returned": float(np.mean(returned)) if returned else None}
    if len(agent) >= 10:
        row["corr_ball"] = float(np.corrcoef(agent, ball)[0, 1])
        row["corr_enemy"] = float(np.corrcoef(agent, enemy)[0, 1])
        row["corr_ball_enemy"] = float(np.corrcoef(ball, enemy)[0, 1])
        row["weights"] = regress(agent, {"ball": ball, "enemy": enemy})
        row["gap_ball"] = float(np.median(np.abs(agent - ball)))
        row["gap_enemy"] = float(np.median(np.abs(agent - enemy)))
    return row


def default_checkpoints(root: Path) -> list[Path]:
    found = []
    for arch in ("cnn", "lstm", "attn_lstm"):
        found += root.glob(f"scratch/{arch}/fog/seed_*/final.pt")
        found += root.glob(f"scratch/{arch}/clean/seed_*/final.pt")
        found += root.glob(f"finetune/{arch}/clean_10.0M_to_fog/seed_*/final.pt")
    return sorted(found)


def describe(row: dict) -> str:
    w = row.get("weights", {})
    return (f"return {row['mean_return']:+.1f}, {row['passes']} passes, weights ball {w.get('ball', float('nan')):+.2f} "
            f"enemy {w.get('enemy', float('nan')):+.2f} (R2 {w.get('r2', float('nan')):.2f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("analysis/decoy_tracking.jsonl"))
    args = parser.parse_args()
    paths = args.checkpoints or default_checkpoints(args.root)
    jobs = [f"{p}::{t}" for p in paths for t in TRANSFORMS]
    failures = run_all(functools.partial(analyse, episodes=args.episodes, seed=args.seed), jobs, args.output,
                       "job", args.workers, describe)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
