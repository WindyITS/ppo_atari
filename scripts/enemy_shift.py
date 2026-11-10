"""Does the agent read the ball's height off the enemy paddle? A dose-response test (E5c, docs/research_plan.md).

    uv run python scripts/enemy_shift.py --workers 4

E5 showed that mirroring or erasing the enemy paddle wrecks every fog-trained agent, but a mirrored paddle is
far outside anything the agent has seen, so it cannot say what the agent does with the cue. Here the cue is
moved a little instead. Each checkpoint plays the frozen evaluation's games on its own view (Player: same env
and action seeds). Every time the ball starts towards the agent (first step moving right after it left the
enemy), the game is saved and the flight is replayed in several branches from the save, with the same action
seed in all but control_b:
    shift_<d>   the enemy paddle the model sees moved by d pixels (d in SHIFTS; + is down), from the save until
                the ball reaches the agent's paddle line (x >= PADDLE_X); the game itself is untouched
    control_b   d = 0 with another action seed (the sampling-noise floor)
In every branch: the agent's paddle height when the ball reaches the paddle line, and whether it returns the
ball (returned / missed / undecided, as in scripts/memory_interventions.py). The gain is the least-squares slope
of (paddle height in shift_d - in shift_0) on d over all events: about 1 if the agent puts its paddle where the
enemy paddle says the ball is, about 0 if it ignores the enemy paddle. Masked views are skipped (the cue is not
shown). Output: JSON lines (default analysis/enemy_shift.jsonl), one row per checkpoint, appended as each
finishes; reruns skip checkpoints already done.
"""

import argparse
import functools
import os
from pathlib import Path
import sys
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pong_rl.analysis import scene  # noqa: E402
from pong_rl.analysis.batch import run_all  # noqa: E402
from pong_rl.analysis.player import Player  # noqa: E402
from pong_rl.wrappers import COURT_Y_MAX, COURT_Y_MIN, PONG_BG_VALUE  # noqa: E402

SHIFTS = (-8, -4, 0, 4, 8)
PADDLE_X = 70.0
RETURN_X = 40.0
MAX_STEPS = 150


PADDLE_MIN_ROWS = 4  # a vertical run of bright pixels at least this long is the paddle; the ball is ~2 rows


def paddle_pixels(region: np.ndarray) -> np.ndarray:
    """Boolean mask of the enemy paddle's pixels in one (rows, columns) crop of the enemy region: per column,
    bright runs of at least PADDLE_MIN_ROWS rows. The ball, which can pass through the region just after the
    enemy hits it, is shorter and is left out."""
    bright = np.abs(region.astype(np.int16) - PONG_BG_VALUE) > scene.THRESHOLD
    mask = np.zeros_like(bright)
    for c in range(bright.shape[1]):
        rows = np.flatnonzero(bright[:, c])
        if not len(rows):
            continue
        runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
        for run in runs:
            if len(run) >= PADDLE_MIN_ROWS:
                mask[run, c] = True
    return mask


def shift_enemy(frame: np.ndarray, d: int) -> np.ndarray:
    """The frame with the enemy paddle moved d rows down inside the court (+ is down). Only the paddle's own
    pixels move: its old place is painted in the background and anything else in the enemy region (the ball
    just after the enemy hit it) stays where it is. Works on (..., 84, 84) arrays."""
    if d == 0:
        return frame
    out = frame.copy()
    flat = out.reshape(-1, *out.shape[-2:])
    for f in flat:
        region = f[COURT_Y_MIN:COURT_Y_MAX, scene.ENEMY_REGION]
        mask = paddle_pixels(region)
        if not mask.any():
            continue
        values = np.where(mask, region, PONG_BG_VALUE)
        moved_values = np.full_like(region, PONG_BG_VALUE)
        moved_mask = np.zeros_like(mask)
        if d > 0:
            moved_values[d:], moved_mask[d:] = values[:-d], mask[:-d]
        else:
            moved_values[:d], moved_mask[:d] = values[-d:], mask[-d:]
        region[mask] = PONG_BG_VALUE
        region[moved_mask] = moved_values[moved_mask]
    return out


def play_branch(player: Player, d: int) -> dict:
    """From the save: shift the enemy cue by d until the ball reaches the paddle line; play the point out."""
    previous_x, reached, paddle_y, ball_y = None, False, None, None
    transform = functools.partial(shift_enemy, d=d)
    for k in range(MAX_STEPS):
        if player.done:
            return {"outcome": "undecided", "paddle_y": paddle_y, "ball_y": ball_y}
        where = player.scene_now()
        x = where["ball_x"]
        if not reached and x == x and x >= PADDLE_X:
            reached = True
            paddle_y = where["agent_y"] if where["agent_y"] == where["agent_y"] else None
            ball_y = where["ball_y"]
        if reached and x == x and previous_x is not None and x < previous_x and x < RETURN_X:
            return {"outcome": "returned", "paddle_y": paddle_y, "ball_y": ball_y}
        step = player.step(transform=None if reached else transform)
        if step.reward != 0:
            return {"outcome": "returned" if step.reward > 0 else "missed", "paddle_y": paddle_y, "ball_y": ball_y}
        previous_x = x if x == x else previous_x
    return {"outcome": "undecided", "paddle_y": paddle_y, "ball_y": ball_y}


def run(path: Path, episodes: int, seed: int, max_events: int) -> dict:
    torch.set_num_threads(1)
    player = Player(path)
    branches = [f"shift_{d}" for d in SHIFTS] + ["control_b"]
    events = []
    try:
        if "mask" in player.view:
            raise ValueError(f"{path}: masked view, the enemy paddle is never shown")
        for game in range(episodes):
            player.reset(seed + game)
            previous_x, armed, count = None, True, 0
            while not player.done:
                where = player.scene_now()
                x = where["ball_x"]
                if x != x or (previous_x is not None and x < previous_x):
                    armed = True  # ball gone or heading to the enemy: the next flight our way may trigger
                incoming = x == x and previous_x is not None and x > previous_x
                if armed and incoming and count < max_events:
                    armed, count = False, count + 1
                    saved = player.save()
                    branch_seed = zlib.crc32(f"{seed + game}:{player.t}".encode())
                    outcomes = {}
                    for name in branches:
                        d = 0 if name == "control_b" else int(name.removeprefix("shift_"))
                        player.load(saved, reseed=branch_seed + (1 if name == "control_b" else 0))
                        outcomes[name] = play_branch(player, d)
                    player.load(saved)  # back to the original game, unperturbed
                    events.append({"game": game, "t": player.t, "ball_x": x, "outcomes": outcomes})
                player.step()
                previous_x = x if x == x else previous_x
    finally:
        player.close()
    return summarize(path, player, events, episodes, seed, max_events)


def summarize(path, player, events, episodes, seed, max_events) -> dict:
    rates = {}
    for name in [f"shift_{d}" for d in SHIFTS] + ["control_b"]:
        outcomes = [e["outcomes"][name]["outcome"] for e in events]
        decided = [o for o in outcomes if o != "undecided"]
        rates[name] = sum(o == "returned" for o in decided) / len(decided) if decided else None
    xs, ys, noise = [], [], []
    for e in events:
        base = e["outcomes"]["shift_0"]["paddle_y"]
        if base is None:
            continue
        b = e["outcomes"]["control_b"]["paddle_y"]
        if b is not None:
            noise.append(b - base)
        for d in SHIFTS:
            if d and e["outcomes"][f"shift_{d}"]["paddle_y"] is not None:
                xs.append(d)
                ys.append(e["outcomes"][f"shift_{d}"]["paddle_y"] - base)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    gain = float((xs * ys).sum() / (xs ** 2).sum()) if len(xs) else None
    per_shift = {str(d): float(np.mean(ys[xs == d])) for d in SHIFTS if d and (xs == d).any()}
    rng = np.random.default_rng(0)
    if len(events) > 5 and len(xs):
        # bootstrap over events: resample the events, recompute the slope
        ev = [[(d, e["outcomes"][f"shift_{d}"]["paddle_y"] - e["outcomes"]["shift_0"]["paddle_y"])
               for d in SHIFTS if d and e["outcomes"]["shift_0"]["paddle_y"] is not None
               and e["outcomes"][f"shift_{d}"]["paddle_y"] is not None] for e in events]
        boots = []
        for _ in range(1000):
            pick = [p for i in rng.integers(0, len(ev), len(ev)) for p in ev[i]]
            if pick:
                a = np.asarray(pick, float)
                boots.append((a[:, 0] * a[:, 1]).sum() / (a[:, 0] ** 2).sum())
        gain_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    else:
        gain_ci = None
    return {"checkpoint": str(path), "arch": player.arch, "train_view": player.train_view, "view": player.view,
            "kind": player.checkpoint.get("kind", "scratch"), "run": player.checkpoint.get("run"),
            "episodes": episodes, "seed": seed, "max_events_per_game": max_events, "shifts": list(SHIFTS),
            "events": len(events), "gain": gain, "gain_ci": gain_ci, "mean_shift_by_d": per_shift,
            "noise_sd": float(np.std(noise)) if noise else None, "return_rate": rates,
            "event_list": events}


def default_checkpoints(root: Path) -> list[Path]:
    found = []
    for arch in ("cnn", "lstm", "attn_lstm"):
        found += root.glob(f"scratch/{arch}/clean/seed_*/final.pt")
        found += root.glob(f"scratch/{arch}/fog/seed_*/final.pt")
        found += root.glob(f"finetune/{arch}/clean_10.0M_to_fog/seed_*/final.pt")
    return sorted(found)


def describe(row: dict) -> str:
    ci = row["gain_ci"] or [float("nan")] * 2
    rates = " ".join(f"{k.removeprefix('shift_')}:{v:.2f}" for k, v in row["return_rate"].items() if v is not None)
    return f"{row['events']} events, gain {row['gain']:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}], return {rates}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-events", type=int, default=20, help="per game")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("analysis/enemy_shift.jsonl"))
    args = parser.parse_args()
    paths = args.checkpoints or default_checkpoints(args.root)
    failures = run_all(functools.partial(run, episodes=args.episodes, seed=args.seed, max_events=args.max_events),
                       paths, args.output, "checkpoint", args.workers, describe)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
