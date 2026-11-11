"""Where the attention-LSTM looks while a ball comes towards it (data for the paper's attention figure).

    uv run python scripts/attention_maps.py summary --rollouts analysis/rollouts --out analysis/attention_maps.json
    uv run python scripts/attention_maps.py strip --checkpoint <final.pt> --out analysis/attention_strip.npz

summary  For every attn_lstm recording on its own view (scripts/record_rollouts.py: the 30 frozen-evaluation
         games), each flight towards the agent is aligned on the step the ball enters the agent's half (x >= 42,
         where the fog starts), from STEPS_BEFORE steps before to STEPS_AFTER after. At each step it measures the
         attention weight (mean over heads) on three targets, located on the unoccluded frame:
           ball         the 3x3 attention cells around the ball (as in scripts/attention_block.py), also while
                        the ball is hidden by the fog
           own_paddle   the agent's paddle column (6), the paddle's row and its neighbours
           enemy_paddle the enemy paddle column (0), likewise
         and, as a reference, the share of the 49 cells each target covers (what uniform attention would give).
         Only flights the agent returns are kept, so every step is a moment of a successful play.
         Output: one JSON object, per recording the mean (and flight count) per target and relative step.
strip    Plays one frozen-evaluation game (env seed --seed, exactly as pong_rl.evaluate) with the checkpoint and
         saves the first returned flight after --after steps: per step the frame the model saw, the
         unoccluded frame, both heads' attention (2, 49) and the ball's position.
Cells: the 84x84 frame is read as a 7x7 grid of 12-pixel cells (row = y // 12, column = x // 12), the convention
of scripts/attention_block.py; each conv cell actually sees an overlapping 36x36 patch.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pong_rl.analysis.rollouts import WINDOW_END, WINDOW_START, load

GRID, CELL = 7, 12
STEPS_BEFORE, STEPS_AFTER = 12, 14
RETURN_WITHIN = 30  # steps after fog entry in which the ball must be seen moving back left to count as returned
TARGETS = ("ball", "own_paddle", "enemy_paddle")


def cell_row(y: float) -> int:
    return int(np.clip(y // CELL, 0, GRID - 1))


def target_mask(target: str, ball_x: float, ball_y: float, agent_y: float, enemy_y: float) -> np.ndarray | None:
    """bool (49,) cells of the target, or None if its position is unknown at this step."""
    mask = np.zeros((GRID, GRID), bool)
    if target == "ball":
        if not (ball_x == ball_x and ball_y == ball_y):
            return None
        row, col = cell_row(ball_y), int(np.clip(ball_x // CELL, 0, GRID - 1))
        mask[max(0, row - 1):row + 2, max(0, col - 1):col + 2] = True
    else:
        y, col = (agent_y, GRID - 1) if target == "own_paddle" else (enemy_y, 0)
        if y != y:
            return None
        row = cell_row(y)
        mask[max(0, row - 1):row + 2, col] = True
    return mask.reshape(-1)


def returned_flights(arrays: dict) -> list[int]:
    """Row index of each fog entry (ball first at x >= WINDOW_START, moving right) whose ball is then seen
    moving left again within RETURN_WITHIN steps, in the same game (the agent hit it back)."""
    x, game = arrays["ball_x"], arrays["game"]
    entries = []
    for i in range(1, len(x)):
        if not (x[i] == x[i] and x[i - 1] == x[i - 1]) or game[i] != game[i - 1]:
            continue
        if x[i - 1] < WINDOW_START <= x[i] < WINDOW_END and x[i] > x[i - 1]:
            end = min(len(x), i + RETURN_WITHIN)
            for j in range(i + 1, end - 1):
                if game[j + 1] != game[i]:
                    break
                if x[j] == x[j] and x[j + 1] == x[j + 1] and x[j + 1] < x[j] and x[j] >= WINDOW_START:
                    entries.append(i)
                    break
    return entries


def summarize_recording(path: Path) -> dict:
    arrays, meta = load(path)
    attention = arrays["attention"].astype(np.float32).mean(axis=1)  # (steps, 49), mean over heads
    steps = range(-STEPS_BEFORE, STEPS_AFTER + 1)
    sums = {t: np.zeros(len(steps)) for t in TARGETS}
    counts = {t: np.zeros(len(steps), int) for t in TARGETS}
    covered = {t: [] for t in TARGETS}
    entries = returned_flights(arrays)
    for entry in entries:
        for k, offset in enumerate(steps):
            i = entry + offset
            if not (0 <= i < len(attention)) or arrays["game"][i] != arrays["game"][entry]:
                continue
            for target in TARGETS:
                mask = target_mask(target, arrays["ball_x"][i], arrays["ball_y"][i], arrays["agent_y"][i],
                                   arrays["enemy_y"][i])
                if mask is None:
                    continue
                sums[target][k] += attention[i][mask].sum()
                counts[target][k] += 1
                covered[target].append(mask.sum() / mask.size)
    return {"recording": Path(path).name, "run": meta["run"], "kind": meta["kind"], "view": meta["view"],
            "train_view": meta["train_view"], "flights": len(entries), "steps": list(steps),
            "mean": {t: (sums[t] / np.maximum(counts[t], 1)).round(4).tolist() for t in TARGETS},
            "count": {t: counts[t].tolist() for t in TARGETS},
            "uniform": {t: round(float(np.mean(covered[t])), 4) if covered[t] else None for t in TARGETS}}


def summary(rollouts: Path, out: Path):
    rows = []
    for path in sorted(rollouts.glob("*__attn_lstm__*__final.npz")):
        if path.name.endswith(".random.npz"):
            continue
        _, meta = load(path)
        if meta["view"] != meta["train_view"]:
            continue
        row = summarize_recording(path)
        rows.append(row)
        ball = row["mean"]["ball"]
        print(f"{row['run']}: {row['flights']} flights; on the ball 8 steps before / at / 8 after fog entry "
              f"{ball[STEPS_BEFORE - 8]:.2f} / {ball[STEPS_BEFORE]:.2f} / {ball[STEPS_BEFORE + 8]:.2f} "
              f"(uniform {row['uniform']['ball']:.2f})", flush=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} recordings to {out}")


def strip(checkpoint: Path, out: Path, seed: int, after: int):
    from pong_rl.analysis.player import Player  # imports torch and the environment; summary needs neither
    player = Player(checkpoint)
    player.reset(seed)
    frames, clean, attention, ball, entry = [], [], [], [], None
    while not player.done:
        seen, unoccluded = player.obs[-1].copy(), player.clean_frame()
        step = player.step()
        frames.append(seen)
        clean.append(unoccluded)
        attention.append(step.internals["attention"])
        ball.append((step.scene.get("ball_x", np.nan), step.scene.get("ball_y", np.nan)))
        t = len(frames) - 1
        x = np.array(ball, dtype=float)[:, 0]
        if entry is None and t >= after and t >= 1 and x[t - 1] < WINDOW_START <= x[t] and x[t] > x[t - 1]:
            entry = t
        if entry is not None:
            xs = x[entry:t + 1]
            returned = any(b < a for a, b in zip(xs[:-1], xs[1:]) if a == a and b == b)
            if returned and t >= entry + STEPS_AFTER:
                break
            if (returned and t < entry + STEPS_AFTER) or t >= entry + RETURN_WITHIN:
                entry = None  # too short a flight to show, or missed: look for the next one
    player.close()
    if entry is None:
        raise SystemExit("no returned flight found; try another --seed or --after")
    rows = slice(entry - STEPS_BEFORE, entry + STEPS_AFTER + 1)
    np.savez_compressed(out, seen=np.array(frames[rows]), clean=np.array(clean[rows]),
                        attention=np.array(attention[rows], dtype=np.float32), ball=np.array(ball[rows]),
                        steps=np.arange(-STEPS_BEFORE, STEPS_AFTER + 1), checkpoint=str(checkpoint),
                        seed=seed, entry=entry)
    print(f"saved steps {entry - STEPS_BEFORE}..{entry + STEPS_AFTER} of game seed {seed} to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--rollouts", type=Path, default=Path("analysis/rollouts"))
    s.add_argument("--out", type=Path, default=Path("analysis/attention_maps.json"))
    t = sub.add_parser("strip")
    t.add_argument("--checkpoint", type=Path, required=True)
    t.add_argument("--out", type=Path, default=Path("analysis/attention_strip.npz"))
    t.add_argument("--seed", type=int, default=10000)
    t.add_argument("--after", type=int, default=200, help="skip the game's first steps (serve, warm-up)")
    args = parser.parse_args()
    if args.command == "summary":
        summary(args.rollouts, args.out)
    else:
        strip(args.checkpoint, args.out, args.seed, args.after)


if __name__ == "__main__":
    main()
