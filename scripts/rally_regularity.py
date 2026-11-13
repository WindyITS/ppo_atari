"""How regular are the rallies? Arrival heights of the ball and the agent's paddle (E17b, docs/research_plan.md).

    uv run python scripts/rally_regularity.py            # on the training host (reads analysis/rollouts/)

For every recording of a final model on its own view (scripts/record_rollouts.py: the 30 frozen-evaluation
games), each incoming pass (pong_rl.analysis.rollouts.incoming_passes) is reduced to its last recorded step,
where the ball reaches the agent's side. Per recording: the number of passes, the standard deviation of the
ball's height and of the agent's paddle height there, the share of balls arriving within 4 px of the median
arrival height, and the correlation between paddle and ball height. A policy that waits at the usual height
returns most balls when this share is high. Output: analysis/rally_regularity.json (one row per recording).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pong_rl.analysis.rollouts import incoming_passes, load

NEAR = 4.0  # px


def summarize(path: Path) -> dict | None:
    arrays, meta = load(path)
    if meta["view"] != meta["train_view"]:
        return None
    ball, paddle = [], []
    for p in incoming_passes(arrays):
        last = p.rows[-1]
        if np.isfinite(arrays["ball_y"][last]) and np.isfinite(arrays["agent_y"][last]):
            ball.append(float(arrays["ball_y"][last]))
            paddle.append(float(arrays["agent_y"][last]))
    if len(ball) < 10:
        return None
    ball, paddle = np.array(ball), np.array(paddle)
    return {"recording": path.name, "run": meta["run"], "arch": meta.get("arch"), "kind": meta["kind"],
            "view": meta["view"], "passes": int(len(ball)), "ball_sd": float(ball.std()),
            "paddle_sd": float(paddle.std()), "near_median": float(np.mean(np.abs(ball - np.median(ball)) < NEAR)),
            "paddle_ball_corr": float(np.corrcoef(paddle, ball)[0, 1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, default=Path("analysis/rollouts"))
    parser.add_argument("--output", type=Path, default=Path("analysis/rally_regularity.json"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.rollouts.glob("*__final.npz")):
        if path.name.endswith((".random.npz", ".tmp.npz")):
            continue
        row = summarize(path)
        if row:
            rows.append(row)
            print(f"{row['run']}: ball sd {row['ball_sd']:.1f}, paddle sd {row['paddle_sd']:.1f}, "
                  f"within {NEAR:.0f} px of median {row['near_median']:.2f}", flush=True)
    args.output.write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} recordings to {args.output}")


if __name__ == "__main__":
    main()
