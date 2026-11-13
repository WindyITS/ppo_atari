"""Does an agent need to move while the ball is still visible? A causal test of early commitment (E17).

    uv run python scripts/commit_hold.py --workers 3

E8 found that the memoryless CNN wins fog + hidden paddle by moving its paddle early, while the ball is still
visible, and barely moving inside the fog; memory agents can also move late. That is a correlation. Here the
early movement is taken away. Each checkpoint plays games on its own view (Player: the frozen evaluation's env
and action seeds). Every time the ball starts towards the agent, the game is saved and the flight is replayed in
three branches with the same action seed:
    control       no intervention
    hold_visible  the agent's action is replaced by NOOP from the save until the ball reaches the fog's edge
                  (x >= 42; the same point on unfogged views), so its paddle stays where it was while the ball
                  is visible on its way; from then on the agent plays freely (the sampling generator still draws
                  at every step, so the free part starts from the same random stream)
    hold_stack    the same, held for STACK more steps after the ball reaches the fog's edge: on fog + hidden
                  paddle the CNN's 4-frame stack then no longer shows the ball when it may move again. (The
                  paddle crosses the court in about 6 steps, so a CNN held only while the ball is visible can
                  still read the ball's path from its last frames and get there in time.)
Per branch: whether the ball is returned (as in scripts/memory_interventions.py) and the distance between the
agent's paddle and the ball when the ball reaches the paddle line. Prediction: an agent that carries the ball
through the fog only in its paddle position (early commitment) loses the ball after hold_visible on fog + hidden
paddle, since it can no longer see the ball when it is allowed to move; an agent with a memory of the ball can
still move late and recovers; on views where the ball stays visible every agent should recover.
Output: JSON lines (default analysis/commit_hold.jsonl), one row per checkpoint; reruns skip done rows.
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

from pong_rl.analysis.batch import run_all  # noqa: E402
from pong_rl.analysis.player import Player  # noqa: E402
from pong_rl.analysis.rollouts import WINDOW_START  # noqa: E402

PADDLE_X, RETURN_X, MAX_STEPS = 70.0, 40.0, 150
NOOP = 0
STACK = 4
BRANCHES = ("control", "hold_visible", "hold_stack")


def play_branch(player: Player, branch: str) -> dict:
    """From the save: hold the paddle (hold_visible) until the ball reaches the fog's edge; play the point out."""
    previous_x, reached, gap, held = None, False, None, 0
    holding, entered = branch != "control", None
    extra = STACK if branch == "hold_stack" else 0
    for k in range(MAX_STEPS):
        if player.done:
            return {"outcome": "undecided", "gap": gap, "held_steps": held}
        where = player.scene_now()
        x = where["ball_x"]
        if entered is None and x == x and x >= WINDOW_START:
            entered = k
        if holding and entered is not None and k >= entered + extra:
            holding = False
        if not reached and x == x and x >= PADDLE_X:
            reached = True
            if where["agent_y"] == where["agent_y"]:
                gap = float(abs(where["agent_y"] - where["ball_y"]))
        if reached and x == x and previous_x is not None and x < previous_x and x < RETURN_X:
            return {"outcome": "returned", "gap": gap, "held_steps": held}
        step = player.step(action=NOOP if holding else None)
        held += int(holding)
        if step.reward != 0:
            return {"outcome": "returned" if step.reward > 0 else "missed", "gap": gap, "held_steps": held}
        previous_x = x if x == x else previous_x
    return {"outcome": "undecided", "gap": gap, "held_steps": held}


def run(path: Path, episodes: int, seed: int, max_events: int) -> dict:
    torch.set_num_threads(1)
    player = Player(path)
    events = []
    try:
        for game in range(episodes):
            player.reset(seed + game)
            previous_x, armed, count = None, True, 0
            while not player.done:
                where = player.scene_now()
                x = where["ball_x"]
                if x != x or (previous_x is not None and x < previous_x):
                    armed = True  # ball gone or heading to the enemy: the next flight our way may trigger
                incoming = x == x and previous_x is not None and x > previous_x and x < WINDOW_START
                if armed and incoming and count < max_events:
                    armed, count = False, count + 1
                    saved = player.save()
                    branch_seed = zlib.crc32(f"{seed + game}:{player.t}".encode())
                    outcomes = {}
                    for name in BRANCHES:
                        player.load(saved, reseed=branch_seed)
                        outcomes[name] = play_branch(player, name)
                    player.load(saved)  # back to the original game, unperturbed
                    events.append({"game": game, "t": player.t, "ball_x": x, "outcomes": outcomes})
                player.step()
                previous_x = x if x == x else previous_x
    finally:
        player.close()
    return summarize(path, player, events, episodes, seed, max_events)


def summarize(path, player, events, episodes, seed, max_events) -> dict:
    per_branch = {}
    for name in BRANCHES:
        outcomes = [e["outcomes"][name]["outcome"] for e in events]
        decided = [o for o in outcomes if o != "undecided"]
        gaps = [e["outcomes"][name]["gap"] for e in events if e["outcomes"][name]["gap"] is not None]
        per_branch[name] = {"return_rate": sum(o == "returned" for o in decided) / len(decided) if decided else None,
                            "median_gap": float(np.median(gaps)) if gaps else None,
                            "held_steps": float(np.mean([e["outcomes"][name]["held_steps"] for e in events]))
                            if events else None}
    change = {}
    for name in BRANCHES[1:]:
        paired = [(e["outcomes"][name]["outcome"] == "returned") - (e["outcomes"]["control"]["outcome"] == "returned")
                  for e in events if "undecided" not in (e["outcomes"][name]["outcome"],
                                                         e["outcomes"]["control"]["outcome"])]
        change[name] = None
        if len(paired) >= 5:
            d = np.asarray(paired, float)
            boots = d[np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))].mean(1)
            change[name] = [float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return {"checkpoint": str(path), "arch": player.arch, "train_view": player.train_view, "view": player.view,
            "kind": player.checkpoint.get("kind", "scratch"), "run": player.checkpoint.get("run"),
            "episodes": episodes, "seed": seed, "max_events_per_game": max_events, "events": len(events),
            "branches": per_branch, "return_change": change}


def default_checkpoints(root: Path) -> list[Path]:
    found = []
    for arch in ("cnn", "lstm", "attn_lstm", "attn_lstm_wide"):
        for view in ("clean", "fog", "fog_mask"):
            found += root.glob(f"scratch/{arch}/{view}/seed_[123]/final.pt")
        for view in ("fog", "fog_mask"):
            if arch.startswith("attn"):
                found += root.glob(f"finetune/{arch}/clean_10.0M_to_{view}/seed_[123]/final.pt")
    return sorted(found)


def describe(row: dict) -> str:
    b = row["branches"]
    return (f"{row['events']} events; returned control {b['control']['return_rate']:.2f}, hold_visible "
            f"{b['hold_visible']['return_rate']:.2f}, hold_stack {b['hold_stack']['return_rate']:.2f} "
            f"(held {b['hold_stack']['held_steps']:.1f} steps)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-events", type=int, default=20, help="per game")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("analysis/commit_hold.jsonl"))
    args = parser.parse_args()
    paths = args.checkpoints or default_checkpoints(args.root)
    failures = run_all(functools.partial(run, episodes=args.episodes, seed=args.seed, max_events=args.max_events),
                       paths, args.output, "checkpoint", args.workers, describe)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
