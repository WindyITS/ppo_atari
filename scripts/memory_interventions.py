"""Causal memory test (E4 in docs/research_plan.md): does wiping what an agent remembers stop it returning the ball?

    uv run python scripts/memory_interventions.py --workers 10        # default checkpoint set, see below
    uv run python scripts/memory_interventions.py --checkpoints checkpoints/scratch/lstm/fog_mask/seed_1/final.pt

Each checkpoint plays the frozen evaluation's games on its own view (Player: same env and action seeds). Every
time an incoming ball (moving towards the agent) first reaches x >= TRIGGER_X, the game is saved and that
point is replayed under several branches, all starting from the save:
    control      no intervention
    control_b    no intervention, different action seed (sampling noise floor)
    wipe_before  memory wiped at the save, while the ball is still visible in the enemy half
    wipe_entry   memory wiped when the ball first reaches the midline x >= 42 (where the fog starts)
    wipe_mid     memory wiped MID_DELAY steps after that
All branches except control_b share one action seed. A recurrent agent's wipe zeroes its LSTM state once; the
cnn's analogue replaces every frame of its stack by the newest one for STACK consecutive steps (no motion, no
past). A branch ends when the ball comes back across x < RETURN_X moving left (returned), when the agent loses
the point (missed), or after MAX_STEPS (undecided). It also records the gap between the paddle and the ball when
the ball reaches the paddle (x >= 70). After the branches the original game continues unperturbed.
Output: one JSON per checkpoint under --out-dir with per-branch totals and per-event outcomes.
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pong_rl.analysis.player import Player, stack_wipe  # noqa: E402

TRIGGER_X = 30.0
ENTRY_X = 42.0
MID_DELAY = 7
RETURN_X = 40.0
PADDLE_X = 70.0
MAX_STEPS = 150
STACK = 4
BRANCHES = ("control", "control_b", "wipe_before", "wipe_entry", "wipe_mid")


def play_branch(player: Player, branch: str) -> dict:
    """Play from the current (just loaded) moment until the ball is returned or the point is lost."""
    wipe_at = 0 if branch == "wipe_before" else None
    previous_x = None
    entered, reached, gap = None, False, None
    cnn_wipe_left = 0
    for k in range(MAX_STEPS):
        if player.done:
            return {"outcome": "undecided", "steps": k, "gap": gap}
        where = player.scene_now()
        x = where["ball_x"]
        if entered is None and x == x and x >= ENTRY_X and (previous_x is None or x > previous_x):
            entered = k
            if branch == "wipe_entry":
                wipe_at = k
            elif branch == "wipe_mid":
                wipe_at = k + MID_DELAY
        if not reached and x == x and x >= PADDLE_X:
            reached, gap = True, abs(where["agent_y"] - where["ball_y"]) if where["agent_y"] == where["agent_y"] else None
        if reached and x == x and previous_x is not None and x < previous_x and x < RETURN_X:
            return {"outcome": "returned", "steps": k, "gap": gap}
        wipe = wipe_at is not None and k == wipe_at
        if player.recurrent:
            step = player.step(wipe=wipe)
        else:
            if wipe:
                cnn_wipe_left = STACK
            step = player.step(transform=stack_wipe if cnn_wipe_left > 0 else None)
            cnn_wipe_left = max(0, cnn_wipe_left - 1)
        if step.reward < 0:
            return {"outcome": "missed", "steps": k + 1, "gap": gap}
        if step.reward > 0:  # the agent scored, so it returned the ball
            return {"outcome": "returned", "steps": k + 1, "gap": gap}
        previous_x = x if x == x else previous_x
    return {"outcome": "undecided", "steps": MAX_STEPS, "gap": gap}


def run(path: str, episodes: int, seed: int, max_events: int) -> dict:
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
                incoming = x == x and previous_x is not None and x > previous_x
                if x != x or (previous_x is not None and x == x and x < previous_x):
                    armed = True  # a new incoming ball may trigger once it heads our way
                if armed and incoming and x >= TRIGGER_X and count < max_events:
                    armed, count = False, count + 1
                    saved = player.save()
                    branch_seed = zlib.crc32(f"{seed + game}:{player.t}".encode())
                    outcomes = {}
                    for branch in BRANCHES:
                        player.load(saved, reseed=branch_seed + (1 if branch == "control_b" else 0))
                        outcomes[branch] = play_branch(player, branch)
                    player.load(saved)  # back to the original game, unperturbed
                    events.append({"game": game, "t": player.t, "ball_y": where["ball_y"],
                                   "outcomes": {b: o["outcome"] for b, o in outcomes.items()},
                                   "gaps": {b: o["gap"] for b, o in outcomes.items()}})
                player.step()
                previous_x = x if x == x else previous_x
    finally:
        player.close()
    totals = {}
    for branch in BRANCHES:
        outcomes = [e["outcomes"][branch] for e in events]
        gaps = [e["gaps"][branch] for e in events if e["gaps"][branch] is not None]
        decided = sum(o != "undecided" for o in outcomes)
        returned = sum(o == "returned" for o in outcomes)
        totals[branch] = {"events": len(outcomes), "returned": returned, "missed": sum(o == "missed" for o in outcomes),
                          "undecided": len(outcomes) - decided,
                          "return_rate": returned / decided if decided else None,
                          "median_gap": float(np.median(gaps)) if gaps else None}
    return {"checkpoint": path, "arch": player.arch, "train_view": player.train_view, "view": player.view,
            "kind": player.checkpoint.get("kind", "scratch"), "run": player.checkpoint.get("run"),
            "episodes": episodes, "seed": seed, "max_events_per_game": max_events,
            "params": {"trigger_x": TRIGGER_X, "entry_x": ENTRY_X, "mid_delay": MID_DELAY, "return_x": RETURN_X,
                       "paddle_x": PADDLE_X, "max_steps": MAX_STEPS, "cnn_stack": STACK},
            "totals": totals, "events": events}


def default_checkpoints(root: Path) -> list[Path]:
    found = []
    for arch in ("cnn", "lstm"):
        found += root.glob(f"scratch/{arch}/*/seed_*/final.pt")
        found += root.glob(f"finetune/{arch}/clean_10.0M_to_fog_mask/seed_*/final.pt")
    found += root.glob("scratch/attn_lstm/clean/seed_*/final.pt")
    return sorted(p for p in found if ".aborted_" not in str(p))


def work(job):
    path, out, episodes, seed, max_events = job
    started = time.time()
    result = run(path, episodes, seed, max_events)
    result["seconds"] = round(time.time() - started, 1)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result) + "\n")
    os.replace(temporary, out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-events", type=int, default=20, help="saved points per game")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/interventions"))
    args = parser.parse_args()
    paths = args.checkpoints or default_checkpoints(args.root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in paths:
        name = "__".join(path.parent.relative_to(args.root).parts) + "__" + path.stem + ".json"
        out = args.out_dir / name
        if not out.is_file():
            jobs.append((str(path), out, args.episodes, args.seed, args.max_events))
    print(f"{len(paths)} checkpoints, {len(jobs)} to run, {args.workers} workers", flush=True)
    failures = 0
    with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(work, job): job for job in jobs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            job = futures[future]
            try:
                result = future.result()
                rates = " ".join(f"{b} {t['return_rate']:.2f}" if t["return_rate"] is not None else f"{b} -"
                                 for b, t in result["totals"].items())
                print(f"[{done}/{len(jobs)}] {job[0]}: {result['totals']['control']['events']} events; {rates} "
                      f"({result['seconds']:.0f} s)", flush=True)
            except Exception as exc:
                failures += 1
                print(f"[{done}/{len(jobs)}] {job[0]}: FAILED {exc!r}", file=sys.stderr, flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
