"""How long the fog hides the ball, and what still gives its position away (step 2 of docs/analysis_plan.md).

Plays every finished fog / fog_mask scratch run of each --archs on its own view, with the same games as the
frozen evaluation (env and action seeds --seed, --seed + 1, ...; stochastic policy, CPU), and reads the
clean screen at every agent step to locate the ball and both paddles:

    uv run python scripts/fog_exposure.py                 # 30 games per checkpoint, 4 CPU processes

For each checkpoint it reports
  - hidden runs: maximal stretches of consecutive agent steps where the ball is on the court but not on the
    agent's screen, split by the ball's direction on entry (incoming = towards the agent's paddle);
  - stack-blind fraction: of the hidden steps, how many had no visible ball in the current and previous
    3 frames, i.e. a 4-frame stack (cnn) holds no ball at all;
  - tracking: while the ball is hidden, how closely each paddle's height follows the ball's height
    (correlation, mean gap in pixels). The enemy paddle is visible to the agent under fog, not fog_mask.
Each output line (default evaluations/fog_exposure.jsonl) holds one checkpoint's summary, its hidden runs
and its per-step trace (positions in pixels of the 84x84 frame, null when not on screen; visible = the ball
is on the agent's screen), for later analyses.

Ball pixels are those differing from the background by more than THRESHOLD; a ball counts as visible when
any such pixel survives the agent's occlusion, so a faint sliver at the fog's edge can be counted as hidden.
"""

import argparse
import collections
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pong_rl.envs import make_env  # noqa: E402
from pong_rl.evaluate import _checkpoint_config, view_name  # noqa: E402
from pong_rl.models import RecurrentAgent, make_agent, zero_lstm_state  # noqa: E402
from pong_rl.train import FRAME_STACK  # noqa: E402
from pong_rl.wrappers import COURT_Y_MAX, COURT_Y_MIN, PONG_BG_VALUE  # noqa: E402

THRESHOLD = 20
BALL_COLUMNS = slice(12, 73)   # between the paddles (enemy x 8-10, its blur reaches 11; agent x 73-75)
ENEMY_COLUMNS = slice(8, 11)
AGENT_COLUMNS = slice(73, 76)
STACK = 4                      # frames in the cnn's stack
TRACE_KEYS = ("ball_x", "ball_y", "enemy_y", "agent_y", "visible")


def _rows_and_columns(frame: np.ndarray, columns: slice):
    court = frame[COURT_Y_MIN:COURT_Y_MAX, columns].astype(np.int16)
    rows, cols = np.nonzero(np.abs(court - PONG_BG_VALUE) > THRESHOLD)
    return rows + COURT_Y_MIN, cols + columns.start


def locate(clean: np.ndarray, seen: np.ndarray) -> dict:
    """Ball and paddle positions on the clean frame; whether the ball is on the agent's (occluded) frame."""
    rows, cols = _rows_and_columns(clean, BALL_COLUMNS)
    enemy_rows, _ = _rows_and_columns(clean, ENEMY_COLUMNS)
    agent_rows, _ = _rows_and_columns(clean, AGENT_COLUMNS)
    seen_rows, _ = _rows_and_columns(seen, BALL_COLUMNS)
    return {
        "ball_x": round(float(cols.mean()), 1) if len(cols) else None,
        "ball_y": round(float(rows.mean()), 1) if len(rows) else None,
        "enemy_y": round(float(enemy_rows.mean()), 1) if len(enemy_rows) else None,
        "agent_y": round(float(agent_rows.mean()), 1) if len(agent_rows) else None,
        "visible": bool(len(seen_rows)) and bool(len(rows)),
    }


def hidden_runs(steps: list[dict]) -> list[dict]:
    """Maximal stretches of steps with the ball on the court but not on the agent's screen."""
    runs, start = [], None
    for t, step in enumerate(steps + [None]):
        hidden = step is not None and step["ball_x"] is not None and not step["visible"]
        if hidden and start is None:
            start = t
        elif not hidden and start is not None:
            before = steps[start - 1]["ball_x"] if start > 0 else None
            first = steps[start]["ball_x"]
            direction = None if before is None else ("incoming" if first > before else "outgoing")
            end = "episode_end" if step is None else ("emerged" if step["ball_x"] is not None else "vanished")
            runs.append({"start": start, "length": t - start, "direction": direction, "end": end})
            start = None
    return runs


def stack_blind(steps: list[dict]) -> int:
    """Hidden steps at which none of the last STACK frames (this one included) showed the ball."""
    window = collections.deque(maxlen=STACK)
    blind = 0
    for step in steps:
        window.append(step["visible"])
        if step["ball_x"] is not None and not step["visible"] and not any(window):
            blind += 1
    return blind


def tracking(steps: list[dict], paddle: str) -> dict:
    pairs = [(s["ball_y"], s[paddle]) for s in steps
             if s["ball_x"] is not None and not s["visible"] and s[paddle] is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "corr": None, "mean_abs_gap": None, "ball_y_std": None}
    ball, pad = np.array(pairs).T
    corr = float(np.corrcoef(ball, pad)[0, 1]) if ball.std() > 0 and pad.std() > 0 else None
    return {"n": len(pairs), "corr": None if corr is None else round(corr, 3),
            "mean_abs_gap": round(float(np.abs(ball - pad).mean()), 2), "ball_y_std": round(float(ball.std()), 2)}


def describe(lengths: list[int]) -> dict:
    if not lengths:
        return {"count": 0}
    ordered = sorted(lengths)
    return {"count": len(ordered), "mean": round(statistics.mean(ordered), 2), "median": statistics.median(ordered),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], "max": ordered[-1]}


def summarize(games: list[dict]) -> dict:
    steps = [step for game in games for step in game["steps"]]
    runs = [run for game in games for run in game["runs"]]
    present = sum(s["ball_x"] is not None for s in steps)
    hidden = sum(s["ball_x"] is not None and not s["visible"] for s in steps)
    blind = sum(game["stack_blind"] for game in games)
    return {
        "games": len(games),
        "mean_return": round(statistics.mean(g["return"] for g in games), 2),
        "steps": len(steps),
        "ball_on_court_steps": present,
        "hidden_steps": hidden,
        "hidden_fraction": round(hidden / present, 3) if present else None,
        "stack_blind_fraction": round(blind / hidden, 3) if hidden else None,
        "runs": {direction: describe([r["length"] for r in runs if r["direction"] == direction])
                 for direction in ("incoming", "outgoing")},
        "run_ends": dict(collections.Counter(r["end"] for r in runs)),
        "enemy_tracking": tracking(steps, "enemy_y"),
        "agent_tracking": tracking(steps, "agent_y"),
    }


@torch.inference_mode()
def play(path: str, episodes: int, seed: int) -> dict:
    """The evaluation's games (evaluate.py's loop, stochastic policy), recording positions every step."""
    torch.set_num_threads(1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = _checkpoint_config(checkpoint)
    arch = config["arch"]
    envs = gym.vector.SyncVectorEnv(
        [make_env(config["env_id"], 0, False, "fog-exposure", use_fog=config["use_fog"],
                  mask_enemy=config["mask_enemy"], frame_stack=FRAME_STACK[arch], tap_clean_frame=True)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    try:
        agent = make_agent(arch, envs)
        agent.load_state_dict(checkpoint["model_state_dict"], strict=True)
        agent.eval()
        games = []
        for index in range(episodes):
            episode_seed = seed + index
            obs, _ = envs.reset(seed=episode_seed)
            generator = torch.Generator(device="cpu").manual_seed(episode_seed)
            state = zero_lstm_state(1, torch.device("cpu"))
            steps, episode_return = [], 0.0
            while True:
                steps.append(locate(envs.envs[0].get_wrapper_attr("clean_frame"), obs[0, -1]))
                tensor = torch.as_tensor(obs, dtype=torch.float32)
                if isinstance(agent, RecurrentAgent):
                    hidden, state = agent.get_states(tensor, state, torch.zeros(1))
                else:
                    hidden = agent.get_hidden(tensor)
                probabilities = torch.softmax(agent.actor(hidden), dim=-1)
                action = torch.multinomial(probabilities, 1, generator=generator).flatten().numpy()
                obs, rewards, terminated, truncated, _ = envs.step(action)
                episode_return += float(rewards[0])
                if terminated[0] or truncated[0]:
                    break
            games.append({"env_seed": episode_seed, "return": episode_return, "steps": steps,
                          "runs": hidden_runs(steps), "stack_blind": stack_blind(steps)})
    finally:
        envs.close()
    return {"arch": arch, "train_view": view_name(config["use_fog"], config["mask_enemy"]),
            "training_seed": config["seed"], "games": games}


def discover(root: Path, archs, views) -> list[Path]:
    return sorted(p for arch in archs for view in views for p in root.glob(f"scratch/{arch}/{view}/seed_*/final.pt")
                  if not any(".aborted_" in part for part in p.parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--archs", nargs="+", default=["cnn", "lstm"])
    parser.add_argument("--views", nargs="+", choices=("fog", "fog_mask"), default=["fog", "fog_mask"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("evaluations/fog_exposure.jsonl"))
    args = parser.parse_args()
    if args.episodes <= 0 or args.seed < 0 or args.workers <= 0:
        parser.error("--episodes and --workers must be positive, --seed nonnegative")
    paths = discover(args.root, args.archs, args.views)
    if not paths:
        parser.error(f"no finished fog / fog_mask scratch runs under {args.root}")
    print(f"{len(paths)} checkpoints x {args.episodes} games, {args.workers} workers", flush=True)
    started, rows = time.time(), []
    with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(play, str(p), args.episodes, args.seed): p for p in paths}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            result = future.result()
            if result["train_view"] != path.parent.parent.name:
                sys.exit(f"{path}: trained on {result['train_view']}, folder says {path.parent.parent.name}")
            summary = summarize(result["games"])
            for game in result["games"]:  # columns, not one object per step, to keep the file small
                steps = game.pop("steps")
                game["trace"] = {key: [s[key] for s in steps] for key in TRACE_KEYS}
            rows.append({"checkpoint": str(path), "run": str(path.parent.relative_to(args.root)), **result,
                         "summary": summary, "protocol": {"episodes": args.episodes, "seed": args.seed,
                                                          "policy": "stochastic", "threshold": THRESHOLD}})
            runs = summary["runs"]
            print(f"{rows[-1]['run']}: return {summary['mean_return']:+.1f}, hidden {summary['hidden_fraction']:.0%}"
                  f" of ball steps, incoming run median {runs['incoming'].get('median')} steps, stack-blind "
                  f"{summary['stack_blind_fraction']:.0%}, enemy-ball corr {summary['enemy_tracking']['corr']} "
                  f"({time.time() - started:.0f} s)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda r: r["run"]):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
