"""Scripted reference players for the fog views (E11, docs/research_plan.md).

    uv run python scripts/scripted_baselines.py

Hand-written policies that play the same games as the frozen evaluation (env seeds --seed, --seed + 1, ...;
same environment and frame skip as training), each told what it may see:
    copy_enemy      moves its paddle to the enemy paddle's height as shown on the fog view (the shortcut alone;
                    needs the enemy paddle, so fog only)
    last_seen       goes to the ball's last height seen on the fog view (memory of a position, no motion)
    tracker_fog     extrapolates the ball's straight-line path, with wall bounces, to the paddle line, from a
                    least-squares fit to its positions seen on the fog view since the last wall bounce (up to
                    8), and keeps that target while the ball is hidden (ideal short memory of the trajectory)
    tracker_clean   the same with the ball always visible (reference without fog)
Positions are read from the frames the policy may see (scene.locate on the seen frame; the agent's own paddle
from the unoccluded frame, as it is never fogged). Paddle control: the paddle has momentum, so the policy
steers by where its paddle will be LOOKAHEAD steps on at its current speed: up (action 2) when the target is
more than DEADBAND px above that, down (3) when below, otherwise no-op. One setting for every policy, LOOKAHEAD 1
and DEADBAND 3: the best tracker_fog return rate among 4 settings on 2 games with seeds 50000+ (kept apart from
the evaluation seeds); with a 5 px dead band the paddle can stop more than half its height off target. Output: JSON lines (default
analysis/scripted_baselines.jsonl), one row per policy and view, with the game score and the share of incoming
balls returned (returns / (returns + points lost)): the CPU opponent wins long rallies, so a player can return
most balls and still lose every point.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pong_rl.analysis import scene
from pong_rl.analysis.rollouts import COURT_BOTTOM, COURT_TOP
from pong_rl.envs import make_env

UP, DOWN, NOOP = 2, 3, 0
DEADBAND = 3.0
LOOKAHEAD = 1.0
PADDLE_X = 73.0
POLICIES = {"copy_enemy": ("fog",), "last_seen": ("fog", "fog_mask"), "tracker_fog": ("fog", "fog_mask"),
            "tracker_clean": ("clean",)}
VIEWS = {"clean": (False, False), "fog": (True, False), "fog_mask": (True, True)}


def bounce(y0: float, vy: float, steps: float) -> float:
    span = COURT_BOTTOM - COURT_TOP
    folded = np.mod((y0 - COURT_TOP) + vy * steps, 2 * span)
    return float(COURT_TOP + (2 * span - folded if folded > span else folded))


class Policy:
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.target, self.last, self.t, self.paddle = (COURT_TOP + COURT_BOTTOM) / 2, None, 0, None
        self.flight = []  # sightings (t, x, y) of the current incoming flight

    def segment(self) -> list:
        """The sightings since the last wall bounce (vertical direction change), for a straight-line fit."""
        points = self.flight[-8:]
        for i in range(len(points) - 2, 0, -1):
            if (points[i][2] - points[i - 1][2]) * (points[i + 1][2] - points[i][2]) < 0:
                return points[i:]
        return points

    def act(self, seen: dict, agent_y: float) -> int:
        ball_seen = seen["ball_x"] == seen["ball_x"]
        self.t += 1
        if self.name == "copy_enemy":
            if seen["enemy_y"] == seen["enemy_y"]:
                self.target = seen["enemy_y"]
        elif self.name == "last_seen":
            if ball_seen:
                self.target = seen["ball_y"]
        else:  # trackers: extrapolate the path of an incoming ball to the paddle line
            if ball_seen:
                x, y = seen["ball_x"], seen["ball_y"]
                if self.last is not None and x < self.last[0]:  # going away: forget the flight, re-centre
                    self.flight, self.target = [], (COURT_TOP + COURT_BOTTOM) / 2
                self.flight.append((self.t, x, y))
                self.last = (x, y, self.t)
                points = self.segment()
                if len(points) >= 2 and points[-1][1] > points[0][1]:
                    t, xs, ys = (np.array(v, float) for v in zip(*points))
                    vx, vy = np.polyfit(t, xs, 1)[0], np.polyfit(t, ys, 1)[0]
                    if vx > 0:
                        self.target = bounce(y, vy, (PADDLE_X - x) / vx)
            elif self.name == "tracker_clean":
                self.flight, self.last = [], None
        if agent_y != agent_y:
            return NOOP
        # the paddle has momentum: steer by where it will be LOOKAHEAD steps on at its current speed
        speed = 0.0 if self.paddle is None else agent_y - self.paddle
        self.paddle = agent_y
        ahead = agent_y + LOOKAHEAD * speed
        if self.target < ahead - DEADBAND:
            return UP
        if self.target > ahead + DEADBAND:
            return DOWN
        return NOOP


def play(name: str, view: str, episodes: int, seed: int) -> dict:
    use_fog, mask_enemy = VIEWS[view]
    env = make_env("ALE/Pong-v5", 0, False, "baseline", use_fog=use_fog, mask_enemy=mask_enemy, frame_stack=1,
                   tap_clean_frame=True)()
    policy = Policy(name)
    returns, returned, missed = [], 0, 0
    try:
        for game in range(episodes):
            obs, _ = env.reset(seed=seed + game)
            policy.reset()
            total, done, previous_x = 0.0, False, None
            while not done:
                clean = np.array(env.get_wrapper_attr("clean_frame"))
                x = scene.locate(clean)["ball_x"]
                # a return: the ball, near the agent's paddle and moving right, starts moving left
                if x == x and previous_x is not None and previous_x >= 66 and x < previous_x:
                    returned += 1
                previous_x = x if x == x else None
                seen = scene.locate(np.asarray(obs)[-1])  # positions on the frame the policy may see
                agent_y = scene.locate(clean)["agent_y"]
                obs, reward, terminated, truncated, _ = env.step(policy.act(seen, agent_y))
                total += float(reward)
                missed += int(reward < 0)
                done = terminated or truncated
            returns.append(total)
    finally:
        env.close()
    return {"policy": name, "view": view, "episodes": episodes, "seed": seed,
            "mean_return": float(np.mean(returns)), "win_rate": float(np.mean(np.asarray(returns) > 0)),
            "balls_returned": returned, "balls_missed": missed,
            "return_rate": returned / (returned + missed) if returned + missed else None,
            "returns": returns}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("analysis/scripted_baselines.jsonl"))
    parser.add_argument("--only", nargs="+", default=None, help="policy:view pairs to run (default all)")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as stream:
        for name, views in POLICIES.items():
            for view in views:
                if args.only and f"{name}:{view}" not in args.only:
                    continue
                row = play(name, view, args.episodes, args.seed)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"{name} on {view}: {row['mean_return']:+.1f}, wins {row['win_rate']:.0%}, "
                      f"balls returned {row['return_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
