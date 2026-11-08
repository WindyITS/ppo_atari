import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch

from pong_rl.models import make_agent

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fog_exposure.py"
spec = importlib.util.spec_from_file_location("fog_exposure", SCRIPT)
fe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe)


def step(x, visible, y=40.0, enemy=40.0):
    return {"ball_x": x, "ball_y": y, "enemy_y": enemy, "agent_y": 50.0, "visible": visible}


class FogExposureTests(unittest.TestCase):
    def test_locate_reads_the_clean_frame_and_visibility_from_the_seen_one(self):
        clean = np.full((84, 84), 87, np.uint8)
        clean[40:42, 50] = 236
        clean[30:38, 8:11] = 236
        seen = clean.copy()
        seen[14:77, 42:72] = 87
        self.assertEqual(fe.locate(clean, seen), {"ball_x": 50.0, "ball_y": 40.5, "enemy_y": 33.5,
                                                  "agent_y": None, "visible": False})
        self.assertTrue(fe.locate(clean, clean)["visible"])
        self.assertIsNone(fe.locate(np.full((84, 84), 87, np.uint8), seen)["ball_x"])

    def test_runs_directions_ends_and_stack_blindness(self):
        steps = ([step(36, True), step(40, True)] + [step(44 + 3 * i, False) for i in range(8)]
                 + [step(72, True)] + [step(68 - 3 * i, False) for i in range(5)]
                 + [step(None, False)] + [step(50, False)])
        runs = fe.hidden_runs(steps)
        self.assertEqual([(r["length"], r["direction"], r["end"]) for r in runs],
                         [(8, "incoming", "emerged"), (5, "outgoing", "vanished"), (1, None, "episode_end")])
        # 8-step run: steps 4..8 of it are blind (3 earlier frames hidden too) -> 5; 5-step run: 2; last: 1
        self.assertEqual(fe.stack_blind(steps), 5 + 2 + 1)
        summary = fe.summarize([{"return": 1.0, "steps": steps, "runs": runs, "stack_blind": fe.stack_blind(steps)}])
        self.assertEqual((summary["hidden_steps"], summary["ball_on_court_steps"]), (14, 17))
        self.assertEqual(summary["runs"]["incoming"]["median"], 8)

    def test_tracking_correlates_paddle_and_ball_heights_on_hidden_steps_only(self):
        steps = [step(50, False, y=20 + i, enemy=22 + i) for i in range(10)] + [step(30, True, y=0, enemy=80)]
        result = fe.tracking(steps, "enemy_y")
        self.assertEqual((result["n"], result["corr"], result["mean_abs_gap"]), (10, 1.0, 2.0))

    def test_real_games_on_the_fog_view(self):
        spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(0, 255, (1, 84, 84), np.uint8),
                                 single_action_space=gym.spaces.Discrete(6))
        torch.manual_seed(0)
        weights = make_agent("lstm", spaces).state_dict()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root, "scratch/lstm/fog/seed_1/final.pt")
            path.parent.mkdir(parents=True)
            torch.save({"arch": "lstm", "model_state_dict": weights, "condition": "fog_1_mask_0",
                        "args": {"arch": "lstm", "env_id": "ALE/Pong-v5", "use_fog": True, "mask_enemy": False,
                                 "seed": 1}}, path)
            output = Path(root, "out.jsonl")
            done = subprocess.run([sys.executable, str(SCRIPT), "--root", root, "--episodes", "1", "--workers", "1",
                                   "--output", str(output)], capture_output=True, text=True, timeout=300)
            self.assertEqual(done.returncode, 0, done.stderr)
            row = json.loads(output.read_text())
            game = row["games"][0]
            trace = game["trace"]
            self.assertEqual(set(trace), set(fe.TRACE_KEYS))
            self.assertTrue(len(set(map(len, trace.values()))) == 1 and len(trace["ball_x"]) > 100)
            hidden = [x for x, v in zip(trace["ball_x"], trace["visible"]) if x is not None and not v]
            self.assertTrue(hidden and all(42 <= x < 72 for x in hidden))  # hidden only inside the fog band
            self.assertTrue(all(v for x, v in zip(trace["ball_x"], trace["visible"])
                                if x is not None and (x < 41 or x > 72)))
            self.assertEqual(row["summary"]["hidden_steps"], len(hidden))
            self.assertEqual(row["train_view"], "fog")


if __name__ == "__main__":
    unittest.main()
