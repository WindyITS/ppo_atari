import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from pong_rl.train import Args, CHECKPOINT_STEPS, main


class ShortGame(gym.Env):
    def __init__(self, channels):
        self.observation_space = gym.spaces.Box(0, 255, (channels, 84, 84), np.uint8)
        self.action_space = gym.spaces.Discrete(6)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(self.observation_space.shape, np.uint8), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(self.observation_space.shape, np.uint8), 1.0, self.steps == 2, False, {}


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])

    def run_training(self, args, milestones):
        def factory(*unused, **kwargs):
            return lambda: gym.wrappers.RecordEpisodeStatistics(ShortGame(kwargs["frame_stack"]))

        def update(agent, *unused):
            # Deterministic marker proves snapshots are taken after completed updates.
            with torch.no_grad():
                agent.actor.bias.add_(1)
            return {"charts/optimizer_steps": 1}

        with (patch("pong_rl.train.parse_args", return_value=args),
              patch("pong_rl.train.CHECKPOINT_STEPS", milestones),
              patch("pong_rl.train.make_env", factory),
              patch("pong_rl.train.update_policy", update),
              patch("pong_rl.train.torch.cuda.is_available", return_value=False),
              patch("pong_rl.train.torch.backends.mps.is_available", return_value=False),
              patch("pong_rl.train.wandb") as tracker,
              contextlib.redirect_stdout(io.StringIO())):
            main()
        return tracker

    def test_zero_and_crossed_milestones_for_both_models(self):
        self.assertEqual(CHECKPOINT_STEPS, (0, 2_500_000, 5_000_000, 7_500_000, 10_000_000))
        for arch in ("cnn", "lstm"):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                previous = Path.cwd()
                try:
                    os.chdir(directory)
                    tracker = self.run_training(Args(arch=arch, total_timesteps=6, num_envs=1), (0,3,5,7))
                    prefix = f"{arch}_fog_0_mask_0_seed_1"
                    files = sorted(Path("checkpoints").glob("*.pt"))
                    self.assertEqual(len(files), 4)  # 0, 3, 5 and final; 7 exceeds the target
                    for milestone, actual_step, updates in ((0,0,0), (3,4,2), (5,6,3)):
                        saved = torch.load(Path("checkpoints") / f"{prefix}_step_{milestone}.pt", weights_only=True)
                        self.assertEqual(saved["checkpoint_step"], milestone)
                        self.assertEqual(saved["global_step"], actual_step)
                        self.assertEqual(saved["updates"], updates)
                        self.assertEqual(saved["optimizer_steps"], updates)
                        torch.testing.assert_close(saved["model_state_dict"]["actor.bias"], torch.full((6,), float(updates)))
                    final = torch.load(Path("checkpoints") / f"{prefix}.pt", weights_only=True)
                    self.assertIsNone(final["checkpoint_step"])
                    self.assertEqual(final["global_step"], 6)
                    self.assertEqual(tracker.save.call_count, 4)
                    self.assertTrue(all(call.kwargs["policy"] == "now" for call in tracker.save.call_args_list))
                    self.assertEqual(list(Path("checkpoints").glob("*.tmp")), [])
                finally:
                    os.chdir(previous)

    def test_one_update_crossing_multiple_milestones_saves_them_all(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                self.run_training(Args(total_timesteps=3, num_envs=3, track=False), (0,1,2,3,4))
                for milestone in (1,2,3):
                    path = Path("checkpoints") / f"cnn_fog_0_mask_0_seed_1_step_{milestone}.pt"
                    saved = torch.load(path, weights_only=True)
                    self.assertEqual(saved["checkpoint_step"], milestone)
                    self.assertEqual(saved["global_step"], 6)
                    self.assertEqual(saved["updates"], 1)
                self.assertEqual(len(list(Path("checkpoints").glob("*.pt"))), 5)
            finally:
                os.chdir(previous)

    def test_save_disabled_creates_no_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                tracker = self.run_training(Args(total_timesteps=2, num_envs=1, save_model=False), (0,1,2))
                self.assertFalse(Path("checkpoints").exists())
                tracker.save.assert_not_called()
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
