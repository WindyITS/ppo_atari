import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from pong_rl.train import Args, main


class TwoStepGame(gym.Env):
    def __init__(self, channels):
        self.observation_space = gym.spaces.Box(0, 255, shape=(channels, 84, 84), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(6)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.timeout = seed % 2 == 0
        self.steps = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self.steps += 1
        ended = self.steps == 2
        return (np.full(self.observation_space.shape, self.steps, dtype=np.uint8),
                1, ended and not self.timeout, ended and self.timeout, {})


class TrainingReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize PyTorch's lazy optimizer/device registry before mocking device probes.
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])

    def test_same_step_vector_outcomes_and_window_metrics_for_both_models(self):
        for arch in ("cnn", "lstm"):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                def factory(*args, **kwargs):
                    return lambda: gym.wrappers.RecordEpisodeStatistics(TwoStepGame(kwargs["frame_stack"]))
                args = Args(arch=arch, total_timesteps=1024, num_envs=8, track=True,
                            save_model=False, report_window_steps=128)
                cwd = Path.cwd()
                try:
                    os.chdir(directory)
                    with (patch("pong_rl.train.parse_args", return_value=args),
                          patch("pong_rl.train.make_env", factory),
                          patch("pong_rl.train.torch.cuda.is_available", return_value=False),
                          patch("pong_rl.train.torch.backends.mps.is_available", return_value=False),
                          patch("pong_rl.train.wandb") as tracker,
                          contextlib.redirect_stdout(io.StringIO())):
                        main()
                    records = [json.loads(line) for line in next(Path("logs", "scratch", arch, "clean", "seed_1").glob("episodes_*.jsonl")).read_text().splitlines()]
                    self.assertEqual(len(records), 512)
                    self.assertEqual(sum(row["outcome"] == "timeout" for row in records), 256)
                    self.assertEqual(sum(row["outcome"] == "win" for row in records), 256)
                    metrics = tracker.log.call_args_list[-1].args[0]
                    self.assertEqual(metrics["charts/window/start_step_exclusive"], 896)
                    self.assertEqual(metrics["charts/window/end_step_inclusive"], 1024)
                    self.assertEqual(metrics["charts/window/episodes"], 64)
                    self.assertEqual(metrics["charts/window/completed_games"], 32)
                    self.assertEqual(metrics["charts/window/timeouts"], 32)
                    self.assertEqual(metrics["charts/window/completed_win_rate"], 1)
                    self.assertEqual(metrics["charts/window/mean_episode_length"], 2)
                finally:
                    os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
