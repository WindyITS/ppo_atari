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

from pong_rl.evaluate import evaluate_checkpoint, validate_output_path, write_report
from pong_rl.models import AGENTS, LSTMAgent, make_agent


class ShortGame(gym.Env):
    def __init__(self, channels, actions):
        self.observation_space = gym.spaces.Box(0, 255, shape=(channels, 84, 84), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(6)
        self.actions = actions

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.seed_value = seed
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self.actions.append((self.seed_value, int(action)))
        self.step_count += 1
        ended = self.step_count == 2
        timeout = self.seed_value % 2 == 0
        return (np.full(self.observation_space.shape, self.step_count, dtype=np.uint8),
                1.0, ended and not timeout, ended and timeout, {})


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_checkpoint(self, path, arch):
        channels = 4 if arch == "cnn" else 1
        env = gym.vector.SyncVectorEnv([lambda: ShortGame(channels, [])])
        model = make_agent(arch, env)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            model.actor.bias[2] = 2
        config = dict(arch=arch, env_id="ALE/Pong-v5", use_fog=True, mask_enemy=True,
                      seed=1, total_timesteps=10_000_000)
        torch.save(dict(arch=arch, args=config, condition="fog_1_mask_1",
                        model_state_dict=model.state_dict(), checkpoint_step=5_000_000,
                        global_step=5_000_128, updates=200), path)
        env.close()

    def test_both_architectures_use_saved_masks_and_reproducible_action_rng(self):
        for arch in ("cnn", "lstm", "attn_lstm"):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                self.make_checkpoint(path, arch)
                before = path.read_bytes()
                traces = []
                for global_seed in (123, 456):
                    torch.manual_seed(global_seed)
                    actions = []
                    def factory(env_id, idx, video, name, *, use_fog, mask_enemy, frame_stack):
                        self.assertTrue(use_fog and mask_enemy)
                        self.assertEqual(frame_stack, 4 if arch == "cnn" else 1)
                        return lambda: ShortGame(frame_stack, actions)
                    with patch("pong_rl.evaluate.make_env", factory), contextlib.redirect_stderr(io.StringIO()):
                        result = evaluate_checkpoint(path, episodes=4, seed=10000)
                    traces.append(actions)
                self.assertEqual(traces[0], traces[1])
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(result["summary"]["wins"], 2)
                self.assertEqual(result["summary"]["timeouts"], 2)
                self.assertEqual(result["summary"]["positive_timeouts"], 2)
                self.assertEqual(result["summary"]["completed_win_rate"], 1)
                self.assertEqual(result["checkpoint_step"], 5_000_000)
                self.assertEqual(result["training_global_step"], 5_000_128)
                self.assertEqual(result["training_updates"], 200)
                self.assertEqual([row["env_seed"] for row in result["episodes"]], [10000,10001,10002,10003])

    def test_lstm_is_frozen_and_memory_is_reset_after_timeout_and_termination(self):
        observed_starts = []
        class MemoryProbe(LSTMAgent):
            def get_states(self, x, state, done):
                if x[0, 0, 0, 0] == 0:
                    observed_starts.append(all(torch.count_nonzero(value).item() == 0 for value in state))
                assert not self.training
                assert not torch.is_grad_enabled()
                # Explicitly carry nonzero memory, including through the last episode step.
                hidden = torch.zeros((1, 128), device=x.device)
                return hidden, (state[0] + 1, state[1] + 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lstm.pt"
            self.make_checkpoint(path, "lstm")
            factory = lambda *args, **kwargs: lambda: ShortGame(1, [])
            with patch("pong_rl.evaluate.make_env", factory), patch.dict(AGENTS, {"lstm": MemoryProbe}), contextlib.redirect_stderr(io.StringIO()):
                result = evaluate_checkpoint(path, episodes=3, seed=10000, policy="greedy")
            self.assertEqual(observed_starts, [True, True, True])
            self.assertEqual([row["outcome"] for row in result["episodes"]], ["timeout", "win", "timeout"])

    def test_output_cannot_replace_checkpoint_or_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            checkpoint.write_bytes(b"weights")
            symlink = Path(directory) / "alias.json"
            symlink.symlink_to(checkpoint)
            hardlink = Path(directory) / "hardlink.json"
            os.link(checkpoint, hardlink)
            for output in (checkpoint, symlink, hardlink):
                with self.assertRaises(ValueError):
                    validate_output_path(output, [checkpoint])
            self.assertEqual(checkpoint.read_bytes(), b"weights")

    def test_unique_atomic_temp_file_cannot_clobber_input(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            checkpoint = Path(directory) / "report.json.tmp"
            checkpoint.write_bytes(b"weights")
            validate_output_path(output, [checkpoint])
            write_report(output, {"complete": False, "results": [1]})
            write_report(output, {"complete": True, "results": [1, 2]})
            self.assertEqual(json.loads(output.read_text()), {"complete": True, "results": [1,2]})
            self.assertEqual(checkpoint.read_bytes(), b"weights")
            self.assertEqual(list(Path(directory).glob(".pong-eval-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
