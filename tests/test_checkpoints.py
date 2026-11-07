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

from pong_rl.train import Args, CHECKPOINT_STEPS, checkpoint_filename, main


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


@contextlib.contextmanager
def inside_temporary_directory():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(previous)


def load(path):
    return torch.load(path, weights_only=True)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])

    def run_training(self, args, milestones, learning_rates=None):
        def factory(*unused, **kwargs):
            return lambda: gym.wrappers.RecordEpisodeStatistics(ShortGame(kwargs["frame_stack"]))

        def update(agent, optimizer, *unused):
            if learning_rates is not None:
                learning_rates.append(optimizer.param_groups[0]["lr"])
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

    def test_schedule_and_readable_names(self):
        self.assertEqual(CHECKPOINT_STEPS, (0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
                                            7_500_000, 10_000_000, 11_000_000, 12_000_000, 13_000_000,
                                            14_000_000, 15_000_000))
        self.assertEqual([checkpoint_filename(step) for step in CHECKPOINT_STEPS],
                         ["step_00.0M.pt", "step_01.0M.pt", "step_02.0M.pt", "step_03.0M.pt",
                          "step_04.0M.pt", "step_05.0M.pt", "step_07.5M.pt", "step_10.0M.pt", "step_11.0M.pt",
                          "step_12.0M.pt", "step_13.0M.pt", "step_14.0M.pt", "step_15.0M.pt"])
        self.assertEqual(checkpoint_filename(3), "step_3.pt")
        self.assertEqual(Args().total_timesteps, 15_000_000)

    def test_zero_and_crossed_milestones_for_both_models(self):
        for arch in ("cnn", "lstm", "attn_lstm", "vit", "vit_lstm"):
            with self.subTest(arch=arch), inside_temporary_directory():
                tracker = self.run_training(Args(num_steps=2, arch=arch, total_timesteps=6, num_envs=1), (0, 3, 5, 7))
                folder = Path("checkpoints", "scratch", arch, "clean", "seed_1")
                self.assertEqual(sorted(p.name for p in Path("checkpoints").rglob("*.pt")),
                                 ["final.pt", "step_00.0M.pt", "step_3.pt", "step_5.pt"])  # 7 exceeds the target
                for name, milestone, actual_step, updates in (("step_00.0M.pt", 0, 0, 0), ("step_3.pt", 3, 4, 2),
                                                              ("step_5.pt", 5, 6, 3)):
                    saved = load(folder / name)
                    self.assertEqual(saved["checkpoint_step"], milestone)
                    self.assertEqual(saved["global_step"], actual_step)
                    self.assertEqual(saved["updates"], updates)
                    self.assertEqual(saved["optimizer_steps"], updates)
                    self.assertEqual((saved["kind"], saved["run"], saved["start_step"], saved["init_from"]),
                                     ("scratch", f"scratch/{arch}/clean/seed_1", 0, None))
                    torch.testing.assert_close(saved["model_state_dict"]["actor.bias"], torch.full((6,), float(updates)))
                final = load(folder / "final.pt")
                self.assertIsNone(final["checkpoint_step"])
                self.assertEqual(final["global_step"], 6)
                self.assertEqual(tracker.save.call_count, 4)
                self.assertTrue(all(call.kwargs["policy"] == "now" for call in tracker.save.call_args_list))
                self.assertEqual(list(Path("checkpoints").rglob("*.tmp")), [])
                self.assertEqual(len(list(Path("logs", "scratch", arch, "clean", "seed_1").glob("episodes_*.jsonl"))), 1)

    def test_one_update_crossing_multiple_milestones_saves_them_all(self):
        with inside_temporary_directory():
            self.run_training(Args(num_steps=2, total_timesteps=3, num_envs=3, track=False), (0, 1, 2, 3, 4))
            folder = Path("checkpoints", "scratch", "cnn", "clean", "seed_1")
            for milestone in (1, 2, 3):
                saved = load(folder / f"step_{milestone}.pt")
                self.assertEqual(saved["checkpoint_step"], milestone)
                self.assertEqual(saved["global_step"], 6)
                self.assertEqual(saved["updates"], 1)
            self.assertEqual(len(list(folder.glob("*.pt"))), 5)

    def test_finetune_continues_parent_steps_with_its_own_folder_and_schedule(self):
        milestones = (0, 3, 5, 7, 9)
        for arch in ("cnn", "lstm", "attn_lstm", "vit", "vit_lstm"):
            with self.subTest(arch=arch), inside_temporary_directory():
                self.run_training(Args(num_steps=2, arch=arch, total_timesteps=6, num_envs=1, track=False), milestones)
                parent_path = Path("checkpoints", "scratch", arch, "clean", "seed_1", "step_3.pt")
                parent = load(parent_path)  # milestone 3, actual step 4, actor.bias == 2
                learning_rates = []
                tracker = self.run_training(
                    Args(num_steps=2, arch=arch, use_fog=True, total_timesteps=9, num_envs=1, init_from=str(parent_path)),
                    milestones, learning_rates)
                folder = Path("checkpoints", "finetune", arch, "clean_3_to_fog", "seed_1")
                self.assertEqual(sorted(p.name for p in folder.glob("*.pt")),
                                 ["final.pt", "step_3.pt", "step_5.pt", "step_7.pt", "step_9.pt"])
                start = load(folder / "step_3.pt")  # the parent's weights, before any fine-tuning
                for key, value in parent["model_state_dict"].items():
                    torch.testing.assert_close(start["model_state_dict"][key], value)
                self.assertEqual((start["global_step"], start["updates"], start["checkpoint_step"]), (4, 0, 3))
                self.assertEqual(start["kind"], "finetune")
                self.assertEqual(start["condition"], "fog_1_mask_0")
                self.assertEqual((start["start_step"], start["steps_since_start"]), (4, 0))
                self.assertEqual(load(folder / "step_9.pt")["steps_since_start"], 6)
                self.assertEqual(start["init_from"]["path"], str(parent_path.resolve()))
                self.assertEqual(start["init_from"]["global_step"], 4)
                self.assertEqual(start["init_from"]["condition_label"], "clean")
                self.assertEqual(len(start["init_from"]["sha256"]), 64)
                for name, actual_step, updates in (("step_5.pt", 6, 1), ("step_7.pt", 8, 2), ("step_9.pt", 10, 3)):
                    saved = load(folder / name)
                    self.assertEqual((saved["global_step"], saved["updates"]), (actual_step, updates))
                    torch.testing.assert_close(saved["model_state_dict"]["actor.bias"],
                                               torch.full((6,), 2.0 + updates))
                self.assertEqual(load(folder / "final.pt")["global_step"], 10)
                # The learning rate restarts at the base value and anneals over steps 4 -> 9 only.
                base = Args(arch=arch).learning_rate
                for actual, expected in zip(learning_rates, (base, base * 0.6, base * 0.2)):
                    self.assertAlmostEqual(actual, expected)
                self.assertEqual(len(learning_rates), 3)
                self.assertEqual(tracker.init.call_args.kwargs["config"]["kind"], "finetune")
                logs = list(Path("logs", "finetune", arch, "clean_3_to_fog", "seed_1").glob("episodes_*.jsonl"))
                self.assertEqual(len(logs), 1)
                metrics = tracker.log.call_args_list[-1].args[0]
                self.assertEqual(metrics["charts/window/start_step_exclusive"], 4)
                self.assertEqual(metrics["charts/steps_since_start"], 6)

    def test_finetune_rejects_wrong_architecture_and_exhausted_budget(self):
        with inside_temporary_directory():
            self.run_training(Args(num_steps=2, total_timesteps=4, num_envs=1, track=False), (0, 3))
            parent = str(Path("checkpoints", "scratch", "cnn", "clean", "seed_1", "step_3.pt"))
            with self.assertRaisesRegex(ValueError, "architecture"):
                self.run_training(Args(num_steps=2, arch="lstm", total_timesteps=9, num_envs=1, init_from=parent), (0, 3))
            with self.assertRaisesRegex(ValueError, "must exceed"):
                self.run_training(Args(num_steps=2, total_timesteps=4, num_envs=1, init_from=parent), (0, 3))
            self.assertFalse(Path("checkpoints", "finetune").exists())

    def test_save_disabled_creates_no_checkpoints(self):
        with inside_temporary_directory():
            tracker = self.run_training(Args(num_steps=2, total_timesteps=2, num_envs=1, save_model=False), (0, 1, 2))
            self.assertFalse(Path("checkpoints").exists())
            tracker.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
