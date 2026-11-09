"""Training variants used by the follow-up experiments: --run-tag folders, the finished-run guard,
long-run milestones, --reinit fine-tunes and --resume of an interrupted run."""

from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch._dynamo  # noqa: F401  (Adam.load_state_dict imports it lazily; the tests mock torch.backends)

import pong_rl.train
from pong_rl.train import LONG_RUN_CHECKPOINT_STEPS, Args, parse_args
from tests.test_checkpoints import CheckpointTests, inside_temporary_directory, load


class TrainingVariantTests(unittest.TestCase):
    run_training = CheckpointTests.run_training

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def parent(self, arch="lstm"):
        self.run_training(Args(num_steps=2, arch=arch, total_timesteps=4, num_envs=1, track=False), (0, 3))
        return Path("checkpoints", "scratch", arch, "clean", "seed_1", "step_3.pt")

    def test_run_tag_gets_its_own_folder_and_keeps_the_original_untouched(self):
        with inside_temporary_directory():
            parent = self.parent()
            args = dict(num_steps=2, arch="lstm", use_fog=True, mask_enemy=True, total_timesteps=8, num_envs=1,
                        track=False, init_from=str(parent))
            self.run_training(Args(**args), (0, 3, 5))
            original = Path("checkpoints", "finetune", "lstm", "clean_3_to_fog_mask", "seed_1")
            before = {p.name: p.read_bytes() for p in original.glob("*.pt")}
            self.run_training(Args(**args, run_tag="ft15M"), (0, 3, 5))
            tagged = Path("checkpoints", "finetune", "lstm", "clean_3_to_fog_mask__ft15M", "seed_1")
            self.assertEqual(sorted(p.name for p in tagged.glob("*.pt")), ["final.pt", "step_3.pt", "step_5.pt"])
            self.assertEqual(load(tagged / "final.pt")["run"], "finetune/lstm/clean_3_to_fog_mask__ft15M/seed_1")
            self.assertEqual({p.name: p.read_bytes() for p in original.glob("*.pt")}, before)

    def test_a_finished_run_is_never_overwritten(self):
        with inside_temporary_directory():
            parent = self.parent()
            final = parent.parent / "final.pt"
            before = final.read_bytes()
            with self.assertRaisesRegex(ValueError, "already holds a finished run"):
                self.run_training(Args(num_steps=2, arch="lstm", total_timesteps=4, num_envs=1, track=False), (0, 3))
            self.assertEqual(final.read_bytes(), before)

    def test_long_runs_save_the_extra_milestones(self):
        self.assertEqual(LONG_RUN_CHECKPOINT_STEPS, (17_500_000, 20_000_000, 22_500_000, 25_000_000, 30_000_000))
        with inside_temporary_directory(), patch("pong_rl.train.LONG_RUN_CHECKPOINT_STEPS", (7, 9, 11)):
            self.run_training(Args(num_steps=2, total_timesteps=10, num_envs=1, track=False), (0, 3, 5))
            folder = Path("checkpoints", "scratch", "cnn", "clean", "seed_1")
            self.assertEqual(sorted(p.name for p in folder.glob("*.pt")),
                             ["final.pt", "step_00.0M.pt", "step_3.pt", "step_5.pt", "step_7.pt", "step_9.pt"])

    def test_reinit_resets_only_the_named_modules(self):
        with inside_temporary_directory():
            parent_path = self.parent()
            parent = load(parent_path)["model_state_dict"]
            # Seed 2: a fresh seed-1 module would equal the parent's own (untrained here) initial weights.
            self.run_training(Args(num_steps=2, arch="lstm", seed=2, use_fog=True, mask_enemy=True,
                                   total_timesteps=8, num_envs=1, track=False, init_from=str(parent_path),
                                   run_tag="reinit", reinit=("lstm", "actor", "critic")), (0, 3, 5))
            start = load(Path("checkpoints", "finetune", "lstm", "clean_3_to_fog_mask__reinit", "seed_2",
                              "step_3.pt"))
            self.assertEqual(start["init_from"]["reinit"], ["actor", "critic", "lstm"])
            state = start["model_state_dict"]
            for key, value in parent.items():
                if key.split(".")[0] == "network":
                    torch.testing.assert_close(state[key], value)
                elif key.endswith("weight") or "weight_" in key:  # biases start at constants either way
                    self.assertFalse(torch.equal(state[key], value), key)

    def test_reinit_rejects_unknown_modules_and_missing_parent(self):
        with inside_temporary_directory():
            parent_path = self.parent("cnn")
            with self.assertRaisesRegex(ValueError, "no module lstm"):
                self.run_training(Args(num_steps=2, total_timesteps=8, num_envs=1, track=False,
                                       init_from=str(parent_path), reinit=("lstm",)), (0, 3))
            with self.assertRaisesRegex(ValueError, "parent"):
                self.run_training(Args(num_steps=2, total_timesteps=8, num_envs=1, track=False, run_tag="x",
                                       reinit=("actor",)), (0, 3))

    def finetune(self, parent, crash_at=None, learning_rates=None):
        """Fine-tune to 12 steps (updates at 6, 8, 10, 12; milestones 5 and 9); optionally crash in an update."""
        args = Args(num_steps=2, arch="lstm", use_fog=True, mask_enemy=True, total_timesteps=12, num_envs=1,
                    track=False, init_from=str(parent), run_tag="ft")
        calls = []
        original = pong_rl.train.save_checkpoint

        def crashing_save(*a, **k):  # crash right after the run's crash_at-th checkpoint is written
            original(*a, **k)
            calls.append(k.get("checkpoint_step"))
            if crash_at is not None and len(calls) == crash_at:
                raise KeyboardInterrupt("simulated power cut")
        with patch("pong_rl.train.save_checkpoint", crashing_save):
            try:
                self.run_training(args, (0, 3, 5, 9), learning_rates)
            except KeyboardInterrupt:
                pass
        return Path("checkpoints", "finetune", "lstm", "clean_3_to_fog_mask__ft", "seed_1")

    def resume(self, path, learning_rates=None):
        with patch("sys.argv", ["pong-rl", "--resume", str(path), "--no-track"]):
            args = parse_args()
        self.assertEqual((args.arch, args.total_timesteps, args.resume), ("lstm", 12, str(path)))
        self.run_training(args, (0, 3, 5, 9), learning_rates)

    def test_resume_continues_an_interrupted_run_on_its_schedule_without_touching_its_files(self):
        with inside_temporary_directory():
            parent = self.parent()
            full_rates = []
            folder = self.finetune(parent, learning_rates=full_rates)
            reference = load(folder / "final.pt")
            self.assertIn("optimizer_state_dict", reference)
            stored = folder.with_name("seed_reference")
            folder.rename(stored)

            cut_rates = []
            self.finetune(parent, crash_at=2, learning_rates=cut_rates)  # crash after step_5.pt
            self.assertEqual(sorted(p.name for p in folder.glob("*.pt")), ["step_3.pt", "step_5.pt"])
            before = {p.name: p.read_bytes() for p in folder.glob("*.pt")}
            with self.assertRaisesRegex(ValueError, "final.pt"):
                self.resume(stored / "final.pt")
            resumed_rates = []
            self.resume(folder / "step_5.pt", resumed_rates)
            self.assertEqual(sorted(p.name for p in folder.glob("*.pt")),
                             ["final.pt", "step_3.pt", "step_5.pt", "step_9.pt"])
            self.assertEqual({n: (folder / n).read_bytes() for n in before}, before)  # untouched
            self.assertEqual(cut_rates + resumed_rates, full_rates)  # same learning-rate schedule
            final = load(folder / "final.pt")
            self.assertEqual((final["global_step"], final["updates"], final["start_step"]),
                             (reference["global_step"], reference["updates"], reference["start_step"]))
            self.assertEqual(final["resumed_from"]["checkpoint"], "step_5.pt")
            self.assertFalse(final["resumed_from"]["fresh_optimizer"])
            self.assertEqual(final["init_from"]["path"], reference["init_from"]["path"])
            # The fake update adds 1 to the actor bias per update: same number of updates as uninterrupted.
            torch.testing.assert_close(final["model_state_dict"]["actor.bias"],
                                       reference["model_state_dict"]["actor.bias"])
            with self.assertRaisesRegex(ValueError, "already holds a finished run"):
                self.resume(folder / "step_5.pt")

    def test_cli_parses_and_validates_the_new_options(self):
        argv = ["pong-rl", "--arch", "lstm", "--init-from", "p.pt", "--run-tag", "ft15M", "--reinit", "lstm", "actor"]
        with patch("sys.argv", argv):
            args = parse_args()
        self.assertEqual((args.run_tag, args.reinit), ("ft15M", ("lstm", "actor")))
        for bad in (["--run-tag", "a/b"], ["--reinit", "actor"]):
            with self.subTest(bad=bad), patch("sys.argv", ["pong-rl", *bad]), \
                    patch("sys.stderr"), self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
