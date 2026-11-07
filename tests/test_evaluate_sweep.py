import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from pong_rl.models import make_agent

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_sweep.py"
spec = importlib.util.spec_from_file_location("evaluate_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

FLAGS = {"clean": (False, False), "mask": (False, True), "fog": (True, False), "fog_mask": (True, True)}


def save(root, relative, arch, view, kind="scratch", init_from=None, weights=None, tag=""):
    """A checkpoint with the metadata the runner checks (tiny unless real weights are given)."""
    path = Path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    fog, mask = FLAGS[view]
    torch.save({"arch": arch, "kind": kind, "init_from": init_from, "tag": tag, "run": "saved/run/name",
                "condition": f"fog_{int(fog)}_mask_{int(mask)}", "checkpoint_step": 1, "global_step": 1,
                "updates": 1, "model_state_dict": weights or {},
                "args": {"arch": arch, "env_id": "ALE/Pong-v5", "use_fog": fog, "mask_enemy": mask, "seed": 1}},
               path)
    return path


def make_tree(root):
    for arch in ("cnn", "lstm"):
        for seed in (1, 2):
            for view in FLAGS:
                save(root, f"scratch/{arch}/{view}/seed_{seed}/final.pt", arch, view, tag=f"{seed}")
            parent = save(root, f"scratch/{arch}/clean/seed_{seed}/step_10.0M.pt", arch, "clean", tag=f"p{seed}")
            record = {"sha256": hashlib.sha256(parent.read_bytes()).hexdigest()}
            for view in ("mask", "fog", "fog_mask"):
                save(root, f"finetune/{arch}/clean_10.0M_to_{view}/seed_{seed}/final.pt", arch, view,
                     kind="finetune", init_from=record, tag=f"{seed}")
    # Must be ignored: an aborted run, another architecture, a parent whose run did not finish,
    # a fine-tune from another parent checkpoint.
    save(root, "scratch/cnn/fog/seed_1.aborted_20261105_101010/final.pt", "cnn", "fog")
    save(root, "scratch/attn_lstm/clean/seed_1/final.pt", "attn_lstm", "clean")
    save(root, "scratch/lstm/clean/seed_3/step_10.0M.pt", "lstm", "clean")
    save(root, "finetune/cnn/clean_05.0M_to_fog/seed_1/final.pt", "cnn", "fog", kind="finetune")


def fake_evaluate(path, *, episodes, seed, policy, device, view, quiet):
    """Stands in for evaluate_checkpoint: same identifying fields, no games."""
    path = Path(path)
    args = torch.load(path, weights_only=True)["args"]
    assert quiet and device == "cpu" and view in FLAGS
    return {"checkpoint": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "run": "saved/run/name", "kind": "scratch", "use_fog": FLAGS[view][0], "mask_enemy": FLAGS[view][1],
            "eval_view": view, "train_view": sweep.view_name(args["use_fog"], args["mask_enemy"]),
            "summary": {"mean_return": 1.0, "completed_win_rate": 0.5}, "episodes": []}


def run_main(root, output, *extra, evaluate=fake_evaluate, workers="0"):
    argv = ["evaluate_sweep.py", "--root", str(root), "--workers", workers, "--episodes", "3",
            "--output", str(output), *extra]
    with patch("sys.argv", argv), patch.object(sweep, "evaluate_checkpoint", evaluate), \
            contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
        try:
            sweep.main()
            code = 0
        except SystemExit as exc:
            code = exc.code
    return out.getvalue(), err.getvalue(), code


class SweepTests(unittest.TestCase):
    def test_discovery_keeps_finished_runs_of_the_requested_archs_and_parent_only(self):
        with tempfile.TemporaryDirectory() as root:
            make_tree(root)
            found = sweep.discover(Path(root), ["cnn", "lstm"], sweep.ROLES, "step_10.0M.pt")
            roles = [role for role, _ in found]
            self.assertEqual((roles.count("scratch"), roles.count("finetune"), roles.count("parent")), (16, 12, 4))
            self.assertFalse(any(".aborted_" in str(p) or "attn_lstm" in str(p) or "seed_3" in str(p)
                                 or "05.0M" in str(p) for _, p in found))

    def test_jobs_cover_every_view_and_own_resolves_to_the_training_view(self):
        with tempfile.TemporaryDirectory() as root:
            make_tree(root)
            found = sweep.discover(Path(root), ["cnn"], ("finetune",), "step_10.0M.pt")
            jobs = sweep.plan_jobs(found, ["own", "clean", "fog", "fog_mask"], Path(root))
            first = [j["eval_view"] for j in jobs if j["run"] == "finetune/cnn/clean_10.0M_to_fog/seed_1"]
            self.assertEqual(first, ["fog", "clean", "fog_mask"])  # own == fog, not repeated
            parents = sweep.plan_jobs(sweep.discover(Path(root), ["cnn"], ("parent",), "step_10.0M.pt"),
                                      ["own"], Path(root))
            self.assertEqual({j["eval_view"] for j in parents}, {"clean"})

    def test_full_run_writes_one_line_per_job_and_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            make_tree(root)
            output = Path(root, "out", "eval.jsonl")
            _, _, code = run_main(root, output)
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 32 * 4)
            self.assertEqual(len({(r["run"], r["checkpoint_file"], r["eval_view"]) for r in rows}), 128)
            row = next(r for r in rows if r["run"] == "finetune/lstm/clean_10.0M_to_mask/seed_2")
            self.assertEqual((row["role"], row["train_view"], row["checkpoint_run"]),
                             ("finetune", "mask", "saved/run/name"))
            self.assertEqual(row["protocol"], {"version": sweep.PROTOCOL_VERSION, "episodes": 3, "seed": 10000,
                                               "policy": "stochastic", "device": "cpu"})
            self.assertIn("128 already in", run_main(root, output)[0])

    def test_resume_after_a_cut_off_line_and_a_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            make_tree(root)
            output = Path(root, "eval.jsonl")
            run_main(root, output)
            data = output.read_bytes()
            output.write_bytes(data[:-200])  # the last row is cut off mid-write
            save(root, "scratch/cnn/fog/seed_2/final.pt", "cnn", "fog", tag="retrained")
            out, _, code = run_main(root, output)
            self.assertEqual(code, 0)
            self.assertIn("5 to run", out)  # the cut-off job + 4 views of the changed checkpoint
            lines = output.read_text().splitlines()
            self.assertEqual(len(lines), 128 + 4)  # the 4 rows of the old checkpoint bytes stay
            for line in lines:
                json.loads(line)  # every line is a whole row
            self.assertIn("128 already in", run_main(root, output)[0])
            # A different protocol is a different set of jobs.
            self.assertIn("128 to run", run_main(root, output, "--episodes", "5", "--dry-run")[0])

    def test_a_failing_job_is_reported_the_others_written_and_the_exit_code_is_1(self):
        def flaky(path, **kwargs):
            if "fog/seed_1" in str(path) and kwargs["view"] == "clean":
                raise RuntimeError("emulator crashed")
            return fake_evaluate(path, **kwargs)
        with tempfile.TemporaryDirectory() as root:
            make_tree(root)
            output = Path(root, "eval.jsonl")
            _, err, code = run_main(root, output, "--roles", "scratch", evaluate=flaky)
            self.assertEqual(code, 1)
            self.assertIn("RuntimeError: emulator crashed", err)
            self.assertEqual(len(output.read_text().splitlines()), 16 * 4 - 2)  # cnn and lstm fog/seed_1

    def test_mismatched_views_are_failures_not_rows(self):
        def wrong(path, **kwargs):
            return {**fake_evaluate(path, **kwargs), "train_view": "clean"}
        with tempfile.TemporaryDirectory() as root:
            save(root, "scratch/cnn/fog/seed_1/final.pt", "cnn", "fog")
            output = Path(root, "eval.jsonl")
            _, err, code = run_main(root, output, "--views", "own", evaluate=wrong)
            self.assertEqual(code, 1)
            self.assertIn("expected fog / fog", err)
            self.assertEqual(output.read_text(), "")

    def test_checkpoints_that_do_not_match_their_folder_stop_the_run_before_any_game(self):
        cases = {
            "scratch checkpoint in a fine-tune folder": lambda root: save(
                root, "finetune/cnn/clean_10.0M_to_mask/seed_1/final.pt", "cnn", "mask"),
            "wrong view": lambda root: save(root, "scratch/cnn/fog/seed_1/final.pt", "cnn", "clean"),
            "wrong arch": lambda root: save(root, "scratch/cnn/fog/seed_1/final.pt", "lstm", "fog"),
            "wrong parent": lambda root: save(root, "finetune/cnn/clean_10.0M_to_mask/seed_1/final.pt", "cnn",
                                              "mask", kind="finetune", init_from={"sha256": "0" * 64}),
        }
        for name, break_tree in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as root:
                make_tree(root)
                break_tree(root)
                output = Path(root, "eval.jsonl")
                _, err, code = run_main(root, output, evaluate=lambda *a, **k: self.fail("played a game"))
                self.assertEqual(code, 2)
                self.assertIn("do not match their folders", err)

    def test_worker_processes_play_real_games(self):
        """The spawn pool path end to end: real checkpoint, real evaluator and Pong, 2 workers."""
        spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(0, 255, (1, 84, 84), np.uint8),
                                 single_action_space=gym.spaces.Discrete(6))
        torch.manual_seed(0)
        weights = make_agent("lstm", spaces).state_dict()
        with tempfile.TemporaryDirectory() as root:
            save(root, "scratch/lstm/fog/seed_1/final.pt", "lstm", "fog", weights=weights)
            output = Path(root, "eval.jsonl")
            # A real subprocess, as on the host: spawn workers must be able to import the script.
            done = subprocess.run([sys.executable, str(SCRIPT), "--root", root, "--workers", "2", "--episodes", "1",
                                   "--output", str(output), "--views", "own", "clean"],
                                  capture_output=True, text=True, timeout=300)
            self.assertEqual(done.returncode, 0, done.stderr)
            rows = sorted((json.loads(line) for line in output.read_text().splitlines()),
                          key=lambda r: r["eval_view"])
            self.assertEqual([(r["eval_view"], r["use_fog"], r["train_view"]) for r in rows],
                             [("clean", False, "fog"), ("fog", True, "fog")])
            self.assertTrue(all(r["summary"]["episodes"] == 1 and r["episodes"][0]["env_seed"] == 10000
                                for r in rows))


if __name__ == "__main__":
    unittest.main()
