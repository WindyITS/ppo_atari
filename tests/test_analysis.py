import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import ale_py
import gymnasium as gym
import numpy as np
import torch

from pong_rl.analysis import scene
from pong_rl.analysis.player import Player, stack_wipe
from pong_rl.analysis.snapshot import restore, snapshot
from pong_rl.envs import make_env
from pong_rl.evaluate import evaluate_checkpoint
from pong_rl.models import make_agent

gym.register_envs(ale_py)


def make_checkpoint(path, arch, use_fog=True, mask_enemy=True, seed=0):
    """A randomly initialised (reproducible) checkpoint with the metadata the evaluator needs."""
    torch.manual_seed(seed)
    channels = 4 if arch == "cnn" else 1
    space = gym.spaces.Box(0, 255, (channels, 84, 84), np.uint8)
    model = make_agent(arch, SimpleNamespace(single_observation_space=space, single_action_space=gym.spaces.Discrete(6)))
    config = dict(arch=arch, env_id="ALE/Pong-v5", use_fog=use_fog, mask_enemy=mask_enemy, seed=1,
                  total_timesteps=15_000_000)
    torch.save(dict(arch=arch, args=config, condition=f"fog_{int(use_fog)}_mask_{int(mask_enemy)}",
                    model_state_dict=model.state_dict(), checkpoint_step=None, global_step=1, updates=1), path)
    return path


class SceneTests(unittest.TestCase):
    def test_positions_visibility_and_edits(self):
        clean = np.full((84, 84), 87, np.uint8)
        clean[40:42, 50] = 236          # ball
        clean[20:28, 8:11] = 236        # enemy paddle, rows 20-27
        clean[60:68, 73:76] = 236       # agent paddle
        seen = clean.copy()
        seen[14:77, 42:72] = 87
        where = scene.locate(clean, seen)
        self.assertEqual((where["ball_x"], where["ball_y"], where["enemy_y"], where["agent_y"]), (50.0, 40.5, 23.5, 63.5))
        self.assertFalse(where["visible"])
        self.assertTrue(scene.locate(clean)["visible"])
        self.assertTrue(np.isnan(scene.locate(np.full((84, 84), 87, np.uint8))["ball_x"]))
        mirrored = scene.locate(scene.mirror_enemy(clean))
        self.assertEqual(mirrored["enemy_y"], 14 + 76 - 23.5)
        self.assertEqual(mirrored["ball_y"], 40.5)
        rows, cols = scene.ball_box(clean, pad=1)
        self.assertTrue(np.isnan(scene.locate(scene.erase(clean, rows, cols))["ball_x"]))
        stack = np.stack([np.full((84, 84), i, np.uint8) for i in range(4)])
        self.assertTrue((stack_wipe(stack) == 3).all())


class SnapshotTests(unittest.TestCase):
    def test_restore_replays_observations_rewards_and_clean_frames_exactly(self):
        env = make_env("ALE/Pong-v5", 0, False, "test", use_fog=True, mask_enemy=True, frame_stack=4,
                       tap_clean_frame=True)()
        rng = np.random.default_rng(0)
        env.reset(seed=3)
        for _ in range(120):
            env.step(int(rng.integers(6)))
        saved = snapshot(env)
        actions = [int(a) for a in rng.integers(6, size=200)]

        def play():
            out = []
            for a in actions:
                obs, reward, terminated, truncated, _ = env.step(a)
                out.append((np.array(obs, copy=True), reward, terminated, truncated,
                            np.array(env.get_wrapper_attr("clean_frame"), copy=True)))
            return out

        first = play()
        for _ in range(37):  # wander off before restoring
            env.step(int(rng.integers(6)))
        restore(env, saved)
        second = play()
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1:4], b[1:4])
            np.testing.assert_array_equal(a[4], b[4])
        env.close()


class PlayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_player_reproduces_the_frozen_evaluation_games(self):
        for arch in ("cnn", "lstm", "attn_lstm"):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as folder:
                path = make_checkpoint(Path(folder, "model.pt"), arch)
                report = evaluate_checkpoint(path, episodes=1, seed=10_000, quiet=True)
                player = Player(path)
                player.reset(10_000)
                steps = []
                while not player.done:
                    steps.append(player.step())
                player.close()
                self.assertEqual(player.episode_return, report["episodes"][0]["episode_return"])
                self.assertEqual(len(steps), report["episodes"][0]["episode_length"])
                self.assertEqual(steps[0].probs.shape, (6,))
                keys = {"cnn": {"features"}, "lstm": {"features", "h", "c"}, "attn_lstm": {"attention", "h", "c"}}
                self.assertEqual(set(steps[0].internals), keys[arch])

    def test_saved_moments_replay_identically_and_wipes_change_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            player = Player(make_checkpoint(Path(folder, "model.pt"), "lstm"))
            player.reset(10_001)
            for _ in range(150):
                player.step()
            saved = player.save()

            def run():
                return [(s.action, s.probs.tobytes(), s.reward, s.scene["ball_x"]) for s in
                        (player.step() for _ in range(80))]

            first = run()
            player.load(saved)
            self.assertEqual(run(), first)
            player.load(saved)
            wiped = player.step(wipe=True)
            player.load(saved)
            plain = player.step()
            self.assertFalse(np.allclose(wiped.internals["h"], plain.internals["h"]))
            player.close()

    def test_views_and_transforms_change_only_what_the_model_sees(self):
        with tempfile.TemporaryDirectory() as folder:
            player = Player(make_checkpoint(Path(folder, "model.pt"), "cnn", use_fog=False, mask_enemy=False),
                            view="fog")
            self.assertEqual((player.train_view, player.view), ("clean", "fog"))
            player.reset(10_002)
            seen = []
            for _ in range(60):
                player.step(transform=lambda obs: seen.append(obs.copy()) or scene.mirror_enemy(obs))
            self.assertTrue(all((obs[..., 14:77, 42:72] == 87).all() for obs in seen))  # fog applied
            player.close()



def analyse_for_test(path):  # module level, so spawned workers can import it
    if path.name == "bad.npz":
        raise RuntimeError("broken recording")
    return {"recording": str(path), "size": len(path.name)}


class BatchTests(unittest.TestCase):
    def test_rows_are_appended_and_finished_recordings_skipped_after_an_interruption(self):
        from pong_rl.analysis.batch import finished, run_all
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder, f"{name}.npz") for name in ("a", "b", "bad", "c")]
            output = Path(folder, "out.jsonl")
            # an earlier run finished "a" and was cut while writing the next row
            output.write_text(json.dumps({"recording": str(paths[0]), "size": -1}) + "\n" + '{"recording": "x')
            self.assertEqual(finished(output, "recording"), {str(paths[0])})
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                failures = run_all(analyse_for_test, paths, output, "recording", workers=2)
            self.assertEqual(failures, 1)
            rows = [json.loads(line) for line in output.read_text().splitlines() if line.startswith('{"recording": "/')
                    or line.startswith('{"recording": "' + folder)]
            self.assertEqual(sorted(r["recording"] for r in rows), sorted(str(p) for p in paths if p.stem != "bad"))
            self.assertEqual([r["size"] for r in rows if r["recording"] == str(paths[0])], [-1])  # not redone
            self.assertEqual(finished(output, "recording"), {str(p) for p in paths if p.stem != "bad"})


class AttentionBlockTests(unittest.TestCase):
    def test_blocked_cells_get_no_attention_and_none_changes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            player = Player(make_checkpoint(Path(folder, "model.pt"), "attn_lstm"))
            player.reset(10_000)
            for _ in range(30):
                player.step()
            blocked = np.zeros(49, bool)
            blocked[[10, 11, 17, 18]] = True
            plain = player.forward(player.obs, player.state)
            same = player.forward(player.obs, player.state, None)
            np.testing.assert_array_equal(plain[2]["attention"], same[2]["attention"])
            masked = player.forward(player.obs, player.state, blocked)
            self.assertTrue((masked[2]["attention"][:, blocked] == 0).all())
            np.testing.assert_allclose(masked[2]["attention"].sum(-1), 1.0, rtol=1e-5)
            self.assertFalse(np.allclose(masked[2]["h"], plain[2]["h"]))
            player.close()
            cnn = Player(make_checkpoint(Path(folder, "cnn.pt"), "cnn"))
            cnn.reset(10_000)
            with self.assertRaisesRegex(ValueError, "attn_lstm"):
                cnn.step(blocked=blocked)
            cnn.close()


class AttentionBlockScriptTests(unittest.TestCase):
    def test_blocks_and_a_full_run_on_a_random_model(self):
        block = load_script("attention_block")
        ball = block.block_around(3, 1).reshape(7, 7)
        self.assertEqual(ball.sum(), 6)                      # column 0 (enemy paddle) never blocked
        self.assertTrue(ball[2:5, 1:3].all() and not ball[:, 0].any())
        self.assertEqual(block.sham_row(3), 6)
        self.assertEqual(block.sham_row(0), 6)
        self.assertEqual(block.sham_row(5), 1)
        self.assertFalse((block.block_around(3, 4) & block.block_around(block.sham_row(3), 4)).any())
        record = load_script("record_rollouts")
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder, "checkpoints", "finetune", "attn_lstm", "clean_10.0M_to_fog_mask", "seed_1")
            run.mkdir(parents=True)
            path = make_checkpoint(run / "final.pt", "attn_lstm", use_fog=True, mask_enemy=True)
            recorded = record.record(str(path), 2, 10_000, saliency_every=0)
            rollouts = Path(folder, "rollouts")
            rollouts.mkdir()
            stem = rollouts / "finetune__attn_lstm__clean_10.0M_to_fog_mask__seed_1__final"
            np.savez_compressed(f"{stem}.npz", **recorded["arrays"])
            Path(f"{stem}.json").write_text(json.dumps(recorded["meta"]))
            self.assertEqual(block.recording_for(path, rollouts), Path(f"{stem}.npz"))
            row = block.run(path, episodes=1, seed=20_000, max_events=3, rollouts=rollouts)
            self.assertEqual(row["events"], 3)
            self.assertEqual(set(row["branches"]), set(block.BRANCHES))
            self.assertEqual(block.branches_for("lstm"), ("control", "control_b", "erase_ball"))
        frame = np.full((1, 84, 84), 87, np.uint8)
        frame[0, 40:42, 30] = 236
        frame[0, 20:28, 8:11] = 236
        erased = block.erase_ball(frame)
        self.assertTrue(np.isnan(scene.locate(erased[0])["ball_x"]))
        self.assertEqual(scene.locate(erased[0])["enemy_y"], 23.5)
        self.assertEqual(frame[0, 40, 30], 236)  # input untouched


class RandomControlTests(unittest.TestCase):
    def test_twin_replays_the_recording_and_feeds_the_probes(self):
        control, record, probes = (load_script(n) for n in ("random_control", "record_rollouts", "probe_memory"))
        with tempfile.TemporaryDirectory() as folder:
            path = make_checkpoint(Path(folder, "model.pt"), "lstm", use_fog=True, mask_enemy=True)
            recorded = record.record(str(path), 2, 10_000, saliency_every=0)
            stem = Path(folder, "lstm_rec")
            np.savez_compressed(f"{stem}.npz", **recorded["arrays"])
            Path(f"{stem}.json").write_text(json.dumps(recorded["meta"]))
            row = control.run(Path(f"{stem}.npz"), init_seed=0, root=Path(folder))
            with np.load(row["random"]) as data:
                memory = data["memory"]
            self.assertEqual(memory.shape, (len(recorded["arrays"]["t"]), 256))
            # make_checkpoint and the twin both initialise from torch seed 0, so here the "untrained" twin is
            # the recorded model itself: step for step its memory must equal the recorded one (alignment check)
            np.testing.assert_array_equal(memory[:, :128], recorded["arrays"]["h"])
            np.testing.assert_array_equal(memory[:, 128:], recorded["arrays"]["c"])
            attn = make_checkpoint(Path(folder, "attn.pt"), "attn_lstm", use_fog=True, mask_enemy=True)
            attn_recorded = record.record(str(attn), 1, 10_000, saliency_every=0)
            attn_stem = Path(folder, "attn_rec")
            np.savez_compressed(f"{attn_stem}.npz", **attn_recorded["arrays"])
            Path(f"{attn_stem}.json").write_text(json.dumps(attn_recorded["meta"]))
            with np.load(control.run(Path(f"{attn_stem}.npz"), init_seed=0, root=Path(folder))["random"]) as data:
                np.testing.assert_array_equal(data["memory"][:, :128], attn_recorded["arrays"]["h"])
            other = control.run(Path(f"{stem}.npz"), init_seed=1, root=Path(folder))
            with np.load(other["random"]) as data:
                self.assertFalse(np.allclose(data["memory"][:, :128].astype(np.float32),
                                             recorded["arrays"]["h"].astype(np.float32)))


class EnemyShiftTests(unittest.TestCase):
    def test_shift_moves_only_the_enemy_paddle_and_branches_replay(self):
        shift = load_script("enemy_shift")
        clean = np.full((84, 84), 87, np.uint8)
        clean[40:42, 50] = 236
        clean[20:28, 8:11] = 236
        clean[60:68, 73:76] = 236
        moved = scene.locate(shift.shift_enemy(clean, 6))
        self.assertEqual((moved["enemy_y"], moved["ball_y"], moved["agent_y"]), (29.5, 40.5, 63.5))
        self.assertEqual(scene.locate(shift.shift_enemy(clean, -4))["enemy_y"], 19.5)
        stack = np.stack([clean] * 4)
        self.assertEqual(shift.shift_enemy(stack, 3).shape, (4, 84, 84))
        # a ball passing through the enemy region (just after the hit) is not moved with the paddle
        near = clean.copy()
        near[40:42, 50] = 87
        near[50:52, 10:12] = 236
        moved = shift.shift_enemy(near, 6)
        self.assertTrue((moved[50:52, 10:12] == 236).all())
        self.assertTrue((moved[26:34, 8:11] == 236).all() and (moved[20:26, 8:11] == 87).all())
        self.assertEqual(int((moved != near).sum()), 2 * 6 * 3)  # 6 rows vacated, 6 filled, 3 columns
        with tempfile.TemporaryDirectory() as folder:
            path = make_checkpoint(Path(folder, "model.pt"), "lstm", use_fog=True, mask_enemy=False)
            row = shift.run(path, episodes=1, seed=10_000, max_events=3)
            self.assertEqual(row["events"], 3)
            self.assertEqual(set(row["return_rate"]), {f"shift_{d}" for d in shift.SHIFTS} | {"control_b"})
            masked = make_checkpoint(Path(folder, "masked.pt"), "cnn", use_fog=True, mask_enemy=True)
            with self.assertRaisesRegex(ValueError, "masked"):
                shift.run(masked, episodes=1, seed=10_000, max_events=1)


class ProbeTests(unittest.TestCase):
    def test_features_constant_on_training_games_do_not_blow_up(self):
        probe = load_script("probe_memory")
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 5))
        y = X[:, 0] * 2 + rng.normal(scale=0.1, size=200)
        X[:150, 4] = 0.0     # a dead unit on the training rows that fires on the test rows
        X[150:, 4] = 5.0
        predict = probe.fit(X[:150], y[:150], 1.0)
        self.assertLess(np.abs(predict(X[150:]) - y[150:]).mean(), 0.5)


def load_script(name):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnalysisScriptTests(unittest.TestCase):
    def test_recorder_view_and_memory_scripts_on_a_random_model(self):
        record, views, memory = (load_script(n) for n in ("record_rollouts", "view_interventions",
                                                          "memory_interventions"))
        with tempfile.TemporaryDirectory() as folder:
            path = make_checkpoint(Path(folder, "model.pt"), "lstm", use_fog=True, mask_enemy=False)
            report = evaluate_checkpoint(path, episodes=1, seed=10_000, quiet=True)
            recorded = record.record(str(path), 1, 10_000, saliency_every=10)
            arrays = recorded["arrays"]
            self.assertEqual(len(arrays["t"]), report["episodes"][0]["episode_length"])
            self.assertEqual(recorded["meta"]["returns"], [report["episodes"][0]["episode_return"]])
            self.assertEqual(arrays["h"].shape, (len(arrays["t"]), 128))
            sampled = ~np.isnan(arrays["saliency_kl"][:, 0])
            self.assertEqual(sampled.sum(), (arrays["t"] % 10 == 0).sum())
            plain = views.run(str(path), "none", 1, 10_000)
            self.assertEqual(plain["returns"], [report["episodes"][0]["episode_return"]])
            result = memory.run(str(path), 1, 10_000, max_events=3)
            self.assertEqual(len(result["events"]), 3)
            self.assertEqual(set(result["totals"]), set(memory.BRANCHES))
            for event in result["events"]:
                self.assertTrue(set(event["outcomes"].values()) <= {"returned", "missed", "undecided"})


if __name__ == "__main__":
    unittest.main()
