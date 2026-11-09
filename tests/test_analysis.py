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
