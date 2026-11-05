import copy
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from pong_rl.models import LSTMAgent, make_agent, zero_lstm_state
from pong_rl.ppo import replay_rallies, replay_rally, update_policy
from pong_rl.parallel_envs import ParallelEnvPool
from pong_rl.rollouts import Rally, RallyCollector
from pong_rl.train import Args


class ScriptedGame(gym.Env):
    def __init__(self, lengths, timeout_at=None):
        self.observation_space = gym.spaces.Box(0, 255, (1, 84, 84), np.uint8)
        self.action_space = gym.spaces.Discrete(6)
        self.points = np.cumsum(lengths).tolist()
        self.timeout_at = timeout_at
        self.resets = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.resets += 1
        return np.ones(self.observation_space.shape, np.uint8), {}

    def step(self, action):
        self.steps += 1
        return (np.full(self.observation_space.shape, self.steps + 1, np.uint8),
                float(self.steps in self.points), self.steps == self.points[-1],
                self.steps == self.timeout_at, {})


class MemoryCounter(LSTMAgent):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.value_memories = []

    def get_action_and_value(self, obs, state, done, action=None):
        n = len(obs)
        zeros = torch.zeros(n)
        return zeros.long(), zeros, zeros, self.get_value(obs, state, done), tuple(s+1 for s in state)

    def get_value(self, obs, state, done):
        self.value_memories.append(state[0].clone())
        return obs[:, 0, 0, 0].reshape(-1, 1)


class RallyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_five_rallies_then_game_remainder_with_paused_lanes_and_memory(self):
        children = [gym.wrappers.RecordEpisodeStatistics(ScriptedGame(lengths))
                    for lengths in ([1, 2, 1, 2, 1, 2, 1], [3, 3, 3])]
        envs = gym.vector.SyncVectorEnv([lambda child=child: child for child in children])
        try:
            collector = RallyCollector(envs, MemoryCounter(), torch.device("cpu"), 1)
            episodes = []
            first = collector.collect(1.0, 1.0, episodes.append)
            self.assertEqual([len(r) for r in first.rallies], [1, 2, 1, 2, 1, 3, 3, 3])
            self.assertEqual(first.steps, 16)
            self.assertEqual(first.completed_rallies, 8)
            self.assertEqual(first.ended_games, 1)
            self.assertEqual(children[0].unwrapped.steps, 7)  # never stepped beyond quota
            self.assertEqual(children[0].unwrapped.resets, 1)
            self.assertEqual(children[1].unwrapped.resets, 2)
            self.assertEqual(first.rallies[1].initial_state[0][0, 0, 0], 1)  # point kept memory
            self.assertEqual(first.rallies[4].returns[-1], 9)  # point bootstraps V(next_obs)=8
            self.assertEqual(first.rallies[-1].returns[-1], 1)  # game termination does not
            self.assertEqual(first.rallies[0].returns[0], 13)  # GAE crosses all five points
            second = collector.collect(1.0, 1.0, episodes.append)
            self.assertEqual([len(r) for r in second.rallies[:2]], [2, 1])
            self.assertEqual(second.completed_rallies, 5)  # two remaining + other lane's three
            self.assertEqual(second.ended_games, 2)
            self.assertEqual(second.rallies[0].initial_state[0][0, 0, 0], 7)
            self.assertEqual(second.rallies[2].initial_state[0].count_nonzero(), 0)
            self.assertEqual(episodes[1].episode_length, 10)
            self.assertEqual(collector.state[0].count_nonzero(), 0)
        finally:
            envs.close()

    def test_parallel_collection_matches_serial_pauses_timeouts_and_memory(self):
        factories = [lambda: gym.wrappers.RecordEpisodeStatistics(ScriptedGame([1,2,1,2,1,2,1])),
                     lambda: gym.wrappers.RecordEpisodeStatistics(ScriptedGame([2,5], timeout_at=4))]
        serial = gym.vector.SyncVectorEnv(factories)
        parallel = ParallelEnvPool(factories)
        try:
            collectors = [RallyCollector(envs, MemoryCounter(), torch.device("cpu"), 1)
                          for envs in (serial, parallel)]
            for _ in range(2):
                episodes = [[], []]
                expected, actual = [collector.collect(0.99, 0.95, events.append)
                                    for collector, events in zip(collectors, episodes)]
                self.assertEqual(episodes[0], episodes[1])
                self.assertEqual(expected.global_step, actual.global_step)
                self.assertEqual(expected.completed_rallies, actual.completed_rallies)
                self.assertEqual(expected.ended_games, actual.ended_games)
                self.assertEqual(len(expected.rallies), len(actual.rallies))
                for left, right in zip(expected.rallies, actual.rallies):
                    for name in ("observations", "actions", "logprobs", "values", "advantages", "returns"):
                        torch.testing.assert_close(getattr(left, name), getattr(right, name))
                    for x, y in zip(left.initial_state, right.initial_state):
                        torch.testing.assert_close(x, y)
        finally:
            serial.close()
            parallel.close()
        self.assertTrue(all(not worker.is_alive() for worker in parallel.processes))

    def test_parallel_worker_initialization_errors_are_reported(self):
        with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError"):
            ParallelEnvPool([lambda: 1 / 0])

    def test_timeout_flushes_partial_rally_and_bootstraps_before_reset(self):
        envs = gym.vector.SyncVectorEnv([lambda: gym.wrappers.RecordEpisodeStatistics(
            ScriptedGame([2, 5], timeout_at=4))])
        try:
            model = MemoryCounter()
            collector = RallyCollector(envs, model, torch.device("cpu"), 1)
            episodes = []
            batch = collector.collect(1.0, 1.0, episodes.append)
            self.assertEqual([len(r) for r in batch.rallies], [2, 2])
            self.assertEqual(batch.completed_rallies, 1)
            self.assertEqual(batch.rallies[-1].returns[-1], 5)  # final obs, not reset obs (=1)
            self.assertEqual(model.value_memories[-1][0,0,0], 4)  # pre-reset h/c
            self.assertEqual(episodes[0].outcome, "timeout")
            self.assertEqual(collector.state[0].count_nonzero(), 0)
        finally:
            envs.close()

    def test_all_game_ending_remainders_and_exact_five_boundary(self):
        for points in range(6, 11):
            with self.subTest(points=points):
                envs = gym.vector.SyncVectorEnv([lambda: gym.wrappers.RecordEpisodeStatistics(
                    ScriptedGame([1] * points))])
                try:
                    collector = RallyCollector(envs, MemoryCounter(), torch.device("cpu"), 1)
                    first = collector.collect(0.99, 0.95, lambda _: None)
                    second = collector.collect(0.99, 0.95, lambda _: None)
                    self.assertEqual(first.completed_rallies, 5)
                    self.assertEqual(second.completed_rallies, points - 5)
                    self.assertEqual(len(second.rallies), points - 5)
                    self.assertEqual(second.ended_games, 1)
                    self.assertTrue(all(len(r) == 1 for r in second.rallies))
                finally:
                    envs.close()

    def test_checkpointed_replay_matches_full_rally_outputs_and_gradients(self):
        for arch in ("lstm", "attn_lstm"):
            with self.subTest(arch=arch):
                self.check_checkpointed_replay(arch)

    def check_checkpointed_replay(self, arch):
        torch.manual_seed(7)
        spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(0, 255, (1,84,84), np.uint8),
                                 single_action_space=gym.spaces.Discrete(6))
        agent = make_agent(arch, spaces)
        reference = copy.deepcopy(agent)
        obs = torch.randint(0, 256, (35, 1, 84, 84), dtype=torch.uint8)
        actions = torch.arange(35) % 6
        initial = zero_lstm_state(1, torch.device("cpu"))
        zeros = torch.zeros(35)
        rally = Rally(obs, actions, zeros, zeros, zeros, zeros, initial)
        logp, entropy, value = replay_rally(agent, rally, torch.device("cpu"))
        _, ref_logp, ref_entropy, ref_value, _ = reference.get_action_and_value(
            obs.float(), initial, zeros, actions)
        torch.testing.assert_close(logp, ref_logp)
        torch.testing.assert_close(entropy, ref_entropy)
        torch.testing.assert_close(value, ref_value.flatten())
        # Only the last step has loss: its gradients must still cross checkpoint boundaries.
        (logp[-1] + value[-1]).backward()
        (ref_logp[-1] + ref_value[-1].sum()).backward()
        for (name, actual), expected in zip(agent.named_parameters(), reference.parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-4)

    def test_batched_replay_matches_individual_outputs_and_full_gradients(self):
        for arch in ("lstm", "attn_lstm"):
            with self.subTest(arch=arch):
                torch.manual_seed(11)
                spaces = SimpleNamespace(
                    single_observation_space=gym.spaces.Box(0, 255, (1,84,84), np.uint8),
                    single_action_space=gym.spaces.Discrete(6))
                agent = make_agent(arch, spaces)
                reference = copy.deepcopy(agent)
                rallies = []
                expected = [[], [], []]
                for length in (35, 7, 1):
                    obs = torch.randint(0, 256, (length, 1, 84, 84), dtype=torch.uint8)
                    actions = torch.arange(length) % 6
                    zeros = torch.zeros(length)
                    initial = tuple(torch.randn_like(s) for s in zero_lstm_state(1, torch.device("cpu")))
                    rallies.append(Rally(obs, actions, zeros, zeros, zeros, zeros, initial))
                    _, logp, entropy, value, _ = reference.get_action_and_value(
                        obs.float(), initial, zeros, actions)
                    for parts, output in zip(expected, (logp, entropy, value.flatten())):
                        parts.append(output)
                # Exercise retained activations as well as checkpoint recomputation.
                actual = replay_rallies(agent, rallies, torch.device("cpu"),
                                        checkpoint_threshold=2048)
                expected = tuple(torch.cat(parts) for parts in expected)
                for output, target in zip(actual, expected):
                    torch.testing.assert_close(output, target, atol=2e-6, rtol=2e-4)
                # Last real step of every lane: excludes all padding, crosses chunk 32.
                ends = torch.tensor([34, 41, 42])
                sum(output[ends].sum() for output in actual).backward()
                sum(output[ends].sum() for output in expected).backward()
                for (name, parameter), target in zip(agent.named_parameters(), reference.parameters()):
                    with self.subTest(parameter=name):
                        torch.testing.assert_close(parameter.grad, target.grad, atol=4e-6, rtol=5e-4)

    def test_both_architectures_update_from_tiny_variable_batches(self):
        for arch in ("cnn", "lstm", "attn_lstm"):
            with self.subTest(arch=arch):
                channels = 4 if arch == "cnn" else 1
                spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(0,255,(channels,84,84), np.uint8),
                                         single_action_space=gym.spaces.Discrete(6))
                agent = make_agent(arch, spaces)
                rallies = []
                for length in (1, 3):
                    obs = torch.randint(0,256,(length,channels,84,84),dtype=torch.uint8)
                    state = zero_lstm_state(1, torch.device("cpu"))
                    with torch.no_grad():
                        if arch == "cnn":
                            actions, logp, _, values = agent.get_action_and_value(obs.float())
                        else:
                            actions, logp, _, values, _ = agent.get_action_and_value(obs.float(), state, torch.zeros(length))
                    rallies.append(Rally(obs, actions, logp, values.flatten(), torch.arange(length).float()+1,
                                         torch.arange(length).float()+2, state))
                batch = SimpleNamespace(rallies=rallies, steps=4)
                before = agent.actor.weight.detach().clone()
                metrics = update_policy(agent, torch.optim.Adam(agent.parameters(), lr=1e-3), batch,
                                        Args(update_epochs=1), torch.device("cpu"))
                self.assertFalse(torch.equal(before, agent.actor.weight))
                self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
                self.assertEqual(metrics["charts/optimizer_steps"], 4 if arch == "cnn" else 2)

    def test_unequal_rallies_are_weighted_by_steps_for_both_architectures(self):
        for arch in ("cnn", "lstm", "attn_lstm"):
            with self.subTest(arch=arch):
                channels = 4 if arch == "cnn" else 1
                spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(0,255,(channels,84,84), np.uint8),
                                         single_action_space=gym.spaces.Discrete(6))
                agent = make_agent(arch, spaces)
                reference = copy.deepcopy(agent)
                rallies, ref_losses = [], []
                for length in (1, 3):
                    obs = torch.randint(0,256,(length,channels,84,84),dtype=torch.uint8)
                    actions = torch.arange(length) % 6
                    zeros = torch.zeros(length)
                    state = zero_lstm_state(1, torch.device("cpu"))
                    if arch == "cnn":
                        _, logp, _, value = reference.get_action_and_value(obs.float(), actions)
                    else:
                        _, logp, _, value, _ = reference.get_action_and_value(obs.float(), state, zeros, actions)
                    advantages = torch.arange(length).float() + 1
                    old_logp = logp.detach().clone()
                    rallies.append(Rally(obs, actions, old_logp, value.detach().flatten().clone(),
                                         advantages, zeros, state))
                    ref_losses.append(-advantages * (logp - old_logp).exp())
                ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
                torch.cat(ref_losses).mean().backward()
                ref_optimizer.step()
                args = Args(update_epochs=1, num_minibatches=1, norm_adv=False,
                            ent_coef=0, vf_coef=0, max_grad_norm=1e6)
                update_policy(agent, torch.optim.SGD(agent.parameters(), lr=0.01),
                              SimpleNamespace(rallies=rallies, steps=4), args, torch.device("cpu"))
                for actual, expected in zip(agent.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
