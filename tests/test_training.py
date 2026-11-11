import copy
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from pong_rl.models import LSTMAgent, make_agent, zero_lstm_state
from pong_rl.ppo import replay_sequences, update_policy
from pong_rl.parallel_envs import ParallelEnvPool
from pong_rl.rollouts import RolloutCollector, Sequence
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


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_every_game_plays_num_steps_and_sequences_split_only_at_game_ends(self):
        children = [gym.wrappers.RecordEpisodeStatistics(ScriptedGame(lengths, timeout_at))
                    for lengths, timeout_at in (([3, 3], None), ([4, 6], 5))]
        envs = gym.vector.SyncVectorEnv([lambda child=child: child for child in children])
        try:
            collector = RolloutCollector(envs, MemoryCounter(), torch.device("cpu"), 1, num_steps=8)
            episodes = []
            batch = collector.collect(1.0, 1.0, episodes.append)
            self.assertEqual([len(s) for s in batch.sequences], [5, 6, 2, 3])  # timeout, end, two tails
            self.assertEqual(batch.steps, 16)
            self.assertEqual(batch.points, 3)
            self.assertEqual(batch.ended_games, 2)
            self.assertEqual(batch.global_step, 16)
            self.assertEqual([e.truncated for e in episodes], [True, False])
            self.assertEqual(batch.sequences[0].returns[-1], 6)  # timeout bootstraps V(final obs)=6
            self.assertEqual(batch.sequences[0].returns[0], 7)   # and GAE crosses its point
            self.assertEqual(batch.sequences[1].returns[0], 2)   # termination: no bootstrap
            self.assertEqual(batch.sequences[2].returns[-1], 3)  # rollout end bootstraps V(obs)=3
            self.assertEqual(batch.sequences[2].initial_state[0].count_nonzero(), 0)  # memory reset
            self.assertEqual(batch.sequences[3].initial_state[0].count_nonzero(), 0)
            self.assertEqual(collector.state[0][0, 0, 0], 2)  # memory continues into next batch
            self.assertEqual(children[0].unwrapped.resets, 2)
            # Each step stores the observation it acted on (ScriptedGame shows step t as value t).
            game = batch.sequences[1]
            self.assertEqual(game.observations[:, 0, 0, 0].tolist(), [1, 2, 3, 4, 5, 6])
            torch.testing.assert_close(game.values, torch.arange(1.0, 7.0))
            tail = batch.sequences[2]  # starts with the reset observation
            self.assertEqual(tail.observations[:, 0, 0, 0].tolist(), [1, 2])
        finally:
            envs.close()

    def test_points_keep_memory_and_memory_continues_across_batches(self):
        envs = gym.vector.SyncVectorEnv([lambda: gym.wrappers.RecordEpisodeStatistics(ScriptedGame([2] * 10))])
        try:
            collector = RolloutCollector(envs, MemoryCounter(), torch.device("cpu"), 1, num_steps=3)
            first = collector.collect(1.0, 1.0, lambda _: None)
            second = collector.collect(1.0, 1.0, lambda _: None)
            self.assertEqual([len(s) for s in first.sequences + second.sequences], [3, 3])
            self.assertEqual((first.points, second.points), (1, 2))
            self.assertEqual(second.sequences[0].initial_state[0][0, 0, 0], 3)  # not reset at points
            self.assertEqual(first.sequences[0].returns[-1], 4)  # rollout end bootstraps V(obs)=4
            self.assertEqual(envs.envs[0].unwrapped.resets, 1)
        finally:
            envs.close()

    def test_timeout_bootstraps_from_final_observation_with_pre_reset_memory(self):
        envs = gym.vector.SyncVectorEnv([lambda: gym.wrappers.RecordEpisodeStatistics(
            ScriptedGame([2, 5], timeout_at=4))])
        try:
            model = MemoryCounter()
            collector = RolloutCollector(envs, model, torch.device("cpu"), 1, num_steps=4)
            episodes = []
            batch = collector.collect(1.0, 1.0, episodes.append)
            self.assertEqual([len(s) for s in batch.sequences], [4])
            self.assertEqual(batch.points, 1)
            self.assertEqual(batch.sequences[0].returns[-1], 5)  # final obs, not reset obs (=1)
            self.assertEqual(model.value_memories[-1][0, 0, 0], 4)  # pre-reset h/c
            self.assertEqual(episodes[0].outcome, "timeout")
            self.assertEqual(collector.state[0].count_nonzero(), 0)
        finally:
            envs.close()

    def test_replaying_each_sequence_reproduces_collection_for_recurrent_agents(self):
        for arch in ("lstm", "attn_lstm", "attn_lstm_wide"):
            with self.subTest(arch=arch):
                torch.manual_seed(3)
                children = [gym.wrappers.RecordEpisodeStatistics(ScriptedGame(lengths, timeout_at))
                            for lengths, timeout_at in (([2, 3], None), ([4, 9], 6), ([20], None))]
                envs = gym.vector.SyncVectorEnv([lambda child=child: child for child in children])
                try:
                    agent = make_agent(arch, envs)
                    collector = RolloutCollector(envs, agent, torch.device("cpu"), 1, num_steps=7)
                    for _ in range(3):  # resets inside rollouts and memory carried across batches
                        batch = collector.collect(0.99, 0.95, lambda _: None)
                        with torch.no_grad():
                            for sequence in batch.sequences:
                                logp, _, value = replay_sequences(agent, [sequence], torch.device("cpu"))
                                torch.testing.assert_close(logp, sequence.logprobs, atol=1e-5, rtol=1e-5)
                                torch.testing.assert_close(value, sequence.values, atol=1e-5, rtol=1e-5)
                finally:
                    envs.close()

    def test_no_done_flags_match_all_zero_flags_including_gradients(self):
        for arch in ("lstm", "attn_lstm", "attn_lstm_wide"):
            with self.subTest(arch=arch):
                torch.manual_seed(5)
                spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(0, 255, (1,84,84), np.uint8),
                                         single_action_space=gym.spaces.Discrete(6))
                agent = make_agent(arch, spaces)
                reference = copy.deepcopy(agent)
                obs = torch.randint(0, 256, (6 * 3, 1, 84, 84)).float()
                state = tuple(torch.randn(1, 3, 128) for _ in range(2))
                hidden, (h, c) = agent.get_states(obs, state, None)
                ref_hidden, (ref_h, ref_c) = reference.get_states(obs, state, torch.zeros(18))
                for actual, expected in ((hidden, ref_hidden), (h, ref_h), (c, ref_c)):
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                (hidden.sum() + h.sum()).backward()
                (ref_hidden.sum() + ref_h.sum()).backward()
                for (name, parameter), target in zip(agent.named_parameters(), reference.parameters()):
                    with self.subTest(parameter=name):
                        torch.testing.assert_close(parameter.grad, target.grad, atol=1e-5, rtol=1e-4)

    def test_target_kl_stops_the_remaining_epochs(self):
        spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(0,255,(4,84,84), np.uint8),
                                 single_action_space=gym.spaces.Discrete(6))
        torch.manual_seed(2)
        agent = make_agent("cnn", spaces)
        obs = torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8)
        with torch.no_grad():
            actions, logp, _, values = agent.get_action_and_value(obs.float())
        sequence = Sequence(obs, actions, logp, values.flatten(), torch.tensor([1.0, -1.0, 2.0, -2.0]),
                            torch.zeros(4), zero_lstm_state(1, torch.device("cpu")))
        batch = SimpleNamespace(sequences=[sequence], steps=4)
        metrics = update_policy(agent, torch.optim.Adam(agent.parameters(), lr=1e-2), batch,
                                Args(update_epochs=4, num_minibatches=2, target_kl=1e-12), torch.device("cpu"))
        self.assertEqual(metrics["charts/optimizer_steps"], 2)  # one epoch of two minibatches

    def test_parallel_collection_matches_serial_resets_timeouts_and_memory(self):
        factories = [lambda: gym.wrappers.RecordEpisodeStatistics(ScriptedGame([1,2,1,2,1,2,1])),
                     lambda: gym.wrappers.RecordEpisodeStatistics(ScriptedGame([2,5], timeout_at=4))]
        serial = gym.vector.SyncVectorEnv(factories, autoreset_mode=gym.vector.AutoresetMode.DISABLED)
        parallel = ParallelEnvPool(factories)
        try:
            collectors = [RolloutCollector(envs, MemoryCounter(), torch.device("cpu"), 1, num_steps=5)
                          for envs in (serial, parallel)]
            for _ in range(3):
                episodes = [[], []]
                expected, actual = [collector.collect(0.99, 0.95, events.append)
                                    for collector, events in zip(collectors, episodes)]
                self.assertEqual(episodes[0], episodes[1])
                self.assertEqual(expected.global_step, actual.global_step)
                self.assertEqual(expected.points, actual.points)
                self.assertEqual(expected.ended_games, actual.ended_games)
                self.assertEqual(len(expected.sequences), len(actual.sequences))
                for left, right in zip(expected.sequences, actual.sequences):
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

    def test_checkpointed_replay_matches_full_sequence_outputs_and_gradients(self):
        for arch in ("lstm", "attn_lstm", "attn_lstm_wide"):
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
        sequence = Sequence(obs, actions, zeros, zeros, zeros, zeros, initial)
        logp, entropy, value = replay_sequences(agent, [sequence], torch.device("cpu"))
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
        for arch in ("lstm", "attn_lstm", "attn_lstm_wide"):
            with self.subTest(arch=arch):
                torch.manual_seed(11)
                spaces = SimpleNamespace(
                    single_observation_space=gym.spaces.Box(0, 255, (1,84,84), np.uint8),
                    single_action_space=gym.spaces.Discrete(6))
                agent = make_agent(arch, spaces)
                reference = copy.deepcopy(agent)
                sequences = []
                expected = [[], [], []]
                for length in (35, 7, 1):
                    obs = torch.randint(0, 256, (length, 1, 84, 84), dtype=torch.uint8)
                    actions = torch.arange(length) % 6
                    zeros = torch.zeros(length)
                    initial = tuple(torch.randn_like(s) for s in zero_lstm_state(1, torch.device("cpu")))
                    sequences.append(Sequence(obs, actions, zeros, zeros, zeros, zeros, initial))
                    _, logp, entropy, value, _ = reference.get_action_and_value(
                        obs.float(), initial, zeros, actions)
                    for parts, output in zip(expected, (logp, entropy, value.flatten())):
                        parts.append(output)
                # Exercise retained activations as well as checkpoint recomputation.
                actual = replay_sequences(agent, sequences, torch.device("cpu"),
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

    def test_all_architectures_update_from_tiny_variable_batches(self):
        for arch in ("cnn", "lstm", "attn_lstm", "attn_lstm_wide"):
            with self.subTest(arch=arch):
                channels = 4 if arch == "cnn" else 1
                spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(0,255,(channels,84,84), np.uint8),
                                         single_action_space=gym.spaces.Discrete(6))
                agent = make_agent(arch, spaces)
                sequences = []
                for length in (1, 3):
                    obs = torch.randint(0,256,(length,channels,84,84),dtype=torch.uint8)
                    state = zero_lstm_state(1, torch.device("cpu"))
                    with torch.no_grad():
                        if arch == "cnn":
                            actions, logp, _, values = agent.get_action_and_value(obs.float())
                        else:
                            actions, logp, _, values, _ = agent.get_action_and_value(obs.float(), state, torch.zeros(length))
                    sequences.append(Sequence(obs, actions, logp, values.flatten(), torch.arange(length).float()+1,
                                              torch.arange(length).float()+2, state))
                batch = SimpleNamespace(sequences=sequences, steps=4)
                before = agent.actor.weight.detach().clone()
                metrics = update_policy(agent, torch.optim.Adam(agent.parameters(), lr=1e-3), batch,
                                        Args(update_epochs=1), torch.device("cpu"))
                self.assertFalse(torch.equal(before, agent.actor.weight))
                self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
                self.assertEqual(metrics["charts/optimizer_steps"], 4 if arch == "cnn" else 2)

    def test_unequal_sequences_are_weighted_by_steps_for_all_architectures(self):
        for arch in ("cnn", "lstm", "attn_lstm", "attn_lstm_wide"):
            with self.subTest(arch=arch):
                channels = 4 if arch == "cnn" else 1
                spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(0,255,(channels,84,84), np.uint8),
                                         single_action_space=gym.spaces.Discrete(6))
                agent = make_agent(arch, spaces)
                reference = copy.deepcopy(agent)
                sequences, ref_losses = [], []
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
                    sequences.append(Sequence(obs, actions, old_logp, value.detach().flatten().clone(),
                                              advantages, zeros, state))
                    ref_losses.append(-advantages * (logp - old_logp).exp())
                ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
                torch.cat(ref_losses).mean().backward()
                ref_optimizer.step()
                args = Args(update_epochs=1, num_minibatches=1, norm_adv=False,
                            ent_coef=0, vf_coef=0, max_grad_norm=1e6)
                update_policy(agent, torch.optim.SGD(agent.parameters(), lr=0.01),
                              SimpleNamespace(sequences=sequences, steps=4), args, torch.device("cpu"))
                for actual, expected in zip(agent.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
