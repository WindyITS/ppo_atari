import io
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from pong_rl.models import Agent, AttentionLSTMAgent, LSTMAgent, WideAttentionLSTMAgent, make_agent, zero_lstm_state
from pong_rl.train import FRAME_STACK, LEARNING_RATE, REINIT_MODULES


def spaces(channels=1):
    return SimpleNamespace(single_observation_space=gym.spaces.Box(0, 255, (channels, 84, 84), np.uint8),
                           single_action_space=gym.spaces.Discrete(6))


def frames(steps, batch):
    return torch.randint(0, 256, (steps * batch, 1, 84, 84)).float()


class AttentionAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(3)
        self.agent = AttentionLSTMAgent(spaces())

    def test_weights_are_distributions_over_the_49_cells(self):
        x = frames(4, 3)
        hidden, (h, c), weights = self.agent.get_states_and_attention(
            x, zero_lstm_state(3, torch.device("cpu")), torch.zeros(12))
        self.assertEqual(hidden.shape, (12, 128))
        self.assertEqual((h.shape, c.shape), ((1, 3, 128), (1, 3, 128)))
        self.assertEqual(weights.shape, (12, 2, 49))
        self.assertTrue(torch.all(weights >= 0))
        torch.testing.assert_close(weights.sum(-1), torch.ones(12, 2))

    def test_one_call_over_a_sequence_equals_step_by_step_calls_with_resets(self):
        steps, batch = 6, 2
        x = frames(steps, batch)
        done = torch.zeros(steps * batch)
        done[3 * batch + 0] = 1  # lane 0 starts a new episode at step 3
        start = (torch.randn(1, batch, 128), torch.randn(1, batch, 128))
        with torch.no_grad():
            hidden, final, weights = self.agent.get_states_and_attention(x, start, done)
            state, step_hidden, step_weights = start, [], []
            for t in range(steps):
                rows = slice(t * batch, (t + 1) * batch)
                out, state, w = self.agent.get_states_and_attention(x[rows], state, done[rows])
                step_hidden.append(out)
                step_weights.append(w)
        torch.testing.assert_close(hidden, torch.cat(step_hidden))
        torch.testing.assert_close(weights, torch.cat(step_weights))
        for actual, expected in zip(final, state):
            torch.testing.assert_close(actual, expected)

    def test_reset_flag_matches_starting_from_empty_memory(self):
        x = frames(3, 1)
        with torch.no_grad():
            carried = (torch.randn(1, 1, 128), torch.randn(1, 1, 128))
            reset, _ = self.agent.get_states(x, carried, torch.tensor([1.0, 0.0, 0.0]))
            fresh, _ = self.agent.get_states(x, zero_lstm_state(1, torch.device("cpu")), torch.zeros(3))
        torch.testing.assert_close(reset, fresh)

    def test_memory_decides_where_to_look(self):
        x = frames(1, 1)
        with torch.no_grad():
            _, _, empty = self.agent.get_states_and_attention(x, zero_lstm_state(1, torch.device("cpu")), torch.zeros(1))
            _, _, remembered = self.agent.get_states_and_attention(
                x, (torch.randn(1, 1, 128) * 3, torch.zeros(1, 1, 128)), torch.zeros(1))
        self.assertFalse(torch.allclose(empty, remembered))

    def test_attention_is_not_stuck_uniform_with_small_initial_memory(self):
        # Regression: without query-key normalisation, an LSTM output of typical early size
        # (~0.04) gave scores near 0 and exactly uniform weights that never became selective.
        x = frames(1, 8)
        with torch.no_grad():
            small_memory = (torch.randn(1, 8, 128) * 0.04, torch.zeros(1, 8, 128))
            _, _, weights = self.agent.get_states_and_attention(x, small_memory, torch.zeros(8))
        entropy = -(weights * weights.log()).sum(-1)
        self.assertLess(entropy.mean().item(), np.log(49) - 0.05)

    def test_the_readout_is_the_only_path_from_the_frame_to_the_memory(self):
        with torch.no_grad():
            self.agent.values.weight.zero_()
            self.agent.values.bias.zero_()
            state = (torch.randn(1, 2, 128), torch.randn(1, 2, 128))
            first, _ = self.agent.get_states(frames(2, 2), state, torch.zeros(4))
            second, _ = self.agent.get_states(frames(2, 2), state, torch.zeros(4))
        torch.testing.assert_close(first, second)

    def test_the_cell_position_reaches_the_keys_and_values(self):
        positions = self.agent.positions
        self.assertEqual(positions.shape, (49, 2))
        # Row-major: cell 0 is the top-left (-1, -1), cell 6 the top-right (1, -1), cell 48 the bottom-right.
        torch.testing.assert_close(positions[0], torch.tensor([-1.0, -1.0]))
        torch.testing.assert_close(positions[6], torch.tensor([1.0, -1.0]))
        torch.testing.assert_close(positions[48], torch.tensor([1.0, 1.0]))
        self.assertEqual(self.agent.keys.in_features, 66)
        self.assertEqual(self.agent.values.in_features, 66)

    def test_checkpoint_round_trip_is_strict_and_position_buffer_is_not_saved(self):
        state = self.agent.state_dict()
        self.assertNotIn("positions", state)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored = make_agent("attn_lstm", spaces())
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        x = frames(2, 1)
        with torch.no_grad():
            expected, _ = self.agent.get_states(x, zero_lstm_state(1, torch.device("cpu")), torch.zeros(2))
            actual, _ = restored.get_states(x, zero_lstm_state(1, torch.device("cpu")), torch.zeros(2))
        torch.testing.assert_close(actual, expected)

    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in self.agent.parameters()), 225_959)


class WideAttentionAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(3)
        self.agent = make_agent("attn_lstm_wide", spaces())

    def test_size_matches_the_lstm_agent_with_the_same_memory(self):
        wide = sum(p.numel() for p in self.agent.parameters())
        lstm = sum(p.numel() for p in LSTMAgent(spaces()).parameters())
        self.assertEqual(wide, 1_984_551)
        self.assertLess(abs(wide - lstm) / lstm, 0.015)
        self.assertEqual(self.agent.lstm.hidden_size, 128)

    def test_eight_heads_and_a_512_unit_relu_layer_feed_the_lstm(self):
        self.assertIsInstance(self.agent, WideAttentionLSTMAgent)
        self.assertEqual((self.agent.heads, self.agent.key_size, self.agent.value_size), (8, 64, 320))
        linear, relu = self.agent.readout_mlp
        self.assertEqual((linear.in_features, linear.out_features), (8 * 320, 512))
        self.assertIsInstance(relu, torch.nn.ReLU)
        self.assertEqual(self.agent.lstm.input_size, 512)

    def test_attention_is_a_distribution_per_head_and_blocking_removes_cells(self):
        x = frames(3, 2)
        blocked = torch.zeros(6, 49, dtype=torch.bool)
        blocked[:, :10] = True
        with torch.no_grad():
            hidden, _, weights = self.agent.get_states_and_attention(
                x, zero_lstm_state(2, torch.device("cpu")), torch.zeros(6))
            _, _, masked = self.agent.get_states_and_attention(
                x, zero_lstm_state(2, torch.device("cpu")), torch.zeros(6), blocked=blocked)
        self.assertEqual((hidden.shape, weights.shape), ((6, 128), (6, 8, 49)))
        torch.testing.assert_close(weights.sum(-1), torch.ones(6, 8))
        self.assertTrue(torch.all(masked[:, :, :10] == 0))
        torch.testing.assert_close(masked.sum(-1), torch.ones(6, 8))

    def test_one_call_over_a_sequence_equals_step_by_step_calls_with_resets(self):
        steps, batch = 5, 2
        x = frames(steps, batch)
        done = torch.zeros(steps * batch)
        done[2 * batch + 1] = 1
        start = (torch.randn(1, batch, 128), torch.randn(1, batch, 128))
        with torch.no_grad():
            hidden, final = self.agent.get_states(x, start, done)
            state, outputs = start, []
            for t in range(steps):
                rows = slice(t * batch, (t + 1) * batch)
                out, state = self.agent.get_states(x[rows], state, done[rows])
                outputs.append(out)
        torch.testing.assert_close(hidden, torch.cat(outputs))
        for actual, expected in zip(final, state):
            torch.testing.assert_close(actual, expected)

    def test_gradients_reach_every_parameter(self):
        _, logp, entropy, value, _ = self.agent.get_action_and_value(
            frames(4, 2), (torch.randn(1, 2, 128), torch.randn(1, 2, 128)), torch.zeros(8))
        (logp.sum() + entropy.sum() + value.sum()).backward()
        for name, parameter in self.agent.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_checkpoint_round_trip_is_strict_and_not_interchangeable_with_attn_lstm(self):
        state = self.agent.state_dict()
        restored = make_agent("attn_lstm_wide", spaces())
        restored.load_state_dict(state, strict=True)
        x = frames(2, 1)
        with torch.no_grad():
            expected, _ = self.agent.get_states(x, zero_lstm_state(1, torch.device("cpu")), torch.zeros(2))
            actual, _ = restored.get_states(x, zero_lstm_state(1, torch.device("cpu")), torch.zeros(2))
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(RuntimeError):
            make_agent("attn_lstm", spaces()).load_state_dict(state, strict=True)

    def test_training_defaults_match_attn_lstm(self):
        self.assertEqual(FRAME_STACK["attn_lstm_wide"], FRAME_STACK["attn_lstm"])
        self.assertEqual(LEARNING_RATE["attn_lstm_wide"], LEARNING_RATE["attn_lstm"])
        self.assertIn("readout_mlp", REINIT_MODULES)


class ExistingCheckpointCompatibilityTests(unittest.TestCase):
    def test_attn_lstm_weight_names_are_unchanged(self):
        self.assertEqual(list(AttentionLSTMAgent(spaces()).state_dict()),
                         ["encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "encoder.2.bias",
                          "encoder.4.weight", "encoder.4.bias", "keys.weight", "keys.bias", "values.weight",
                          "values.bias", "query.weight", "query.bias", "lstm.weight_ih_l0", "lstm.weight_hh_l0",
                          "lstm.bias_ih_l0", "lstm.bias_hh_l0", "actor.weight", "actor.bias", "critic.weight",
                          "critic.bias"])

    def test_cnn_and_lstm_weight_names_and_sizes_are_unchanged(self):
        cnn = Agent(spaces(4))
        lstm = LSTMAgent(spaces(1))
        encoder = ["network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias",
                   "network.4.weight", "network.4.bias", "network.7.weight", "network.7.bias"]
        self.assertEqual(list(cnn.state_dict()),
                         encoder + ["actor.weight", "actor.bias", "critic.weight", "critic.bias"])
        self.assertEqual(list(lstm.state_dict()),
                         encoder + ["lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0",
                                    "actor.weight", "actor.bias", "critic.weight", "critic.bias"])
        self.assertEqual(sum(p.numel() for p in cnn.parameters()), 1_687_719)
        self.assertEqual(sum(p.numel() for p in lstm.parameters()), 2_007_591)


if __name__ == "__main__":
    unittest.main()
