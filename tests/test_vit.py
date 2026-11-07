import io
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from pong_rl.models import ViTAgent, ViTEncoder, ViTLSTMAgent, make_agent, zero_lstm_state
from unittest.mock import patch

from pong_rl.train import FRAME_STACK, Args, parse_args


def spaces(channels):
    return SimpleNamespace(single_observation_space=gym.spaces.Box(0, 255, (channels, 84, 84), np.uint8),
                           single_action_space=gym.spaces.Discrete(6))


def frames(n, channels):
    return torch.randint(0, 256, (n, channels, 84, 84)).float()


def count(model):
    return sum(p.numel() for p in model.parameters())


class ViTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(3)

    def test_frame_stacks_match_the_cnn_and_lstm_counterparts(self):
        self.assertEqual(FRAME_STACK["vit"], FRAME_STACK["cnn"])
        self.assertEqual(FRAME_STACK["vit_lstm"], FRAME_STACK["lstm"])

    def test_learning_rate_defaults_per_architecture_and_can_be_overridden(self):
        expected = {"cnn": 1e-3, "lstm": 1e-3, "attn_lstm": 1e-3, "vit": 2.5e-4, "vit_lstm": 2.5e-4}
        for arch, lr in expected.items():
            with self.subTest(arch=arch):
                self.assertEqual(Args(arch=arch).learning_rate, lr)
                with patch("sys.argv", ["pong-rl", "--arch", arch]):
                    self.assertEqual(parse_args().learning_rate, lr)
        with patch("sys.argv", ["pong-rl", "--arch", "vit", "--learning-rate", "5e-4"]):
            self.assertEqual(parse_args().learning_rate, 5e-4)
        self.assertEqual(Args(arch="vit", learning_rate=1e-3).learning_rate, 1e-3)

    def test_sizes_match_the_cnn_and_lstm_within_3_percent(self):
        cnn, lstm = count(make_agent("cnn", spaces(4))), count(make_agent("lstm", spaces(1)))
        vit, vit_lstm = count(make_agent("vit", spaces(4))), count(make_agent("vit_lstm", spaces(1)))
        self.assertEqual((cnn, lstm), (1_687_719, 2_007_591))
        self.assertEqual((vit, vit_lstm), (1_733_223, 1_990_119))
        self.assertLess(abs(vit / cnn - 1), 0.03)
        self.assertLess(abs(vit_lstm / lstm - 1), 0.03)

    def test_vit_lstm_shares_the_lstm_agents_memory_and_heads(self):
        vit_lstm, lstm = make_agent("vit_lstm", spaces(1)), make_agent("lstm", spaces(1))
        for name in ("lstm", "actor", "critic"):
            ours = {k: v.shape for k, v in getattr(vit_lstm, name).state_dict().items()}
            theirs = {k: v.shape for k, v in getattr(lstm, name).state_dict().items()}
            self.assertEqual(ours, theirs, name)

    def test_encoder_outputs_512_relu_features_and_a_cls_attention_distribution(self):
        encoder = ViTEncoder(4)
        features, weights = encoder(frames(3, 4) / 255.0, need_weights=True)
        self.assertEqual(features.shape, (3, 512))
        self.assertTrue(torch.all(features >= 0))
        self.assertEqual(weights.shape, (3, 5, 49))
        self.assertTrue(torch.all(weights >= 0))
        # The CLS row also attends to itself, so the 49 patch weights sum to at most 1.
        self.assertTrue(torch.all(weights.sum(-1) <= 1 + 1e-5))
        torch.testing.assert_close(encoder(frames(3, 4) / 255.0).shape, features.shape)

    def test_need_weights_does_not_change_the_features(self):
        encoder = ViTEncoder(1)
        x = frames(4, 1) / 255.0
        with torch.no_grad():
            torch.testing.assert_close(encoder(x), encoder(x, need_weights=True)[0], atol=1e-5, rtol=1e-5)

    def test_every_patch_and_position_affects_the_output(self):
        encoder = ViTEncoder(1)
        x = frames(1, 1) / 255.0
        base = encoder(x)
        for row, col in ((0, 0), (3, 3), (6, 6)):
            changed = x.clone()
            changed[..., row * 12:(row + 1) * 12, col * 12:(col + 1) * 12] = 0.5
            self.assertFalse(torch.allclose(encoder(changed), base), (row, col))
        # Position embeddings break permutation symmetry: swapping two patches changes the output.
        swapped = x.clone()
        swapped[..., :12, :12], swapped[..., :12, 12:24] = x[..., :12, 12:24], x[..., :12, :12]
        self.assertFalse(torch.allclose(encoder(swapped), base))

    def test_vit_agent_outputs(self):
        agent = ViTAgent(spaces(4))
        action, logp, entropy, value = agent.get_action_and_value(frames(5, 4))
        self.assertEqual((action.shape, logp.shape, entropy.shape, value.shape), ((5,), (5,), (5,), (5, 1)))
        hidden, weights = agent.get_hidden_and_attention(frames(5, 4))
        self.assertEqual((hidden.shape, weights.shape), ((5, 512), (5, 5, 49)))
        # Near-uniform policy at initialisation, as for the CNN.
        self.assertGreater(entropy.mean().item(), np.log(6) - 0.01)

    def test_vit_lstm_sequence_equals_step_by_step_calls_with_resets(self):
        agent = ViTLSTMAgent(spaces(1))
        steps, batch = 5, 2
        x = frames(steps * batch, 1)
        done = torch.zeros(steps * batch)
        done[2 * batch + 1] = 1
        start = (torch.randn(1, batch, 128), torch.randn(1, batch, 128))
        with torch.no_grad():
            hidden, final, weights = agent.get_states_and_attention(x, start, done)
            state, outs = start, []
            for t in range(steps):
                rows = slice(t * batch, (t + 1) * batch)
                out, state = agent.get_states(x[rows], state, done[rows])
                outs.append(out)
        self.assertEqual(weights.shape, (steps * batch, 5, 49))
        torch.testing.assert_close(hidden, torch.cat(outs), atol=1e-5, rtol=1e-5)
        for actual, expected in zip(final, state):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_vit_lstm_memory_changes_the_output(self):
        agent = ViTLSTMAgent(spaces(1))
        x = frames(1, 1)
        with torch.no_grad():
            empty, _ = agent.get_states(x, zero_lstm_state(1, torch.device("cpu")), torch.zeros(1))
            remembered, _ = agent.get_states(x, (torch.randn(1, 1, 128), torch.randn(1, 1, 128)), torch.zeros(1))
        self.assertFalse(torch.allclose(empty, remembered))

    def test_gradients_reach_every_parameter(self):
        for arch, channels in (("vit", 4), ("vit_lstm", 1)):
            with self.subTest(arch=arch):
                agent = make_agent(arch, spaces(channels))
                if arch == "vit":
                    _, logp, entropy, value = agent.get_action_and_value(frames(3, 4))
                else:
                    _, logp, entropy, value, _ = agent.get_action_and_value(
                        frames(6, 1), zero_lstm_state(2, torch.device("cpu")), torch.zeros(6))
                (logp.sum() + entropy.sum() + value.sum()).backward()
                missing = [n for n, p in agent.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
                self.assertEqual(missing, [])

    def test_checkpoint_round_trip_is_strict(self):
        for arch, channels in (("vit", 4), ("vit_lstm", 1)):
            with self.subTest(arch=arch):
                agent = make_agent(arch, spaces(channels))
                buffer = io.BytesIO()
                torch.save(agent.state_dict(), buffer)
                buffer.seek(0)
                restored = make_agent(arch, spaces(channels))
                restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
                x = frames(2, channels)
                with torch.no_grad():
                    if arch == "vit":
                        torch.testing.assert_close(restored.get_hidden(x), agent.get_hidden(x))
                    else:
                        state = zero_lstm_state(1, torch.device("cpu"))
                        torch.testing.assert_close(restored.get_states(x, state, torch.zeros(2))[0],
                                                   agent.get_states(x, state, torch.zeros(2))[0])

    def test_existing_cnn_and_lstm_checkpoint_keys_are_unchanged(self):
        self.assertIn("network.7.weight", make_agent("cnn", spaces(4)).state_dict())
        self.assertIn("network.7.weight", make_agent("lstm", spaces(1)).state_dict())


if __name__ == "__main__":
    unittest.main()
