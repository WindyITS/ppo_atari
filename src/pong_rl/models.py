import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

LSTM_HIDDEN_SIZE = 128

# (h, c) of the LSTM, each of shape (num_layers=1, batch, LSTM_HIDDEN_SIZE)
LSTMState = tuple[torch.Tensor, torch.Tensor]


def zero_lstm_state(batch_size: int, device: torch.device) -> LSTMState:
    """All-zero (h, c) state, i.e. an empty memory (also useful for memory-wipe tests)."""
    shape = (1, batch_size, LSTM_HIDDEN_SIZE)
    return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


def layer_init(layer: nn.Conv2d | nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def nature_cnn(in_channels: int) -> nn.Sequential:
    """Nature DQN encoder: (B, in_channels, 84, 84) -> (B, 512)."""
    return nn.Sequential(
        layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(64 * 7 * 7, 512)),
        nn.ReLU(),
    )


class Agent(nn.Module):
    """Nature CNN Actor-Critic taking 4 stacked 84x84 frames."""

    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        assert envs.single_observation_space.shape is not None
        self.network = nature_cnn(envs.single_observation_space.shape[0])
        assert isinstance(envs.single_action_space, gym.spaces.Discrete)
        self.actor = layer_init(nn.Linear(512, int(envs.single_action_space.n)), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the 512-dim latent representation (useful for linear probing)."""
        return self.network(x / 255.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.get_hidden(x))

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        hidden = self.get_hidden(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class LSTMAgent(nn.Module):
    """Nature CNN on a single 84x84 frame, followed by an LSTM that carries memory across steps.

    Adapted from CleanRL's ppo_atari_lstm.py. All methods take observations as a flattened,
    time-major sequence: x[t * batch + b] is step t of sequence b, where batch = lstm_state[0].shape[1].
    done[t * batch + b] = 1 means that observation is the first of a new episode, so the LSTM
    state is reset to zero before processing it.
    """

    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        assert envs.single_observation_space.shape is not None
        self.network = nature_cnn(envs.single_observation_space.shape[0])
        self.lstm = nn.LSTM(512, LSTM_HIDDEN_SIZE)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        assert isinstance(envs.single_action_space, gym.spaces.Discrete)
        self.actor = layer_init(nn.Linear(LSTM_HIDDEN_SIZE, int(envs.single_action_space.n)), std=0.01)
        self.critic = layer_init(nn.Linear(LSTM_HIDDEN_SIZE, 1), std=1.0)

    def get_states(
        self, x: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor
    ) -> tuple[torch.Tensor, LSTMState]:
        """Returns the per-step LSTM outputs (T * batch, hidden_size), useful for linear probing,
        and the (h, c) state after the last step."""
        assert done.numel() == x.shape[0], "need one done flag per observation"
        features = self.network(x / 255.0)
        batch_size = lstm_state[0].shape[1]
        features = features.reshape((-1, batch_size, self.lstm.input_size))
        done = done.float().reshape((-1, batch_size))
        outputs = []
        for f, d in zip(features, done):
            keep = (1.0 - d).view(1, -1, 1)
            out, lstm_state = self.lstm(f.unsqueeze(0), (keep * lstm_state[0], keep * lstm_state[1]))
            outputs.append(out)
        return torch.flatten(torch.cat(outputs), 0, 1), lstm_state

    def get_value(self, x: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        lstm_state: LSTMState,
        done: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state
