import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Conv2d | nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Nature CNN Actor-Critic taking 4 stacked 84x84 frames."""

    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
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
