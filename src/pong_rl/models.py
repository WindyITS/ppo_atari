import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

LSTM_HIDDEN_SIZE = 128
# Attention + LSTM: a few heads, each reading the 7x7 conv map with its own query.
ATTENTION_HEADS = 2
ATTENTION_KEY_SIZE = 32
ATTENTION_VALUE_SIZE = 64
CONV_GRID = 7  # the Nature CNN's last conv layer outputs a 7x7 map for 84x84 inputs
CONV_CHANNELS = 64
# Vision transformer: 12x12 patches of the 84x84 frame give the same 7x7 grid (49 tokens) as the CNN.
# Sized so vit (4 frames) and vit_lstm (1 frame + LSTM) are within 3% of cnn and lstm respectively.
VIT_PATCH = 12
VIT_WIDTH = 160
VIT_DEPTH = 5
VIT_HEADS = 5  # 32 dimensions per head
VIT_MLP_RATIO = 4
FEATURES = 512  # latent size every encoder hands to the heads or the LSTM

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


def nature_convs(in_channels: int) -> list[nn.Module]:
    """Nature DQN convolutions: (B, in_channels, 84, 84) -> (B, 64, 7, 7)."""
    return [
        layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, CONV_CHANNELS, 3, stride=1)),
        nn.ReLU(),
    ]


def nature_cnn(in_channels: int) -> nn.Sequential:
    """Nature DQN encoder: (B, in_channels, 84, 84) -> (B, 512).

    Module indices (network.0, .2, .4, .7) are part of saved checkpoints; keep them stable.
    """
    return nn.Sequential(
        *nature_convs(in_channels),
        nn.Flatten(),
        layer_init(nn.Linear(CONV_CHANNELS * CONV_GRID * CONV_GRID, 512)),
        nn.ReLU(),
    )


class TransformerBlock(nn.Module):
    """Pre-norm transformer block without dropout: x + attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, width: int, heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_ratio * width), nn.GELU(), nn.Linear(mlp_ratio * width, width))

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        y = self.norm1(x)
        y, weights = self.attention(y, y, y, need_weights=need_weights, average_attn_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x)), weights


class ViTEncoder(nn.Module):
    """Vision transformer encoder: (B, in_channels, 84, 84) in [0, 1] -> (B, 512).

    Each 12x12 patch (all input channels together) becomes one token; a learned CLS token and learned
    position embeddings are added, and the CLS output after the last block, layer-normalised and
    projected to 512 features with a ReLU (as in nature_cnn), is the frame's representation.
    forward(x, need_weights=True) also returns the CLS token's attention over the 49 patches in the last
    block, shaped (B, heads, 49), patches in row-major order (row = y, column = x).
    """

    def __init__(self, in_channels: int, width: int = VIT_WIDTH, depth: int = VIT_DEPTH, heads: int = VIT_HEADS,
                 mlp_ratio: int = VIT_MLP_RATIO, patch: int = VIT_PATCH):
        super().__init__()
        assert 84 % patch == 0 and width % heads == 0
        self.tokens = (84 // patch) ** 2
        # A stride-patch convolution is the standard patch embedding: one linear map per patch.
        self.patch_embed = nn.Conv2d(in_channels, width, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.zeros(1, self.tokens + 1, width))
        self.blocks = nn.ModuleList(TransformerBlock(width, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.project = layer_init(nn.Linear(width, FEATURES))
        # Standard ViT initialisation (truncated normal, std 0.02) everywhere except the output projection,
        # which uses the same orthogonal init as nature_cnn's 512-unit layer.
        for module in [self.patch_embed, *self.blocks.modules()]:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.trunc_normal_(block.attention.in_proj_weight, std=0.02)
            nn.init.zeros_(block.attention.in_proj_bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, 49, width)
        tokens = torch.cat((self.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1) + self.position
        weights = None
        for i, block in enumerate(self.blocks):
            tokens, weights = block(tokens, need_weights=need_weights and i == len(self.blocks) - 1)
        features = F.relu(self.project(self.norm(tokens[:, 0])))
        if not need_weights:
            return features
        assert weights is not None
        return features, weights[:, :, 0, 1:]  # CLS row, patch columns: (B, heads, 49)


def init_lstm(lstm: nn.LSTM):
    for name, param in lstm.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, 1.0)


class Agent(nn.Module):
    """Nature CNN Actor-Critic taking 4 stacked 84x84 frames."""

    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        assert envs.single_observation_space.shape is not None
        self.network = self.make_encoder(envs.single_observation_space.shape[0])
        assert isinstance(envs.single_action_space, gym.spaces.Discrete)
        self.actor = layer_init(nn.Linear(FEATURES, int(envs.single_action_space.n)), std=0.01)
        self.critic = layer_init(nn.Linear(FEATURES, 1), std=1.0)

    @staticmethod
    def make_encoder(in_channels: int) -> nn.Module:
        """(B, in_channels, 84, 84) in [0, 1] -> (B, 512)."""
        return nature_cnn(in_channels)

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the 512-dim latent representation (useful for linear probing)."""
        return self.network(x / 255.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.get_hidden(x))

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        hidden = self.get_hidden(x)
        logits = self.actor(hidden)
        # Actions are sampled here or replayed from the collector. Avoid repeated
        # CUDA validation synchronizations; PPO checks aggregate losses for finiteness.
        probs = Categorical(logits=logits, validate_args=False)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class RecurrentAgent(nn.Module):
    """PPO interface shared by agents that carry LSTM memory (h, c) across steps.

    All methods take observations as a flattened, time-major sequence: x[t * batch + b] is step t
    of sequence b, where batch = lstm_state[0].shape[1]. done[t * batch + b] = 1 means that
    observation is the first of a new episode, so the state is reset to zero before processing it.
    done=None means no resets anywhere in the sequence.
    Subclasses implement get_states and define actor/critic heads on its outputs.
    """

    actor: nn.Linear
    critic: nn.Linear

    def get_states(
        self, x: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor | None
    ) -> tuple[torch.Tensor, LSTMState]:
        raise NotImplementedError

    def get_value(self, x: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor | None) -> torch.Tensor:
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        lstm_state: LSTMState,
        done: torch.Tensor | None,
        action: torch.Tensor | None = None,
    ):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        # Actions are sampled here or replayed from the collector. Avoid repeated
        # CUDA validation synchronizations; PPO checks aggregate losses for finiteness.
        probs = Categorical(logits=logits, validate_args=False)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


class LSTMAgent(RecurrentAgent):
    """Nature CNN on a single 84x84 frame, followed by an LSTM that carries memory across steps.

    Adapted from CleanRL's ppo_atari_lstm.py.
    """

    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        assert envs.single_observation_space.shape is not None
        self.network = self.make_encoder(envs.single_observation_space.shape[0])
        self.lstm = nn.LSTM(FEATURES, LSTM_HIDDEN_SIZE)
        init_lstm(self.lstm)
        assert isinstance(envs.single_action_space, gym.spaces.Discrete)
        self.actor = layer_init(nn.Linear(LSTM_HIDDEN_SIZE, int(envs.single_action_space.n)), std=0.01)
        self.critic = layer_init(nn.Linear(LSTM_HIDDEN_SIZE, 1), std=1.0)

    @staticmethod
    def make_encoder(in_channels: int) -> nn.Module:
        """(B, in_channels, 84, 84) in [0, 1] -> (B, 512)."""
        return nature_cnn(in_channels)

    def get_states(
        self, x: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor | None
    ) -> tuple[torch.Tensor, LSTMState]:
        """Returns the per-step LSTM outputs (T * batch, hidden_size), useful for linear probing,
        and the (h, c) state after the last step."""
        features = self.network(x / 255.0)
        batch_size = lstm_state[0].shape[1]
        features = features.reshape((-1, batch_size, self.lstm.input_size))
        if done is None:
            # No resets inside the sequence: one cuDNN call over all steps.
            out, lstm_state = self.lstm(features, lstm_state)
            return torch.flatten(out, 0, 1), lstm_state
        assert done.numel() == x.shape[0], "need one done flag per observation"
        done = done.float().reshape((-1, batch_size))
        outputs = []
        for f, d in zip(features, done):
            keep = (1.0 - d).view(1, -1, 1)
            out, lstm_state = self.lstm(f.unsqueeze(0), (keep * lstm_state[0], keep * lstm_state[1]))
            outputs.append(out)
        return torch.flatten(torch.cat(outputs), 0, 1), lstm_state


class AttentionLSTMAgent(RecurrentAgent):
    """Nature CNN convolutions on a single frame, read through spatial attention by an LSTM.

    Simplified from Mott et al. (2019), "Towards Interpretable Reinforcement Learning Using
    Attention Augmented Agents". Each step:
      1. The convolutions turn the frame into a 7x7 map of 64 features (49 cells); two fixed
         channels add each cell's x and y position in [-1, 1].
      2. Every cell gives a key and a value per head.
      3. Each head's query comes from the previous hidden state h(t-1): memory decides where to look.
      4. Softmax over the 49 scores q . k / sqrt(key size) gives the attention weights. q and k are
         layer-normalised per head first (no learned scale), so the scores are not near zero at
         initialisation; without this the weights stay uniform and the agent cannot learn where to look.
      5. The weighted sums of values, concatenated over heads, are the LSTM's only input.
    get_states_and_attention also returns the weights, shaped (T * batch, heads, 49), cells in
    row-major order (row = y, column = x). Its optional `blocked` (bool, (T * batch, 49)) removes cells from
    every head's softmax (an analysis intervention: the agent cannot look there); None changes nothing.
    """

    def __init__(self, envs: gym.vector.VectorEnv, heads: int = ATTENTION_HEADS,
                 key_size: int = ATTENTION_KEY_SIZE, value_size: int = ATTENTION_VALUE_SIZE):
        super().__init__()
        assert envs.single_observation_space.shape is not None
        self.heads, self.key_size, self.value_size = heads, key_size, value_size
        self.encoder = nn.Sequential(*nature_convs(envs.single_observation_space.shape[0]))
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, CONV_GRID), torch.linspace(-1, 1, CONV_GRID), indexing="ij")
        # Fixed, not learned: kept out of checkpoints so loading stays strict and simple.
        self.register_buffer("positions", torch.stack((xs, ys), dim=-1).reshape(CONV_GRID * CONV_GRID, 2),
                             persistent=False)
        cell_size = CONV_CHANNELS + 2
        self.keys = layer_init(nn.Linear(cell_size, heads * key_size), std=1.0)
        self.values = layer_init(nn.Linear(cell_size, heads * value_size), std=1.0)
        self.query = layer_init(nn.Linear(LSTM_HIDDEN_SIZE, heads * key_size), std=1.0)
        self.lstm = nn.LSTM(heads * value_size, LSTM_HIDDEN_SIZE)
        init_lstm(self.lstm)
        # The time loop calls torch.lstm_cell with these weights: one layer, biases, no projection.
        assert self.lstm.num_layers == 1 and self.lstm.bias and self.lstm.proj_size == 0
        assert isinstance(envs.single_action_space, gym.spaces.Discrete)
        self.actor = layer_init(nn.Linear(LSTM_HIDDEN_SIZE, int(envs.single_action_space.n)), std=0.01)
        self.critic = layer_init(nn.Linear(LSTM_HIDDEN_SIZE, 1), std=1.0)

    def get_states_and_attention(
        self, x: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor | None,
        blocked: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LSTMState, torch.Tensor]:
        """Per-step LSTM outputs (T * batch, hidden), the final (h, c), and attention weights."""
        batch = lstm_state[0].shape[1]
        assert done is None or done.numel() == x.shape[0], "need one done flag per observation"
        cells = self.encoder(x / 255.0).flatten(2).transpose(1, 2)  # (T*B, 49, 64)
        cells = torch.cat((cells, self.positions.expand(cells.shape[0], -1, -1)), dim=-1)
        n = cells.shape[1]
        # Keys and values depend only on the frame, so compute them for all steps at once,
        # laid out (T, B, heads, cells, size) so each step's attention is two batched matmuls.
        keys = F.layer_norm(self.keys(cells).reshape(-1, batch, n, self.heads, self.key_size),
                            (self.key_size,)).permute(0, 1, 3, 2, 4)
        values = self.values(cells).reshape(-1, batch, n, self.heads, self.value_size).permute(0, 1, 3, 2, 4)
        keeps = None if done is None else (1.0 - done.float()).reshape((-1, batch, 1))
        # (T, B, 1, cells): broadcast over heads; -inf scores get zero weight after the softmax
        masks = None if blocked is None else blocked.reshape(-1, batch, 1, n)
        h, c = lstm_state[0][0], lstm_state[1][0]
        # One fused cell per step (same parameters as self.lstm): much less per-step overhead
        # than calling nn.LSTM on a length-1 sequence inside this loop.
        lstm_weights = (self.lstm.weight_ih_l0, self.lstm.weight_hh_l0, self.lstm.bias_ih_l0, self.lstm.bias_hh_l0)
        scale = 1.0 / math.sqrt(self.key_size)
        outputs, weights = [], []
        for t, (k, v) in enumerate(zip(keys, values)):
            if keeps is not None:
                h, c = keeps[t] * h, keeps[t] * c
            # The query needs this step's starting memory, so attention runs inside the time loop.
            q = F.layer_norm(self.query(h).reshape(batch, self.heads, self.key_size), (self.key_size,))
            scores = torch.matmul(k, q.unsqueeze(-1)).squeeze(-1) * scale  # (B, heads, 49)
            if masks is not None:
                scores = scores.masked_fill(masks[t], float("-inf"))
            attention = scores.softmax(dim=-1)
            readout = torch.matmul(attention.unsqueeze(-2), v).reshape(batch, self.heads * self.value_size)
            h, c = torch.lstm_cell(readout, (h, c), *lstm_weights)
            outputs.append(h)
            weights.append(attention)
        return (torch.stack(outputs).flatten(0, 1), (h.unsqueeze(0), c.unsqueeze(0)),
                torch.flatten(torch.stack(weights), 0, 1))

    def get_states(
        self, x: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor | None
    ) -> tuple[torch.Tensor, LSTMState]:
        hidden, lstm_state, _ = self.get_states_and_attention(x, lstm_state, done)
        return hidden, lstm_state


class ViTAgent(Agent):
    """Vision transformer Actor-Critic taking 4 stacked 84x84 frames: the cnn agent with a ViT encoder."""

    @staticmethod
    def make_encoder(in_channels: int) -> nn.Module:
        return ViTEncoder(in_channels)

    def get_hidden_and_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """512-dim latent and the CLS attention over the 49 patches in the last block, (B, heads, 49)."""
        return self.network(x / 255.0, need_weights=True)


class ViTLSTMAgent(LSTMAgent):
    """Vision transformer on a single 84x84 frame, followed by the same LSTM as the lstm agent."""

    @staticmethod
    def make_encoder(in_channels: int) -> nn.Module:
        return ViTEncoder(in_channels)

    def get_states_and_attention(
        self, x: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor | None
    ) -> tuple[torch.Tensor, LSTMState, torch.Tensor]:
        """get_states plus each frame's CLS attention over the 49 patches, (T * batch, heads, 49).
        The encoder sees only the frame, so the weights do not depend on memory."""
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        with torch.no_grad():
            _, weights = self.network(x / 255.0, need_weights=True)
        return hidden, lstm_state, weights


AGENTS = {"cnn": Agent, "lstm": LSTMAgent, "attn_lstm": AttentionLSTMAgent, "vit": ViTAgent, "vit_lstm": ViTLSTMAgent}


def make_agent(arch: str, envs: gym.vector.VectorEnv) -> nn.Module:
    return AGENTS[arch](envs)
