"""On-policy batches ending at five points per environment or an actual game end."""

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import torch

from pong_rl.metrics import Episode
from pong_rl.models import LSTMState, RecurrentAgent, zero_lstm_state


@dataclass
class Rally:
    observations: torch.Tensor  # CPU uint8; image storage does not occupy accelerator memory
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    initial_state: LSTMState

    def __len__(self):
        return len(self.actions)


@dataclass
class RallyBatch:
    rallies: list[Rally]
    global_step: int
    completed_rallies: int
    ended_games: int

    @property
    def steps(self):
        return sum(map(len, self.rallies))


def advantages_from_rewards(rewards, values, bootstrap, gamma, gae_lambda):
    """One game's contiguous batch. Points are NOT terminal; only its final tail is cut."""
    advantages = torch.empty(len(rewards), dtype=torch.float32)
    last = 0.0
    next_value = bootstrap
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * gae_lambda * last
        advantages[t] = last
        next_value = values[t]
    return advantages


class RallyCollector:
    """Step only active lanes, leaving completed lanes paused until the policy update.

    SyncVectorEnv supplies spaces and owns the envs; stepping/resetting its individual
    children is deliberate: its vector step API cannot pause lanes. No vector step is
    used after initialization. Every batch is collected with a single frozen policy.
    """

    def __init__(self, envs, agent, device, seed, rallies_per_update=5, start_step=0):
        if rallies_per_update <= 0:
            raise ValueError("rallies_per_update must be positive")
        self.envs, self.agent, self.device = envs, agent, device
        self.quota = rallies_per_update
        self.observations = list(envs.reset(seed=seed)[0])
        self.state = zero_lstm_state(len(envs.envs), device)
        self.global_step = start_step  # nonzero when fine-tuning continues a parent's step count

    @torch.no_grad()
    def collect(self, gamma: float, gae_lambda: float,
                on_episode: Callable[[Episode], None]) -> RallyBatch:
        n = len(self.envs.envs)
        active = list(range(n))
        counts = [0] * n
        ended = [False] * n
        terminated = [False] * n
        sequences = [[] for _ in range(n)]
        pending = [[] for _ in range(n)]
        starts = [self._state_for(i) for i in range(n)]
        while active:
            obs = torch.as_tensor(np.stack([self.observations[i] for i in active]),
                                  device=self.device, dtype=torch.float32)
            if isinstance(self.agent, RecurrentAgent):
                action, logp, _, value, state = self.agent.get_action_and_value(
                    obs, tuple(s[:, active] for s in self.state),
                    torch.zeros(len(active), device=self.device))
                for target, source in zip(self.state, state):
                    target[:, active] = source
            else:
                action, logp, _, value = self.agent.get_action_and_value(obs)
            actions, logps, values = action.cpu().tolist(), logp.cpu().tolist(), value.flatten().cpu().tolist()
            self.global_step += len(active)
            results = (self.envs.step_active(active, actions)
                       if hasattr(self.envs, "step_active") else
                       [self.envs.envs[i].step(action) for i, action in zip(active, actions)])
            remaining = []
            for slot, i in enumerate(active):
                next_obs, reward, term, trunc, info = results[slot]
                pending[i].append((self.observations[i].copy(), actions[slot], logps[slot], values[slot], float(reward)))
                self.observations[i] = next_obs
                point = reward != 0  # raw Pong reward: +1 or -1 for each point
                counts[i] += int(point)
                ended[i], terminated[i] = bool(term or trunc), bool(term)
                if point or ended[i]:
                    sequences[i].append((pending[i], starts[i]))
                    pending[i] = []
                    starts[i] = self._state_for(i)
                if ended[i]:
                    episode = info["episode"]  # RecordEpisodeStatistics is the outer wrapper
                    on_episode(Episode(self.global_step, float(episode["r"]),
                                       int(episode["l"]), bool(term), bool(trunc)))
                elif counts[i] < self.quota:
                    remaining.append(i)
            active = remaining

        rallies = []
        for i in range(n):
            # A point boundary still bootstraps. A timeout bootstraps its final
            # observation with pre-reset memory; a genuine termination does not.
            bootstrap = 0.0
            if not terminated[i]:
                obs = torch.as_tensor(np.asarray(self.observations[i])[None],
                                      device=self.device, dtype=torch.float32)
                if isinstance(self.agent, RecurrentAgent):
                    value = self.agent.get_value(obs, tuple(s[:, i:i+1] for s in self.state),
                                                 torch.zeros(1, device=self.device))
                else:
                    value = self.agent.get_value(obs)
                bootstrap = value.item()
            transitions = [row for sequence, _ in sequences[i] for row in sequence]
            values = torch.tensor([row[3] for row in transitions])
            advantages = advantages_from_rewards([row[4] for row in transitions], values,
                                                 bootstrap, gamma, gae_lambda)
            offset = 0
            for sequence, state in sequences[i]:
                end = offset + len(sequence)
                rallies.append(Rally(torch.from_numpy(np.stack([row[0] for row in sequence])),
                                     torch.tensor([row[1] for row in sequence]),
                                     torch.tensor([row[2] for row in sequence]), values[offset:end],
                                     advantages[offset:end], advantages[offset:end] + values[offset:end], state))
                offset = end
            if ended[i]:
                self.observations[i], _ = self.envs.envs[i].reset()
                for state in self.state:
                    state[:, i] = 0
        return RallyBatch(rallies, self.global_step, sum(counts), sum(ended))

    def _state_for(self, i):
        return tuple(s[:, i:i+1].detach().cpu().clone() for s in self.state)
