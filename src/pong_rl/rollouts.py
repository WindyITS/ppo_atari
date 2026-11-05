"""Fixed-length on-policy rollouts: every game plays num_steps steps, then the PPO update."""

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import torch

from pong_rl.metrics import Episode
from pong_rl.models import LSTMState, RecurrentAgent, zero_lstm_state


@dataclass
class Sequence:
    """Consecutive steps of one game inside a rollout, ending at a game end or the rollout's end."""

    observations: torch.Tensor  # CPU uint8; uploaded to the accelerator once per update
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    initial_state: LSTMState  # memory before the first step (zero after a game reset)

    def __len__(self):
        return len(self.actions)


@dataclass
class Batch:
    sequences: list[Sequence]
    global_step: int
    points: int  # points scored by either side during the rollout
    ended_games: int

    @property
    def steps(self):
        return sum(map(len, self.sequences))


def advantages_from_rewards(rewards, values, bootstrap, gamma, gae_lambda):
    """GAE over one sequence. Points are NOT terminal; only the sequence's end is cut."""
    advantages = torch.empty(len(rewards), dtype=torch.float32)
    last = 0.0
    next_value = bootstrap
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * gae_lambda * last
        advantages[t] = last
        next_value = values[t]
    return advantages


class RolloutCollector:
    """Every game plays exactly num_steps steps per batch with one frozen policy.

    Each step asks the policy for all games at once. A game that ends inside the rollout is reset
    immediately (its memory too) and keeps playing. The batch is cut into sequences at game ends
    only, so the PPO update replays each sequence from its saved memory. GAE bootstraps at the
    rollout's end and at a timeout (final observation, pre-reset memory); a genuine game end has
    zero bootstrap value. Memory carries over to the next rollout.

    envs is a SyncVectorEnv with autoreset disabled (its games are stepped and reset individually)
    or a ParallelEnvPool.
    """

    def __init__(self, envs, agent, device, seed, num_steps=256, start_step=0):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self.envs, self.agent, self.device = envs, agent, device
        self.num_steps = num_steps
        self.observations = list(envs.reset(seed=seed)[0])
        self.state = zero_lstm_state(len(envs.envs), device)
        self.global_step = start_step  # nonzero when fine-tuning continues a parent's step count

    @torch.no_grad()
    def collect(self, gamma: float, gae_lambda: float,
                on_episode: Callable[[Episode], None]) -> Batch:
        n = len(self.envs.envs)
        recurrent = isinstance(self.agent, RecurrentAgent)
        pending = [[] for _ in range(n)]
        starts = [self._state_for(i) for i in range(n)]
        finished = []  # (transitions, initial memory, bootstrap value)
        points = ended_games = 0
        for _ in range(self.num_steps):
            obs = torch.as_tensor(np.stack(self.observations), device=self.device, dtype=torch.float32)
            if recurrent:
                action, logp, _, value, self.state = self.agent.get_action_and_value(
                    obs, self.state, torch.zeros(n, device=self.device))
            else:
                action, logp, _, value = self.agent.get_action_and_value(obs)
            actions, logps, values = action.cpu().tolist(), logp.cpu().tolist(), value.flatten().cpu().tolist()
            self.global_step += n
            results = (self.envs.step_each(actions) if hasattr(self.envs, "step_each") else
                       [env.step(a) for env, a in zip(self.envs.envs, actions)])
            for i, (next_obs, reward, term, trunc, info) in enumerate(results):
                pending[i].append((self.observations[i].copy(), actions[i], logps[i], values[i], float(reward)))
                points += int(reward != 0)  # raw Pong reward: +1 or -1 for each point
                if term or trunc:
                    bootstrap = 0.0 if term else self._value(i, next_obs)
                    finished.append((pending[i], starts[i], bootstrap))
                    episode = info["episode"]  # RecordEpisodeStatistics is the outer wrapper
                    on_episode(Episode(self.global_step, float(episode["r"]),
                                       int(episode["l"]), bool(term), bool(trunc)))
                    ended_games += 1
                    self.observations[i], _ = self.envs.envs[i].reset()
                    for state in self.state:
                        state[:, i] = 0
                    pending[i], starts[i] = [], self._state_for(i)
                else:
                    self.observations[i] = next_obs
        # Rollout end: bootstrap unfinished games from their current observation and memory, in one call.
        if any(pending):
            obs = torch.as_tensor(np.stack(self.observations), device=self.device, dtype=torch.float32)
            if recurrent:
                last_values = self.agent.get_value(obs, self.state, torch.zeros(n, device=self.device))
            else:
                last_values = self.agent.get_value(obs)
            last_values = last_values.flatten().cpu().tolist()
            for i in range(n):
                if pending[i]:
                    finished.append((pending[i], starts[i], last_values[i]))

        sequences = []
        for transitions, state, bootstrap in finished:
            values = torch.tensor([row[3] for row in transitions])
            advantages = advantages_from_rewards([row[4] for row in transitions], values,
                                                 bootstrap, gamma, gae_lambda)
            sequences.append(Sequence(torch.from_numpy(np.stack([row[0] for row in transitions])),
                                      torch.tensor([row[1] for row in transitions]),
                                      torch.tensor([row[2] for row in transitions]), values,
                                      advantages, advantages + values, state))
        return Batch(sequences, self.global_step, points, ended_games)

    def _state_for(self, i):
        return tuple(s[:, i:i+1].detach().cpu().clone() for s in self.state)

    def _value(self, i, observation):
        obs = torch.as_tensor(np.asarray(observation)[None], device=self.device, dtype=torch.float32)
        if isinstance(self.agent, RecurrentAgent):
            return self.agent.get_value(obs, tuple(s[:, i:i+1] for s in self.state),
                                        torch.zeros(1, device=self.device)).item()
        return self.agent.get_value(obs).item()
