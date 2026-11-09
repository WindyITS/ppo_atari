"""An instrumented frozen agent playing one game at a time, with interventions.

Player reproduces the frozen evaluation (pong_rl.evaluate) step for step: same environment, reset seed,
action generator (CPU torch.Generator seeded with the game's seed) and sampling call, so with no
intervention it plays exactly the evaluation's games. On top it exposes, at each step:
  - scene: ball and paddle positions on the unoccluded frame (pong_rl.analysis.scene.locate)
  - the policy's probabilities, the value and the model's internals (cnn: 512 features; lstm: 512 encoder
    features fed to the LSTM, h and c; attn_lstm: h, c and the attention weights)
and accepts interventions on what the model sees (transform) or remembers (wipe), plus snapshots of the
whole game (environment, memory, action generator) to replay a moment under different interventions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import ale_py
import gymnasium as gym
import numpy as np
import torch

from pong_rl.analysis import scene
from pong_rl.analysis.snapshot import restore, snapshot
from pong_rl.envs import make_env
from pong_rl.evaluate import VIEWS, _checkpoint_config, view_name
from pong_rl.models import AttentionLSTMAgent, RecurrentAgent, make_agent, zero_lstm_state
from pong_rl.train import FRAME_STACK

gym.register_envs(ale_py)

Transform = Callable[[np.ndarray], np.ndarray]


@dataclass
class Step:
    """What happened at one agent step (positions describe the frame the model saw before acting)."""
    t: int
    scene: dict
    action: int
    probs: np.ndarray
    value: float
    reward: float
    done: bool
    internals: dict = field(default_factory=dict)


class Player:
    def __init__(self, checkpoint: str | Path, view: str = "own"):
        torch.set_num_threads(1)
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = _checkpoint_config(data)
        self.path = Path(checkpoint)
        self.checkpoint = data
        self.arch = config["arch"]
        self.train_view = view_name(config["use_fog"], config["mask_enemy"])
        self.view = self.train_view if view == "own" else view
        use_fog, mask_enemy = VIEWS[self.view]
        self.env = make_env(config["env_id"], 0, False, "analysis", use_fog=use_fog, mask_enemy=mask_enemy,
                            frame_stack=FRAME_STACK[self.arch], tap_clean_frame=True)()
        spaces = SimpleNamespace(single_observation_space=self.env.observation_space,
                                 single_action_space=self.env.action_space)
        self.agent = make_agent(self.arch, spaces)
        self.agent.load_state_dict(data["model_state_dict"], strict=True)
        self.agent.eval()
        self.recurrent = isinstance(self.agent, RecurrentAgent)
        self.obs = None

    # ----- one game -----
    def reset(self, seed: int):
        obs, _ = self.env.reset(seed=seed)
        self.obs = np.array(obs, copy=True)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.state = zero_lstm_state(1, torch.device("cpu"))
        self.t = 0
        self.episode_return = 0.0
        self.done = False

    def clean_frame(self) -> np.ndarray:
        return np.array(self.env.get_wrapper_attr("clean_frame"), copy=True)

    @torch.inference_mode()
    def forward(self, obs: np.ndarray, state=None):
        """Policy outputs for an observation (stack, 84, 84) from a memory state, without side effects.
        Returns (probs (6,), value, internals, next state)."""
        x = torch.as_tensor(obs[None], dtype=torch.float32)
        internals = {}
        if isinstance(self.agent, AttentionLSTMAgent):
            hidden, state, attention = self.agent.get_states_and_attention(x, state, None)
            internals["attention"] = attention[0].numpy().copy()
        elif self.recurrent:
            features = self.agent.network(x / 255.0)
            internals["features"] = features[0].numpy().copy()
            out, state = self.agent.lstm(features.unsqueeze(0), state)
            hidden = out[0]
        else:
            hidden = self.agent.get_hidden(x)
            internals["features"] = hidden[0].numpy().copy()
        if self.recurrent:
            internals["h"] = state[0][0, 0].numpy().copy()
            internals["c"] = state[1][0, 0].numpy().copy()
        logits = self.agent.actor(hidden)
        probs = torch.softmax(logits, dim=-1)
        return probs, float(self.agent.critic(hidden)[0, 0]), internals, state

    def step(self, transform: Transform | None = None, wipe: bool = False, action: int | None = None) -> Step:
        """Play one agent step. transform changes what the model sees (not the game); wipe empties the memory
        before this step (recurrent agents); action overrides the sampled action (the generator still draws)."""
        if self.done:
            raise RuntimeError("the game is over; call reset")
        where = scene.locate(self.clean_frame(), self.obs[-1])
        seen = self.obs if transform is None else transform(self.obs)
        if wipe and self.recurrent:
            self.state = zero_lstm_state(1, torch.device("cpu"))
        probs, value, internals, self.state = self.forward(seen, self.state)
        # Same sampling call as pong_rl.evaluate, so unperturbed games match the frozen evaluation.
        sampled = int(torch.multinomial(probs, 1, generator=self.generator).flatten()[0])
        chosen = sampled if action is None else int(action)
        obs, reward, terminated, truncated, _ = self.env.step(chosen)
        self.obs = np.array(obs, copy=True)
        self.done = bool(terminated or truncated)
        self.episode_return += float(reward)
        record = Step(self.t, where, chosen, probs[0].numpy().copy(), value, float(reward), self.done, internals)
        self.t += 1
        return record

    # ----- snapshots -----
    def save(self):
        """Everything needed to replay the game from this moment."""
        return {"env": snapshot(self.env), "obs": self.obs.copy(),
                "state": (self.state[0].clone(), self.state[1].clone()),
                "generator": self.generator.get_state(), "t": self.t,
                "episode_return": self.episode_return, "done": self.done}

    def load(self, saved, reseed: int | None = None):
        """Return to a saved moment; reseed gives the action generator a new seed instead of its saved state."""
        restore(self.env, saved["env"])
        self.obs = saved["obs"].copy()
        self.state = (saved["state"][0].clone(), saved["state"][1].clone())
        if reseed is None:
            self.generator.set_state(saved["generator"])
        else:
            self.generator = torch.Generator(device="cpu").manual_seed(reseed)
        self.t, self.episode_return, self.done = saved["t"], saved["episode_return"], saved["done"]

    def close(self):
        self.env.close()


def stack_wipe(obs: np.ndarray) -> np.ndarray:
    """CNN analogue of a memory wipe: every frame of the stack replaced by the newest one (no motion, no past)."""
    return np.repeat(obs[-1:], obs.shape[0], axis=0)
