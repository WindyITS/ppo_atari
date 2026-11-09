"""Save and restore the complete state of a single (non-vector) environment built by pong_rl.envs.make_env.

A snapshot holds the emulator state (ALE cloneState) and a deep copy of every wrapper's own attributes
(frame stack, max-pool buffer, clean-frame tap, episode statistics), so after restore the environment
produces exactly the observations, rewards and end flags it would have produced from the snapshot point.
Pong without sticky actions has no randomness after reset, so the emulator RNG need not be saved.
"""

import copy

import gymnasium as gym


def _wrappers(env: gym.Env):
    while isinstance(env, gym.Wrapper):
        yield env
        env = env.env


def snapshot(env: gym.Env):
    states = [{key: copy.deepcopy(value) for key, value in vars(w).items() if key != "env"} for w in _wrappers(env)]
    return states, env.unwrapped.ale.cloneState()


def restore(env: gym.Env, saved) -> None:
    states, ale_state = saved
    wrappers = list(_wrappers(env))
    if len(wrappers) != len(states):
        raise ValueError("snapshot was taken from a differently wrapped environment")
    for wrapper, state in zip(wrappers, states):
        for key, value in state.items():
            setattr(wrapper, key, copy.deepcopy(value))
    env.unwrapped.ale.restoreState(ale_state)
