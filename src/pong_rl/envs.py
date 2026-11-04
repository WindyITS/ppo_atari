import gymnasium as gym
import numpy as np

from pong_rl.wrappers import EnemyMaskWrapper, FogWrapper


def make_env(
    env_id: str,
    idx: int,
    capture_video: bool,
    run_name: str,
    use_fog: bool = False,
    mask_enemy: bool = False,
):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, frameskip=1)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
        )
        if use_fog:
            env = FogWrapper(env)
        if mask_enemy:
            env = EnemyMaskWrapper(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        return env

    return thunk
