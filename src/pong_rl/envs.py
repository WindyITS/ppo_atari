import gymnasium as gym

from pong_rl.wrappers import EnemyMaskWrapper, FogWrapper


def make_env(
    env_id: str,
    idx: int,
    capture_video: bool,
    run_name: str,
    use_fog: bool = False,
    mask_enemy: bool = False,
    frame_stack: int = 4,
):
    def thunk():
        # Match DeepMind 2013/2015 & CleanRL PongNoFrameskip-v4:
        # frameskip=1 (handled by AtariPreprocessing) and repeat_action_probability=0.0 (no sticky actions)
        if capture_video and idx == 0:
            env = gym.make(
                env_id,
                frameskip=1,
                repeat_action_probability=0.0,
                render_mode="rgb_array",
            )
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(
                env_id,
                frameskip=1,
                repeat_action_probability=0.0,
            )
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
        # stack_size=1 still adds the channel dim: (1, 84, 84)
        env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
