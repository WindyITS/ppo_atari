import ale_py
import gymnasium as gym

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", frameskip=1)

env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.AtariPreprocessing(
    env,
    noop_max=30,
    frame_skip=4,
    screen_size=84,
    terminal_on_life_loss=False,
    grayscale_obs=True,
)
env = gym.wrappers.FrameStackObservation(env, stack_size=4)

obs, info = env.reset()

print(env.observation_space)
print(env.action_space)
print(env.unwrapped.get_action_meanings())  # type: ignore
