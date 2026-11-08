import gymnasium as gym
import numpy as np

# In 84x84 grayscale Pong (after AtariPreprocessing):
# - Background pixel intensity is 87.
# - Top wall ends at y=13, bottom wall starts at y=77 -> playable court is y in [14, 77).
# - Enemy (CPU) paddle is at x in [8, 10] -> masking x in [6, 12) completely hides it.
# - Agent (player) paddle is at x in [73, 75].
# - Midline of the court is x=42 -> agent's half of the field (before agent paddle) is x in [42, 72).
# - The resize blends objects touching a wall into the wall's edge rows (y=13 and y=77),
#   so those rows must be restored to their clean values (176 and 147) or they leak positions.
PONG_BG_VALUE = 87
COURT_Y_MIN = 14
COURT_Y_MAX = 77
TOP_WALL_EDGE_Y = COURT_Y_MIN - 1
TOP_WALL_EDGE_VALUE = 176
BOTTOM_WALL_EDGE_Y = COURT_Y_MAX
BOTTOM_WALL_EDGE_VALUE = 147


class CleanFrameTap(gym.ObservationWrapper):
    """Passes observations through unchanged and keeps the latest one in clean_frame, so analyses can read
    the unoccluded screen while the agent sees the occluded one (insert it before the occlusion wrappers)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.clean_frame = None

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.clean_frame = observation.copy()
        return observation


class CourtColumnMaskWrapper(gym.ObservationWrapper):
    """Hides everything in the court columns x in [x_start, x_end), including the wall edge rows."""

    def __init__(self, env: gym.Env, x_start: int, x_end: int):
        super().__init__(env)
        self.x_start = x_start
        self.x_end = x_end

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = observation.copy()
        cols = slice(self.x_start, self.x_end)
        obs[..., COURT_Y_MIN:COURT_Y_MAX, cols] = PONG_BG_VALUE
        obs[..., TOP_WALL_EDGE_Y, cols] = TOP_WALL_EDGE_VALUE
        obs[..., BOTTOM_WALL_EDGE_Y, cols] = BOTTOM_WALL_EDGE_VALUE
        return obs


class FogWrapper(CourtColumnMaskWrapper):
    """Paints over the agent's half of the court so the ball is hidden there,
    while keeping the agent's own paddle (x >= 72) visible."""

    def __init__(self, env: gym.Env, x_start: int = 42, x_end: int = 72):
        super().__init__(env, x_start, x_end)


class EnemyMaskWrapper(CourtColumnMaskWrapper):
    """Masks out the enemy (left) paddle with the background color so it is invisible."""

    def __init__(self, env: gym.Env, x_start: int = 6, x_end: int = 12):
        super().__init__(env, x_start, x_end)
