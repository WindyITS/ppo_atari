import gymnasium as gym
import numpy as np

# In 84x84 grayscale Pong (after AtariPreprocessing):
# - Background pixel intensity is 87.
# - Top wall ends at y=13, bottom wall starts at y=77 -> playable court is y in [14, 77).
# - Enemy (CPU) paddle is at x in [8, 10] -> masking x in [6, 12) completely hides it.
# - Agent (player) paddle is at x in [73, 75].
# - Midline of the court is x=42 -> agent's half of the field (before agent paddle) is x in [42, 72).
PONG_BG_VALUE = 87
COURT_Y_MIN = 14
COURT_Y_MAX = 77


class FogWrapper(gym.ObservationWrapper):
    """Paints over the agent's half of the court so the ball is hidden there,
    while keeping the agent's own paddle (x >= 72) visible."""

    def __init__(
        self,
        env: gym.Env,
        x_start: int = 42,
        x_end: int = 72,
        y_start: int = COURT_Y_MIN,
        y_end: int = COURT_Y_MAX,
        fill_value: int = PONG_BG_VALUE,
    ):
        super().__init__(env)
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.fill_value = fill_value

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = observation.copy()
        obs[..., self.y_start : self.y_end, self.x_start : self.x_end] = self.fill_value
        return obs


class EnemyMaskWrapper(gym.ObservationWrapper):
    """Masks out the enemy (left) paddle with the background color so it is invisible."""

    def __init__(
        self,
        env: gym.Env,
        x_start: int = 6,
        x_end: int = 12,
        y_start: int = COURT_Y_MIN,
        y_end: int = COURT_Y_MAX,
        fill_value: int = PONG_BG_VALUE,
    ):
        super().__init__(env)
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.fill_value = fill_value

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = observation.copy()
        obs[..., self.y_start : self.y_end, self.x_start : self.x_end] = self.fill_value
        return obs
