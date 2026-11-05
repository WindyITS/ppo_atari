"""Episode outcomes and reporting windows, independent of PPO's training targets."""

from collections import deque
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Episode:
    global_step: int
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool

    def __post_init__(self):
        if self.global_step < 0 or self.episode_length <= 0:
            raise ValueError("episode steps must be nonnegative and length must be positive")
        if not math.isfinite(self.episode_return):
            raise ValueError("episode return must be finite")
        if not (self.terminated or self.truncated):
            raise ValueError("an episode record must have an end flag")

    @property
    def outcome(self) -> str:
        # ALE can report both flags on the last frame. A genuine game over takes precedence.
        if self.terminated:
            return "win" if self.episode_return > 0 else "loss" if self.episode_return < 0 else "draw"
        return "timeout"

    def to_dict(self) -> dict:
        return {**asdict(self), "outcome": self.outcome}


def summarize_episodes(episodes) -> dict:
    episodes = list(episodes)
    count = len(episodes)
    outcomes = [ep.outcome for ep in episodes]
    wins, losses, draws, timeouts = (outcomes.count(name) for name in ("win", "loss", "draw", "timeout"))
    completed = wins + losses + draws
    return {
        "episodes": count,
        "completed_games": completed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "positive_timeouts": sum(ep.outcome == "timeout" and ep.episode_return > 0 for ep in episodes),
        "mean_return": sum(ep.episode_return for ep in episodes) / count if count else None,
        "mean_episode_length": sum(ep.episode_length for ep in episodes) / count if count else None,
        "completed_win_rate": wins / completed if completed else None,
        "timeout_fraction": timeouts / count if count else None,
        "positive_return_fraction": sum(ep.episode_return > 0 for ep in episodes) / count if count else None,
    }


class EpisodeWindow:
    """Episodes ending in (max(start_step, step - window_steps), step], in aggregate env steps.

    start_step is the run's first step (nonzero for fine-tuning), so no window reaches before it.

    These are training observations, potentially produced by different policy versions.
    Ongoing episodes are censored: counts and lengths must accompany the averages.
    """

    def __init__(self, window_steps: int, start_step: int = 0):
        if window_steps <= 0 or start_step < 0:
            raise ValueError("window_steps must be positive and start_step nonnegative")
        self.window_steps = window_steps
        self.start_step = start_step
        self.episodes: deque[Episode] = deque()
        self._last_added_step = -1
        self._last_reported_step = -1

    def add(self, episode: Episode):
        if episode.global_step < max(self._last_added_step, self._last_reported_step):
            raise ValueError("episode end steps must be nondecreasing")
        self._last_added_step = episode.global_step
        self.episodes.append(episode)

    def metrics(self, global_step: int) -> dict:
        if global_step < max(self._last_added_step, self._last_reported_step, self.start_step):
            raise ValueError("report step must not precede recorded episodes or the run's start")
        self._last_reported_step = global_step
        start = max(self.start_step, global_step - self.window_steps)
        while self.episodes and self.episodes[0].global_step <= start:
            self.episodes.popleft()
        return {
            "start_step_exclusive": start,
            "end_step_inclusive": global_step,
            "observed_window_steps": global_step - start,
            "full_window": global_step - self.start_step >= self.window_steps,
            **summarize_episodes(self.episodes),
        }
