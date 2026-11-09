"""Reading the recordings written by scripts/record_rollouts.py.

The key unit is an incoming pass: the stretch during which the ball, moving towards the agent, crosses the
agent's half (WINDOW_START <= x < WINDOW_END, exactly where the fog lies). Defining passes by position rather
than by visibility gives the same window on every view, so clean- and fog-trained models compare directly.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

WINDOW_START = 42.0   # the fog's first column (and the court's midline)
WINDOW_END = 72.0     # the fog's last column + 1; the agent's paddle starts at 73
COURT_TOP, COURT_BOTTOM = 14.0, 76.0


def load(path: str | Path) -> tuple[dict, dict]:
    """(arrays, meta) for a recording; path may name the .npz, the .json or the common stem."""
    stem = str(path).removesuffix(".npz").removesuffix(".json")
    with np.load(stem + ".npz") as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads(Path(stem + ".json").read_text())
    return arrays, meta


@dataclass
class Pass:
    game: int
    rows: np.ndarray        # row indices of the pass's steps in the recording, in time order
    entry_y: float          # ball height at the first step inside the window
    entry_vy: float         # vertical speed (pixels per step) just before entering
    exit_y: float           # ball height at the last step inside the window (where it meets the paddle line)
    hidden: np.ndarray      # per step: ball not on the model's frame


def incoming_passes(arrays: dict, min_steps: int = 3) -> list[Pass]:
    """All incoming passes through the window with at least min_steps steps."""
    x, y, game = arrays["ball_x"], arrays["ball_y"], arrays["game"]
    passes = []
    n = len(x)
    i = 1
    while i < n:
        inside = x[i] == x[i] and WINDOW_START <= x[i] < WINDOW_END
        moving_in = x[i - 1] == x[i - 1] and x[i] > x[i - 1] and game[i] == game[i - 1]
        if inside and moving_in and not (WINDOW_START <= x[i - 1] < WINDOW_END):
            j = i
            while (j + 1 < n and game[j + 1] == game[i] and x[j + 1] == x[j + 1]
                   and WINDOW_START <= x[j + 1] < WINDOW_END and x[j + 1] >= x[j]):
                j += 1
            rows = np.arange(i, j + 1)
            if len(rows) >= min_steps:
                vy = y[i] - y[i - 1] if y[i - 1] == y[i - 1] else np.nan
                passes.append(Pass(int(game[i]), rows, float(y[i]), float(vy), float(y[j]),
                                   ~arrays["visible"][rows]))
            i = j + 1
        else:
            i += 1
    return passes


def extrapolate(y0: float, vy: float, steps: np.ndarray) -> np.ndarray:
    """Ball height after `steps` steps from height y0 at speed vy, bouncing off the walls (ideal reflection)."""
    span = COURT_BOTTOM - COURT_TOP
    raw = (y0 - COURT_TOP) + vy * steps
    folded = np.mod(raw, 2 * span)
    return COURT_TOP + np.where(folded > span, 2 * span - folded, folded)


def hits(arrays: dict) -> list[dict]:
    """Moments the agent returns the ball: the ball was near the agent's paddle (x >= 68) moving right and the
    next position moves left. offset = ball_y - agent_y at the last step before the bounce."""
    x, y, a, game = arrays["ball_x"], arrays["ball_y"], arrays["agent_y"], arrays["game"]
    out = []
    for i in range(1, len(x) - 1):
        if game[i] != game[i + 1] or game[i] != game[i - 1]:
            continue
        if x[i] == x[i] and x[i] >= 68 and x[i - 1] == x[i - 1] and x[i] >= x[i - 1] \
                and x[i + 1] == x[i + 1] and x[i + 1] < x[i] and a[i] == a[i]:
            out.append({"row": i, "game": int(game[i]), "offset": float(y[i] - a[i]),
                        "speed_in": float(x[i] - x[i - 1]), "vy_out": float(y[i + 1] - y[i])
                        if y[i + 1] == y[i + 1] else float("nan")})
    return out
