"""Where the ball and the paddles are, read from an 84x84 grayscale Pong frame (after AtariPreprocessing).

Same conventions as scripts/fog_exposure.py: object pixels differ from the background (87) by more than
THRESHOLD; the ball is searched between the paddles, so a ball touching a paddle can be missed for a step.
"""

import numpy as np

from pong_rl.wrappers import COURT_Y_MAX, COURT_Y_MIN, PONG_BG_VALUE

THRESHOLD = 20
BALL_COLUMNS = slice(12, 73)   # between the paddles (enemy x 8-10, its blur reaches 11; agent x 73-75)
ENEMY_COLUMNS = slice(8, 11)
AGENT_COLUMNS = slice(73, 76)
ENEMY_REGION = slice(6, 12)    # what EnemyMaskWrapper paints over
FOG_COLUMNS = slice(42, 72)    # what FogWrapper paints over
SCORE_ROWS = slice(0, COURT_Y_MIN - 1)  # the score digits sit above the top wall
NAN = float("nan")


def _rows_and_columns(frame: np.ndarray, columns: slice):
    court = frame[COURT_Y_MIN:COURT_Y_MAX, columns].astype(np.int16)
    rows, cols = np.nonzero(np.abs(court - PONG_BG_VALUE) > THRESHOLD)
    return rows + COURT_Y_MIN, cols + columns.start


def locate(clean: np.ndarray, seen: np.ndarray | None = None) -> dict:
    """Positions on the clean frame (NaN when absent); visible = some ball pixel is on the seen frame.

    Keys: ball_x, ball_y, enemy_y, agent_y (centre rows), enemy_top, enemy_bottom, agent_top, agent_bottom,
    visible. With seen=None the ball counts as visible whenever it is on the clean frame.
    """
    rows, cols = _rows_and_columns(clean, BALL_COLUMNS)
    enemy_rows, _ = _rows_and_columns(clean, ENEMY_COLUMNS)
    agent_rows, _ = _rows_and_columns(clean, AGENT_COLUMNS)
    present = len(cols) > 0
    visible = present if seen is None else present and len(_rows_and_columns(seen, BALL_COLUMNS)[0]) > 0

    def centre(values):
        return float(values.mean()) if len(values) else NAN

    return {
        "ball_x": centre(cols), "ball_y": centre(rows),
        "enemy_y": centre(enemy_rows), "agent_y": centre(agent_rows),
        "enemy_top": float(enemy_rows.min()) if len(enemy_rows) else NAN,
        "enemy_bottom": float(enemy_rows.max()) if len(enemy_rows) else NAN,
        "agent_top": float(agent_rows.min()) if len(agent_rows) else NAN,
        "agent_bottom": float(agent_rows.max()) if len(agent_rows) else NAN,
        "visible": bool(visible),
    }


def ball_box(clean: np.ndarray, pad: int = 2) -> tuple[slice, slice] | None:
    """(rows, columns) around the ball on the clean frame, widened by pad pixels; None without a ball."""
    rows, cols = _rows_and_columns(clean, BALL_COLUMNS)
    if not len(cols):
        return None
    return (slice(max(COURT_Y_MIN, rows.min() - pad), min(COURT_Y_MAX, rows.max() + pad + 1)),
            slice(max(BALL_COLUMNS.start, cols.min() - pad), min(BALL_COLUMNS.stop, cols.max() + pad + 1)))


def mirror_enemy(frame: np.ndarray) -> np.ndarray:
    """The frame with the enemy paddle's columns flipped top to bottom inside the court, so the paddle shows
    at the mirrored height (row r -> COURT_Y_MIN + COURT_Y_MAX - 1 - r). Works on (..., 84, 84) arrays."""
    out = frame.copy()
    region = out[..., COURT_Y_MIN:COURT_Y_MAX, ENEMY_REGION]
    out[..., COURT_Y_MIN:COURT_Y_MAX, ENEMY_REGION] = region[..., ::-1, :]
    return out


def erase(frame: np.ndarray, rows: slice, columns: slice) -> np.ndarray:
    """The frame with a rectangle painted in the background value (court area only)."""
    out = frame.copy()
    out[..., rows, columns] = PONG_BG_VALUE
    return out
