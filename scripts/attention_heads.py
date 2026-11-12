"""Per-head attention around fog entry (E12c), for the attn_lstm and attn_lstm_wide recordings on their own view.

    uv run python scripts/attention_heads.py

scripts/attention_maps.py averages attention over heads; with the wide agent's 8 heads (the small agent has 2)
a few heads that track the ball are diluted by the others, so this prints, per recording and per head:
    ball before fog   attention on the ball's 3x3 cells 8 to 4 steps before the ball enters the fog
                      (returned flights only), mean over heads, best head, and heads above 0.4
    inside fog        the best head's attention on the hidden ball's cells, steps 1-8 after entry
    enemy paddle      attention on the enemy paddle's cells (every 7th step), mean and best head
Uniform attention would give about 0.18 on the ball and 0.06 on a paddle.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from attention_maps import STEPS_BEFORE, returned_flights, target_mask  # noqa: E402
from pong_rl.analysis.rollouts import load  # noqa: E402

for arch in ("attn_lstm", "attn_lstm_wide"):
    for path in sorted(Path("analysis/rollouts").glob(f"*__{arch}__*__final.npz")):
        if path.name.endswith(".random.npz"):
            continue
        arrays, meta = load(path)
        if meta["view"] != meta["train_view"]:
            continue
        att = arrays["attention"].astype(np.float32)  # (steps, heads, 49)
        heads = att.shape[1]
        out = {}
        for name, offsets in (("ball_before", range(-8, -3)), ("ball_inside", range(1, 9))):
            sums, n = np.zeros(heads), 0
            for entry in returned_flights(arrays):
                for o in offsets:
                    i = entry + o
                    if not (0 <= i < len(att)) or arrays["game"][i] != arrays["game"][entry]:
                        continue
                    m = target_mask("ball", arrays["ball_x"][i], arrays["ball_y"][i], arrays["agent_y"][i],
                                    arrays["enemy_y"][i])
                    if m is None:
                        continue
                    sums += att[i][:, m].sum(axis=1)
                    n += 1
            out[name] = sums / max(n, 1)
        enemy = []
        for i in range(0, len(att), 7):
            m = target_mask("enemy_paddle", 0, 0, arrays["agent_y"][i], arrays["enemy_y"][i])
            if m is not None:
                enemy.append(att[i][:, m].sum(axis=1))
        enemy = np.mean(enemy, axis=0)
        b = out["ball_before"]
        print(f"{meta['run']:55s} heads {heads}: ball before fog mean {b.mean():.2f} best {b.max():.2f} "
              f"(heads > 0.4: {(b > 0.4).sum()}); inside fog best {out['ball_inside'].max():.2f}; "
              f"enemy paddle mean {enemy.mean():.2f} best {enemy.max():.2f}")
