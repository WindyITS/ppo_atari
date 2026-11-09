"""Linear probes: is the hidden ball's height decodable from a model's internal state? (E3, docs/research_plan.md)

    uv run python scripts/probe_memory.py                       # every recording in analysis/rollouts
    uv run python scripts/probe_memory.py --match fog_mask

For every incoming pass of the ball through the agent's half (pong_rl.analysis.rollouts.incoming_passes; under
fog this is where the ball is hidden) two targets are decoded at each step of the pass:
    now   the ball's current height
    exit  the height at which the ball leaves the window, i.e. where it meets the paddle line
from these inputs (whatever the recording holds):
    memory   lstm / attn_lstm: LSTM state [h, c]; cnn: its 512 features (computed from 4 stacked frames)
    frame    lstm: the encoder's 512 features of the current frame alone, before the LSTM (no memory)
    paddle   the agent's own paddle height (a position the agent controls, so it can carry information)
    physics  ideal bouncing extrapolation from the ball's height and speed when it entered the window (a
             reference: what perfect memory of the entry would give; not a probe)
Probes are ridge regressions, evaluated by 5-fold cross-validation over games (the ridge penalty picked by an
inner 4-fold split of the training games). Reported: R^2 and mean absolute error in pixels, overall and by
steps since entering the window (0-3: the cnn's stack still holds the ball under fog; 4-7; 8+).
Output: JSON lines (default analysis/probes.jsonl), one row per recording.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from pong_rl.analysis.rollouts import extrapolate, incoming_passes, load

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
BUCKETS = ((0, 3), (4, 7), (8, 99))
FOLDS = 5


def fit(X, y, alpha):
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-4, np.inf, std)  # features constant on the training games (dead units) are dropped
    Z = (X - mean) / std
    ym = y.mean()
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return lambda Xn: ((Xn - mean) / std) @ beta + ym


def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def cross_validated(X, y, groups, rng):
    """Out-of-fold predictions, folds over groups (games); alpha chosen on an inner split of the training games."""
    games = np.unique(groups)
    rng.shuffle(games)
    outer = np.array_split(games, FOLDS)
    prediction = np.full(len(y), np.nan)
    for test_games in outer:
        test = np.isin(groups, test_games)
        train_games = np.setdiff1d(games, test_games)
        inner = np.array_split(train_games, 4)
        scores = []
        for alpha in ALPHAS:
            s = []
            for val_games in inner:
                val = np.isin(groups, val_games)
                tr = ~test & ~val
                if tr.sum() < 10 or val.sum() < 5:
                    continue
                s.append(r2(y[val], fit(X[tr], y[tr], alpha)(X[val])))
            scores.append(np.mean(s) if s else -np.inf)
        alpha = ALPHAS[int(np.argmax(scores))]
        prediction[test] = fit(X[~test], y[~test], alpha)(X[test])
    return prediction


def summarize(y, p, tau):
    out = {"r2": r2(y, p), "mae": float(np.abs(y - p).mean()), "n": int(len(y))}
    for lo, hi in BUCKETS:
        m = (tau >= lo) & (tau <= hi)
        if m.sum() > 20:
            out[f"r2_{lo}_{hi}"] = r2(y[m], p[m])
            out[f"mae_{lo}_{hi}"] = float(np.abs(y[m] - p[m]).mean())
    return out


def probe(path: Path, seed: int = 0) -> dict:
    arrays, meta = load(path)
    passes = incoming_passes(arrays)
    rows = np.concatenate([p.rows for p in passes])
    tau = np.concatenate([np.arange(len(p.rows)) for p in passes])
    groups = arrays["game"][rows]
    targets = {"now": arrays["ball_y"][rows].astype(float),
               "exit": np.concatenate([np.full(len(p.rows), p.exit_y) for p in passes])}
    hidden = np.concatenate([p.hidden for p in passes])
    inputs = {}
    if "h" in arrays:
        inputs["memory"] = np.concatenate([arrays["h"], arrays["c"]], axis=1)[rows].astype(float)
        if "features" in arrays:
            inputs["frame"] = arrays["features"][rows].astype(float)
    else:
        inputs["memory"] = arrays["features"][rows].astype(float)
    inputs["paddle"] = arrays["agent_y"][rows, None].astype(float)
    keep = np.isfinite(inputs["paddle"][:, 0]) & np.isfinite(targets["now"])
    entry_y = np.concatenate([np.full(len(p.rows), p.entry_y) for p in passes])
    entry_vy = np.concatenate([np.full(len(p.rows), p.entry_vy) for p in passes])
    keep &= np.isfinite(entry_vy)
    rng = np.random.default_rng(seed)
    result = {"recording": str(path), **{k: meta[k] for k in ("arch", "train_view", "view", "kind", "run",
                                                              "checkpoint", "mean_return")},
              "passes": len(passes), "steps": int(keep.sum()), "hidden_fraction": float(hidden[keep].mean()),
              "probes": {}}
    for target, y in targets.items():
        y = y[keep]
        t = tau[keep]
        result["probes"][target] = {}
        for name, X in inputs.items():
            prediction = cross_validated(X[keep], y, groups[keep], rng)
            result["probes"][target][name] = summarize(y, prediction, t)
        lengths = np.concatenate([np.full(len(p.rows), len(p.rows) - 1) for p in passes])[keep]
        horizon = t if target == "now" else lengths
        result["probes"][target]["physics"] = summarize(y, extrapolate(entry_y[keep], entry_vy[keep], horizon), t)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, default=Path("analysis/rollouts"))
    parser.add_argument("--match", default="", help="only recordings whose name contains this")
    parser.add_argument("--output", type=Path, default=Path("analysis/probes.jsonl"))
    args = parser.parse_args()
    paths = sorted(p for p in args.rollouts.glob("*.npz") if args.match in p.name and not p.name.endswith(".tmp.npz"))
    if not paths:
        sys.exit("no recordings found")
    rows = []
    for path in paths:
        row = probe(path)
        rows.append(row)
        m = row["probes"]["exit"]
        print(f"{path.stem}: exit R2 " + ", ".join(f"{k} {v['r2']:.2f}" for k, v in m.items())
              + f" | late (8+) memory R2 {m['memory'].get('r2_8_99', float('nan')):.2f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
