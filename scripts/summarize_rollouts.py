"""Behaviour, saliency, attention and LSTM-gate statistics of every recording (E7, E8, E9 in docs/research_plan.md).

    uv run python scripts/summarize_rollouts.py              # analysis/rollouts -> analysis/rollout_summary.jsonl

Per recording (scripts/record_rollouts.py):
  play       mean return, points, incoming passes and agent hits per point, policy entropy, action shares
  passes     on incoming passes through the agent's half (pong_rl.analysis.rollouts): spread of the entry and
             exit heights (how stereotyped the rallies are), paddle movement and the share made in the last 4
             steps, distance from the paddle to the exit height 4 steps in and at the exit
  hits       where on its paddle the agent meets the ball (ball - paddle centre) and the outgoing vertical speed
  saliency   mean KL (policy change) and |value change| when each object is erased, split by the ball being
             in the agent's half on an incoming pass or not
  attention  attn_lstm: attention mass on the 7x7 cells holding the ball, the enemy paddle and the agent paddle
             (cell of an object = the cell its centre falls in; 12-pixel cells), split the same way
  gates      lstm: forget-gate activity recomputed from the checkpoint weights, the recorded encoder features and
             the previous step's h; mean, share of units with mean f > 0.9, median per-unit half-life in steps
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from pong_rl.analysis.rollouts import hits, incoming_passes, load

ACTIONS = ("noop", "fire", "up", "down", "up_fire", "down_fire")  # Pong: RIGHT moves the paddle up, LEFT down
CELL = 12


def in_window_mask(arrays, passes):
    mask = np.zeros(len(arrays["t"]), bool)
    for p in passes:
        mask[p.rows] = True
    return mask


def pass_stats(arrays, passes):
    if not passes:
        return {}
    agent = arrays["agent_y"]
    moves, lates, gap4, gap_exit = [], [], [], []
    for p in passes:
        ys = agent[p.rows]
        if not np.isfinite(ys).all() or len(ys) < 6:
            continue
        steps = np.abs(np.diff(ys))
        moves.append(steps.sum())
        lates.append(steps[-4:].sum())
        gap4.append(abs(ys[4] - p.exit_y))
        gap_exit.append(abs(ys[-1] - p.exit_y))
    total = float(np.sum(moves))
    return {"count": len(passes), "entry_y_std": float(np.std([p.entry_y for p in passes])),
            "exit_y_std": float(np.std([p.exit_y for p in passes])),
            "mean_length": float(np.mean([len(p.rows) for p in passes])),
            "hidden_fraction": float(np.mean(np.concatenate([p.hidden for p in passes]))),
            "paddle_move": float(np.mean(moves)) if moves else None,
            "late_share": float(np.sum(lates) / total) if total > 0 else None,
            "gap_after_4": float(np.median(gap4)) if gap4 else None,
            "gap_at_exit": float(np.median(gap_exit)) if gap_exit else None}


def saliency_stats(arrays, window, objects):
    out = {}
    sampled = np.isfinite(arrays["saliency_kl"][:, 0])
    for name, part in (("all", sampled), ("window", sampled & window), ("outside", sampled & ~window)):
        if part.sum() < 5:
            continue
        out[name] = {"steps": int(part.sum()),
                     "kl": dict(zip(objects, arrays["saliency_kl"][part].mean(0).round(5).tolist())),
                     "value": dict(zip(objects, arrays["saliency_value"][part].mean(0).round(5).tolist()))}
    return out


def cell_of(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    cell = np.full(len(x), -1)
    cell[ok] = (np.clip(y[ok] // CELL, 0, 6) * 7 + np.clip(x[ok] // CELL, 0, 6)).astype(int)
    return cell


def attention_stats(arrays, window):
    weights = arrays["attention"].astype(np.float32)  # (N, heads, 49)
    n = len(weights)
    cells = {"ball": cell_of(arrays["ball_x"], arrays["ball_y"]),
             "enemy": cell_of(np.full(n, 9.0), arrays["enemy_y"]),
             "agent": cell_of(np.full(n, 74.0), arrays["agent_y"])}
    out = {}
    for name, part in (("all", np.ones(n, bool)), ("window", window), ("outside", ~window)):
        row = {}
        for obj, cell in cells.items():
            ok = part & (cell >= 0)
            if ok.sum():
                mass = weights[np.flatnonzero(ok), :, cell[ok]]  # (k, heads)
                row[obj] = mass.mean(0).round(4).tolist()
        row["uniform"] = round(1 / 49, 4)
        out[name] = row
    return out


def gate_stats(arrays, checkpoint, window):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    w_ih, w_hh = state["lstm.weight_ih_l0"].numpy(), state["lstm.weight_hh_l0"].numpy()
    bias = (state["lstm.bias_ih_l0"] + state["lstm.bias_hh_l0"]).numpy()
    hidden = w_hh.shape[1]
    forget = slice(hidden, 2 * hidden)  # PyTorch gate order: input, forget, cell, output
    x = arrays["features"].astype(np.float32)
    h = arrays["h"].astype(np.float32)
    h_prev = np.vstack([np.zeros((1, hidden), np.float32), h[:-1]])
    first = np.r_[True, arrays["game"][1:] != arrays["game"][:-1]]
    h_prev[first] = 0
    f = 1 / (1 + np.exp(-(x @ w_ih[forget].T + h_prev @ w_hh[forget].T + bias[forget])))
    out = {}
    for name, part in (("all", np.ones(len(f), bool)), ("window", window), ("outside", ~window)):
        unit_mean = f[part].mean(0)
        half_life = np.log(0.5) / np.log(np.clip(unit_mean, 1e-6, 1 - 1e-6))
        out[name] = {"mean_f": float(unit_mean.mean()), "units_f_over_0.9": int((unit_mean > 0.9).sum()),
                     "median_half_life": float(np.median(half_life)),
                     "top10_half_life": float(np.sort(half_life)[-10:].mean())}
    return out


def summarize(path: Path, root: Path) -> dict:
    arrays, meta = load(path)
    passes = incoming_passes(arrays)
    window = in_window_mask(arrays, passes)
    probs = arrays["probs"].astype(np.float32)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1))).sum(1).mean())
    points = int((arrays["reward"] != 0).sum())
    hit_list = hits(arrays)
    offsets = np.array([h["offset"] for h in hit_list])
    vy = np.array([h["vy_out"] for h in hit_list])
    row = {"recording": path.stem, **{k: meta[k] for k in ("arch", "train_view", "view", "kind", "run", "checkpoint",
                                                            "mean_return", "steps")},
           "play": {"points": points, "won": int((arrays["reward"] > 0).sum()),
                    "passes_per_point": len(passes) / points if points else None,
                    "hits_per_point": len(hit_list) / points if points else None,
                    "entropy": entropy,
                    "actions": dict(zip(ACTIONS, np.bincount(arrays["action"].astype(int), minlength=6)
                                        .astype(float) / len(arrays["action"])))},
           "passes": pass_stats(arrays, passes),
           "hits": {"count": len(hit_list),
                    "abs_offset_median": float(np.median(np.abs(offsets))) if len(offsets) else None,
                    "offset_quartiles": np.percentile(offsets, [25, 50, 75]).tolist() if len(offsets) else None,
                    "abs_vy_out_median": float(np.nanmedian(np.abs(vy))) if len(vy) else None},
           "saliency": saliency_stats(arrays, window, meta.get("objects", ["ball", "enemy", "agent", "score"]))}
    if "attention" in arrays:
        row["attention"] = attention_stats(arrays, window)
    if "h" in arrays and "features" in arrays:
        checkpoint = Path(meta["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        row["gates"] = gate_stats(arrays, checkpoint, window)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, default=Path("analysis/rollouts"))
    parser.add_argument("--root", type=Path, default=Path("."), help="folder the recorded checkpoint paths are under")
    parser.add_argument("--match", default="")
    parser.add_argument("--output", type=Path, default=Path("analysis/rollout_summary.jsonl"))
    args = parser.parse_args()
    paths = sorted(p for p in args.rollouts.glob("*.npz") if args.match in p.name and not p.name.endswith(".tmp.npz"))
    if not paths:
        sys.exit("no recordings found")
    rows = []
    for path in paths:
        rows.append(summarize(path, args.root))
        r = rows[-1]
        print(f"{path.stem}: return {r['mean_return']:+.1f}, passes/point {r['play']['passes_per_point']:.2f}, "
              f"late share {r['passes'].get('late_share')}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
