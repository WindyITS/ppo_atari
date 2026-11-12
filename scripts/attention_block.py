"""Does the attention-LSTM remember only what it looked at? A causal test (E10, docs/research_plan.md).

    uv run python scripts/attention_block.py --workers 4

Each attn_lstm checkpoint plays games on its own view (Player; env seeds --seed, --seed + 1, ..., chosen apart
from the frozen evaluation's 10000-10029 so the probe below is tested on games it was not fit on). Every time
the ball starts towards the agent, the game is saved and the flight is replayed in four branches:
    control      no intervention
    control_b    no intervention, another action seed (sampling-noise floor)
    block_ball   while the ball is visible on its way to the fog (from the save until it reaches x >= 42), the
                 agent may not attend to the 3x3 block of attention cells around the ball (columns limited to
                 1-5, between the paddles, so neither paddle's cells are ever blocked)
    block_sham   the same 3x3 block at the vertically mirrored height (moved further if it would overlap the
                 ball's block): as much attention removed, but not from the ball
    erase_ball   over the same steps, the ball is painted out of the frame the model sees (its 1-pixel-padded
                 box): nothing about the ball reaches the network at all. Compared with block_ball it measures
                 how much of the ball still gets past the attention block; run also on the plain lstm (which
                 has only control, control_b and erase_ball) as a reference without attention
All branches except control_b share one action seed. Per branch: whether the ball is returned (as in
scripts/memory_interventions.py) and how well a linear probe reads the hidden ball's height from memory while
the ball is in the agent's half and hidden. The probe (ridge on [h, c] -> ball height, penalty chosen by
game-grouped cross-validation) is fit on the checkpoint's own recording (scripts/record_rollouts.py, same
steps: hidden ball inside the agent's half). In the control branch the script also records how much attention
went to the ball's own cell while it was visible, for the correlational version of the question.
Output: JSON lines (default analysis/attention_block.jsonl; the first run, attn_lstm only and without erase_ball,
is archive/analysis/attention_block.v1.jsonl on the host), one row per checkpoint; reruns skip done rows.
"""

import argparse
import functools
import os
from pathlib import Path
import sys
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pong_rl.analysis import scene  # noqa: E402
from pong_rl.analysis.batch import run_all  # noqa: E402
from pong_rl.analysis.player import Player  # noqa: E402
from pong_rl.analysis.rollouts import WINDOW_END, WINDOW_START, incoming_passes, load  # noqa: E402

GRID, CELL = 7, 12
BLOCK_COLUMNS = (1, 5)  # attention-grid columns strictly between the paddles (enemy: 0, agent: 6)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
PADDLE_X, RETURN_X, MAX_STEPS = 70.0, 40.0, 150
BRANCHES = ("control", "control_b", "block_ball", "block_sham", "erase_ball")
LSTM_BRANCHES = ("control", "control_b", "erase_ball")


def branches_for(arch: str) -> tuple:
    return BRANCHES if arch.startswith("attn_lstm") else LSTM_BRANCHES


def erase_ball(obs: np.ndarray) -> np.ndarray:
    """The observation with the ball painted out of every frame (box padded by 1 pixel)."""
    out = obs.copy()
    for frame in out.reshape(-1, *out.shape[-2:]):
        box = scene.ball_box(frame, pad=1)
        if box is not None:
            frame[box] = scene.PONG_BG_VALUE
    return out


def block_around(row: int, col: int) -> np.ndarray:
    """bool (49,): the 3x3 block of cells centred on (row, col), clipped to the grid and to BLOCK_COLUMNS."""
    mask = np.zeros((GRID, GRID), bool)
    rows = slice(max(0, row - 1), min(GRID, row + 2))
    cols = slice(max(BLOCK_COLUMNS[0], col - 1), min(BLOCK_COLUMNS[1] + 1, col + 2))
    mask[rows, cols] = True
    return mask.reshape(-1)


def ball_cell(x: float, y: float) -> tuple[int, int]:
    return int(np.clip(y // CELL, 0, GRID - 1)), int(np.clip(x // CELL, 0, GRID - 1))


def sham_row(row: int) -> int:
    """A row for the sham block: the mirrored height, pushed away so the two 3x3 blocks do not overlap."""
    mirrored = GRID - 1 - row
    if abs(mirrored - row) >= 3:
        return mirrored
    return row + 3 if row + 3 <= GRID - 1 else row - 3


# ----- the probe -----

def fit(X, y, alpha):
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-4, np.inf, std)
    Z = (X - mean) / std
    ym = y.mean()
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return lambda Xn: ((np.atleast_2d(Xn) - mean) / std) @ beta + ym


def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def train_probe(recording: Path) -> tuple:
    """Ridge probe [h, c] -> ball height on the hidden steps of incoming passes; returns (predict, cv R^2)."""
    arrays, _ = load(recording)
    passes = incoming_passes(arrays)
    rows = np.concatenate([p.rows[p.hidden] for p in passes])
    X = np.concatenate([arrays["h"], arrays["c"]], axis=1)[rows].astype(float)
    y = arrays["ball_y"][rows].astype(float)
    groups = arrays["game"][rows]
    keep = np.isfinite(y)
    X, y, groups = X[keep], y[keep], groups[keep]
    games = np.unique(groups)
    folds = np.array_split(np.random.default_rng(0).permutation(games), 5)
    best, best_score = ALPHAS[0], -np.inf
    for alpha in ALPHAS:
        predictions = np.full(len(y), np.nan)
        for fold in folds:
            test = np.isin(groups, fold)
            predictions[test] = fit(X[~test], y[~test], alpha)(X[test])
        score = r2(y, predictions)
        if score > best_score:
            best, best_score = alpha, score
    return fit(X, y, best), best_score


# ----- the experiment -----

def play_branch(player: Player, branch: str, probe) -> dict:
    previous_x, reached = None, False
    errors, truth, preds, looked = [], [], [], []
    for k in range(MAX_STEPS):
        if player.done:
            return {"outcome": "undecided", "errors": errors, "truth": truth, "preds": preds, "looked": looked}
        where = player.scene_now()
        x, y = where["ball_x"], where["ball_y"]
        blocked, transform = None, None
        before_fog = x == x and x < WINDOW_START
        if before_fog and branch == "erase_ball":
            transform = erase_ball
        if before_fog and branch in ("block_ball", "block_sham"):
            row, col = ball_cell(x, y)
            blocked = block_around(row if branch == "block_ball" else sham_row(row), col)
        if not reached and x == x and x >= PADDLE_X:
            reached = True
        if reached and x == x and previous_x is not None and x < previous_x and x < RETURN_X:
            return {"outcome": "returned", "errors": errors, "truth": truth, "preds": preds, "looked": looked}
        step = player.step(transform=transform, blocked=blocked)
        if before_fog and branch == "control" and "attention" in step.internals:
            row, col = ball_cell(x, y)
            looked.append(float(step.internals["attention"][:, row * GRID + col].sum()))
        inside = x == x and WINDOW_START <= x < WINDOW_END and not where["visible"] and not reached
        if inside:
            p = float(probe(np.concatenate([step.internals["h"], step.internals["c"]]))[0])
            preds.append(p)
            truth.append(float(y))
            errors.append(abs(p - y))
        if step.reward != 0:
            return {"outcome": "returned" if step.reward > 0 else "missed", "errors": errors, "truth": truth,
                    "preds": preds, "looked": looked}
        previous_x = x if x == x else previous_x
    return {"outcome": "undecided", "errors": errors, "truth": truth, "preds": preds, "looked": looked}


def recording_for(checkpoint: Path, rollouts: Path) -> Path:
    """The recording of a checkpoint (scripts/record_rollouts.py names it after the run path and file stem)."""
    parts = checkpoint.parts
    i = parts.index("checkpoints")
    name = "__".join(parts[i + 1:-1]) + "__" + checkpoint.stem
    return rollouts / f"{name}.npz"


def run(path: Path, episodes: int, seed: int, max_events: int, rollouts: Path) -> dict:
    torch.set_num_threads(1)
    recording = recording_for(path, rollouts)
    probe, probe_r2 = train_probe(recording)
    player = Player(path)
    events = []
    try:
        for game in range(episodes):
            player.reset(seed + game)
            previous_x, armed, count = None, True, 0
            while not player.done:
                where = player.scene_now()
                x = where["ball_x"]
                if x != x or (previous_x is not None and x < previous_x):
                    armed = True
                incoming = x == x and previous_x is not None and x > previous_x and x < WINDOW_START
                if armed and incoming and count < max_events:
                    armed, count = False, count + 1
                    saved = player.save()
                    branch_seed = zlib.crc32(f"{seed + game}:{player.t}".encode())
                    outcomes = {}
                    for branch in branches_for(player.arch):
                        player.load(saved, reseed=branch_seed + (1 if branch == "control_b" else 0))
                        outcomes[branch] = play_branch(player, branch, probe)
                    player.load(saved)
                    events.append({"game": game, "t": player.t, "outcomes": outcomes})
                player.step()
                previous_x = x if x == x else previous_x
    finally:
        player.close()
    return summarize(path, player, events, probe_r2, recording, episodes, seed)


def summarize(path, player, events, probe_r2, recording, episodes, seed) -> dict:
    per_branch = {}
    names = branches_for(player.arch)
    for b in names:
        outcomes = [e["outcomes"][b]["outcome"] for e in events]
        decided = [o for o in outcomes if o != "undecided"]
        truth = np.concatenate([e["outcomes"][b]["truth"] for e in events]) if events else np.array([])
        preds = np.concatenate([e["outcomes"][b]["preds"] for e in events]) if events else np.array([])
        per_branch[b] = {"return_rate": sum(o == "returned" for o in decided) / len(decided) if decided else None,
                         "probe_mae": float(np.mean(np.abs(preds - truth))) if len(truth) else None,
                         "probe_r2": r2(truth, preds) if len(truth) > 20 else None, "hidden_steps": int(len(truth))}
    # paired per-event differences against control (events where both have hidden steps)
    paired = {}
    for b in names[1:]:
        d = [np.mean(e["outcomes"][b]["errors"]) - np.mean(e["outcomes"]["control"]["errors"])
             for e in events if e["outcomes"][b]["errors"] and e["outcomes"]["control"]["errors"]]
        r = [(e["outcomes"][b]["outcome"] == "returned") - (e["outcomes"]["control"]["outcome"] == "returned")
             for e in events if "undecided" not in (e["outcomes"][b]["outcome"], e["outcomes"]["control"]["outcome"])]
        paired[b] = {"mae_change": bootstrap(np.asarray(d, float)), "return_change": bootstrap(np.asarray(r, float))}
    # correlational: attention on the ball's cell before the fog vs probe error in the fog (control branch)
    looked = [np.mean(e["outcomes"]["control"]["looked"]) for e in events
              if e["outcomes"]["control"]["looked"] and e["outcomes"]["control"]["errors"]]
    error = [np.mean(e["outcomes"]["control"]["errors"]) for e in events
             if e["outcomes"]["control"]["looked"] and e["outcomes"]["control"]["errors"]]
    correlation = spearman(np.asarray(looked), np.asarray(error)) if len(looked) > 10 else None
    return {"checkpoint": str(path), "arch": player.arch, "train_view": player.train_view, "view": player.view,
            "kind": player.checkpoint.get("kind", "scratch"), "run": player.checkpoint.get("run"),
            "episodes": episodes, "seed": seed, "events": len(events), "recording": str(recording),
            "probe_cv_r2": probe_r2, "branches": per_branch, "paired": paired,
            "looked_vs_error": {"spearman": correlation, "n": len(looked),
                                "looked_mean": float(np.mean(looked)) if looked else None}}


def bootstrap(d: np.ndarray, n: int = 2000) -> list | None:
    if len(d) < 5:
        return None
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def default_checkpoints(root: Path) -> list[Path]:
    found = []
    for arch in ("attn_lstm", "lstm"):
        for view in ("fog_mask", "fog"):
            found += root.glob(f"scratch/{arch}/{view}/seed_*/final.pt")
            found += root.glob(f"finetune/{arch}/clean_10.0M_to_{view}/seed_*/final.pt")
    return sorted(found)


def describe(row: dict) -> str:
    b, p = row["branches"], row["paired"]
    def f(v, d=2):
        return "–" if v is None else f"{v:.{d}f}"
    return (f"{row['events']} events, probe cv R2 {f(row['probe_cv_r2'])}; return " +
            " ".join(f"{k}:{f(v['return_rate'])}" for k, v in b.items()) + "; probe MAE " +
            " ".join(f"{k}:{f(v['probe_mae'], 1)}" for k, v in b.items()) +
            f"; looked-vs-error rho {f(row['looked_vs_error']['spearman'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=None)
    parser.add_argument("--rollouts", type=Path, default=Path("analysis/rollouts"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--max-events", type=int, default=20, help="per game")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("analysis/attention_block.jsonl"))
    args = parser.parse_args()
    paths = [p.resolve().relative_to(Path.cwd()) if p.is_absolute() else p
             for p in (args.checkpoints or default_checkpoints(args.root))]
    failures = run_all(functools.partial(run, episodes=args.episodes, seed=args.seed, max_events=args.max_events,
                                         rollouts=args.rollouts), paths, args.output, "checkpoint", args.workers,
                       describe)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
