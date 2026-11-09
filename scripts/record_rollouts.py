"""Record the frozen-evaluation games of many checkpoints with everything later analyses need (E2, E7 in
docs/research_plan.md).

    uv run python scripts/record_rollouts.py --workers 10              # finals + clean parents, own view
    uv run python scripts/record_rollouts.py --runs scratch/lstm/fog_mask/seed_1 --files step_05.0M.pt

Each checkpoint plays --episodes games (env and action seeds --seed, --seed + 1, ...: the frozen evaluation's
games) on its own view, through pong_rl.analysis.player.Player. One compressed .npz per checkpoint under
--out-dir (name: run path with / -> __, then the file stem) holds one row per agent step:
    game, t, action, reward (int), value, probs (6 actions), done
    ball_x, ball_y, enemy_y, agent_y (positions on the unoccluded frame, NaN when absent), visible
    features (cnn: 512 latent; lstm: 512 encoder output fed to the LSTM), h, c (recurrent), attention
    (attn_lstm: heads x 49 cells), all float16
    saliency_kl, saliency_value (every --saliency-every steps, NaN elsewhere): for each object in OBJECTS,
    the KL divergence from the policy to the policy with that object erased from every frame the model sees
    (same memory state), and the absolute change in value.
A .json next to it holds the checkpoint's identity and per-game returns. Existing outputs are skipped.
"""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pong_rl.analysis import scene  # noqa: E402
from pong_rl.analysis.player import Player  # noqa: E402
from pong_rl.wrappers import COURT_Y_MAX, COURT_Y_MIN, PONG_BG_VALUE  # noqa: E402

OBJECTS = ("ball", "enemy", "agent", "score")
POSITIONS = ("ball_x", "ball_y", "enemy_y", "agent_y")


def object_mask(frame: np.ndarray, name: str) -> np.ndarray:
    """Boolean mask of an object's pixels (with a 1-pixel margin) on one seen 84x84 frame."""
    mask = np.zeros(frame.shape, bool)
    if name == "score":
        mask[: COURT_Y_MIN - 1, :] = True
        return mask
    columns = {"ball": scene.BALL_COLUMNS, "enemy": scene.ENEMY_REGION, "agent": slice(72, 77)}[name]
    court = frame[COURT_Y_MIN:COURT_Y_MAX, columns].astype(np.int16)
    rows, cols = np.nonzero(np.abs(court - PONG_BG_VALUE) > scene.THRESHOLD)
    if len(rows):
        top, bottom = rows.min() + COURT_Y_MIN - 1, rows.max() + COURT_Y_MIN + 2
        left, right = cols.min() + columns.start - 1, cols.max() + columns.start + 2
        mask[max(COURT_Y_MIN, top):min(COURT_Y_MAX, bottom), max(columns.start, left):min(columns.stop, right)] = True
    return mask


def erase_object(obs: np.ndarray, name: str) -> np.ndarray:
    out = obs.copy()
    for i, frame in enumerate(out):
        frame[object_mask(frame, name)] = PONG_BG_VALUE
    return out


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    p, q = p.clamp_min(1e-8), q.clamp_min(1e-8)
    return float((p * (p.log() - q.log())).sum())


def record(path: str, episodes: int, seed: int, saliency_every: int, view: str = "own") -> dict:
    player = Player(path, view=view)
    rows = {key: [] for key in ("game", "t", "action", "reward", "value", "done", "visible", *POSITIONS)}
    probs, internals = [], {}
    sal_kl, sal_value = [], []
    returns = []
    try:
        for game in range(episodes):
            player.reset(seed + game)
            while not player.done:
                before = player.state
                obs = player.obs
                if saliency_every and player.t % saliency_every == 0:
                    base_probs, base_value, _, _ = player.forward(obs, before)
                    kls, dvs = [], []
                    for name in OBJECTS:
                        p, v, _, _ = player.forward(erase_object(obs, name), before)
                        kls.append(kl(base_probs[0], p[0]))
                        dvs.append(abs(v - base_value))
                    sal_kl.append(kls)
                    sal_value.append(dvs)
                else:
                    sal_kl.append([np.nan] * len(OBJECTS))
                    sal_value.append([np.nan] * len(OBJECTS))
                step = player.step()
                rows["game"].append(game)
                rows["t"].append(step.t)
                rows["action"].append(step.action)
                rows["reward"].append(step.reward)
                rows["value"].append(step.value)
                rows["done"].append(step.done)
                rows["visible"].append(step.scene["visible"])
                for key in POSITIONS:
                    rows[key].append(step.scene[key])
                probs.append(step.probs)
                for key, value in step.internals.items():
                    internals.setdefault(key, []).append(value)
            returns.append(player.episode_return)
    finally:
        player.close()
    arrays = {
        "game": np.array(rows["game"], np.int16), "t": np.array(rows["t"], np.int32),
        "action": np.array(rows["action"], np.int8), "reward": np.array(rows["reward"], np.int8),
        "value": np.array(rows["value"], np.float32), "done": np.array(rows["done"], bool),
        "visible": np.array(rows["visible"], bool), "probs": np.array(probs, np.float16),
        "saliency_kl": np.array(sal_kl, np.float32), "saliency_value": np.array(sal_value, np.float32),
        **{key: np.array(rows[key], np.float32) for key in POSITIONS},
        **{key: np.array(values, np.float16) for key, values in internals.items()},
    }
    meta = {"checkpoint": str(Path(path)), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "arch": player.arch, "train_view": player.train_view, "view": player.view,
            "kind": player.checkpoint.get("kind", "scratch"), "run": player.checkpoint.get("run"),
            "global_step": player.checkpoint.get("global_step"), "episodes": episodes, "seed": seed,
            "policy": "stochastic", "returns": returns, "mean_return": float(np.mean(returns)),
            "steps": len(rows["t"]), "saliency_every": saliency_every, "objects": list(OBJECTS)}
    return {"arrays": arrays, "meta": meta}


def output_name(root: Path, path: Path, view: str) -> str:
    stem = "__".join(path.parent.relative_to(root).parts) + "__" + path.stem
    return stem if view == "own" else f"{stem}__on_{view}"


def default_checkpoints(root: Path) -> list[Path]:
    """Scratch and fine-tune finals of cnn, lstm, attn_lstm (untagged, finished) and the clean 10M parents."""
    found = []
    for arch in ("cnn", "lstm", "attn_lstm"):
        found += root.glob(f"scratch/{arch}/*/seed_*/final.pt")
        found += root.glob(f"finetune/{arch}/clean_10.0M_to_*/seed_*/final.pt")
        found += root.glob(f"scratch/{arch}/clean/seed_*/step_10.0M.pt")
    return sorted(p for p in found if ".aborted_" not in str(p) and "__" not in p.parent.parent.name
                  and (p.parent / "final.pt").is_file())


def work(job):
    path, out, episodes, seed, saliency_every, view = job
    torch.set_num_threads(1)
    started = time.time()
    result = record(path, episodes, seed, saliency_every, view)
    temporary = Path(str(out) + ".tmp.npz")
    np.savez_compressed(temporary, **result["arrays"])
    os.replace(temporary, Path(str(out) + ".npz"))
    result["meta"]["seconds"] = round(time.time() - started, 1)
    Path(str(out) + ".json").write_text(json.dumps(result["meta"], indent=1) + "\n")
    return result["meta"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--runs", nargs="+", default=None, help="run folders under --root (default: finals + parents)")
    parser.add_argument("--files", nargs="+", default=["final.pt"], help="checkpoint files of each --runs folder")
    parser.add_argument("--view", default="own", choices=("own", "clean", "mask", "fog", "fog_mask"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--saliency-every", type=int, default=8, help="0 disables saliency")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/rollouts"))
    args = parser.parse_args()
    if args.runs:
        paths = [args.root / run / name for run in args.runs for name in args.files]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            parser.error("missing checkpoints: " + ", ".join(missing))
    else:
        paths = default_checkpoints(args.root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in paths:
        out = args.out_dir / output_name(args.root, path, args.view)
        if not Path(str(out) + ".npz").is_file():
            jobs.append((str(path), out, args.episodes, args.seed, args.saliency_every, args.view))
    print(f"{len(paths)} checkpoints, {len(jobs)} to record, {args.workers} workers", flush=True)
    failures = 0
    with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(work, job): job for job in jobs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            job = futures[future]
            try:
                meta = future.result()
                print(f"[{done}/{len(jobs)}] {job[0]}: mean return {meta['mean_return']:+.1f}, {meta['steps']} steps, "
                      f"{meta['seconds']:.0f} s", flush=True)
            except Exception as exc:  # report and carry on with the others
                failures += 1
                print(f"[{done}/{len(jobs)}] {job[0]}: FAILED {exc!r}", file=sys.stderr, flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
