"""Untrained-network control for the probes (E3b, docs/research_plan.md; AtariARI's random-CNN baseline).

    uv run python scripts/random_control.py --workers 2 --match fog

For each recording (scripts/record_rollouts.py) the same games are played again by the trained agent (Player
reproduces them exactly; checked against the recorded actions and ball positions) while every observation is
also fed to a randomly initialised network of the same architecture (torch seed --init-seed), carrying its own
memory across steps exactly as the trained one does. Its state is what the probes read from the trained model:
[h, c] for lstm and attn_lstm, the 512 features for cnn. Saved next to the recording as <name>.random.npz with
one row per recorded step (key "memory"); scripts/probe_memory.py then adds it as the probe input "random".
Existing outputs are skipped.
"""

import argparse
import functools
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pong_rl.analysis.player import Player  # noqa: E402
from pong_rl.analysis.rollouts import load  # noqa: E402
from pong_rl.models import make_agent  # noqa: E402


def random_twin(checkpoint: str, folder: str, seed: int) -> str:
    """A copy of the checkpoint with freshly initialised (seeded) weights of the same architecture."""
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    arch = data.get("arch", data["args"]["arch"])
    channels = 4 if arch == "cnn" else 1
    spaces = type("Spaces", (), {"single_observation_space": gym.spaces.Box(0, 255, (channels, 84, 84), np.uint8),
                                 "single_action_space": gym.spaces.Discrete(6)})()
    torch.manual_seed(seed)
    data["model_state_dict"] = make_agent(arch, spaces).state_dict()
    path = Path(folder, "random.pt")
    torch.save(data, path)
    return str(path)


def memory_of(internals: dict) -> np.ndarray:
    if "h" in internals:
        return np.concatenate([internals["h"], internals["c"]])
    return internals["features"]


def run(recording: Path, init_seed: int, root: Path) -> dict:
    torch.set_num_threads(1)
    arrays, meta = load(recording)
    checkpoint = Path(meta["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    games = np.unique(arrays["game"])
    seed = int(meta.get("seed", 10_000))
    player = Player(checkpoint, view=meta.get("view", "own"))
    with tempfile.TemporaryDirectory() as folder:
        twin = Player(random_twin(str(checkpoint), folder, init_seed), view=player.view)
        rows, actions, ball_x = [], [], []
        try:
            for game in games:
                player.reset(seed + int(game))
                state = None
                while not player.done:
                    obs = player.obs.copy()
                    _, _, internals, state = twin.forward(obs, state)
                    rows.append(memory_of(internals).astype(np.float16))
                    step = player.step()
                    actions.append(step.action)
                    ball_x.append(step.scene["ball_x"])
        finally:
            player.close()
            twin.close()
    actions, ball_x = np.asarray(actions), np.asarray(ball_x, float)
    if len(actions) != len(arrays["action"]) or not (actions == arrays["action"]).all() \
            or not np.allclose(ball_x, arrays["ball_x"].astype(float), equal_nan=True, atol=0.01):
        raise RuntimeError(f"{recording.name}: replay does not match the recording")
    out = recording.with_name(recording.stem + ".random.npz")
    temporary = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(temporary, memory=np.stack(rows), init_seed=init_seed)
    os.replace(temporary, out)
    return {"recording": str(recording), "random": str(out), "steps": len(rows)}


def main():
    from pong_rl.analysis.batch import run_all
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, default=Path("analysis/rollouts"))
    parser.add_argument("--root", type=Path, default=Path("."), help="folder the recorded checkpoint paths are under")
    parser.add_argument("--match", default="", help="only recordings whose name contains this")
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("analysis/random_control.jsonl"))
    args = parser.parse_args()
    paths = sorted(p for p in args.rollouts.glob("*.npz") if args.match in p.name
                   and not p.name.endswith((".tmp.npz", ".random.npz"))
                   and not p.with_name(p.stem + ".random.npz").exists())
    if not paths:
        sys.exit("nothing to do")
    failures = run_all(functools.partial(run, init_seed=args.init_seed, root=args.root), paths, args.output,
                       "recording", args.workers, lambda row: f"{row['steps']} steps")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
