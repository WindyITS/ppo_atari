"""Evaluate frozen checkpoints under an explicit, shared protocol, on their own view or any other."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import gymnasium as gym
import torch

from pong_rl.envs import make_env
from pong_rl.metrics import Episode, summarize_episodes
from pong_rl.models import RecurrentAgent, make_agent, zero_lstm_state
from pong_rl.train import FRAME_STACK

# Test views: (use_fog, mask_enemy). "own" (the default) means the view the checkpoint was trained on.
VIEWS = {"clean": (False, False), "mask": (False, True), "fog": (True, False), "fog_mask": (True, True)}


def view_name(use_fog: bool, mask_enemy: bool) -> str:
    return next(name for name, flags in VIEWS.items() if flags == (use_fog, mask_enemy))


def _checkpoint_config(checkpoint: dict) -> dict:
    args = checkpoint["args"]
    arch = checkpoint.get("arch", args.get("arch", "cnn"))
    if arch not in FRAME_STACK or ("arch" in args and args["arch"] != arch):
        raise ValueError("unsupported or inconsistent checkpoint architecture")
    for key in ("env_id", "use_fog", "mask_enemy", "seed"):
        if key not in args:
            raise ValueError(f"checkpoint is missing args.{key}")
    if not isinstance(args["use_fog"], bool) or not isinstance(args["mask_enemy"], bool):
        raise ValueError("checkpoint masks must be boolean")
    condition = f"fog_{int(args['use_fog'])}_mask_{int(args['mask_enemy'])}"
    if checkpoint.get("condition", condition) != condition:
        raise ValueError("checkpoint condition disagrees with its saved arguments")
    return {**args, "arch": arch}


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    episodes: int = 20,
    seed: int = 10_000,
    policy: str = "stochastic",
    device: str = "cpu",
    view: str = "own",
    quiet: bool = False,
) -> dict:
    """view: "own" plays the checkpoint's training view; clean/mask/fog/fog_mask plays that view instead.
    quiet: no per-episode progress lines on stderr."""
    if view != "own" and view not in VIEWS:
        raise ValueError(f"view must be own or one of {', '.join(VIEWS)}")
    if episodes <= 0 or seed < 0 or seed + episodes - 1 >= 2**63:
        raise ValueError("episodes must be positive and seeds must fit in [0, 2**63)")
    if policy not in ("stochastic", "greedy"):
        raise ValueError("policy must be stochastic or greedy")
    path = Path(checkpoint_path)
    # Read once so the hash always identifies exactly the checkpoint evaluated.
    data = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    config = _checkpoint_config(checkpoint)
    arch = config["arch"]
    eval_flags = (config["use_fog"], config["mask_enemy"]) if view == "own" else VIEWS[view]
    with _evaluation_torch_settings():
        return _run_episodes(path, data, checkpoint, config, arch, episodes, seed, policy, device, eval_flags, quiet)


@contextmanager
def _evaluation_torch_settings():
    """Single-threaded, deterministic cuDNN while evaluating; the caller's settings are restored."""
    threads, deterministic = torch.get_num_threads(), torch.backends.cudnn.deterministic
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.backends.cudnn.deterministic = deterministic


def _run_episodes(path, data, checkpoint, config, arch, episodes, seed, policy, device, eval_flags, quiet) -> dict:
    use_fog, mask_enemy = eval_flags
    envs = gym.vector.SyncVectorEnv(
        [make_env(config["env_id"], 0, False, "evaluation", use_fog=use_fog,
                  mask_enemy=mask_enemy, frame_stack=FRAME_STACK[arch])],
        # Reset explicitly with the next evaluation seed after either end flag.
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    try:
        agent = make_agent(arch, envs)
        agent.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if any(not torch.isfinite(param).all().item() for param in agent.parameters()):
            raise ValueError("checkpoint contains non-finite weights")
        agent.to(device).eval()
        raw_env = envs.envs[0].unwrapped
        frame_cap = raw_env.ale.getInt("max_num_frames_per_episode") if hasattr(raw_env, "ale") else None
        records = []
        total_steps = 0
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            obs, _ = envs.reset(seed=episode_seed)
            # A separate action generator makes sampling independent of model initialization.
            generator = torch.Generator(device="cpu").manual_seed(episode_seed)
            state = zero_lstm_state(1, torch.device(device))
            episode_return, episode_length = 0.0, 0
            while True:
                tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                if isinstance(agent, RecurrentAgent):
                    hidden, state = agent.get_states(tensor, state, torch.zeros(1, device=device))
                else:
                    hidden = agent.get_hidden(tensor)
                logits = agent.actor(hidden)
                if policy == "greedy":
                    action = logits.argmax(dim=-1).cpu().numpy()
                else:
                    probabilities = torch.softmax(logits, dim=-1).cpu()
                    action = torch.multinomial(probabilities, 1, generator=generator).flatten().numpy()
                obs, rewards, terminated, truncated, _ = envs.step(action)
                episode_return += float(rewards[0])
                episode_length += 1
                total_steps += 1
                if terminated[0] or truncated[0]:
                    record = Episode(total_steps, episode_return, episode_length,
                                     bool(terminated[0]), bool(truncated[0]))
                    records.append({"env_seed": episode_seed,
                                    "action_seed": episode_seed if policy == "stochastic" else None,
                                    **record.to_dict()})
                    if not quiet:
                        print(f"{path.name}: episode {episode_index + 1}/{episodes}, "
                              f"return={episode_return}, length={episode_length}, outcome={record.outcome}",
                              file=sys.stderr, flush=True)
                    break
        summary = summarize_episodes(Episode(**{key: row[key] for key in
            ("global_step", "episode_return", "episode_length", "terminated", "truncated")}) for row in records)
        return {
            "checkpoint": str(path.resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "arch": arch,
            "training_seed": config["seed"],
            "training_target_timesteps": config.get("total_timesteps"),
            "training_global_step": checkpoint.get("global_step"),
            "checkpoint_step": checkpoint.get("checkpoint_step"),
            "run": checkpoint.get("run"),
            "kind": checkpoint.get("kind", "scratch"),
            "init_from": checkpoint.get("init_from"),
            "training_updates": checkpoint.get("updates"),
            "env_id": config["env_id"],
            # use_fog / mask_enemy describe the view the episodes were played on; train_* the training view.
            "use_fog": use_fog,
            "mask_enemy": mask_enemy,
            "eval_view": view_name(use_fog, mask_enemy),
            "train_use_fog": config["use_fog"],
            "train_mask_enemy": config["mask_enemy"],
            "train_view": view_name(config["use_fog"], config["mask_enemy"]),
            "max_num_frames_per_episode": frame_cap,
            "summary": summary,
            "episodes": records,
        }
    finally:
        envs.close()


def software_versions() -> dict:
    return {"python": platform.python_version(), "torch": str(torch.__version__), "gymnasium": gym.__version__,
            "ale-py": version("ale-py"), "numpy": version("numpy")}


def validate_output_path(output: Path | None, checkpoints: list[Path]):
    if output is None:
        return
    if output.exists() and not output.is_file():
        raise ValueError("output must be a file")
    for checkpoint in checkpoints:
        if output.resolve() == checkpoint.resolve() or (output.exists() and output.samefile(checkpoint)):
            raise ValueError("output must not overwrite an input checkpoint")


def write_report(output: Path, report: dict):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".pong-eval-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000, help="first evaluation seed; incremented each episode")
    parser.add_argument("--policy", choices=("stochastic", "greedy"), default="stochastic")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--view", choices=("own", *VIEWS), default="own",
                        help="view to play: own (each checkpoint's training view) or a fixed one for all")
    parser.add_argument("--output", type=Path, help="JSON report; saved after each checkpoint")
    args = parser.parse_args()
    if args.episodes <= 0 or args.seed < 0 or args.seed + args.episodes - 1 >= 2**63:
        parser.error("episodes must be positive and evaluation seeds must fit in [0, 2**63)")
    if any(not path.is_file() for path in args.checkpoints):
        parser.error("all checkpoint paths must be existing files")
    try:
        validate_output_path(args.output, args.checkpoints)
    except ValueError as exc:
        parser.error(str(exc))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "protocol": {
            "episodes_per_checkpoint": args.episodes,
            "env_seeds": list(range(args.seed, args.seed + args.episodes)),
            "policy": args.policy,
            "action_rng": "CPU torch.Generator reset to env_seed per episode" if args.policy == "stochastic" else None,
            "device": args.device,
            "view": args.view,
            "software_versions": software_versions(),
            "frozen_weights": True,
            "checkpoint_paths": [str(path.resolve()) for path in args.checkpoints],
            "completed_win_rate_denominator": "completed_games (timeouts excluded; draws included)",
        },
        "results": [],
    }
    for index, path in enumerate(args.checkpoints):
        report["results"].append(evaluate_checkpoint(path, episodes=args.episodes, seed=args.seed,
                                                    policy=args.policy, device=args.device, view=args.view))
        report["complete"] = index == len(args.checkpoints) - 1
        if args.output:
            write_report(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
