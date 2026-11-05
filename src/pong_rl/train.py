# Adapted from CleanRL's ppo_atari.py and ppo_atari_lstm.py:
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import argparse
import hashlib
import io
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import wandb

from pong_rl.envs import make_env
from pong_rl.models import make_agent
from pong_rl.rollouts import RallyCollector
from pong_rl.ppo import update_policy
from pong_rl.metrics import Episode, EpisodeWindow

gym.register_envs(ale_py)

# Frames per observation: the CNN sees motion through stacking, the recurrent agents through memory
FRAME_STACK = {"cnn": 4, "lstm": 1, "attn_lstm": 1}
# Milestones in aggregate environment steps. A from-scratch run saves its untrained weights at 0;
# a fine-tuning run saves its starting (parent) weights and then every later milestone.
CHECKPOINT_STEPS = (0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 7_500_000, 10_000_000, 11_000_000, 12_000_000, 13_000_000, 14_000_000, 15_000_000)
CONDITION_LABELS = {(False, False): "clean", (False, True): "mask", (True, False): "fog", (True, True): "fog_mask"}


def condition_id(use_fog: bool, mask_enemy: bool) -> str:
    """Machine-readable condition stored inside checkpoints and checked by pong-eval."""
    return f"fog_{int(use_fog)}_mask_{int(mask_enemy)}"


def condition_label(use_fog: bool, mask_enemy: bool) -> str:
    """Human-readable condition used in folder names."""
    return CONDITION_LABELS[(bool(use_fog), bool(mask_enemy))]


def step_label(step: int) -> str:
    """7_500_000 -> 07.5M, so names sort by step. Other step counts stay exact."""
    if step % 100_000 == 0:
        return f"{step // 1_000_000:02d}.{step % 1_000_000 // 100_000}M"
    return str(step)


def checkpoint_filename(step: int) -> str:
    return f"step_{step_label(step)}.pt"


@dataclass
class Args:
    exp_name: str = "ppo_pong"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "pong-rl"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = True
    report_window_steps: int = 500_000
    init_from: str | None = None

    # Algorithm specific arguments
    env_id: str = "ALE/Pong-v5"
    arch: str = "cnn"
    use_fog: bool = False
    mask_enemy: bool = False
    total_timesteps: int = 15_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    rallies_per_update: int = 5
    rollout_scheme: str = "complete_rallies"
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    replay_checkpoint_threshold: int = 2048
    parallel_envs: bool = False


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_pong")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--arch",
        choices=list(FRAME_STACK),
        default="cnn",
        help="cnn: Nature CNN on 4 stacked frames; lstm: Nature CNN + LSTM on 1 frame per step; "
             "attn_lstm: LSTM reading the CNN's 7x7 feature map through spatial attention, 1 frame per step",
    )
    parser.add_argument("--use-fog", action="store_true", help="Enable FogWrapper on agent half")
    parser.add_argument("--mask-enemy", action="store_true", help="Enable EnemyMaskWrapper on enemy paddle")
    parser.add_argument("--total-timesteps", type=int, default=15_000_000,
                        help="target aggregate environment step; a fine-tune continues its parent's step count")
    parser.add_argument("--init-from", type=str, default=None,
                        help="checkpoint to fine-tune from (same --arch); fresh Adam, LR annealed over the fine-tune")
    parser.add_argument("--parallel-envs", action="store_true",
                        help="collect active Pong lanes in separate CPU processes")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rallies-per-update", type=int, default=5,
                        help="complete rallies per env, flushed early when its game ends")
    parser.add_argument("--replay-checkpoint-threshold", type=int, default=2048,
                        help="retain replay activations up to this many padded steps; 0 always recomputes")
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project-name", type=str, default="pong-rl")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-window-steps", type=int, default=500_000,
                        help="aggregate environment-step window for episodes ending in (step-window, step]")
    parsed = parser.parse_args()
    for name in ("report_window_steps", "rallies_per_update", "num_envs", "total_timesteps"):
        if getattr(parsed, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if parsed.replay_checkpoint_threshold < 0:
        parser.error("--replay-checkpoint-threshold must be nonnegative")

    return Args(
        exp_name=parsed.exp_name,
        seed=parsed.seed,
        replay_checkpoint_threshold=parsed.replay_checkpoint_threshold,
        parallel_envs=parsed.parallel_envs,
        arch=parsed.arch,
        use_fog=parsed.use_fog,
        mask_enemy=parsed.mask_enemy,
        total_timesteps=parsed.total_timesteps,
        num_envs=parsed.num_envs,
        rallies_per_update=parsed.rallies_per_update,
        track=parsed.track,
        wandb_project_name=parsed.wandb_project_name,
        wandb_entity=parsed.wandb_entity,
        capture_video=parsed.capture_video,
        save_model=parsed.save_model,
        report_window_steps=parsed.report_window_steps,
        init_from=parsed.init_from,
    )


def load_parent(path: str, arch: str) -> tuple[dict, dict]:
    """Load a checkpoint to fine-tune from: (weights, metadata recorded in every child checkpoint)."""
    data = Path(path).read_bytes()
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    parent_args = checkpoint["args"]
    parent_arch = checkpoint.get("arch", parent_args.get("arch", "cnn"))
    if parent_arch != arch:
        raise ValueError(f"--arch {arch} does not match the parent checkpoint's architecture ({parent_arch})")
    if checkpoint.get("global_step") is None:
        raise ValueError("parent checkpoint does not record its global_step")
    metadata = {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "run": checkpoint.get("run"),
        "condition": condition_id(parent_args["use_fog"], parent_args["mask_enemy"]),
        "condition_label": condition_label(parent_args["use_fog"], parent_args["mask_enemy"]),
        "seed": parent_args["seed"],
        "global_step": int(checkpoint["global_step"]),
        "checkpoint_step": checkpoint.get("checkpoint_step"),
        "updates": checkpoint.get("updates"),
    }
    return checkpoint["model_state_dict"], metadata


def save_checkpoint(agent, args, run, global_step, updates, optimizer_steps, checkpoint_step=None):
    """Save evaluation weights atomically; distinguish requested milestone from actual experience.

    checkpoint_step=None writes final.pt, the end-of-training model (also the run's completion marker).
    """
    name = "final.pt" if checkpoint_step is None else checkpoint_filename(checkpoint_step)
    path = Path("checkpoints") / run["dir"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save({"model_state_dict": agent.state_dict(), "args": vars(args),
                        "arch": args.arch, "condition": run["condition"], "seed": args.seed,
                        "run": run["dir"], "kind": run["kind"], "start_step": run["start_step"],
                        "init_from": run["init_from"], "steps_since_start": global_step - run["start_step"],
                        "global_step": global_step, "updates": updates,
                        "optimizer_steps": optimizer_steps, "checkpoint_step": checkpoint_step}, stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Saved model checkpoint to {path} (actual step={global_step}, milestone={checkpoint_step})")
    if args.track:
        wandb.save(str(path), policy="now")


def main():
    # Limit CPU thread contention when running multiple parallel training jobs
    torch.set_num_threads(1)

    args = parse_args()
    for name in ("num_envs", "rallies_per_update", "total_timesteps", "num_minibatches", "update_epochs", "report_window_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    condition = condition_id(args.use_fog, args.mask_enemy)
    label = condition_label(args.use_fog, args.mask_enemy)
    parent_state, parent = load_parent(args.init_from, args.arch) if args.init_from else (None, None)
    kind = "finetune" if parent else "scratch"
    # A fine-tune continues its parent's step count, so both kinds share one step axis. Its starting
    # weights are labelled by the parent's milestone (e.g. 10.0M), else by the parent's actual step.
    start_step = parent["global_step"] if parent else 0
    initial_checkpoint = parent["checkpoint_step"] if parent and parent["checkpoint_step"] is not None else start_step
    # The folder names the parent checkpoint, so fine-tunes from different parents never share files.
    run_label = f"{parent['condition_label']}_{step_label(initial_checkpoint)}_to_{label}" if parent else label
    if start_step >= args.total_timesteps:
        raise ValueError(f"--total-timesteps ({args.total_timesteps}) must exceed the starting step ({start_step})")
    run = {"dir": f"{kind}/{args.arch}/{run_label}/seed_{args.seed}", "kind": kind,
           "condition": condition, "start_step": start_step, "init_from": parent}
    tag = f"{kind}_{args.arch}_{run_label}"
    started = int(time.time())
    run_name = f"{tag}__seed_{args.seed}__{started}"
    episode_window = EpisodeWindow(args.report_window_steps, start_step)
    log_dir = Path("logs") / run["dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    episode_log_path = log_dir / f"episodes_{started}.jsonl"
    print(f"Episode outcomes: {episode_log_path}")

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config={**vars(args), "kind": kind, "run": run["dir"], "start_step": start_step, "init_from": parent},
            group=tag,
            name=run_name,
            save_code=True,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[{run_name}] Using device: {device}")

    # Env setup
    from pong_rl.parallel_envs import ParallelEnvPool

    envs = (ParallelEnvPool if args.parallel_envs else gym.vector.SyncVectorEnv)(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                use_fog=args.use_fog,
                mask_enemy=args.mask_enemy,
                frame_stack=FRAME_STACK[args.arch],
            )
            for i in range(args.num_envs)
        ],
        **({} if args.parallel_envs else {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}),
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert envs.single_observation_space.shape is not None

    agent = make_agent(args.arch, envs).to(device)
    if parent_state is not None:
        agent.load_state_dict(parent_state, strict=True)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    def on_episode(episode):
        episode_window.add(episode)
        with open(episode_log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"arch": args.arch, "seed": args.seed, "condition": condition,
                                     "run": run["dir"], **episode.to_dict()}, allow_nan=False) + "\n")
        print(f"[{tag}|s{args.seed}] step={episode.global_step}, return={episode.episode_return}, "
              f"length={episode.episode_length}, outcome={episode.outcome}, "
              f"terminated={int(episode.terminated)}, truncated={int(episode.truncated)}")
        if args.track:
            wandb.log({"charts/episodic_return": episode.episode_return,
                       "charts/episodic_length": episode.episode_length,
                       "charts/episode_terminated": int(episode.terminated),
                       "charts/episode_truncated": int(episode.truncated),
                       "charts/episode_outcome": episode.outcome, "global_step": episode.global_step})

    start_time = time.time()
    iteration = 0
    global_step = start_step
    optimizer_steps = 0
    pending_checkpoints = iter(step for step in CHECKPOINT_STEPS
                               if initial_checkpoint < step <= args.total_timesteps)
    next_checkpoint = next(pending_checkpoints, None)
    try:
        if args.save_model:
            # Save the starting weights (random, or the parent's) before collecting experience or updating.
            save_checkpoint(agent, args, run, start_step, 0, 0, checkpoint_step=initial_checkpoint)
        collector = RallyCollector(envs, agent, device, args.seed, args.rallies_per_update, start_step)
        # Finish the last batch at natural boundaries; actual steps may exceed the target.
        while collector.global_step < args.total_timesteps:
            iteration += 1
            if args.anneal_lr:
                # Anneal over this run's own steps, so a fine-tune restarts from the base learning rate.
                optimizer.param_groups[0]["lr"] = args.learning_rate * max(
                    0.0, 1.0 - (collector.global_step - start_step) / (args.total_timesteps - start_step))
            collection_started = time.perf_counter()
            batch = collector.collect(args.gamma, args.gae_lambda, on_episode)
            collection_seconds = time.perf_counter() - collection_started
            global_step = batch.global_step
            update_started = time.perf_counter()
            losses = update_policy(agent, optimizer, batch, args, device)
            update_seconds = time.perf_counter() - update_started
            optimizer_steps += losses.pop("charts/optimizer_steps")
            # Save the first completed policy update at/after each milestone. A
            # long batch may cross multiple milestones; none should be skipped.
            while next_checkpoint is not None and global_step >= next_checkpoint:
                if args.save_model:
                    save_checkpoint(agent, args, run, global_step, iteration,
                                    optimizer_steps, checkpoint_step=next_checkpoint)
                next_checkpoint = next(pending_checkpoints, None)
            sps = int((global_step - start_step) / (time.time() - start_time))
            window_metrics = episode_window.metrics(global_step)
            print(f"[{tag}|s{args.seed}] update={iteration}, step={global_step}/{args.total_timesteps}, "
                  f"batch_steps={batch.steps}, rallies={batch.completed_rallies}, "
                  f"ended_games={batch.ended_games}, SPS={sps}, "
                  f"collect_s={collection_seconds:.3f}, update_s={update_seconds:.3f}")
            if args.track:
                wandb.log({**losses,
                           "charts/learning_rate": optimizer.param_groups[0]["lr"],
                           "charts/SPS": sps, "global_step": global_step,
                           "timing/collection_seconds": collection_seconds,
                           "timing/update_seconds": update_seconds,
                           "charts/steps_since_start": global_step - start_step,
                           "charts/updates": iteration, "charts/optimizer_steps": optimizer_steps,
                           "charts/batch/steps": batch.steps,
                           "charts/batch/completed_rallies": batch.completed_rallies,
                           "charts/batch/sequences": len(batch.rallies),
                           "charts/batch/max_sequence_steps": max(map(len, batch.rallies)),
                           "charts/batch/ended_games": batch.ended_games,
                           "charts/budget_overshoot_steps": max(0, global_step - args.total_timesteps),
                           **{f"charts/window/{key}": value for key, value in window_metrics.items()}})
    finally:
        envs.close()

    if args.save_model:
        save_checkpoint(agent, args, run, global_step, iteration, optimizer_steps)

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
