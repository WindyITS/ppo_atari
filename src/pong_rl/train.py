# Adapted from CleanRL's ppo_atari.py and ppo_atari_lstm.py:
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
import random
import re
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
from pong_rl.rollouts import RolloutCollector
from pong_rl.ppo import update_policy
from pong_rl.metrics import Episode, EpisodeWindow

gym.register_envs(ale_py)

# Frames per observation: the CNN sees motion through stacking, the recurrent agents through memory
FRAME_STACK = {"cnn": 4, "lstm": 1, "attn_lstm": 1, "attn_lstm_wide": 1, "vit": 4, "vit_lstm": 1}
# Default initial Adam learning rate per architecture (annealed over the run; --learning-rate overrides it).
# The CNN-based agents keep the 1e-3 the first sweep was trained with (attn_lstm_wide too, so it differs from
# attn_lstm only in size). The vision transformers use 2.5e-4, the standard PPO Atari rate (Schulman et al.
# 2017, CleanRL), inside the 1e-4 to 3e-4 range ViTs usually train well with; 1e-3 is high for a transformer.
LEARNING_RATE = {"cnn": 1e-3, "lstm": 1e-3, "attn_lstm": 1e-3, "attn_lstm_wide": 1e-3, "vit": 2.5e-4,
                 "vit_lstm": 2.5e-4}
# Milestones in aggregate environment steps. A from-scratch run saves its untrained weights at 0;
# a fine-tuning run saves its starting (parent) weights and then every later milestone.
CHECKPOINT_STEPS = (0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 7_500_000, 10_000_000, 11_000_000, 12_000_000, 13_000_000, 14_000_000, 15_000_000)
# Extra milestones for runs that go past 15M steps (e.g. a fine-tune with a 15M budget ends at 25M).
LONG_RUN_CHECKPOINT_STEPS = (17_500_000, 20_000_000, 22_500_000, 25_000_000, 30_000_000)
# Top-level agent modules --reinit may reset to fresh (seeded) weights when fine-tuning.
REINIT_MODULES = ("network", "encoder", "lstm", "actor", "critic", "keys", "values", "query", "readout_mlp")
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
    run_tag: str | None = None  # appended to the run folder name, so a variant never shares files with a run
    reinit: tuple[str, ...] = ()  # modules of the parent re-initialised before fine-tuning
    resume: str | None = None  # milestone checkpoint of an interrupted run to continue in the same folder

    # Algorithm specific arguments
    env_id: str = "ALE/Pong-v5"
    arch: str = "cnn"
    use_fog: bool = False
    mask_enemy: bool = False
    total_timesteps: int = 15_000_000
    learning_rate: float | None = None  # None: LEARNING_RATE[arch]
    num_envs: int = 16
    num_steps: int = 256  # steps each game plays per batch, then the PPO update
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    replay_checkpoint_threshold: int = 2048
    parallel_envs: bool = False

    def __post_init__(self):
        if self.learning_rate is None:
            self.learning_rate = LEARNING_RATE[self.arch]


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_pong")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--arch",
        choices=list(FRAME_STACK),
        default="cnn",
        help="cnn: Nature CNN on 4 stacked frames; lstm: Nature CNN + LSTM on 1 frame per step; "
             "attn_lstm: LSTM reading the CNN's 7x7 feature map through spatial attention, 1 frame per step; "
             "attn_lstm_wide: attn_lstm with 8 wider heads and a 512-unit layer after the readout (lstm-sized); "
             "vit: vision transformer on 4 stacked frames; vit_lstm: vision transformer + LSTM on 1 frame per step",
    )
    parser.add_argument("--use-fog", action="store_true", help="Enable FogWrapper on agent half")
    parser.add_argument("--mask-enemy", action="store_true", help="Enable EnemyMaskWrapper on enemy paddle")
    parser.add_argument("--total-timesteps", type=int, default=15_000_000,
                        help="target aggregate environment step; a fine-tune continues its parent's step count")
    parser.add_argument("--init-from", type=str, default=None,
                        help="checkpoint to fine-tune from (same --arch); fresh Adam, LR annealed over the fine-tune")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="suffix for the run folder (letters, digits, '.', '-', '_'), e.g. a 15M-budget "
                             "fine-tune: finetune/lstm/clean_10.0M_to_fog_mask__ft15M/seed_1")
    parser.add_argument("--reinit", nargs="+", choices=REINIT_MODULES, default=(),
                        help="with --init-from: re-initialise these top-level modules (fresh weights seeded by "
                             "--seed) and load the rest from the parent")
    parser.add_argument("--resume", type=str, default=None,
                        help="continue an interrupted run from one of its milestone checkpoints, in the same folder "
                             "and on the same learning-rate schedule; all other settings are read from the "
                             "checkpoint (only --track/--no-track and --parallel-envs apply). Restores the Adam "
                             "state when the checkpoint holds it, else starts a fresh Adam (recorded).")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="initial Adam learning rate, annealed over the run; default depends on --arch: "
                             + ", ".join(f"{arch} {lr:g}" for arch, lr in LEARNING_RATE.items()))
    parser.add_argument("--ent-coef", type=float, default=0.02, help="PPO entropy bonus coefficient")
    parser.add_argument("--num-steps", type=int, default=Args.num_steps,
                        help="steps each game plays per batch before the PPO update")
    parser.add_argument("--update-epochs", type=int, default=Args.update_epochs, help="PPO epochs per batch")
    parser.add_argument("--num-minibatches", type=int, default=Args.num_minibatches,
                        help="minibatches per epoch")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="stop a batch's remaining epochs once an epoch's mean approx KL exceeds this")
    parser.add_argument("--parallel-envs", action="store_true",
                        help="run each game in its own CPU process")
    parser.add_argument("--num-envs", type=int, default=Args.num_envs, help="games played side by side")
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
    if parsed.resume is not None:
        return resume_args(parsed.resume, track=parsed.track, parallel_envs=parsed.parallel_envs)
    for name in ("report_window_steps", "num_envs", "total_timesteps"):
        if getattr(parsed, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if parsed.learning_rate is not None and (not np.isfinite(parsed.learning_rate) or parsed.learning_rate <= 0):
        parser.error("--learning-rate must be finite and positive")
    if not np.isfinite(parsed.ent_coef) or parsed.ent_coef < 0:
        parser.error("--ent-coef must be finite and nonnegative")
    for name in ("update_epochs", "num_minibatches", "num_steps"):
        if getattr(parsed, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if parsed.target_kl is not None and (not np.isfinite(parsed.target_kl) or parsed.target_kl <= 0):
        parser.error("--target-kl must be finite and positive")
    if parsed.replay_checkpoint_threshold < 0:
        parser.error("--replay-checkpoint-threshold must be nonnegative")
    if parsed.run_tag is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", parsed.run_tag):
        parser.error("--run-tag may only contain letters, digits, '.', '-' and '_'")
    if parsed.reinit and parsed.init_from is None:
        parser.error("--reinit needs --init-from")

    return Args(
        exp_name=parsed.exp_name,
        seed=parsed.seed,
        learning_rate=parsed.learning_rate,
        ent_coef=parsed.ent_coef,
        num_steps=parsed.num_steps,
        update_epochs=parsed.update_epochs,
        num_minibatches=parsed.num_minibatches,
        target_kl=parsed.target_kl,
        replay_checkpoint_threshold=parsed.replay_checkpoint_threshold,
        parallel_envs=parsed.parallel_envs,
        arch=parsed.arch,
        use_fog=parsed.use_fog,
        mask_enemy=parsed.mask_enemy,
        total_timesteps=parsed.total_timesteps,
        num_envs=parsed.num_envs,
        track=parsed.track,
        wandb_project_name=parsed.wandb_project_name,
        wandb_entity=parsed.wandb_entity,
        capture_video=parsed.capture_video,
        save_model=parsed.save_model,
        report_window_steps=parsed.report_window_steps,
        init_from=parsed.init_from,
        run_tag=parsed.run_tag,
        reinit=tuple(parsed.reinit),
    )


def resume_args(path: str, **overrides) -> Args:
    """The Args of the run a checkpoint belongs to, to continue it (--resume)."""
    saved = torch.load(path, map_location="cpu", weights_only=True)["args"]
    names = {field.name for field in dataclasses.fields(Args)}
    values = {key: value for key, value in saved.items() if key in names}
    values["reinit"] = tuple(values.get("reinit") or ())
    return Args(**{**values, **overrides, "resume": str(path)})


def load_resume(path: str, args: Args) -> dict:
    """A milestone checkpoint of an unfinished run, checked to belong to the run described by args."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_step") is None or checkpoint.get("run") is None:
        raise ValueError("--resume needs a milestone checkpoint (step_*.pt) of a run, not final.pt")
    folder = Path("checkpoints") / checkpoint["run"]
    if Path(path).resolve().parent != folder.resolve():
        raise ValueError(f"--resume: {path} is not inside its own run folder {folder}")
    if checkpoint.get("arch", args.arch) != args.arch or checkpoint["seed"] != args.seed:
        raise ValueError("--resume: checkpoint does not match the run's arch and seed")
    if checkpoint["global_step"] >= args.total_timesteps:
        raise ValueError("--resume: the checkpoint is already at the run's total steps")
    return checkpoint


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
    if "num_steps" not in parent_args or "rallies_per_update" in parent_args:
        print(f"Warning: {path} was trained with rally-based batches, not fixed-length rollouts; "
              "this fine-tune mixes collection schemes.")
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


def save_checkpoint(agent, args, run, global_step, updates, optimizer_steps, checkpoint_step=None, optimizer=None):
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
                        "optimizer_steps": optimizer_steps, "checkpoint_step": checkpoint_step,
                        # Adam's moments, so an interrupted run can be resumed exactly (--resume)
                        **({"optimizer_state_dict": optimizer.state_dict()} if optimizer is not None else {}),
                        "resumed_from": run.get("resumed_from")}, stream)
            # On disk before the rename, so a power cut cannot leave an empty or partial checkpoint behind it.
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
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
    for name in ("num_envs", "num_steps", "total_timesteps", "num_minibatches", "update_epochs", "report_window_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    condition = condition_id(args.use_fog, args.mask_enemy)
    label = condition_label(args.use_fog, args.mask_enemy)
    resumed = load_resume(args.resume, args) if args.resume else None
    if resumed is not None:
        # Continue the interrupted run: its folder, step origin (learning-rate schedule) and parent.
        parent_state, parent = None, resumed["init_from"]
        kind, start_step = resumed["kind"], resumed["start_step"]
        initial_checkpoint = resumed["checkpoint_step"]
        run_label = resumed["run"].split("/")[2]
        run = {"dir": resumed["run"], "kind": kind, "condition": condition, "start_step": start_step,
               "init_from": parent,
               "resumed_from": {"checkpoint": Path(args.resume).name, "global_step": int(resumed["global_step"]),
                                "fresh_optimizer": "optimizer_state_dict" not in resumed,
                                "earlier": resumed.get("resumed_from")}}
    else:
        parent_state, parent = load_parent(args.init_from, args.arch) if args.init_from else (None, None)
        kind = "finetune" if parent else "scratch"
        # A fine-tune continues its parent's step count, so both kinds share one step axis. Its starting
        # weights are labelled by the parent's milestone (e.g. 10.0M), else by the parent's actual step.
        start_step = parent["global_step"] if parent else 0
        initial_checkpoint = parent["checkpoint_step"] if parent and parent["checkpoint_step"] is not None else start_step
        # The folder names the parent checkpoint, so fine-tunes from different parents never share files.
        run_label = f"{parent['condition_label']}_{step_label(initial_checkpoint)}_to_{label}" if parent else label
        if args.run_tag:
            run_label = f"{run_label}__{args.run_tag}"
        if args.reinit:
            if parent is None:
                raise ValueError("reinit needs a parent checkpoint (init_from)")
            parent["reinit"] = sorted(args.reinit)
        if start_step >= args.total_timesteps:
            raise ValueError(f"--total-timesteps ({args.total_timesteps}) must exceed the starting step ({start_step})")
        run = {"dir": f"{kind}/{args.arch}/{run_label}/seed_{args.seed}", "kind": kind,
               "condition": condition, "start_step": start_step, "init_from": parent}
    # Steps already taken when this process starts (a resumed run continues from its checkpoint).
    first_step = int(resumed["global_step"]) if resumed is not None else start_step
    if args.save_model and (Path("checkpoints") / run["dir"] / "final.pt").exists():
        # Never overwrite a finished run's checkpoints: a variant needs its own --run-tag.
        raise ValueError(f"checkpoints/{run['dir']} already holds a finished run; choose a --run-tag")
    tag = f"{kind}_{args.arch}_{run_label}"
    started = int(time.time())
    run_name = f"{tag}__seed_{args.seed}__{started}"
    episode_window = EpisodeWindow(args.report_window_steps, first_step)
    log_dir = Path("logs") / run["dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    episode_log_path = log_dir / f"episodes_{started}.jsonl"
    update_log_path = log_dir / f"updates_{started}.jsonl"  # per-update PPO diagnostics and timing
    print(f"Episode outcomes: {episode_log_path}")

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config={**vars(args), "kind": kind, "run": run["dir"], "start_step": start_step, "init_from": parent,
                    "resumed_from": run.get("resumed_from")},
            group=tag,
            name=run_name,
            save_code=True,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    # Seeding (a resumed run draws fresh games and actions rather than replaying the run's first ones)
    seed = args.seed if resumed is None else (args.seed * 1_000_003 + first_step) % 2**31
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    if parent_state is not None and args.reinit:
        unknown = [name for name in args.reinit if not isinstance(getattr(agent, name, None), torch.nn.Module)
                   or not any(True for _ in getattr(agent, name).parameters())]
        if unknown:
            raise ValueError(f"--reinit: {args.arch} has no module {', '.join(unknown)}")
        # The agent was just built after seeding, so the skipped modules keep fresh, seeded weights.
        kept = {key: value for key, value in parent_state.items() if key.split(".")[0] not in args.reinit}
        missing, unexpected = agent.load_state_dict(kept, strict=False)
        if unexpected or any(key.split(".")[0] not in args.reinit for key in missing):
            raise ValueError("parent checkpoint does not match the agent outside the re-initialised modules")
    elif parent_state is not None:
        agent.load_state_dict(parent_state, strict=True)
    if resumed is not None:
        agent.load_state_dict(resumed["model_state_dict"], strict=True)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if resumed is not None and "optimizer_state_dict" in resumed:
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
    elif resumed is not None:
        print(f"Warning: {args.resume} holds no optimizer state; resuming with a fresh Adam")

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
    iteration = int(resumed["updates"]) if resumed is not None else 0
    global_step = first_step
    optimizer_steps = int(resumed["optimizer_steps"]) if resumed is not None else 0
    pending_checkpoints = iter(step for step in (*CHECKPOINT_STEPS, *LONG_RUN_CHECKPOINT_STEPS)
                               if initial_checkpoint < step <= args.total_timesteps)
    next_checkpoint = next(pending_checkpoints, None)
    try:
        if args.save_model and resumed is None:
            # Save the starting weights (random, or the parent's) before collecting experience or updating.
            save_checkpoint(agent, args, run, start_step, 0, 0, checkpoint_step=initial_checkpoint, optimizer=optimizer)
        collector = RolloutCollector(envs, agent, device, seed, args.num_steps, first_step)
        # The last batch is always completed, so steps may exceed the target by less than one batch.
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
                                    optimizer_steps, checkpoint_step=next_checkpoint, optimizer=optimizer)
                next_checkpoint = next(pending_checkpoints, None)
            sps = int((global_step - first_step) / (time.time() - start_time))
            window_metrics = episode_window.metrics(global_step)
            print(f"[{tag}|s{args.seed}] update={iteration}, step={global_step}/{args.total_timesteps}, "
                  f"batch_steps={batch.steps}, points={batch.points}, "
                  f"ended_games={batch.ended_games}, SPS={sps}, "
                  f"collect_s={collection_seconds:.3f}, update_s={update_seconds:.3f}")
            with open(update_log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"update": iteration, "global_step": global_step,
                                         "batch_steps": batch.steps, "sequences": len(batch.sequences),
                                         "points": batch.points, "ended_games": batch.ended_games,
                                         "learning_rate": optimizer.param_groups[0]["lr"],
                                         "optimizer_steps": optimizer_steps,
                                         "collect_s": collection_seconds, "update_s": update_seconds,
                                         "wall_s": time.time() - start_time,
                                         # Strict JSON: e.g. explained variance is NaN for constant returns.
                                         **{key: value if math.isfinite(value) else None
                                            for key, value in losses.items()}}, allow_nan=False) + "\n")
            if args.track:
                wandb.log({**losses,
                           "charts/learning_rate": optimizer.param_groups[0]["lr"],
                           "charts/SPS": sps, "global_step": global_step,
                           "timing/collection_seconds": collection_seconds,
                           "timing/update_seconds": update_seconds,
                           "charts/steps_since_start": global_step - start_step,
                           "charts/updates": iteration, "charts/optimizer_steps": optimizer_steps,
                           "charts/batch/steps": batch.steps,
                           "charts/batch/points": batch.points,
                           "charts/batch/sequences": len(batch.sequences),
                           "charts/batch/ended_games": batch.ended_games,
                           "charts/budget_overshoot_steps": max(0, global_step - args.total_timesteps),
                           **{f"charts/window/{key}": value for key, value in window_metrics.items()}})
    finally:
        envs.close()

    if args.save_model:
        save_checkpoint(agent, args, run, global_step, iteration, optimizer_steps, optimizer=optimizer)

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
