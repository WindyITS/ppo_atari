"""Short isolated training benchmark, without W&B or checkpoint writes.

Example: uv run python scripts/benchmark_training.py --parallel-envs --arch attn_lstm --gpu-util
"""

import argparse
import json
import subprocess
import time

import gymnasium as gym
import numpy as np
import torch

from pong_rl.envs import make_env
from pong_rl.models import make_agent
from pong_rl.parallel_envs import ParallelEnvPool
from pong_rl.ppo import update_policy
from pong_rl.rollouts import RolloutCollector
from pong_rl.train import FRAME_STACK, Args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--parallel-envs", action="store_true")
    parser.add_argument("--replay-checkpoint-threshold", type=int, default=2048)
    parser.add_argument("--arch", choices=["cnn", "lstm", "attn_lstm"], default="attn_lstm")
    parser.add_argument("--num-envs", type=int, default=Args.num_envs)
    parser.add_argument("--num-steps", type=int, default=Args.num_steps, help="steps per game per batch")
    parser.add_argument("--update-epochs", type=int, default=Args.update_epochs)
    parser.add_argument("--num-minibatches", type=int, default=Args.num_minibatches)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--gpu-util", action="store_true",
                        help="sample nvidia-smi GPU utilization every 100 ms during measured iterations")
    parsed = parser.parse_args()
    if not 0 <= parsed.warmup < parsed.iterations:
        parser.error("require 0 <= warmup < iterations")
    if parsed.replay_checkpoint_threshold < 0:
        parser.error("checkpoint threshold must be nonnegative")
    for name in ("num_envs", "num_steps", "update_epochs", "num_minibatches"):
        if getattr(parsed, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(parsed.seed)
    np.random.seed(parsed.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    args = Args(arch=parsed.arch, replay_checkpoint_threshold=parsed.replay_checkpoint_threshold,
                num_envs=parsed.num_envs, num_steps=parsed.num_steps,
                update_epochs=parsed.update_epochs, num_minibatches=parsed.num_minibatches)
    factories = [make_env("ALE/Pong-v5", i, False, "benchmark", frame_stack=FRAME_STACK[parsed.arch])
                 for i in range(parsed.num_envs)]
    envs = (ParallelEnvPool(factories) if parsed.parallel_envs else
            gym.vector.SyncVectorEnv(factories, autoreset_mode=gym.vector.AutoresetMode.DISABLED))
    rows = []

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    try:
        agent = make_agent(parsed.arch, envs).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        collector = RolloutCollector(envs, agent, device, parsed.seed, args.num_steps)
        print(json.dumps({"configuration": vars(parsed), "device": str(device),
                          "torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}), flush=True)
        sampler = None
        for iteration in range(parsed.iterations):
            if parsed.gpu_util and iteration == parsed.warmup:
                sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu",
                                            "--format=csv,noheader,nounits", "-lms", "100"],
                                           stdout=subprocess.PIPE, text=True)
            synchronize()
            start = time.perf_counter()
            batch = collector.collect(args.gamma, args.gae_lambda, lambda _: None)
            synchronize()
            collected = time.perf_counter()
            losses = update_policy(agent, optimizer, batch, args, device)
            synchronize()
            end = time.perf_counter()
            row = {"iteration": iteration, "steps": batch.steps, "sequences": len(batch.sequences),
                   "optimizer_steps": losses["charts/optimizer_steps"],
                   "collect": collected - start, "update": end - collected}
            rows.append(row)
            print(json.dumps(row), flush=True)
        gpu_util = None
        if sampler is not None:
            sampler.terminate()
            samples = [float(line) for line in sampler.communicate()[0].split() if line.strip()]
            gpu_util = sum(samples) / len(samples) if samples else None
    finally:
        envs.close()
    measured = rows[parsed.warmup:]
    collection = sum(row["collect"] for row in measured)
    update = sum(row["update"] for row in measured)
    steps = sum(row["steps"] for row in measured)
    print(json.dumps({"summary": True, "arch": parsed.arch, "num_envs": parsed.num_envs,
                      "num_steps": parsed.num_steps, "gpu_util_percent": gpu_util,
                      "sps": steps / (collection + update), "collect_sps": steps / collection,
                      "update_sps": steps / update, "collection_seconds": collection,
                      "update_seconds": update, "steps": steps,
                      "peak_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}),
          flush=True)


if __name__ == "__main__":
    main()
