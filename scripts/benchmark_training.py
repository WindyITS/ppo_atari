"""Short isolated training benchmark, without W&B or checkpoint writes.

Example: uv run python scripts/benchmark_training.py --parallel-envs
"""

import argparse
import json
import time

import gymnasium as gym
import numpy as np
import torch

from pong_rl.envs import make_env
from pong_rl.models import make_agent
from pong_rl.parallel_envs import ParallelEnvPool
from pong_rl.ppo import update_policy
from pong_rl.rollouts import RallyCollector
from pong_rl.train import Args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--parallel-envs", action="store_true")
    parser.add_argument("--replay-checkpoint-threshold", type=int, default=2048)
    parsed = parser.parse_args()
    if not 0 <= parsed.warmup < parsed.iterations:
        parser.error("require 0 <= warmup < iterations")
    if parsed.replay_checkpoint_threshold < 0:
        parser.error("checkpoint threshold must be nonnegative")
    torch.set_num_threads(1)
    torch.manual_seed(99)
    np.random.seed(99)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Args(arch="attn_lstm", replay_checkpoint_threshold=parsed.replay_checkpoint_threshold)
    factories = [make_env("ALE/Pong-v5", i, False, "benchmark", frame_stack=1) for i in range(8)]
    envs = (ParallelEnvPool(factories) if parsed.parallel_envs else
            gym.vector.SyncVectorEnv(factories, autoreset_mode=gym.vector.AutoresetMode.DISABLED))
    rows = []

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    try:
        agent = make_agent("attn_lstm", envs).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        collector = RallyCollector(envs, agent, device, 99)
        print(json.dumps({"configuration": vars(parsed), "device": str(device),
                          "torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}), flush=True)
        for iteration in range(parsed.iterations):
            synchronize()
            start = time.perf_counter()
            batch = collector.collect(args.gamma, args.gae_lambda, lambda _: None)
            synchronize()
            collected = time.perf_counter()
            update_policy(agent, optimizer, batch, args, device)
            synchronize()
            end = time.perf_counter()
            row = {"iteration": iteration, "steps": batch.steps,
                   "collect": collected - start, "update": end - collected}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        envs.close()
    measured = rows[parsed.warmup:]
    collection = sum(row["collect"] for row in measured)
    update = sum(row["update"] for row in measured)
    print(json.dumps({"summary": True, "sps": sum(row["steps"] for row in measured) / (collection + update),
                      "collection_seconds": collection, "update_seconds": update}), flush=True)


if __name__ == "__main__":
    main()
