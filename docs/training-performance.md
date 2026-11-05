# Training performance — 2026-11-05

Measured on an RTX 3070 with 20 CPU cores, seed 99, `--parallel-envs`, in short benchmarks
(`--iterations 14`: 14 updates, the first two excluded) at an untrained policy. Learning was not measured, and later
training may differ. W&B, checkpoint writes and subprocess startup are excluded.

## Current defaults

Each batch: 16 games × 256 steps, then 4 epochs × 4 minibatches.

| Architecture | Before (rally batches, 8 games) | Now | Speed-up |
|---|---|---|---|
| `cnn` | 2,011 steps/s | 5,283 steps/s | 2.6× |
| `lstm` | 1,714 steps/s | 5,193 steps/s | 3.0× |
| `attn_lstm` | 984 steps/s | 2,069 steps/s | 2.1× |

"Before" is the code at `d4a5150`: five rallies per game, 8 games, 512-step rally cap, 4 × 4.

## What changed

| Change | Effect |
|---|---|
| Upload each batch's images to the GPU once per update | Before, every minibatch of every epoch gathered and copied images from CPU memory: about 80% of `cnn` update CPU time in a profile. One profiled `cnn` update took 0.63 s before (3,499-step batch) and 0.23 s after (3,103-step batch) |
| `lstm` replays a sequence with one cuDNN call | Sequences never cross a game reset, so the per-step loop is unnecessary. `lstm` update throughput with 8 games: 3,395 → 7,715 steps/s |
| `attn_lstm` uses a fused LSTM cell per step and batched matmuls for attention | Time loop 1.6× faster in a micro-benchmark (e.g. 74.8 → 46.1 ms for 128 steps × 10 sequences), same parameters (checkpoints load unchanged), outputs equal to within 5.1e-6 |
| 16 games instead of 8 | Of 8, 16, 24 and 32 games, 16 was fastest for `cnn` and `lstm`; for `attn_lstm`, 16 and 24 were within 3%. Above 20 game processes the CPU is oversubscribed and collection slows |
| Fixed-length rollouts instead of rally batches | No game waits for the slowest one; every policy call has a full batch of 16 |

## Rally batches versus fixed-length rollouts

Single job, optimized code, 16 games, 4 epochs:

| Architecture | Rally (256-step cap), 400-step minibatches | Fixed 256 steps, 400-step minibatches | Fixed 256 steps, 4 minibatches (default) |
|---|---|---|---|
| `cnn` | 4,515 steps/s | 5,263 steps/s | 5,283 steps/s |
| `lstm` | 3,915 steps/s | 4,334 steps/s | 5,193 steps/s |
| `attn_lstm` | 1,963 steps/s | 1,004 steps/s | 2,069 steps/s |

Rally batches vary in size, so they used step-sized minibatches; fixed-length batches always have
4,096 steps, so 4 minibatches of 1,024 steps each. `attn_lstm` needs few, wide minibatches with
fixed-length rollouts: its time loop runs once per step of the longest sequence in a minibatch, and
sequences are up to 256 steps long.

## GPU utilization

A single job alternates between collecting (the GPU mostly waits for the games) and updating
(the games wait), whatever the rollout scheme: mean utilization was 37–47% with the current defaults.
Running several jobs at once fills the gaps. With 4 simultaneous jobs of 8 games each, utilization was
71–72% for `cnn` and 88–93% for `attn_lstm` under both schemes. Utilization was sampled from
`nvidia-smi` (every 100 ms for single jobs, 200 ms for concurrent jobs), so treat it as approximate.
`run_all.sh` runs 8 jobs of 16 games each without `--parallel-envs`; that configuration was not benchmarked.

## Not adopted

| Option | Result |
|---|---|
| Mixed precision (fp16 autocast) | Nature CNN forward+backward 1.4–1.6× faster on batches of 256–320, slower on batches of 8–16 as used while collecting; convolutions are a minority of update time |
| Non-deterministic cuDNN | Convolutions about 15% faster, but runs would no longer be reproducible |
| Step-sized minibatches | Unnecessary with fixed-length rollouts: every batch has 4,096 steps |

Raw records: [fixed-length rollouts and the comparisons above](benchmarks/training_fixed_steps_2026-11-05.json)
and [the earlier rally-batch optimization](benchmarks/training_2026-11-05.json).

## Reproduce

```bash
uv run python scripts/benchmark_training.py --parallel-envs --iterations 14 --arch attn_lstm --gpu-util
uv run python scripts/benchmark_training.py --parallel-envs --iterations 14 --arch cnn --num-envs 8
uv run python scripts/benchmark_training.py --replay-checkpoint-threshold 0
```

Run benchmarks without competing GPU workloads. Parallel workers need additional host RAM,
so budget carefully when running many training jobs at once. Each update holds the batch's images
on the GPU (about 116 MB at the defaults for `cnn`; peak allocated memory for `attn_lstm` was 433 MiB).

Training logs include `collect_s` and `update_s` (W&B: `timing/collection_seconds` and
`timing/update_seconds`). Each run also writes
`updates_<time>.jsonl` next to its episode log, with per-update PPO diagnostics (approximate KL, clip
fraction, explained variance), batch shape and timing.
