# Training optimization measurements — 2026-11-05

On the RTX 3070, seed 99, attention + LSTM, eight environments and unchanged PPO
hyperparameters, throughput improved from 702 to 1,079 aggregate environment
steps/sec in matched short benchmarks (12 updates, first two excluded). The
optimized run sustained 1,073 steps/sec over 30 updates, excluding its first two.
These are early-training measurements, not guarantees for longer rallies later.
W&B, checkpoint writes and subprocess startup are excluded from these benchmarks.

## Experiments

| Experiment | Result | Decision |
|---|---|---|
| Retain small-batch activations and reduce CPU–GPU waits | Fixed-batch PPO time: 1.628 → 0.890 s; peak allocated GPU memory: 89 → 200 MiB | Keep; retain at most 2048 padded steps, recompute above that |
| Split optimizer groups into rallies of similar lengths | 1.043 s versus 0.890 s without splitting | Discard; extra passes outweighed padding savings |
| Threaded collection | 671 collection steps/sec versus 2232 serial | Discard |
| Subprocess collection | 3197 collection steps/sec on the same 11,403 steps | Keep as optional `--parallel-envs` |
| Distribution validation and static replay indices | Small fixed-batch gain (0.892 → 0.884 s); validation-free collection also helped in end-to-end tests | Keep; reject nonfinite aggregate PPO losses |
| Compile encoder | Failed: no C compiler installed | Leave disabled |

Fixed-batch PPO results are medians of three measured repetitions after one warmup.
Early serial measurements overlapped CPU tests, so the exact percentage gain is
approximate. Raw records are in [the benchmark JSON](benchmarks/training_2026-11-05.json).
No mixed precision, recurrence truncation, epoch-count changes or environment-count
changes were used. Full-rally output and gradient comparisons pass. Subprocess tests
cover paused lanes, timeouts, reset memory, advantages and worker startup failures.
The full suite passed: 36 tests and 99 subtests.

## Reproduce

```bash
uv run python scripts/benchmark_training.py --parallel-envs --iterations 30
uv run python scripts/benchmark_training.py --replay-checkpoint-threshold 0
```

The second command tests the memory-saving replay setting with current code; it is
not the original baseline commit, which also synchronized metrics more frequently.
Run benchmarks without competing GPU workloads. Parallel workers require additional
host RAM and should be budgeted carefully when running many training jobs at once.

Training logs now include `collect_s` and `update_s`, with the same fields in W&B.
The optimized pilot uses:

```bash
uv run pong-rl --arch attn_lstm --seed 99 --total-timesteps 3000000 --parallel-envs
```
