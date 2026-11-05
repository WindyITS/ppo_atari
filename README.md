# Pong RL: Memory vs. Visual Shortcuts under Occlusion

Course project investigating whether a pixel-based PPO agent in Atari Pong (`ALE/Pong-v5`) tracks hidden ball state in its latent representations or relies on visual shortcuts (such as tracking the opponent's paddle) when the ball is occluded.

![Wrapper Conditions](assets/four_states_grid.png)

## Project Structure

```text
.
├── assets/                 # Visualizations of the 4 wrapper states
├── scripts/
│   └── render_states.py    # Generates the 2x2 grid of wrapper screenshots
├── src/pong_rl/
│   ├── __init__.py
│   ├── envs.py             # Environment factory (AtariPreprocessing + custom wrappers + FrameStack)
│   ├── models.py           # Nature CNN and CNN + LSTM Actor-Critics, exposing hidden states for linear probing
│   ├── train.py            # CleanRL-based PPO training loop (feedforward or recurrent) with W&B tracking & checkpointing
│   ├── rollouts.py         # Five-rally batches, game-end remainders, episode reporting and GAE
│   ├── ppo.py              # Variable-size PPO batches and full-rally recurrent replay
│   └── wrappers.py         # Custom `FogWrapper` (agent half) and `EnemyMaskWrapper` (enemy paddle)
├── pyproject.toml          # Project dependencies (managed with `uv`)
└── run_all.sh              # Launches all 4 conditions x 2 seeds (8 runs) per architecture
```

## Setup

Install dependencies with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Architectures

| `--arch` | Input per step | Memory | Reference |
|---|---|---|---|
| `cnn` (default) | 4 stacked 84x84 frames | none beyond the 4 frames | CleanRL `ppo_atari.py` |
| `lstm` | 1 84x84 frame | LSTM (128 units) over the whole episode | CleanRL `ppo_atari_lstm.py` |

Both use the same Nature CNN encoder, PPO hyperparameters and collection rule. Each environment collects
**five complete rallies**, or its remaining rallies when the game ends, before the next PPO update.
A point is detected from Pong's nonzero reward. With eight environments, a normal batch contains 40 rallies;
an environment that finishes early pauses while the others finish their batches. Every transition in an
update therefore comes from the same frozen policy, and no unfinished rally is carried across an update.

The CNN shuffles observations into minibatches. The LSTM replays each complete rally in order, starting
from its saved memory; gradients cover the whole rally, with no fixed 128-step cutoff. Memory continues
across points and updates and resets only when a game terminates or times out. Gradients stop between
rallies. Activation checkpointing recomputes 32-step chunks to reduce accelerator memory use; those compute
chunks **do not detach memory or truncate gradients**. Very long rallies still take more time and memory.

Both architectures use four PPO epochs and up to four nonempty minibatches per epoch. Recurrent minibatches
group whole rallies by their lengths; CNN minibatches shuffle steps. Losses are weighted by actual step
counts, and advantages are normalized over the actual collected batch. GAE continues across scoring points.
True game termination has zero bootstrap value; a timeout uses the final observation and pre-reset memory.
A timeout can end a partial rally, whose available transitions are included in the game-ending remainder.

The timestep target is a **soft aggregate environment-step budget**: the last batch is completed at its
natural boundaries, so the actual count can exceed the target by the final batch. Learning-rate annealing
uses actual collected steps. W&B and checkpoints record actual steps and update counts; W&B also records
batch lengths, completed rallies, maximum sequence length and budget overshoot. Compare actual experience
counts when comparing runs. These changes apply only to newly started training processes.

## Running Training

### Single Run

```bash
# 1. Baseline (Fog OFF, Enemy Mask OFF)
uv run pong-rl --seed 1

# 2. Enemy paddle masked
uv run pong-rl --mask-enemy --seed 1

# 3. Fog on agent's half of the court
uv run pong-rl --use-fog --seed 1

# 4. Both Fog and Enemy Mask enabled
uv run pong-rl --use-fog --mask-enemy --seed 1
```

Add `--arch lstm` to any of the commands above to train the CNN + LSTM agent instead.

Useful flags:
- `--total-timesteps 20000000` (default: `20_000_000`)
- `--rallies-per-update 5` (default: five rallies per environment, or its game-ending remainder)
- `--no-track` (disable Weights & Biases logging for quick local testing)
- `--report-window-steps 500000` (report episodes ending in the latest 500,000 aggregate environment steps)

### Training reports

`charts/window/*` uses the interval `(max(0, global_step - report_window_steps), global_step]`.
An episode belongs to a window according to its **end step**; its return and length cover the whole
episode, including any part before that window. The window reports its bounds, warm-up coverage,
episode count, mean return, mean episode length, completed games, wins, losses, draws, and timeouts.
Ongoing episodes are not included, so compare episode counts and lengths alongside the averages.
An empty window has undefined averages (`null`), rather than a misleading zero score.

`completed_win_rate = wins / completed_games`, including draws in the denominator and excluding
timeouts. When no games completed, that rate is undefined. `positive_return_fraction` counts every
positive-return episode, including timeouts, and is explicitly **not** a win rate. If both termination
and truncation flags are true, genuine game termination takes precedence.

Every episode is also written to `logs/{run_name}_episodes.jsonl`, with its exact end flags, outcome,
return, length, training seed and condition. Raw W&B `charts/episodic_return` remains available;
it describes a changing training policy, not a frozen-model evaluation.

These logging changes apply to newly started processes. Existing training jobs keep their loaded
code. Historical return/length logs cannot reliably recover true wins versus timeouts; do not infer
those outcomes solely from an episode's length.

### Evaluate saved models

```bash
# Evaluate every completed checkpoint with the same 20 environment seeds and sampling policy:
uv run pong-eval checkpoints/*.pt --episodes 20 --seed 10000 \
  --policy stochastic --device cpu --output evaluations/full_sweep.json

# Optional alternative protocol: always choose the most probable action.
uv run pong-eval checkpoints/cnn_fog_1_mask_1_seed_1.pt --policy greedy
```

Evaluation freezes the weights (`eval` and inference mode), restores each checkpoint's architecture
and visibility condition, and uses four input frames for CNN and one for LSTM. Each episode resets
the environment to `seed + episode_index` and clears LSTM memory after either completion or timeout.
Stochastic evaluation resets a separate CPU action generator to that episode's seed, independently
of model initialization. Use the same episode count, seed, policy and device for comparisons;
shared seeds do not promise bitwise identical results across devices or software versions.

The existing Atari time limit is preserved and recorded in the report. The JSON contains separate
completed wins/losses/draws and timeouts, the completed-game win-rate denominator, every episode's
score/length/end flags, checkpoint hashes, and the evaluation protocol. Report output cannot alias
an input checkpoint. It is saved atomically after each checkpoint, with `complete: false` until all
requested models have been evaluated. CPU is the default so evaluation can run alongside training
without using its GPU, though it still consumes CPU resources.

The checkpoint evaluator also works with the current sweep's existing saved models; no training
restart or change to the optimizer, learning-rate schedule, or training time limit is required.

### Full Experiment Sweep (4 conditions × 2 seeds = 8 runs per architecture)

The default seeds are 1 and 2. Running both architectures covers 8 cases and 16 training runs in total.

```bash
# Runs all 8 jobs in parallel by default:
bash run_all.sh

# Limit to 4 simultaneous jobs:
MAX_PARALLEL=4 bash run_all.sh

# Same sweep with the CNN + LSTM agent:
ARCH=lstm bash run_all.sh
```

Trained checkpoints are saved to `checkpoints/{cnn|lstm}_fog_{0|1}_mask_{0|1}_seed_{1|2}.pt` and logs to `logs/`.
