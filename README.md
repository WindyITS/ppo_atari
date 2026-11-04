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
│   └── wrappers.py         # Custom `FogWrapper` (agent half) and `EnemyMaskWrapper` (enemy paddle)
├── pyproject.toml          # Project dependencies (managed with `uv`)
└── run_all.sh              # Launches all 4 conditions x 3 seeds (12 runs) in parallel
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

Both use the same Nature CNN encoder and PPO hyperparameters. The LSTM agent resets its memory at episode
boundaries and is trained with full 128-step sequences per env (minibatches are whole envs, not random steps),
so its memory spans the whole episode while gradients flow back through at most 128 steps.

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
- `--total-timesteps 5000000` (default: `5_000_000`)
- `--no-track` (disable Weights & Biases logging for quick local testing)

### Full Experiment Sweep (4 conditions × 3 seeds = 12 runs)

```bash
# Runs 6 jobs in parallel by default (2 waves of 6):
bash run_all.sh

# Or run all 12 simultaneously:
MAX_PARALLEL=12 bash run_all.sh

# Same sweep with the CNN + LSTM agent:
ARCH=lstm bash run_all.sh
```

Trained checkpoints are saved to `checkpoints/{cnn|lstm}_fog_{0|1}_mask_{0|1}_seed_{1|2|3}.pt` and logs to `logs/`.
