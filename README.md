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
│   ├── models.py           # CNN, CNN + LSTM and attention + LSTM Actor-Critics, exposing hidden states and attention
│   ├── train.py            # CleanRL-based PPO training loop (feedforward or recurrent) with W&B tracking & checkpointing
│   ├── rollouts.py         # Five-rally batches, game-end remainders, episode reporting and GAE
│   ├── ppo.py              # Variable-size PPO batches and full-rally recurrent replay
│   └── wrappers.py         # Custom `FogWrapper` (agent half) and `EnemyMaskWrapper` (enemy paddle)
├── pyproject.toml          # Project dependencies (managed with `uv`)
└── run_all.sh              # Two-stage sweep: from scratch, then fine-tuning from the clean runs
```

## Setup

Install dependencies with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run the tests (pytest is in the `dev` dependency group, installed by `uv sync`):

```bash
uv run pytest
```

## Architectures

| `--arch` | Input per step | Memory | Reference |
|---|---|---|---|
| `cnn` (default) | 4 stacked 84x84 frames | none beyond the 4 frames | CleanRL `ppo_atari.py` |
| `lstm` | 1 84x84 frame | LSTM (128 units) over the whole episode | CleanRL `ppo_atari_lstm.py` |
| `attn_lstm` | 1 84x84 frame | LSTM (128 units) that reads the frame through spatial attention | Mott et al. 2019, simplified |

| `--arch` | Parameters | Image to LSTM |
|---|---|---|
| `cnn` | 1,687,719 | (no LSTM) whole frame flattened to 512 features |
| `lstm` | 2,007,591 | whole frame flattened to 512 features |
| `attn_lstm` | 225,959 | weighted read of the 7x7 conv map, 128 numbers |

### Attention + LSTM (`attn_lstm`)

Based on Mott et al. (2019), *Towards Interpretable Reinforcement Learning Using Attention Augmented
Agents*, simplified. Each step:

1. The three Nature CNN convolutions turn the frame into a 7x7 map of 64 features (49 cells; no flatten
   and no 512-unit dense layer). Two fixed channels add each cell's x and y position in [-1, 1].
2. Each cell gives a key (32 numbers) and a value (64 numbers) per head; there are 2 heads.
3. Each head's query comes from the previous LSTM output h(t-1), so memory decides where to look.
4. A softmax over the 49 scores `q · k / sqrt(32)` gives each head's attention weights. Queries and keys
   are layer-normalised per head first (query-key normalisation, no learned scale). Without it the scores
   start near zero, the weights stay uniform and, in a supervised check, never became selective.
5. The weighted sums of values, concatenated over heads (2 x 64 = 128), are the LSTM's only input.
   Actor and critic read the LSTM output, as in `lstm`.

`AttentionLSTMAgent.get_states_and_attention` returns the weights, shaped `(steps * batch, 2, 49)`,
with cells in row-major order (row = y, column = x), for analysing where the agent looks. Each cell of
the 7x7 map summarises an overlapping 36x36-pixel patch, with centres 8 pixels apart. The model is much
smaller than the other two because it drops the 3136-to-512 dense layer; keep this in mind when comparing.

All three use the same Nature CNN convolutions, PPO hyperparameters and collection rule.
Each environment collects **five rallies**, its game-ending remainder, or a partial rally
that reaches **512 environment steps** (`--max-rally-steps`), whichever stops the lane first.
A point is detected from Pong's nonzero reward. With eight environments, a normal batch
contains 40 rallies; finished lanes pause while the others reach their boundaries. Every
transition in an update comes from the same frozen policy. A capped rally resumes under
the next updated policy; no pending transitions are carried into that next batch.

The CNN shuffles observations into minibatches. Both recurrent agents (`lstm`, `attn_lstm`)
replay each collected sequence in order from its saved memory. Gradients cover the full
sequence, up to the 512-step cap, and stop at point, cap or game boundaries. Memory continues
across points, caps and updates and resets only when a game terminates or times out. A cap
bootstraps from the current observation and memory; it does not end/reset the game, add a
reward or report an episode outcome. Logs and W&B record `capped_rallies` separately from
actual points and game endings.

Replay retains activations for batches of up to 2048 padded steps. Larger batches use
activation checkpointing to recompute 32-step chunks and reduce accelerator memory use;
those compute chunks do not detach memory or truncate gradients within a sequence.

All architectures use four PPO epochs and up to four nonempty minibatches per epoch. Recurrent minibatches
batch collected sequences together with padding masked out of the loss; CNN minibatches shuffle steps. Losses are weighted by actual step
counts, and advantages are normalized over the actual collected batch. GAE continues across scoring points.
True game termination has zero bootstrap value; a timeout uses the final observation and pre-reset memory.
A timeout can end a partial rally, whose available transitions are included in the game-ending remainder.

The timestep target is a **soft aggregate environment-step budget**: the last batch is completed at its
point, cap or game boundaries, so the actual count can exceed the target by the final batch. Learning-rate annealing
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

Add `--arch lstm` (CNN + LSTM) or `--arch attn_lstm` (attention + LSTM) to any of the commands above.

Useful flags:
- `--total-timesteps 15000000` (default: `15_000_000`, matching the last checkpoint milestone)
- `--init-from PATH` (fine-tune from a checkpoint; see [Fine-tuning](#fine-tuning))
- `--rallies-per-update 5` (default: five rallies per environment, stopping early on a game end or rally cap)
- `--max-rally-steps 512` (maximum steps in one collected rally segment; game and memory continue afterward)
- `--learning-rate 0.001` (initial learning rate; the existing linear annealing schedule remains enabled)
- `--ent-coef 0.02` (entropy bonus coefficient)
- `--no-track` (disable Weights & Biases logging for quick local testing)
- `--parallel-envs` (one CPU process per game; preserves individual pauses and resets; uses more host memory)
- `--replay-checkpoint-threshold 2048` (retain activations for small replay batches; `0` always recomputes to save memory)
- `--report-window-steps 500000` (report episodes ending in the latest 500,000 aggregate environment steps)

### Training performance

Use `--parallel-envs` for a single GPU training job to collect games in separate CPU
processes. Leave it off when running many simultaneous jobs unless CPU and RAM allow
one worker per environment per job. Small recurrent replay batches retain activations;
`--replay-checkpoint-threshold 0` restores always-on activation recomputation when memory
is limited. Both paths preserve gradients within each collected sequence and the same PPO optimizer groups.
Training logs and W&B report collection and update times separately.

Run a short benchmark without W&B, checkpoints, or changing existing runs:

```bash
uv run python scripts/benchmark_training.py --parallel-envs
uv run python scripts/benchmark_training.py --replay-checkpoint-threshold 0
```

The first two iterations are excluded from benchmark summaries. Compare identical
settings and avoid running other GPU workloads during measurements.

### Training reports

`charts/window/*` uses the interval `(max(start, global_step - report_window_steps), global_step]`,
where `start` is 0 from scratch and the parent's step when fine-tuning.
An episode belongs to a window according to its **end step**; its return and length cover the whole
episode, including any part before that window. The window reports its bounds, warm-up coverage,
episode count, mean return, mean episode length, completed games, wins, losses, draws, and timeouts.
Ongoing episodes are not included, so compare episode counts and lengths alongside the averages.
An empty window has undefined averages (`null`), rather than a misleading zero score.

`completed_win_rate = wins / completed_games`, including draws in the denominator and excluding
timeouts. When no games completed, that rate is undefined. `positive_return_fraction` counts every
positive-return episode, including timeouts, and is explicitly **not** a win rate. If both termination
and truncation flags are true, genuine game termination takes precedence.

Every episode is also written to `logs/<run folder>/episodes_<start time>.jsonl` (same folders as the
checkpoints below), with its exact end flags, outcome, return, length, training seed and condition. Raw W&B `charts/episodic_return` remains available;
it describes a changing training policy, not a frozen-model evaluation.

These logging changes apply to newly started processes. Existing training jobs keep their loaded
code. Historical return/length logs cannot reliably recover true wins versus timeouts; do not infer
those outcomes solely from an episode's length.

### Checkpoints during training

Snapshots are taken at **0, 1M, 2M, 3M, 4M, 5M, 7.5M, 10M, 11M, 12M, 13M, 14M and 15M aggregate
environment steps**; milestones beyond `--total-timesteps` are not requested. A from-scratch run saves its untrained
weights at 0. Each snapshot is saved after the first PPO update that reaches or passes
its milestone, preserving the rally-quota and step-cap collection rules, so its actual step count can be above the
filename's milestone. If one batch crosses multiple milestones, those files hold the same weights.

Each run has its own folder, `checkpoints/<scratch|finetune>/<arch>/<condition>/seed_<n>/`, where the
condition is `clean`, `mask`, `fog` or `fog_mask`. A fine-tune's folder also names its parent
checkpoint, e.g. `clean_10.0M_to_fog`, so fine-tunes from different parents never share files:

```text
checkpoints/
├── scratch/lstm/fog_mask/seed_1/
│   ├── step_00.0M.pt  step_01.0M.pt  ...  step_10.0M.pt  ...  step_15.0M.pt
│   └── final.pt
└── finetune/lstm/clean_10.0M_to_fog_mask/seed_1/
    ├── step_10.0M.pt   # the parent's weights, before fine-tuning
    ├── step_11.0M.pt  ...  step_15.0M.pt
    └── final.pt
```

Each file records `checkpoint_step` (requested milestone), `global_step` (actual experience), `updates`,
`optimizer_steps`, `run`, `kind`, `start_step` and `steps_since_start` (experience gathered by this run
itself), and for fine-tunes `init_from` (parent path, hash, condition and step). In a fine-tune, `updates`
and `optimizer_steps` count from its own start while `global_step` continues from the parent. Its first
file (`step_10.0M.pt`, `steps_since_start = 0`) holds the parent's weights but records the fine-tune's
condition, so evaluating it is a zero-shot test of the clean model on that condition; filter on
`kind` and `steps_since_start` when grouping checkpoints by condition. `final.pt` is the end-of-training model; with a budget that ends on a milestone it
holds the same weights as the last snapshot, and it marks the run as complete. Snapshots are written
atomically and uploaded to W&B when tracking is enabled. `--no-save-model` disables all saves.
All these files work with `pong-eval`.

### Fine-tuning

```bash
uv run pong-rl --arch lstm --use-fog --mask-enemy --seed 1 \
  --init-from checkpoints/scratch/lstm/clean/seed_1/step_10.0M.pt
```

Fine-tuning loads all weights from a checkpoint of the same architecture and trains on the new
condition with a fresh Adam optimizer, using the same hyperparameters as training from scratch. The
step count continues from the parent (about 10M to 15M), so fine-tuned and from-scratch runs share one
step axis and the same total experience at 15M. W&B also logs `charts/steps_since_start`, the
experience gathered under the new condition, and SPS counts only this run's own steps. The learning rate restarts at its base value and
anneals to zero over the fine-tune's own steps. Note that a 10M snapshot from a 15M run was taken
while its learning rate was still decaying, unlike a model trained with a 10M budget.

### Evaluate saved models

```bash
# Evaluate every final model (scratch and fine-tuned) with the same 20 seeds and sampling policy:
uv run pong-eval checkpoints/*/*/*/seed_*/final.pt --episodes 20 --seed 10000 \
  --policy stochastic --device cpu --output evaluations/full_sweep.json

# Optional alternative protocol: always choose the most probable action.
uv run pong-eval checkpoints/scratch/cnn/fog_mask/seed_1/final.pt --policy greedy
```

Evaluation freezes the weights (`eval` and inference mode), restores each checkpoint's architecture
and visibility condition, and uses four input frames for CNN and one for `lstm` and `attn_lstm`. Each episode
resets the environment to `seed + episode_index` and clears recurrent memory after either completion or timeout.
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

### Full Experiment Sweep

`run_all.sh` runs two stages (bash 4.3 or newer):

1. **scratch**: every architecture × condition (`clean`, `mask`, `fog`, `fog_mask`) × seed, to 15M steps.
2. **finetune**: each finished clean scratch run's `step_10.0M.pt`, fine-tuned on `mask`, `fog` and
   `fog_mask` to 15M (another parent milestone: `PARENT_CHECKPOINT=step_05.0M.pt`).

Defaults: `ARCHS="cnn lstm attn_lstm"`, `SEEDS="1 2 3"`, 8 parallel jobs, i.e. 36 scratch and 27 fine-tuning runs.
Fine-tuning starts after every scratch run has finished, and only from parents whose run completed.
Runs whose `final.pt` exists are skipped, so rerunning the script resumes an interrupted sweep; an
unfinished run's checkpoint and log folders are first moved aside to `<folder>.aborted_<time>`.

```bash
bash run_all.sh                        # both stages
STAGES=scratch bash run_all.sh         # only stage 1 (or STAGES=finetune)
ARCHS=lstm SEEDS="1" bash run_all.sh   # a subset
MAX_PARALLEL=6 bash run_all.sh         # at most 6 simultaneous jobs
```

Each run's console output goes to `logs/<run folder>/train.log`, next to its episode log.
