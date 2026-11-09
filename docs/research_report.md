# How the CNN, LSTM and attention-LSTM agents learned to play occluded Pong

Research report, started 2026-11-09. Questions and experiments: `docs/research_plan.md`. Each experiment's
prediction, written before its results, and its verdict live in the Flywheel graph ("Pong occlusion: how
CNN, LSTM and attention-LSTM agents learn to play"). **Draft:** sections marked *pending* wait for jobs
interrupted by a power cut at the training host on 2026-11-09.

## Setup in one paragraph

PPO on ALE Pong (84x84 grayscale, frame skip 4), 3 training seeds per cell. Views: **clean**; **mask** (the
enemy paddle's columns blanked); **fog** (the agent's half of the court, columns 42-71, blanked: the ball
disappears as it comes towards the agent and reappears only at its paddle); **fog + mask** (both).
Architectures: **cnn** (Nature CNN on 4 stacked frames, no memory), **lstm** (Nature CNN on 1 frame + LSTM
128) and **attn_lstm** (1 frame; the LSTM reads a 7x7 convolutional map through 2 attention heads whose
queries come from its own state). Scratch runs: 15M steps. Fine-tunes: from the clean model at 10M, 5M more
steps on the new view, fresh Adam, learning rate restarted at 1e-3 and annealed over the 5M. Frozen
evaluation: 30 games, environment seeds 10000-10029, stochastic policy. Return is points won minus lost
(-21 to +21).

## Q1 Performance

Frozen models on their own training view (final checkpoints; mean return over 3 seeds, win rate per seed):

| view | cnn scratch | cnn fine-tune | lstm scratch | lstm fine-tune | attn_lstm scratch | attn_lstm fine-tune |
|---|---|---|---|---|---|---|
| clean | +20.8 | - | +20.8 | - | +19.7 | - |
| mask | +20.8 | +20.7 | +20.6 | +20.7 | +6.6 (0/100/100%) | +20.0 |
| fog | +20.8 | +20.3 | +20.4 | +19.0 | +6.8 (100/100/0%) | +18.6 |
| fog + mask | +13.5 (100/100/43%) | **-15.5** (0/0/0%) | +20.2 | **-8.9** (33/0/0%) | -7.0 (0/100/0%) | **+20.0** (100/100/100%) |

- Every architecture learns mask and fog. **Fog + mask separates them:** from scratch, the LSTM wins with
  every seed; the memoryless CNN wins with 2 of 3 seeds (+19.8, +19.1, +1.6), which is surprising; the
  attention-LSTM with 1 of 3.
- **Transfer reverses between architectures.** Fine-tuning the clean model onto fog + mask fails for the CNN
  and the LSTM but succeeds for every attention-LSTM seed (+19.5 to +20.6, within about 3M steps), where its
  scratch runs mostly fail. Clean pre-training helps the attention-LSTM and hurts the other two. Q6 asks
  whether that failure is the clean weights or the fine-tuning recipe.
- **Models are specialists.** The clean models lose about -20 on fog and fog + mask. On mask the clean CNN
  still wins (+20.0 to +20.7), the clean LSTM is mixed (+18.8, -10.9, +6.8) and the clean attention-LSTM
  collapses (-15 to -21). Every fog + mask model, of every architecture, loses about -20 on every other view,
  including clean: seeing the ball in its own half is out of distribution for them.
- **Fog-only models need the enemy paddle.** Fog-trained models of all three architectures (scratch and
  fine-tuned) lose about -20 on fog + mask, i.e. when only the enemy paddle is taken away. This is the first
  evidence for the shortcut of Q3: under fog, the agents read where the ball went off the enemy paddle.
  E5 tests it causally (*pending*).

## Q4 Memory: do the agents carry the hidden ball, and use it?

**E4, causal memory wipe** (`scripts/memory_interventions.py`). Every incoming ball in the 30 evaluation
games is replayed from an exact emulator snapshot taken when it crosses x = 30, under one untouched branch
and several wipe branches: the LSTM state set to zero (for the CNN: the 4-frame stack filled with the current
frame for 4 steps, so it loses all motion) before the fog, as the ball enters the fog, or 7 steps later.
Share of replayed balls the agent returns, 600 replays per model, mean of 3 seeds:

| model | no wipe | wipe before fog | wipe at fog entry | wipe 7 steps in |
|---|---|---|---|---|
| lstm fog + mask, scratch | 0.98 | 0.97 | **0.30** | **0.17** |
| lstm fog, scratch | 0.98 | 0.98 | 0.85 | 0.95 |
| lstm clean / mask, scratch | 0.99 | 0.99 | 0.98-0.99 | 0.98-0.99 |
| lstm fog + mask, fine-tuned (failed) | 0.61 | 0.47 | 0.37 | 0.42 |
| cnn fog + mask, scratch | 0.88 | 0.88 | 0.87 | 0.88 |
| cnn fog + mask, fine-tuned (failed) | 0.47 | 0.46 | 0.46 | 0.46 |
| cnn clean / mask / fog, scratch | 0.99 | 0.99 | 0.98-0.99 | 0.73-0.92 |
| attn_lstm clean, scratch | 0.95 | 0.84 | 0.65 | 0.56 |

A second untouched branch with different action noise matches the first within 0.04 everywhere.

- **The fog + mask LSTM carries the hidden ball in its memory and acts on it.** Wiping memory as the ball
  disappears drops returns from 0.98 to 0.30 (per seed 0.58 / 0.20 / 0.13), and to 0.17 seven steps in.
  Wiping before the fog costs nothing, because the ball is seen again.
- **The fog-only LSTM barely needs memory** (-0.04 to -0.25 at entry): it can rebuild its estimate from
  something still visible, consistent with the enemy-paddle shortcut.
- **The CNN on fog + mask uses no trace of the ball at all**: no wipe changes its return rate. Its 88%
  comes from a memoryless policy. How it does this is the object of E8 (*pending*).
- **The CNN needs its frame stack just before the hit** on views where the ball is visible (0.99 to
  0.73-0.92): the stack is how it reads the ball's speed.
- **The attention-LSTM uses its memory even with nothing hidden** (clean view: down to 0.32 for one seed):
  its memory chooses where the attention heads look, so a wipe blinds it until the heads find the ball again.
- **The failed LSTM fine-tunes do use their memory** (-0.13 to -0.30): they learned a memory, a worse one.

**E3, linear probes** (*pending*). First look, attention-LSTM fine-tuned on fog + mask: its memory
decodes the hidden ball's current height with R^2 0.93 (error about 1.4 px), still 0.90 eight or more
steps into the fog; the height where the ball will reach the paddle decodes less well (R^2 0.49, against
0.90 for ideal physics from the entry point).

## Q2 Perception, Q3 Shortcut, Q5 Strategy

*Pending:* E5 (enemy-paddle decoy), E7 (per-object saliency, attention mass per object), E8 (play
statistics), E9 (LSTM memory timescales).

## Q6 Transfer

*Pending:* E6. Nine fine-tunes from the clean 10M parents to fog + mask with the same 15M budget and
learning-rate schedule as a scratch run (`ft15M` for cnn and lstm; `reinit_core` for lstm: clean encoder,
fresh LSTM and heads). They reached about 20.4M of 25M steps before the power cut. Training returns at that
point (not frozen evaluations): lstm `ft15M` +21 / +20 / +12, cnn `ft15M` winning games (+11, +6), lstm
`reinit_core` mixed (+20, -14). If the frozen evaluation confirms it, the original fine-tunes' failure came
from the short 5M recipe with a restarted learning rate, not from the clean weights.

## Reproducing

Scripts (all in `scripts/`, outputs under `analysis/` on the host): `record_rollouts.py` (E2),
`memory_interventions.py` (E4), `view_interventions.py` (E5), `probe_memory.py` (E3),
`summarize_rollouts.py` (E7-E9), `run_controls.sh` (E6), `report_tables.py` (all tables). Interrupted
training runs continue with `pong-rl --resume <milestone checkpoint>`; the analyses skip finished work.
