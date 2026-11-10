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
- **The CNN on fog + mask uses no remembered trace of the ball**: no wipe changes its return rate. Its 88%
  comes from committing early (E8 below): it moves while the ball is visible and its paddle position then
  carries the information through the fog.
- **The CNN needs its frame stack just before the hit** on views where the ball is visible (0.99 to
  0.73-0.92): the stack is how it reads the ball's speed.
- **The attention-LSTM uses its memory even with nothing hidden** (clean view: down to 0.32 for one seed):
  its memory chooses where the attention heads look, so a wipe blinds it until the heads find the ball again.
- **The failed LSTM fine-tunes do use their memory** (-0.13 to -0.30): they learned a memory, a worse one.

**E3, linear probes** (`scripts/probe_memory.py`). Ridge regressions, cross-validated over games, decode the
ball's height while it crosses the agent's half from each model's internal state. R^2 eight or more steps
into the agent's half (per seed):

| model | memory ([h, c]; cnn: its features) | current frame only (lstm encoder) |
|---|---|---|
| lstm fog + mask, scratch | 0.87 / 0.97 / 0.87 | at most 0.65, mostly negative |
| lstm fog + mask, fine-tuned (failed) | 0.57 / 0.61 / 0.43 | about 0 |
| attn_lstm fog + mask, fine-tuned | 0.90 / 0.89 / 0.98 | - |
| cnn fog + mask, scratch | 0.57 / 0.50 / 0.40 | - |
| cnn fog, scratch (no memory) | 0.97 / 0.97 / 0.96 | - |
| lstm fog, scratch | 0.98 / 0.99 / 0.98 | 0.97 / 0.99 / 0.96 |

- With the ball hidden and nothing else to go on (fog + mask), the LSTM's memory holds the ball's height while
  the frame holds none. With E4 this makes the memory both present and causally used.
- The failed LSTM fine-tunes keep a weaker trace (0.43-0.61), matching their partial use of memory in E4.
- **Under fog alone, decodability is not memory:** the hidden ball decodes just as well from the memoryless
  CNN and from the LSTM's current frame, because the visible enemy paddle tracks it.
- The fog + mask CNN still carries some of the ball (0.40-0.57) with no memory and the ball out of its stack;
  its own paddle, already moving towards the right height, is the likely carrier (E8).

## Q2 Perception and Q3 the enemy-paddle shortcut

**E5, decoy** (`scripts/view_interventions.py`): whole evaluation games in which the model sees the enemy
paddle mirrored top to bottom or erased; the game itself is unchanged. Mean return:

| model | plain | mirrored enemy | erased enemy |
|---|---|---|---|
| fog-trained, every architecture, scratch and fine-tuned | +6.8 to +20.8 | -20.7 to -20.9 | -20.1 to -20.8 |
| cnn clean | +20.8 | +19.9 | +20.4 |
| lstm clean | +20.8 | -8.5 (+6, -18, -14) | +5.1 |
| attn_lstm clean | +19.7 | -18.6 | -18.9 |
| mask / fog + mask models | unchanged (sanity check) | | |

- **Every fog-trained agent depends on the enemy paddle completely**: without a correct enemy paddle it loses
  every point, although the real ball still appears at its paddle.
- **Clean models differ by architecture.** The CNN ignores the enemy paddle; the LSTM partly relies on it
  (mirroring, a misleading cue, hurts more than erasing); the attention-LSTM relies on it almost entirely
  with the ball in full view, which is why it collapses on the mask view.
- **E7 saliency agrees:** erasing the enemy paddle while the ball is in the agent's half changes the
  fog agents' policies strongly (KL 0.44-1.15), the clean CNN's hardly (0.02-0.05). The attention-LSTM's two
  heads put 10-15x the uniform weight on the enemy paddle's cell, on the clean and on the fog view.
- **How the cue is read** (E5c, an enemy paddle shifted by -8..+8 px with the ball untouched): *pending*.

## Q5 Strategy

**E8, play statistics** while the ball crosses the agent's half (`scripts/summarize_rollouts.py`):

| model | paddle-to-arrival distance 4 steps in | paddle movement per pass | share in the last 4 steps | arrival-height sd |
|---|---|---|---|---|
| cnn fog + mask, scratch | **6.0 / 6.3 / 9.5 px** | 20 / 9 / 29 px | **0.04 / 0.10 / 0.19** | 7.1 / 6.4 / 11.1 px |
| lstm fog + mask, scratch | 11.4 / 42.7 / 18.2 | 32 / 32 / 11 | 0.05 / 0.56 / 0.70 | 5.1 / 5.4 / 2.9 |
| attn_lstm fog + mask, fine-tuned | 16.8 | 26-33 | 0.25-0.55 | 3.9-6.4 |
| failed fog + mask fine-tunes (cnn, lstm) | 9-30 | 36-45 | 0.29-0.35 | 12.5-16.9 |
| clean and fog models, all architectures | 13-18 typically | 20-62 | 0.30-0.85 | 2.6-13 |

- **The memoryless CNN wins fog + mask by committing early**: four steps into the fog its paddle is already
  6-10 px from where the ball will arrive (others: 13-18), and it barely moves afterwards. Its paddle is its
  memory. This is why wiping its frame stack does nothing (E4).
- Memory agents can wait and move late (LSTM seeds 2 and 3 make 56-70% of their movement in the last 4 steps).
- **Winning fog + mask agents keep rallies predictable**: the height at which the ball arrives varies by
  2.9-9.3 px, against 12.5-16.9 for the failed fine-tunes (whether this is cause or effect is open).
- Winners take about one incoming ball per point: most points are won with the first return.

**E9, LSTM memory timescales** (forget gates recomputed from the weights on the recorded inputs). The fog +
mask LSTMs trained from scratch have no near-permanent units: their 10 slowest units forget with half-lives
of 18.5 / 16.5 / 8.3 steps, about the 14 steps the ball stays hidden. Clean LSTMs carry units with forget
gates near 1 (half-lives of thousands of steps or more), and every fine-tune keeps them (fog + mask fine-tunes:
12371 / 199 / 467427 steps).

## Q6 Transfer

*Pending:* E6. Nine fine-tunes from the clean 10M parents to fog + mask with the same 15M budget and
learning-rate schedule as a scratch run (`ft15M` for cnn and lstm; `reinit_core` for lstm: clean encoder,
fresh LSTM and heads). They reached about 20.4M of 25M steps before the power cut. Training returns at that
point (not frozen evaluations): lstm `ft15M` +21 / +20 / +12, cnn `ft15M` winning games (+11, +6), lstm
`reinit_core` mixed (+20, -14). If the frozen evaluation confirms it, the original fine-tunes' failure came
from the short 5M recipe with a restarted learning rate, not from the clean weights.

## Reproducing

Scripts (all in `scripts/`, outputs under `analysis/` on the host): `record_rollouts.py` (E2),
`memory_interventions.py` (E4), `view_interventions.py` (E5), `decoy_tracking.py` (E5b), `enemy_shift.py`
(E5c), `probe_memory.py` (E3),
`summarize_rollouts.py` (E7-E9), `run_controls.sh` (E6), `report_tables.py` (all tables). Interrupted
training runs continue with `pong-rl --resume <milestone checkpoint>`; the analyses skip finished work.
