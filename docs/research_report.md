# How the CNN, LSTM and attention-LSTM agents learned to play occluded Pong

Research report, started 2026-11-09. Questions and experiments: `docs/research_plan.md`. Each experiment's
prediction, written before its results, and its verdict live in the Flywheel graph ("Pong occlusion: how
CNN, LSTM and attention-LSTM agents learn to play"). Runs interrupted by power cuts at the training host
(2026-11-09 and 2026-11-10) were resumed and have all completed.

## Main findings

1. **Under fog, every agent plays through the enemy paddle.** With the ball hidden in its own half, each
   fog-trained agent (all three architectures, scratch and fine-tuned) loses every point if the enemy paddle
   is erased or mirrored, and most points if it is moved by only 4 px. It does not copy the paddle's height;
   it reads something finer from it, plausibly where the ball met it (which sets the return angle).
2. **With nothing visible to go on (fog + mask), the memory agents remember the ball.** Its memory holds the hidden
   ball's height (R^2 0.87-0.97 deep into the fog) and wiping it as the ball disappears drops returns from 0.98
   to 0.30. Its slowest memory units forget over 8-18 steps, about the 14 steps the ball is hidden.
3. **The attention-LSTM remembers only what its attention let in.** Blocking its attention to the ball while the
   ball is still visible, before the fog, wipes the ball from its memory and makes it miss most balls (3 of 4
   models); the same block placed elsewhere does nothing. In 2 of the 4, the block does as much damage as
   erasing the ball from the screen.
4. **The memoryless CNN wins fog + mask by committing early.** It moves while the ball is still visible and
   holds its paddle there; the paddle itself carries the information through the fog. Wiping its frame stack
   changes nothing.
5. **Clean models differ in what they look at.** The clean CNN ignores the enemy paddle entirely; the clean
   attention-LSTM's heads stare at it and it collapses without it; the clean LSTM sits in between.
6. **Fine-tuning from the clean model does not hurt.** The earlier failure on fog + mask came from the short
   fine-tuning recipe: given the scratch budget and learning-rate schedule, every fine-tune wins, the LSTM's
   mostly faster than from scratch and the CNN's more reliably.
7. **Every model is a specialist.** Fog + mask models lose on every other view, clean ones on every fogged one.

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
  E5 confirms it causally (below).

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
| attn_lstm fog + mask, curriculum | 0.96 | 0.73 (0.59-0.98) | **0.14** | **0.17** |
| attn_lstm fog + mask, scratch (s2, the seed that learned) | 0.96 | 0.58 | 0.20 | 0.35 |
| attn_lstm fog, curriculum | 0.93 | 0.75 | 0.40 | 0.44 |
| attn_lstm fog, scratch (s1, s2) | 0.96 | 0.87 | 0.28 | 0.34 |

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
- **The attention-LSTM on the fog views depends on its memory most of all** (added 2026-11-11): wiping it at
  fog entry leaves 0.06-0.20 of balls returned on fog + mask and 0.18-0.61 on fog, where the LSTM agent lost
  little (0.85). Even with the enemy paddle in view, it does not rebuild the ball from the shortcut after a
  wipe. Caveat: unlike the LSTM agent, it is also hurt by a wipe before the fog (0.58-0.87), because its
  memory sets its attention query; a wipe both erases what it remembers and disturbs where it looks. The
  attention block (E10) separates the two.
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

**E3b, untrained-network control** (`scripts/random_control.py`): the same games replayed while every frame is
also fed to a randomly initialised network of the same architecture, which keeps its own memory. Its state
decodes the hidden ball as well as the trained models' under fog (0.92-0.99) and as well as the fog + mask
CNN's (0.39-0.55): those decodings come from the input (the enemy paddle; the CNN's own committed paddle).
**Only the fog + mask memory agents beat it**: trained memory 0.87-0.98 against 0.27-0.58 untrained (LSTM and
attention-LSTM); the failed LSTM fine-tunes 0.43-0.61 against 0.05-0.11.

**E10, does the attention-LSTM remember only what it looked at?** (`scripts/attention_block.py`). Each flight
towards the agent is replayed from a snapshot while, as long as the ball is visible before the fog, the agent
may not attend to the 3x3 attention cells around the ball (neither paddle's cells are ever blocked); a sham
branch blocks the same number of cells at the mirrored height, and a third erases the ball from the screen.
The probe is fit on the model's own recorded games and applied to the replays (new game seeds).

| attention-LSTM on fog + mask | balls returned: control / ball blocked / sham / ball erased | memory error on the hidden ball, px: control / blocked / sham / erased |
|---|---|---|
| fine-tune s1 | 0.99 / **0.06** / 0.99 / 0.10 | 1.2 / **32.0** / 2.4 / 21.9 |
| fine-tune s2 | 0.95 / **0.14** / 0.96 / 0.14 | 0.7 / **9.4** / 0.8 / 22.1 |
| fine-tune s3 | 1.00 / 1.00 / 1.00 / 0.04 | 0.4 / 2.9 / 0.5 / 10.6 |
| scratch s2 | 0.97 / **0.54** / 0.97 / 0.04 | 1.4 / **18.6** / 1.9 / 27.4 |

- **Memory holds what attention let in.** Blocking attention to the ball leaves memory without it and the
  agent misses (3 of 4 models), while the sham block changes nothing: the effect is specific to the ball.
- In fine-tunes s1 and s2 the block is as damaging as erasing the ball altogether: attention is the whole gate
  to memory. In the other two, part of the ball still gets past the block (fine-tune s3 loses nothing with
  the block and everything with the erasure), plausibly through neighbouring cells whose receptive fields
  cover the ball.
- With the enemy paddle visible (fog), the attention block mostly matters less (0 to -0.23 of returns in 4 of
  5 models), yet erasing the ball still hurts every fog model (attention-LSTM 0.20-0.49, LSTM 0.48-0.95 of
  balls returned): even with the shortcut, the agents also use the ball they saw.
- Passive correlation (attention on the ball before the fog against memory error inside it) shows no
  consistent relation (Spearman -0.12 to +0.50): only the intervention is informative.

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
- **Attention over a flight** (`scripts/attention_maps.py`, added 2026-11-11; returned flights of the 30
  evaluation games, aligned on the ball entering the agent's half). Weight on the ball's 3x3 cells (uniform:
  0.18) in the 8 steps before the fog / steps 1-8 inside it: fog + mask models 0.35-0.80 / 0.01-0.24 (the
  heads track the ball cell by cell, then let it go). Fog models instead turn to the enemy paddle inside the
  fog (0.27-0.69, uniform about 0.06); clean models watch it throughout (0.29-0.62). Paper figure `attention`.
- **How the cue is read: E5c, a small shift.** Each incoming flight is replayed from a snapshot with the enemy
  paddle the model sees moved by -8, -4, +4 or +8 px (only the paddle's pixels; the ball is never moved), with
  the same action noise as the unshifted replay. The gain is how far the agent's paddle moves per pixel of
  shift: +1 would mean it copies the enemy paddle's height.

  | model | gain per seed | balls returned, unshifted -> shifted by 4-8 px |
  |---|---|---|
  | cnn clean | -0.04 / -0.01 / -0.01 | 0.99-1.00 -> 0.95-1.00 |
  | lstm clean | -0.03 / -0.62 / -0.33 | 0.98-1.00 -> 0.20-0.97 (mostly hurt by an upward shift) |
  | attn_lstm clean | -1.54 / -0.44 / -0.68 | 0.94-0.97 -> 0.15-0.96 |
  | cnn fog (scratch; fine-tune) | -0.37 / -0.58 / -0.58; -0.34 / -0.20 / -1.24 | 0.94-1.00 -> 0.01-0.69 |
  | lstm fog (scratch; fine-tune) | -2.03 / -0.78 / -1.92; -0.06 / -2.53 / -1.27 | 0.85-1.00 -> 0.01-0.86 |
  | attn_lstm fog (scratch s1, s2; fine-tune) | -1.06 / +2.32; -1.10 / -0.67 / -2.04 | 0.90-0.96 -> 0.12-0.71 |

  Half a paddle height of shift costs the fog agents most of their returns, while the clean CNN does not
  notice (causal confirmation that it ignores the enemy paddle). **The fog agents do not copy the enemy
  paddle's height:** 16 of the 17 that play move the *opposite* way, and asymmetrically (an upward shift
  typically sends their paddle about 20 px down). They read something finer from where the enemy paddle is.
  In Pong the point where the ball meets a paddle sets the return angle, so one reading is that they infer
  the ball's direction from where it met the enemy paddle (a paddle drawn lower makes the hit look higher on
  it). That fits the sign, but is not yet tested directly.
- E5b, which tried to read the same thing from correlations under the mirror, was inconclusive (too few and
  too noisy passes per game).
- **E11, scripted players** (`scripts/scripted_baselines.py`, the same 30 games; the CPU wins long rallies, so
  the measure is the share of balls returned): copying the enemy paddle returns 0.55 under fog, going to the
  ball's last seen height 0.66, extrapolating its seen path (ideal short memory) 0.88, and the same with no fog
  0.96. The shortcut alone is strong but not enough; the fog + mask LSTM (0.98) beats ideal extrapolation and
  the memoryless CNN (0.88) matches it.

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

## Q6 Transfer: why did fine-tuning from the clean model fail on fog + mask?

**E6, training controls** (`scripts/run_controls.sh`). The original fine-tunes had 5M steps on fog + mask
with the learning rate restarted at 1e-3 and annealed to 0 over those 5M; scratch runs had 15M. Nine new
fine-tunes from the same clean parents get the scratch budget and schedule (15M steps): `ft15M` (cnn, lstm:
all weights from the parent) and `reinit_core` (lstm: clean encoder, fresh LSTM and heads). Frozen fog + mask
return after k M steps on fog + mask (per seed):

| run | 2M | 3M | 5M | 7.5M | 10M | 15M |
|---|---|---|---|---|---|---|
| lstm scratch | -8.7 / -8.2 / -18.5 | +13.6 / +13.7 / -5.4 | +16.2 / +18.8 / +16.8 | +16.3 / +19.5 / +16.9 | +17.7 / +17.3 / +20.0 | +20.4 / +19.3 / +20.8 |
| lstm fine-tune, original 5M | -13.4 / -14.2 / -16.6 | -12.2 / -14.3 / -15.1 | -3.8 / -9.5 / -13.5 | | | |
| lstm `ft15M` | **+9.1 / +13.8** / -17.4 | +12.5 / +17.9 / -16.7 | +16.2 / +19.5 / -15.3 | +18.5 / +19.0 / -14.1 | +19.9 / +17.6 / +13.4 | **+20.7 / +20.9 / +18.9** |
| lstm `reinit_core` | -13.3 / +13.7 / -15.9 | +15.2 / +18.5 / -16.8 | +17.2 / +19.9 / -15.5 | +17.8 / +18.8 / -13.8 | +18.8 / +19.6 / -9.8 | +20.0 / +20.9 / +0.2 |
| cnn scratch | -20.4 / -20.1 / -20.0 | -19.2 / -19.8 / -20.1 | -18.4 / -17.4 / -17.7 | -15.0 / -15.5 / -13.4 | +9.9 / +13.3 / +5.2 | +19.8 / +19.1 / +1.6 |
| cnn fine-tune, original 5M | -17.4 / -18.5 / -19.6 | -17.6 / -17.7 / -18.2 | -14.6 / -16.1 / -15.8 | | | |
| cnn `ft15M` | -18.7 / -18.1 / -18.4 | -18.6 / -17.2 / -17.2 | -17.6 / -13.3 / -16.6 | -1.3 / +12.4 / -14.4 | +5.7 / +9.7 / +7.3 | **+13.3 / +20.5 / +19.9** |

- **There is no negative transfer.** With the same budget and schedule as scratch, every fine-tune from the
  clean model wins in the end: the LSTM's as well as scratch (+18.9 to +20.9), the CNN's better than
  scratch (all three seeds win; scratch leaves one at +1.6).
- **The CNN simply needed more steps.** From scratch it is still at -17.8 after 5M steps on fog + mask and
  takes off between 7.5M and 10M; the 5M fine-tune could not have got there. The original comparison
  (5M fine-tune against 15M scratch) was unfair to it.
- **For the LSTM the short recipe was the main problem.** With the scratch schedule, two of the three
  seeds learn *faster* than scratch (+9.1 and +13.8 after 2M steps, against -8.7 and -8.2), where the original
  5M fine-tunes from the same parents were still at -13 to -14. The third seed lags until about 10M, then
  catches up.
- **The clean encoder is not the problem.** A fresh LSTM on the clean encoder also learns fast for two
  seeds; it is again seed 3 that stalls (+0.2 at 15M). Seed 3's clean parent is slow to adapt in both
  variants; it is also the clean LSTM with the most near-permanent memory units (E9), a single case.
- Caveats: three seeds; the original and new fine-tunes share parent and seed but not the training
  randomness, so the attribution to the schedule rests on two seeds diverging within 2M steps. The last 2.5M
  steps of the controls resumed after power cuts with a fresh optimizer (recorded in the checkpoints).

## Reproducing

Scripts (all in `scripts/`, outputs under `analysis/` on the host): `record_rollouts.py` (E2),
`memory_interventions.py` (E4), `view_interventions.py` (E5), `decoy_tracking.py` (E5b), `enemy_shift.py`
(E5c), `probe_memory.py` (E3), `random_control.py` (E3b), `attention_block.py` (E10), `scripted_baselines.py`
(E11),
`summarize_rollouts.py` (E7-E9), `run_controls.sh` (E6), `report_tables.py` (all tables). Interrupted
training runs continue with `pong-rl --resume <milestone checkpoint>`; the analyses skip finished work.
