# Research plan: how the agents learned to play occluded Pong

Started 2026-11-09. Continues `docs/analysis_plan.md` (steps 1-3 done there). Three architectures:
`cnn` (4 stacked frames, no memory), `lstm` (1 frame + LSTM memory) and `attn_lstm` (1 frame read through
memory-steered spatial attention, 7-9x smaller). ViTs are out of scope. Mirrored as a Flywheel graph
("Pong occlusion: how CNN, LSTM and attention-LSTM agents learn to play").

Rules: never delete or overwrite existing code or checkpoints; new runs use tagged folders
(`train.py --run-tag`), new analyses are new scripts and output files.

## Questions

- **Q1 Performance.** How well does each frozen model play every view (all three architectures)?
- **Q2 Perception.** What on the screen drives each policy: ball, own paddle, enemy paddle, score?
- **Q3 Shortcut.** Do fog-trained (and clean-trained) agents read the ball's height off the enemy paddle?
- **Q4 Memory.** Do the recurrent agents carry the hidden ball in memory, and use it? How does the CNN
  play fog + mask without memory?
- **Q5 Strategy.** What does each agent's play look like (paddle movement, where it hits the ball)?
- **Q6 Transfer.** Why does fine-tuning from the clean model fail on fog + mask: the learning-rate
  schedule, the budget, or the clean weights (and which part of them)?

## Experiments

| ID | Question | Method | Output |
|----|----------|--------|--------|
| E1 | Q1 | Frozen evaluation of `attn_lstm` (finals, parents, equal steps), same protocol as the cnn/lstm sweep | `evaluations/sweep_stochastic.jsonl` |
| E2 | Q2-Q5 | Record the frozen-evaluation games of every checkpoint of interest: object positions from the unoccluded frame, actions, action probabilities, values and internal states | `analysis/rollouts/` (host) |
| E3 | Q4 | Linear probes: decode the hidden ball's height (now and where it will reappear) from LSTM state / CNN features, against baselines | `analysis/probes.json` |
| E4 | Q4 | Causal memory tests from saved emulator states: wipe the LSTM state when the ball enters the fog (or replace the CNN's older frames by the current one), replay the same point, compare return rates | `analysis/interventions/` |
| E5 | Q3 | Enemy-paddle decoy: play full games where the enemy paddle the agent sees is mirrored vertically (or frozen); compare with the plain and masked views | `analysis/decoy.jsonl` |
| E5b | Q3 | Decoy tracking: under the mirrored enemy paddle, regress the agent's paddle height at the paddle line on the ball's true height and on the enemy paddle as seen | `analysis/decoy_tracking.jsonl` |
| E5c | Q3 | Enemy-paddle shift, dose-response: replay each incoming flight from a snapshot with the enemy paddle the model sees moved by -8..+8 px; gain = paddle displacement per pixel of shift (1 = reads the ball off the enemy paddle) | `analysis/enemy_shift.jsonl` |
| E6 | Q6 | Training controls on fog + mask from the clean 10M parents, 15M-step budget and schedule matched to scratch: `ft15M` (cnn, lstm) and `reinit_core` (lstm, fresh LSTM and heads on the clean encoder) | `checkpoints/finetune/*/clean_10.0M_to_fog_mask__*` |
| E7 | Q2 | Perturbation saliency per object (blur each object, measure policy change); attention maps of `attn_lstm` per object | `analysis/saliency.json` |
| E8 | Q5 | Behaviour statistics from E2: paddle movement while the ball is hidden, hit offsets, rally lengths | `analysis/behaviour.json` |
| E9 | Q4, Q6 | LSTM memory timescale (forget-gate activity) and unit statistics across scratch, parent and fine-tuned models | `analysis/lstm_dynamics.json` |

| E10 | Q4 | Attention block: replay each incoming flight with the attention-LSTM unable to attend to the ball's 3x3 cells before the fog (vs a same-size sham block); measure memory probe error inside the fog and return rate; plus attention-on-ball vs probe error per flight | `analysis/attention_block.jsonl` |

Decision rules are written in each experiment's Flywheel node before results come in.
