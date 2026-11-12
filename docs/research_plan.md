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
| E11 | Q3, Q4 | Scripted reference players on the evaluation games: copy the enemy paddle (the shortcut alone), go to the ball's last seen height, extrapolate its seen path (ideal short memory), the same with no fog | `analysis/scripted_baselines.jsonl` |
| E13 | Q2, Q4 | Attention over a flight: attention weight of attn_lstm on the ball, own paddle and enemy paddle, aligned on fog entry, plus one recorded flight (`scripts/attention_maps.py`) | `analysis/attention_maps.json` |
| E12 | Q5 | Size control for the attention agent: attn_lstm_wide (8 heads, key 64, ReLU values of 320 units, a 512-unit ReLU layer after the readout; 1.98M parameters vs lstm 2.01M, same 128-unit memory) trained on fog_mask from scratch and fine-tuned from its clean 10M checkpoint, 3 seeds, the attn_lstm protocol (`scripts/run_wide.sh`); other views only if it learns. **Done 2026-11-12:** scratch fog_mask +19.5 / -11.1 / -4.2 (no seed stalls, 1 of 3 wins), curriculum 3/3; other views not run | `checkpoints/{scratch,finetune}/attn_lstm_wide/` |
| E12b | Q1-Q3 | The wide agent on the remaining views, so every claim about attention and memory can rest on a size-matched agent (user, 2026-11-12): scratch mask and fog 15M, clean 10M -> mask / fog fine-tunes, 3 seeds; launched 2026-11-12 11:45 | `VIEWS="clean mask fog fog_mask" FT_VIEWS="mask fog fog_mask" scripts/run_wide.sh` |
| E12c | Q4 | The memory and attention analyses (E2 recordings, E3 probes, E4 wipes, E10 attention block, E13 attention over a flight) on the wide models: clean and fog_mask now, mask and fog when E12b ends; then E5 / E5c (enemy-paddle shortcut) on the wide fog models | `scripts/run_wide_analyses.sh` |

Decision rules are written in each experiment's Flywheel node before results come in.
