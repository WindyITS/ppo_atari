# Where the data is

Every result of the project, where it lives, which script writes it and where it is used. The experiments are
described in `docs/research_plan.md` (what and why) and `docs/research_report.md` (what came out).

## Rules

- **The training host is the source of every result** (`ssh computing`, `~/projects/ppo_atari`). This
  checkout holds a copy of the light part at the same paths: `bash scripts/sync_results.sh`. Results are
  never edited by hand; a result made locally goes to the host with `sync_results.sh --push <file>`.
- **One current file per experiment, with a plain name.** A superseded version is not deleted: it moves to
  `archive/` with a line in `archive/README.md` saying what it was and why it was replaced.
- **Results only grow.** Every script appends rows and skips work already in its output, so a rerun resumes
  and adding models adds rows. Each row names its model (`run`, `arch`, `train_view`, `kind`; evaluation rows
  also the checkpoint's `sha256`), so one file can hold every architecture and readers filter rows.
- Data folders are gitignored; code, docs, figures and the paper are in git.

## Folders

| folder | content |
|---|---|
| `checkpoints/{scratch,finetune}/<arch>/<view or clean_10.0M_to_<view>>/seed_<n>/` | models: `step_XX.XM.pt` milestones and `final.pt`. `arch`: `cnn`, `lstm`, `attn_lstm`, `attn_lstm_wide`. A `__<tag>` suffix marks a control run (E6: `__ft15M`, `__reinit_core`) |
| `logs/{scratch,finetune}/…/train.log` | training log of each run, same layout as `checkpoints/` |
| `logs/sweeps/` | launchers of training sweeps and evaluations (`wide.log`, `eval_*.log`, `controls.log`, …) |
| `logs/analysis/` | one log per analysis run |
| `evaluations/` | frozen evaluations (30 games per model and view) |
| `analysis/` | every other result, one file per experiment (table below) |
| `analysis/rollouts/` | raw recordings of the evaluation games (host only, ~4 GB) |
| `archive/` | superseded results, aborted training runs, old tooling; `archive/README.md` lists every item |
| `wandb/` | Weights & Biases run folders. Their links to the ViT checkpoints are broken, but the checkpoints themselves are on W&B (`windyits-bocconi-university/pong-rl`) |

Locally there is also `checkpoints/finetune/attn_lstm/clean_10.0M_to_fog_mask/seed_{1,2}/final.pt`, copied from
the host for the attention strip of Figure `attention`.

## Results by experiment

| exp | question | script | output | used in |
|---|---|---|---|---|
| E1, E12 | how well each model plays each view | `evaluate_sweep.py` | `evaluations/sweep_stochastic.jsonl` (one row per model x view, with its 30 games) | Fig. `returns`; paper 3.1, 3.6; report Q1 |
| E6 | fine-tune controls (15M steps, reinitialised core) | `run_controls.sh`, `evaluate_sweep.py --finetune-tag` | `evaluations/controls_stochastic.jsonl`; models in `checkpoints/finetune/*/clean_10.0M_to_fog_mask__*` | paper 3.1; report Q6 |
| (step 2) | how long the fog hides the ball | `fog_exposure.py` | `evaluations/fog_exposure.jsonl` | paper 3.3; report Q5 |
| E2 | recordings of the evaluation games | `record_rollouts.py` | `analysis/rollouts/<run>__final.npz` + `.json` (host) | input of E3, E7-E9, E13 |
| E3 | is the hidden ball in memory? | `probe_memory.py` | `analysis/probes.jsonl` | paper 3.3; report Q4 |
| E3b | probes against an untrained twin | `random_control.py`, then `probe_memory.py --output analysis/probes_random.jsonl` | twins `analysis/rollouts/*.random.npz` (log `analysis/random_control.jsonl`); probes `analysis/probes_random.jsonl` | Fig. `probes`; paper 3.3; report Q4 |
| E4 | wiping memory as the ball enters the fog | `memory_interventions.py` | `analysis/interventions/<run>.json` | Fig. `wipes`; paper 3.3; report Q4 |
| E5 | erase / mirror the enemy paddle | `view_interventions.py` | `analysis/view_interventions.jsonl` | paper 3.2; report Q1, Q2/Q3 |
| E5b | does the agent follow the mirrored paddle? | `decoy_tracking.py` | `analysis/decoy_tracking.jsonl` | report Q2/Q3 |
| E5c | moving the enemy paddle by a few pixels | `enemy_shift.py` | `analysis/enemy_shift.jsonl` | Fig. `shift`; paper 3.2; report Q2/Q3 |
| E7-E9 | play statistics, saliency, attention mass, LSTM timescales | `summarize_rollouts.py` | `analysis/rollout_summary.jsonl` | paper 3.3 (timescales), 3.4 (early commitment); report Q2/Q3 (saliency), Q5 (behaviour, timescales), Q6 |
| E10 | blocking attention to the ball | `attention_block.py` | `analysis/attention_block.jsonl` | Fig. `block`; paper 3.5; report Q4 |
| E11 | scripted reference players | `scripted_baselines.py` | `analysis/scripted_baselines.jsonl` | report Q2/Q3 |
| E13 | attention over a flight | `attention_maps.py summary` / `strip`; `attention_heads.py` (per head, printed) | `analysis/attention_maps.json` (attn_lstm), `analysis/attention_maps_attn_lstm_wide.json`, `analysis/attention_strip.npz` | Fig. `attention`; paper 3.5; report Q4 |
| E12c | E2-E4, E10, E13 on attn_lstm_wide | `run_wide_analyses.sh` | rows added to the files above | report Q4 (E12c) |
| all | every table of the report | `report_tables.py` | `analysis/report.json` (derived; regenerate after new results) | report |

Figures: `scripts/make_figures.py` reads `analysis/` and `evaluations/` and writes `figures/<name>.pdf/.png`
(`environments` and `views` need no data). The paper (`paper/main.tex`) includes them from `figures/`.
