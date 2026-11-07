# Analysis plan

Decided 2026-11-07, while the `vit` / `vit_lstm` sweep trains on `computing`.

**Focus:** `cnn` (4 stacked frames, no memory) vs `lstm` (1 frame + LSTM memory). The `vit` / `vit_lstm`
pair will get the same analysis once trained, giving a 2×2 of vision (CNN vs ViT) × memory (frame stack vs
LSTM) at matched size. `attn_lstm` is set aside: it is 7–9× smaller than the others, so its results are
confounded with size.

Current numbers come from training games (sampling policy, weights changing), not a frozen evaluation.

## Steps

1. **Frozen evaluation.** Evaluate all `cnn` and `lstm` checkpoints (24 scratch + 18 fine-tunes, plus the
   6 clean 10M parents) with fixed seeds and frozen weights, on CPU so it runs alongside training.
   Every number on the dashboard becomes an evaluated one. Each checkpoint plays all 4 views (train view ×
   test view grid), 30 games with seeds 10000–10029 shared by every model, sampling policy; greedy as a
   check on the scratch finals (only 19 distinct games out of 30 under greedy play: the start depends only on
   the reset no-op count). Runner: `scripts/evaluate_sweep.py` (4 CPU processes, resumable): 48 checkpoints ×
   4 views = 192 jobs.
2. **How long the fog hides the ball.** Count the agent steps the ball spends inside the fog band
   (x 42–72) per crossing. If it is 4 or fewer, the CNN's 4-frame stack already bridges the gap, and memory
   is not needed by construction; then consider a wider fog.
3. **Fine-tune vs scratch at equal steps.** On each target view, compare a fine-tune after k steps with a
   scratch run after the same k steps, to tell whether clean pre-training helps or hurts (negative
   transfer / loss of plasticity).
4. **Does the LSTM use its memory?**
   - Memory wipe: reset the LSTM state while the ball is inside the fog and measure the drop in play.
   - Linear probe: decode the hidden ball's position from the LSTM state while it is occluded.
5. **Dashboard.** Add the evaluated numbers and the results of steps 2–4 to the Pong occlusion sweep
   dashboard (https://claude.ai/artifact/TjkpB2vLGaLkxzdWErEjjw).
