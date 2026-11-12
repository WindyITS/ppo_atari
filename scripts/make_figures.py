"""Figures for the paper, from the analysis outputs (docs/research_report.md).

    bash scripts/sync_results.sh                     # local copy of the host's analysis/ and evaluations/
    uv run python scripts/make_figures.py

--data is the analysis/ folder (enemy_shift.jsonl, attention_block.jsonl, probes_random.jsonl, interventions/,
attention_maps.json, attention_strip.npz) and --evaluations the evaluations/ folder (sweep_stochastic.jsonl);
docs/data.md describes both. Each figure is
written as PDF (for the paper) and PNG (for a quick look); --only picks some of them. Models are the final
checkpoints. "Curriculum" is attn_lstm trained on clean Pong for 10M steps, then fine-tuned on the view.

    views    the four views of one moment of a game, with the hidden regions and the hidden ball marked
    returns  return of every model on the view it was trained on (dots: seeds)
    shift    balls returned when the enemy paddle the model sees is moved (E5c)
    wipes    balls returned when the memory is wiped before, at and inside the fog (E4)
    probes   R^2 of a linear probe for the hidden ball's height: trained vs untrained network (E3, E3b)
    block    the attention agent on fog + hidden paddle: attention blocked on the ball before the fog (E10)
    attention  where the attention agent looks during a flight (E13; attention_maps.json, attention_strip.npz)
    environments  the four views in colour: one flight as a time-lapse trail on the game screen, and the 84x84
                  frame the agent receives (needs no data: a scripted player plays the game)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

# Categorical slots 1-3 of the reference palette (validated all-pairs, colour-blind safe); one per agent.
COLOURS = {"cnn": "#2a78d6", "lstm": "#eb6834", "attn_lstm": "#1baf7a"}
NEUTRAL = "#8a8985"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"cnn": "CNN (4 frames)", "lstm": "CNN + LSTM", "attn_lstm": "CNN + attention + LSTM"}
VIEW_NAMES = {"clean": "clean", "mask": "hidden paddle", "fog": "fog", "fog_mask": "fog + hidden paddle"}
VIEWS = ("clean", "mask", "fog", "fog_mask")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSON-lines file; a torn last line (a power cut mid-write) is skipped."""
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def model_key(row: dict) -> tuple[str, str] | None:
    """(agent, view) of a final checkpoint in the paper, or None for runs the paper leaves out.

    cnn and lstm count from scratch only; attn_lstm counts as "attn_lstm" from scratch and as "curriculum"
    when fine-tuned from its clean 10M checkpoint. Tagged control runs (folder names with "__") are left out.
    """
    run = row["run"]
    kind, arch, label = run.split("/")[:3]
    if "__" in label:
        return None
    if kind == "scratch":
        return arch, label
    if arch == "attn_lstm" and label.startswith("clean_10.0M_to_"):
        return "curriculum", label.removeprefix("clean_10.0M_to_")
    return None


def colour(agent: str) -> str:
    return COLOURS["attn_lstm" if agent == "curriculum" else agent]


def name(agent: str) -> str:
    return "attention + LSTM, curriculum" if agent == "curriculum" else NAMES[agent]


def save(fig, out: Path, stem: str):
    for suffix in (".pdf", ".png"):
        fig.savefig(out / f"{stem}{suffix}")
    plt.close(fig)
    print(f"wrote {out / stem}.pdf/.png")


def dots(ax, x: float, values, c: str, width: float = 0.5, hollow: bool = False):
    """A thin bar at the mean with one dot per seed."""
    values = np.asarray(values, float)
    ax.bar(x, values.mean(), width, color=c, alpha=0.28, edgecolor="none")
    ax.hlines(values.mean(), x - width / 2, x + width / 2, color=c, linewidth=2)
    jitter = np.linspace(-width / 5, width / 5, len(values)) if len(values) > 1 else [0.0]
    ax.scatter(x + np.asarray(jitter), values, s=18, zorder=3, linewidths=1.2,
               facecolor="white" if hollow else c, edgecolor=c)


# --- figures -----------------------------------------------------------------------------------------------

def fig_views(data: Path, out: Path):
    """One moment of a game under each view: the same seed and actions, so the four frames match."""
    from pong_rl.analysis import scene
    from pong_rl.envs import make_env
    from pong_rl.wrappers import COURT_Y_MAX, COURT_Y_MIN
    flags = {"clean": (False, False), "mask": (False, True), "fog": (True, False), "fog_mask": (True, True)}
    envs = {v: make_env("ALE/Pong-v5", 0, False, "figure", use_fog=f, mask_enemy=m, frame_stack=1,
                        tap_clean_frame=True)() for v, (f, m) in flags.items()}
    frames = {v: env.reset(seed=7)[0] for v, env in envs.items()}
    rng = np.random.default_rng(0)
    try:
        for _ in range(2000):  # play until the ball is deep in the fog, heading for the agent
            clean = np.asarray(envs["clean"].get_wrapper_attr("clean_frame"))
            where = scene.locate(clean)
            if where["ball_x"] == where["ball_x"] and 52 <= where["ball_x"] <= 62:
                break
            action = int(rng.choice([0, 2, 3]))
            frames = {v: env.step(action)[0] for v, env in envs.items()}
    finally:
        for env in envs.values():
            env.close()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.25))
    for ax, view in zip(axes, VIEWS):
        ax.imshow(np.asarray(frames[view])[-1], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(VIEW_NAMES[view], color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        hidden = []
        if flags[view][0]:
            hidden.append((42, 72))
        if flags[view][1]:
            hidden.append((6, 12))
        for x0, x1 in hidden:
            ax.add_patch(Rectangle((x0 - 0.5, COURT_Y_MIN - 0.5), x1 - x0, COURT_Y_MAX - COURT_Y_MIN,
                                   fill=False, edgecolor="#e0a400", linewidth=1.0, linestyle=(0, (3, 2))))
        if flags[view][0]:
            ax.add_patch(Circle((where["ball_x"], where["ball_y"]), 3.2, fill=False, edgecolor="#e0a400",
                                linewidth=1.0))
    fig.text(0.5, 0.06, "Dashed: region painted over (the ball and the enemy paddle vanish there). "
             "Circle: where the hidden ball really is.", ha="center", color=INK_2, fontsize=7.5)
    save(fig, out, "views")


def fig_returns(data: Path, out: Path):
    rows = read_jsonl(EVALUATIONS / "sweep_stochastic.jsonl")
    returns = defaultdict(list)
    for row in rows:
        key = model_key(row)
        if key is None or row["checkpoint_file"] != "final.pt" or row["eval_view"] != key[1]:
            continue
        returns[key].append(row["summary"]["mean_return"])
    agents = ("cnn", "lstm", "attn_lstm", "curriculum")
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for j, view in enumerate(VIEWS):
        for i, agent in enumerate(agents):
            values = returns.get((agent, view))
            if values:
                dots(ax, j + (i - 1.5) * 0.2, values, colour(agent), width=0.17, hollow=agent == "curriculum")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(VIEWS)), [VIEW_NAMES[v] for v in VIEWS])
    ax.set_xlabel("view the model was trained and tested on")
    ax.set_ylim(-21.5, 21.5)
    ax.set_yticks([-21, -10, 0, 10, 21])
    ax.set_ylabel("mean return (30 games)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colour(a), markersize=5,
                          markerfacecolor="white" if a == "curriculum" else colour(a)) for a in agents]
    ax.legend(handles, [name(a) for a in agents], ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=7.5, handletextpad=0.3, columnspacing=1.2)
    save(fig, out, "returns")


def fig_shift(data: Path, out: Path):
    # Models that never learned their view (unshifted return rate below 0.5) have no play to disturb.
    rows = [r for r in read_jsonl(data / "enemy_shift.jsonl") if model_key(r) and r["return_rate"]["shift_0"] >= 0.5]
    shifts = (-8, -4, 0, 4, 8)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, view in zip(axes, ("clean", "fog")):
        for agent in ("cnn", "lstm", "attn_lstm"):
            # On fog the attention agent counts from scratch and with the curriculum alike: same question.
            group = [r for r in rows if model_key(r)[1] == view
                     and model_key(r)[0] in ((agent, "curriculum") if agent == "attn_lstm" else (agent,))]
            if not group:
                continue
            curves = np.array([[r["return_rate"][f"shift_{s}"] for s in shifts] for r in group])
            for curve in curves:
                ax.plot(shifts, curve, color=colour(agent), alpha=0.25, linewidth=0.8)
            ax.plot(shifts, curves.mean(0), color=colour(agent), linewidth=2, marker="o", markersize=4,
                    label=f"{NAMES[agent]} ({len(group)})")
        ax.set_title(f"trained on {VIEW_NAMES[view]}", color=INK)
        ax.set_xticks(shifts)
        ax.set_xlabel("enemy paddle shown shifted by (px)")
        ax.set_ylim(0, 1.03)
    axes[0].set_ylabel("share of balls returned")
    handles = [plt.Line2D([], [], color=COLOURS[a], linewidth=2, marker="o", markersize=4) for a in COLOURS]
    fig.legend(handles, [NAMES[a] for a in COLOURS], ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.98),
               fontsize=7.5)
    attention = sum(model_key(r)[0] in ("attn_lstm", "curriculum") and model_key(r)[1] == "fog" for r in rows)
    fig.text(0.5, -0.1, f"Thick: mean; thin: single models (the attention agent on fog: {attention} models, from "
             "scratch and curriculum). Only the paddle's pixels move, never the ball.", ha="center", color=INK_2,
             fontsize=7)
    save(fig, out, "shift")


def fig_wipes(data: Path, out: Path):
    branches = ("control", "wipe_before", "wipe_entry", "wipe_mid")
    labels = ("no wipe", "wipe before fog", "wipe at fog entry", "wipe 7 steps in")
    rates = defaultdict(list)
    for path in sorted((data / "interventions").glob("*.json")):
        row = json.loads(path.read_text())
        key = model_key(row)
        if key is None or row["view"] != key[1]:
            continue
        rates[key].append([row["totals"][b]["return_rate"] for b in branches])
    views, agents = ("fog", "fog_mask"), ("cnn", "lstm", "attn_lstm", "curriculum")
    fig, axes = plt.subplots(len(views), len(agents), figsize=(7.2, 4.6), sharey=True, sharex=True)
    for row, view in enumerate(views):
        for col, agent in enumerate(agents):
            ax = axes[row][col]
            ax.set_ylim(0, 1.03)
            if row == 0:
                title = "attention + LSTM, from scratch" if agent == "attn_lstm" else name(agent)
                ax.set_title(title.replace(", ", ",\n"), color=INK, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{VIEW_NAMES[view]}\nshare of balls returned")
            ax.set_xticks(range(len(branches)), labels, rotation=35, ha="right")
            if (agent, view) not in rates:
                ax.text(1.5, 0.5, "no model\nlearned this view", ha="center", va="center", color=INK_2,
                        fontsize=7)
                continue
            values = np.array(rates[(agent, view)])
            for i in range(len(branches)):
                dots(ax, i, values[:, i], colour(agent) if i else NEUTRAL, width=0.6, hollow=agent == "curriculum")
            ax.text(0.02, 0.03, f"{len(values)} model{'s' if len(values) > 1 else ''}", transform=ax.transAxes,
                    fontsize=6.5, color=INK_2)
    fig.subplots_adjust(hspace=0.15)
    save(fig, out, "wipes")


def fig_probes(data: Path, out: Path):
    bins = (("r2_0_3", "0-3"), ("r2_4_7", "4-7"), ("r2_8_99", "8+"))
    curves = defaultdict(lambda: defaultdict(list))
    for row in read_jsonl(data / "probes_random.jsonl"):
        key = model_key(row)
        if key is None or key[0] == "attn_lstm":
            continue  # attn_lstm from scratch fails on fog + mask; the curriculum agent stands for it
        for source in ("memory", "random"):
            curves[key][source].append([row["probes"]["now"][source][b] for b, _ in bins])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, view in zip(axes, ("fog", "fog_mask")):
        for agent in ("cnn", "lstm", "curriculum"):
            if (agent, view) not in curves:
                continue
            for source, style in (("memory", "-"), ("random", (0, (3, 2)))):
                mean = np.mean(curves[(agent, view)][source], axis=0)
                ax.plot(range(len(bins)), mean, color=colour(agent), linestyle=style, linewidth=2 if
                        source == "memory" else 1.4, marker="o" if source == "memory" else None, markersize=4,
                        label=name(agent) if source == "memory" else None)
        ax.set_title(f"{VIEW_NAMES[view]}", color=INK)
        ax.set_xticks(range(len(bins)), [b for _, b in bins])
        ax.set_xlabel("steps since the ball disappeared")
        ax.set_ylim(-0.05, 1.02)
    axes[0].set_ylabel("probe $R^2$, hidden ball height")
    handles, names = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in names:
                handles.append(handle)
                names.append(label)
    handles += [plt.Line2D([], [], color=INK_2, linewidth=2), plt.Line2D([], [], color=INK_2, linestyle=(0, (3, 2)))]
    names += ["trained network", "untrained network, same frames"]
    axes[0].legend(handles, names, fontsize=7, loc="lower left")
    save(fig, out, "probes")


def fig_block(data: Path, out: Path):
    branches = ("control", "block_sham", "block_ball", "erase_ball")
    labels = ("no block", "block elsewhere", "block the ball", "erase the ball")
    models = []
    for row in read_jsonl(data / "attention_block.jsonl"):
        key = model_key(row)
        if key is None or key[1] != "fog_mask" or row["arch"] != "attn_lstm":
            continue
        if row["branches"]["control"]["return_rate"] < 0.5:
            continue  # scratch seeds that never learned the view: nothing to block
        models.append((key[0], row["run"].split("/")[-1].replace("seed_", "s"), row["branches"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for agent, seed, result in models:
        hollow = agent == "curriculum"
        for ax, metric in zip(axes, ("return_rate", "probe_mae")):
            values = [result[b][metric] for b in branches]
            ax.plot(range(len(branches)), values, color=COLOURS["attn_lstm"], alpha=0.5, linewidth=0.9)
            ax.scatter(range(len(branches)), values, s=20, zorder=3, color=COLOURS["attn_lstm"],
                       facecolor="white" if hollow else COLOURS["attn_lstm"], linewidths=1.2)
    for ax in axes:
        ax.set_xticks(range(len(branches)), labels, rotation=20, ha="right")
        ax.set_xlim(-0.3, len(branches) - 0.7)
    curriculum = sum(agent == "curriculum" for agent, _, _ in models)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLOURS["attn_lstm"], markerfacecolor=face)
               for face in ("white", COLOURS["attn_lstm"])]
    axes[0].legend(handles, [f"curriculum ({curriculum} seeds)", f"from scratch ({len(models) - curriculum} seed)"],
                   fontsize=7, loc="lower left")
    fig.subplots_adjust(wspace=0.35)
    axes[0].set_ylabel("share of balls returned")
    axes[0].set_ylim(0, 1.03)
    axes[1].set_ylabel("memory error on the hidden ball (px)")
    fig.suptitle("attention + LSTM on fog + hidden paddle: attention changed just before the fog",
                 fontsize=8.5, color=INK, y=1.02)
    save(fig, out, "block")


TARGET_COLOURS = {"ball": "#eda100", "enemy_paddle": "#6250d6", "own_paddle": "#e87ba4"}  # slots 4, 7, 5
TARGET_NAMES = {"ball": "the ball (3x3 cells)", "enemy_paddle": "opponent's paddle", "own_paddle": "own paddle"}
STRIP_STEPS = (-8, -4, -1, 0, 6, 13)


def strip_title(k: int) -> str:
    return "0: enters fog" if k == 0 else f"{k:+d}"


def fig_attention(data: Path, out: Path):
    """Top: one returned flight of the curriculum attention agent on fog + hidden paddle (attention_strip.npz).
    Bottom: attention on each target, aligned on fog entry, per view (attention_maps.json; models that play)."""
    from matplotlib.gridspec import GridSpec
    strip = np.load(data / "attention_strip.npz")
    rows = [r for r in json.loads((data / "attention_maps.json").read_text())
            if r["flights"] >= 100 and r["view"] in ("clean", "fog", "fog_mask")
            and model_key({"run": r["run"]}) is not None]
    fig = plt.figure(figsize=(7.2, 4.4))
    grid = GridSpec(2, 6, figure=fig, height_ratios=(1, 1.2), hspace=0.42, wspace=0.12)
    rgb = matplotlib.colors.to_rgb(COLOURS["attn_lstm"])
    steps = list(strip["steps"])
    for column, k in enumerate(STRIP_STEPS):
        ax = fig.add_subplot(grid[0, column])
        i = steps.index(k)
        ax.imshow(strip["seen"][i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        weights = strip["attention"][i].mean(axis=0).reshape(7, 7)  # mean over the two heads
        overlay = np.zeros((7, 7, 4))
        overlay[..., :3] = rgb
        overlay[..., 3] = np.clip(weights / 0.4, 0, 1) * 0.85
        ax.imshow(overlay, extent=(-0.5, 83.5, 83.5, -0.5), interpolation="nearest")
        x, y = strip["ball"][i]
        if x == x:  # the ball is a few pixels: circle it (inside the fog it is hidden from the agent)
            ax.add_patch(Circle((x, y), 3.5, fill=False, edgecolor=TARGET_COLOURS["ball"], linewidth=1.0,
                                linestyle="-" if x < 42 else (0, (1.5, 1))))
        ax.set_title(strip_title(k), fontsize=7.5, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.5, 0.535, "steps from the ball entering the fog. Green: attention (mean of the two heads, per "
             "cell of the 7x7 map); circle: the ball (dotted: hidden)", ha="center", color=INK_2, fontsize=7)
    panels = (("clean", "clean (aligned at the midline)"), ("fog", "fog"), ("fog_mask", "fog + hidden paddle"))
    shared = None
    for column, (view, title) in enumerate(panels):
        ax = fig.add_subplot(grid[1, 2 * column:2 * column + 2], sharey=shared)
        shared = shared or ax
        models = [r for r in rows if r["view"] == view]
        ks = np.array(models[0]["steps"])
        if view != "clean":
            ax.axvspan(0, ks[-1], color=GRID, alpha=0.6, linewidth=0)
            ax.text(ks[-1] - 0.3, 0.97, "ball hidden", ha="right", va="top", fontsize=7, color=INK_2)
        ax.axhline(models[0]["uniform"]["ball"], color=NEUTRAL, linewidth=0.8, linestyle=(0, (2, 2)))
        for target, c in TARGET_COLOURS.items():
            curves = np.array([r["mean"][target] for r in models])
            for curve in curves:
                ax.plot(ks, curve, color=c, linewidth=0.6, alpha=0.3)
            ax.plot(ks, curves.mean(axis=0), color=c, linewidth=2)
        ax.set_title(f"{title}: {len(models)} models", fontsize=8, color=INK)
        ax.set_xlim(ks[0], ks[-1])
        ax.set_ylim(0, 1)
        if column == 0:
            ax.set_ylabel("attention weight")
        else:
            ax.tick_params(labelleft=False)
    fig.text(0.5, 0.02, "steps from the ball entering the agent's half (thin lines: models; thick: mean)",
             ha="center", color=INK, fontsize=8)
    handles = [plt.Line2D([], [], color=c, linewidth=2) for c in TARGET_COLOURS.values()]
    handles.append(plt.Line2D([], [], color=NEUTRAL, linewidth=0.8, linestyle=(0, (2, 2))))
    fig.legend(handles, [f"on {n}" for n in TARGET_NAMES.values()] + ["uniform attention on 3x3 cells"],
               loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    save(fig, out, "attention")


# --- environments: the game screen in colour ----------------------------------------------------------------

SCREEN_COURT = (34, 194)          # rows of the 210x160 screen between the walls
SCREEN_BG = (144, 72, 17)
SCREEN_BALL = (236, 236, 236)
TO_SCREEN = 160 / 84              # AtariPreprocessing resizes the whole 210x160 screen to 84x84
SCREEN_FOG = (42 * TO_SCREEN, 72 * TO_SCREEN)     # columns 42-71 of the 84x84 frame
SCREEN_MASK = (6 * TO_SCREEN, 12 * TO_SCREEN)     # columns 6-11


def screen_ball(rgb: np.ndarray) -> tuple[float, float] | None:
    """Centre (x, y) of the ball on the 210x160 screen, None if it is not on the court."""
    rows = slice(*SCREEN_COURT)
    ys, xs = np.where((rgb[rows] == SCREEN_BALL).all(axis=-1))
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean() + SCREEN_COURT[0])


def record_flight(seed: int = 3, warmup: int = 15):
    """Four lockstep games (one per view), played by a scripted player that moves its paddle towards the
    ball's true height, until one flight from the opponent's paddle to a return by the agent is complete.
    Returns the colour screens, the ball's screen positions and each view's 84x84 frames, one per step."""
    from pong_rl.analysis import scene
    from pong_rl.envs import make_env
    flags = {"clean": (False, False), "mask": (False, True), "fog": (True, False), "fog_mask": (True, True)}
    envs = {v: make_env("ALE/Pong-v5", 0, False, "figure", use_fog=f, mask_enemy=m, frame_stack=1,
                        tap_clean_frame=True)() for v, (f, m) in flags.items()}
    seen = {v: env.reset(seed=seed)[0] for v, env in envs.items()}
    screens, balls, frames = [], [], {v: [] for v in VIEWS}
    start = None
    try:
        for t in range(5000):
            clean = np.asarray(envs["clean"].get_wrapper_attr("clean_frame"))
            where = scene.locate(clean)
            rgb = envs["clean"].unwrapped.ale.getScreenRGB().copy()
            screens.append(rgb)
            balls.append(screen_ball(rgb))
            for v in VIEWS:
                frames[v].append(np.asarray(seen[v])[-1].copy())
            x = [b[0] if b else np.nan for b in balls]
            if t > warmup and start is None and len(x) > 2 and x[-2] == x[-2] and x[-1] > x[-2] and x[-2] < 40:
                start = t - 1  # the ball has just left the opponent's paddle
            if start is not None and len(x) > start + 3 and x[-1] == x[-1] and x[-2] == x[-2]:
                if x[-1] < x[-2] and x[-2] > SCREEN_FOG[1]:  # moving back left after the agent's paddle
                    end = t + 3
                    if end - start > 12:
                        break
                    start = None
                if x[-1] < x[-2] and x[-2] < SCREEN_FOG[0]:  # turned before the agent: not this flight
                    start = None
            target = where["ball_y"] if where["ball_y"] == where["ball_y"] else 46.0
            agent = where["agent_y"]
            action = 0 if agent != agent or abs(target - agent) < 2 else (2 if target < agent else 3)
            for v, env in envs.items():
                obs, _, terminated, truncated, _ = env.step(action)
                seen[v] = obs
                if terminated or truncated:
                    raise RuntimeError("the game ended before a returned flight; try another seed")
        else:
            raise RuntimeError("no returned flight found")
        for _ in range(3):  # a few steps of the return
            for v, env in envs.items():
                seen[v] = env.step(0)[0]
            rgb = envs["clean"].unwrapped.ale.getScreenRGB().copy()
            screens.append(rgb)
            balls.append(screen_ball(rgb))
            for v in VIEWS:
                frames[v].append(np.asarray(seen[v])[-1].copy())
    finally:
        for env in envs.values():
            env.close()
    span = slice(start, len(screens))
    return screens[span], balls[span], {v: f[span] for v, f in frames.items()}


SCREEN_CROP = (31, 197)           # court plus a sliver of each wall; drops the score bar
SCREEN_ENEMY = (213, 130, 74)
VIEW_SUBTITLES = {"clean": "everything visible", "mask": "opponent's paddle hidden",
                  "fog": "agent's half hidden", "fog_mask": "both hidden"}


def mist(height: int, width: int, x0: float, x1: float, alpha: float = 0.42, feather: float = 2.5) -> np.ndarray:
    """RGBA overlay: a soft white haze over columns [x0, x1) with feathered edges."""
    xs = np.arange(width) + 0.5
    profile = 1 / (1 + np.exp(-(xs - x0) / feather * 2)) * 1 / (1 + np.exp((xs - x1) / feather * 2))
    rgba = np.ones((height, width, 4))
    rgba[..., 3] = np.clip(alpha * np.broadcast_to(profile[None], (height, width)), 0, 1)
    return rgba


def paddle_box(rgb: np.ndarray, colour: tuple, columns: slice) -> tuple[float, float, float, float] | None:
    """(x, y, width, height) of a paddle on the screen, from its colour inside the court."""
    court = rgb[SCREEN_COURT[0]:SCREEN_COURT[1], columns]
    ys, xs = np.where((court == colour).all(axis=-1))
    if len(xs) == 0:
        return None
    return (xs.min() + columns.start - 0.5, ys.min() + SCREEN_COURT[0] - 0.5, xs.max() - xs.min() + 1,
            ys.max() - ys.min() + 1)


def fig_environments(data: Path, out: Path):
    screens, balls, _ = record_flight()
    hit = max(range(len(balls)), key=lambda i: balls[i][0] if balls[i] else -1)  # at the agent's paddle
    path = [b for b in balls[:hit + 1] if b]
    ys = [y for _, y in path]
    bounces = [path[i] for i in range(1, len(path) - 1) if (ys[i] - ys[i - 1]) * (ys[i + 1] - ys[i]) < 0]
    corners = [path[0], *bounces, path[-1]]
    # Both paddles at the moment each touches the ball: the opponent's from the flight's first step, the rest
    # (agent's paddle and the ball at its paddle) from the hit.
    scene_rgb = screens[hit].copy()
    scene_rgb[SCREEN_COURT[0]:SCREEN_COURT[1], :21] = screens[0][SCREEN_COURT[0]:SCREEN_COURT[1], :21]
    top, bottom = SCREEN_CROP
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), gridspec_kw={"wspace": 0.05})
    for ax, view in zip(axes, VIEWS):
        fog, mask = view in ("fog", "fog_mask"), view in ("mask", "fog_mask")
        frame = scene_rgb.copy()
        court = slice(*SCREEN_COURT)
        enemy = paddle_box(frame, SCREEN_ENEMY, slice(0, int(SCREEN_MASK[1]) + 2))
        if fog:  # painted over, as in the agent's view: the ball vanishes
            frame[court, int(round(SCREEN_FOG[0])):int(round(SCREEN_FOG[1]))] = SCREEN_BG
        if mask:
            frame[court, int(round(SCREEN_MASK[0])):int(round(SCREEN_MASK[1]))] = SCREEN_BG
        ax.imshow(frame, interpolation="nearest", extent=(-0.5, 159.5, 209.5, -0.5))
        if fog:
            ax.imshow(mist(210, 160, *SCREEN_FOG), interpolation="bilinear", extent=(-0.5, 159.5, 209.5, -0.5))
            ax.text(np.mean(SCREEN_FOG), bottom - 7, "fog", color="white", ha="center", va="bottom",
                    fontsize=7.5, alpha=0.95)
        if mask and enemy:
            ax.add_patch(Rectangle(enemy[:2], enemy[2], enemy[3], fill=False, edgecolor=(0.84, 0.51, 0.29),
                                   linewidth=0.9, linestyle=(0, (2, 1.5))))
        # The flight: solid where this view shows the ball, dashed where it hides it.
        def hidden(x):
            return (fog and SCREEN_FOG[0] <= x < SCREEN_FOG[1]) or (mask and SCREEN_MASK[0] <= x < SCREEN_MASK[1])
        # Straight lines between the flight's key points (start, wall bounces, end), cut where the view hides it.
        for (xa, ya), (xb, yb) in zip(corners[:-1], corners[1:]):
            ts = np.linspace(0, 1, 200)
            xs, ys = xa + (xb - xa) * ts, ya + (yb - ya) * ts
            gone = np.array([hidden(x) for x in xs])
            cuts = np.flatnonzero(np.diff(gone.astype(int))) + 1
            for run in np.split(np.arange(len(ts)), cuts):
                run = np.append(run, min(run[-1] + 1, len(ts) - 1))
                g = gone[run[0]]
                ax.plot(xs[run], ys[run], color="white", linewidth=1.0 if g else 1.3, alpha=0.85 if g else 0.95,
                        linestyle=(0, (2.2, 1.6)) if g else "-", solid_capstyle="round", dash_capstyle="round")
        (xa, ya), (xb, yb) = corners[-2], corners[-1]
        ax.annotate("", xy=(xb, yb), xytext=(xb - (xb - xa) * 0.08, yb - (yb - ya) * 0.08),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=0.0, mutation_scale=7))
        if view == "clean":
            agent = paddle_box(frame, (92, 186, 92), slice(130, 150))
            if enemy:
                ax.text(enemy[0] + enemy[2] + 3, enemy[1] + enemy[3] / 2, "opponent", color="white",
                        fontsize=6.5, va="center", ha="left")
            if agent:
                ax.text(agent[0] + agent[2], agent[1] - 3, "agent", color="white", fontsize=6.5,
                        va="bottom", ha="right")
        ax.set_xlim(-0.5, 159.5)
        ax.set_ylim(bottom - 0.5, top - 0.5)
        ax.text(0.5, 1.13, VIEW_NAMES[view], transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=INK)
        ax.text(0.5, 1.03, VIEW_SUBTITLES[view], transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
                color=INK_2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [plt.Line2D([], [], color=INK_2, linewidth=1.3),
               plt.Line2D([], [], color=INK_2, linewidth=1.0, linestyle=(0, (2.2, 1.6))),
               plt.Rectangle((0, 0), 1, 1, facecolor="#c9a58a", edgecolor="none"),
               plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=(0.84, 0.51, 0.29), linewidth=0.9,
                             linestyle=(0, (2, 1.5)))]
    fig.legend(handles, ["ball path the agent sees", "ball path hidden from it", "fog over the agent's half",
                         "hidden opponent paddle"], loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02), handlelength=2.2, columnspacing=1.6)
    save(fig, out, "environments")


EVALUATIONS = Path("evaluations")  # set from --evaluations in main()

FIGURES = {"views": fig_views, "returns": fig_returns, "shift": fig_shift, "wipes": fig_wipes,
           "probes": fig_probes, "block": fig_block, "attention": fig_attention,
           "environments": fig_environments}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("analysis"))
    parser.add_argument("--evaluations", type=Path, default=Path("evaluations"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--only", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    args = parser.parse_args()
    global EVALUATIONS
    EVALUATIONS = args.evaluations
    args.out.mkdir(parents=True, exist_ok=True)
    for figure in args.only:
        FIGURES[figure](args.data, args.out)


if __name__ == "__main__":
    main()
