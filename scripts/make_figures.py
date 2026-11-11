"""Figures for the paper, from the analysis outputs (docs/research_report.md).

    uv run python scripts/make_figures.py --data analysis --out figures

--data holds the host's analysis/ folder (enemy_shift.jsonl, attention_block_v2.jsonl, probes_random.jsonl,
interventions/, scripted_baselines_v2_*.jsonl) and host_evaluations/sweep_stochastic.jsonl. Each figure is
written as PDF (for the paper) and PNG (for a quick look); --only picks some of them. Models are the final
checkpoints. "Curriculum" is attn_lstm trained on clean Pong for 10M steps, then fine-tuned on the view.

    views    the four views of one moment of a game, with the hidden regions and the hidden ball marked
    returns  return of every model on the view it was trained on (dots: seeds)
    shift    balls returned when the enemy paddle the model sees is moved (E5c)
    wipes    balls returned when the memory is wiped before, at and inside the fog (E4)
    probes   R^2 of a linear probe for the hidden ball's height: trained vs untrained network (E3, E3b)
    block    the attention agent on fog + hidden paddle: attention blocked on the ball before the fog (E10)
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
    rows = read_jsonl(data / "host_evaluations" / "sweep_stochastic.jsonl")
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
    for row in read_jsonl(data / "attention_block_v2.jsonl"):
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


FIGURES = {"views": fig_views, "returns": fig_returns, "shift": fig_shift, "wipes": fig_wipes,
           "probes": fig_probes, "block": fig_block}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("analysis"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--only", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for figure in args.only:
        FIGURES[figure](args.data, args.out)


if __name__ == "__main__":
    main()
