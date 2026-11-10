"""Collect the research outputs into tables (markdown on stdout, JSON for the dashboard).

    uv run python scripts/report_tables.py --out analysis/report.json

Reads, when present: evaluations/sweep_stochastic.jsonl (frozen evaluation), analysis/interventions/*.json (E4),
analysis/view_interventions.jsonl (E5), analysis/probes.jsonl (E3), analysis/rollout_summary.jsonl (E7-E9),
evaluations/controls_stochastic.jsonl (E6). Groups are architecture x training view x model kind; values are
means over seeds with the per-seed values alongside.
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def kind_of(row: dict) -> str:
    run = row.get("run") or ""
    checkpoint = str(row.get("checkpoint") or row.get("recording") or "")
    if checkpoint.endswith(("step_10.0M.pt", "step_10.0M")) and row.get("kind", "scratch") == "scratch":
        return "parent"
    if "__" in run.split("/")[2 if run.count("/") >= 3 else 0]:
        return run.split("/")[2].split("__", 1)[1]
    return row.get("kind", "scratch")


def seed_of(row: dict) -> int | None:
    run = row.get("run") or row.get("checkpoint") or ""
    for part in run.split("/"):
        if part.startswith("seed_"):
            return int(part[5:])
    return None


def bootstrap_diff(a: np.ndarray, b: np.ndarray, rng, n=2000) -> tuple[float, float, float]:
    """Paired difference mean(b - a) over events with a 95% bootstrap interval."""
    d = b.astype(float) - a.astype(float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def interventions(folder: Path, rng) -> list[dict]:
    out = []
    for path in sorted(folder.glob("*.json")):
        r = json.loads(path.read_text())
        events = r["events"]
        returned = {b: np.array([e["outcomes"][b] == "returned" for e in events
                                 if e["outcomes"]["control"] != "undecided" and e["outcomes"][b] != "undecided"])
                    for b in r["totals"]}
        row = {"run": r["run"], "arch": r["arch"], "train_view": r["train_view"], "kind": kind_of(r),
               "seed": seed_of(r), "events": len(events),
               "rate": {b: t["return_rate"] for b, t in r["totals"].items()}}
        control = np.array([e["outcomes"]["control"] == "returned" for e in events])
        for b in ("control_b", "wipe_before", "wipe_entry", "wipe_mid"):
            other = np.array([e["outcomes"][b] == "returned" for e in events])
            row[f"diff_{b}"] = bootstrap_diff(control, other, rng)
        out.append(row)
    return out


def group(rows: list[dict], keys=("arch", "train_view", "kind")) -> dict:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return dict(sorted(groups.items()))


def fmt(v, digits=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, default=Path("analysis"))
    parser.add_argument("--evaluations", type=Path, default=Path("evaluations"))
    parser.add_argument("--out", type=Path, default=Path("analysis/report.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    report = {}

    # E1 / frozen evaluation, own view
    sweep = read_jsonl(args.evaluations / "sweep_stochastic.jsonl")
    own = [r for r in sweep if r.get("checkpoint_file") == "final.pt" and r["eval_view"] == r["train_view"]]
    if own:
        print("\n## Frozen evaluation, own view (finals)\n\n| arch | view | kind | mean return | win rate per seed |\n|---|---|---|---|---|")
        table = []
        for (arch, view, kind), rs in group([{**r, "kind": r["role"]} for r in own]).items():
            rs.sort(key=lambda r: r["training_seed"])
            ret = np.mean([r["summary"]["mean_return"] for r in rs])
            wins = [r["summary"]["completed_win_rate"] for r in rs]
            table.append({"arch": arch, "view": view, "kind": kind, "mean_return": ret, "win_rates": wins})
            print(f"| {arch} | {view} | {kind} | {ret:+.1f} | {' / '.join(f'{w:.0%}' for w in wins)} |")
        report["own_view"] = table

    # E4 memory wipes
    folder = args.analysis / "interventions"
    if folder.is_dir():
        rows = interventions(folder, rng)
        report["interventions"] = rows
        print("\n## E4 memory wipe: return rate of the replayed ball (mean over seeds) and paired change vs control\n")
        print("| arch | view | kind | seeds | control | control_b | wipe before | wipe entry | wipe mid |\n|---|---|---|---|---|---|---|---|---|")
        for (arch, view, kind), rs in group(rows).items():
            cells = []
            for b in ("control", "control_b", "wipe_before", "wipe_entry", "wipe_mid"):
                rates = [r["rate"][b] for r in rs if r["rate"][b] is not None]
                cells.append(f"{np.mean(rates):.2f}" if rates else "–")
            print(f"| {arch} | {view} | {kind} | {len(rs)} | " + " | ".join(cells) + " |")
        print("\nPer seed, change in return rate vs control [95% CI]:\n")
        for r in sorted(rows, key=lambda r: (r["arch"], r["train_view"], r["kind"], r["seed"] or 0)):
            parts = [f"{b.removeprefix('diff_')} {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]"
                     for b, d in r.items() if b.startswith("diff_")]
            print(f"- {r['arch']} {r['train_view']} {r['kind']} s{r['seed']} ({r['events']} events, control "
                  f"{fmt(r['rate']['control'])}): " + "; ".join(parts))

    # E5 view interventions
    views = read_jsonl(args.analysis / "view_interventions.jsonl")
    if views:
        print("\n## E5 enemy-paddle decoy: mean return (seed values)\n\n| arch | view | kind | none | mirror enemy | erase enemy |\n|---|---|---|---|---|---|")
        by = collections.defaultdict(dict)
        for r in views:
            key = (r["arch"], r["train_view"], kind_of(r))
            by[key].setdefault(r["transform"], []).append((seed_of(r), r["mean_return"]))
        table = []
        for key in sorted(by):
            cells = []
            entry = {"arch": key[0], "view": key[1], "kind": key[2]}
            for t in ("none", "mirror_enemy", "erase_enemy"):
                vals = sorted(by[key].get(t, []))
                entry[t] = [v for _, v in vals]
                cells.append(f"{np.mean([v for _, v in vals]):+.1f} ({', '.join(f'{v:+.0f}' for _, v in vals)})" if vals else "–")
            table.append(entry)
            print(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(cells) + " |")
        report["decoy"] = table

    # E3 probes
    probes = read_jsonl(args.analysis / "probes.jsonl")
    if probes:
        print("\n## E3 probes: R^2 for the ball's current height, by steps into the agent's half (0-3 / 4-7 / 8+)\n")
        print("| arch | view | kind | seed | return | hidden | memory | frame-only | own paddle | exit: memory |\n|---|---|---|---|---|---|---|---|---|---|")
        table = []
        for r in sorted(probes, key=lambda r: (r["arch"], r["train_view"], r.get("kind", ""), seed_of(r) or 0)):
            now = r["probes"]["now"]

            def b(m):
                return "/".join(fmt(m.get(k)) for k in ("r2_0_3", "r2_4_7", "r2_8_99")) if m else "–"
            exit_memory = r["probes"]["exit"].get("memory", {})
            print(f"| {r['arch']} | {r['train_view']} | {kind_of(r)} | {seed_of(r)} | {r['mean_return']:+.1f} | "
                  f"{r['hidden_fraction']:.2f} | {b(now.get('memory'))} | {b(now.get('frame'))} | "
                  f"{b(now.get('paddle'))} | {fmt(exit_memory.get('r2'))} |")
            table.append({"arch": r["arch"], "view": r["train_view"], "kind": kind_of(r), "seed": seed_of(r),
                          "mean_return": r["mean_return"], "hidden_fraction": r["hidden_fraction"],
                          "now": now, "exit": r["probes"]["exit"]})
        report["probes"] = table

    # E7-E9 summaries
    summary = read_jsonl(args.analysis / "rollout_summary.jsonl")
    if summary:
        report["summary"] = summary
        print("\n## E7 saliency (KL when erased; all sampled steps / ball in agent's half) and E8 play\n")
        print("| arch | view | kind | seed | return | ball | enemy | own paddle | score | passes/pt | late share | |offset| |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(summary, key=lambda r: (r["arch"], r["train_view"], kind_of(r), seed_of(r) or 0)):
            s = r["saliency"]
            def kl(obj):
                return f"{s['all']['kl'][obj]:.3f}/{s.get('window', {}).get('kl', {}).get(obj, float('nan')):.3f}"
            p = r["passes"]
            print(f"| {r['arch']} | {r['train_view']} | {kind_of(r)} | {seed_of(r)} | {r['mean_return']:+.1f} | "
                  f"{kl('ball')} | {kl('enemy')} | {kl('agent')} | {kl('score')} | "
                  f"{fmt(r['play']['passes_per_point'])} | {fmt(p.get('late_share'))} | {fmt(r['hits']['abs_offset_median'], 1)} |")
        gates = [r for r in summary if "gates" in r]
        if gates:
            print("\n## E9 LSTM forget gates (all steps): mean f, units with mean f > 0.9, median / top-10 half-life (steps)\n")
            for r in sorted(gates, key=lambda r: (r["train_view"], kind_of(r), seed_of(r) or 0)):
                g = r["gates"]["all"]
                print(f"- lstm {r['train_view']} {kind_of(r)} s{seed_of(r)}: f {g['mean_f']:.3f}, "
                      f"{g['units_f_over_0.9']} units > 0.9, half-life {g['median_half_life']:.1f} / {g['top10_half_life']:.1f}")
        attention = [r for r in summary if "attention" in r]
        if attention:
            print("\n## attn_lstm attention mass per object (heads), all steps (uniform = 0.020)\n")
            for r in attention:
                a = r["attention"]["all"]
                print(f"- {r['train_view']} {kind_of(r)} s{seed_of(r)} (return {r['mean_return']:+.1f}): "
                      + ", ".join(f"{k} {v}" for k, v in a.items() if k != "uniform"))

    # E5b decoy tracking
    tracking = read_jsonl(args.analysis / "decoy_tracking.jsonl")
    if tracking:
        print("\n## E5b decoy tracking: correlation of the agent's paddle (when the ball reaches it) with the ball's "
              "height / with the enemy cue as seen; per seed\n")
        print("| arch | view | kind | transform | return | passes | corr paddle-ball | corr paddle-enemy cue | corr ball-cue |"
              "\n|---|---|---|---|---|---|---|---|---|")
        by = collections.defaultdict(list)
        for r in tracking:
            by[(r["arch"], r["train_view"], r["kind"], r["transform"])].append(r)
        table = []
        for key in sorted(by):
            rs = sorted(by[key], key=lambda r: seed_of(r) or 0)

            def cells(k):
                return " / ".join(fmt(r.get(k)) for r in rs)
            print(f"| {' | '.join(key)} | {np.mean([r['mean_return'] for r in rs]):+.1f} | "
                  f"{' / '.join(str(r['passes']) for r in rs)} | {cells('corr_ball')} | {cells('corr_enemy')} | "
                  f"{cells('corr_ball_enemy')} |")
            table.append({"arch": key[0], "view": key[1], "kind": key[2], "transform": key[3],
                          **{k: [r.get(k) for r in rs] for k in ("mean_return", "passes", "corr_ball", "corr_enemy",
                                                                 "corr_ball_enemy", "gap_ball", "gap_enemy")}})
        report["decoy_tracking"] = table

    # E6 training controls: fog_mask return after k M steps on fog_mask, scratch vs each kind of fine-tune
    controls = read_jsonl(args.evaluations / "controls_stochastic.jsonl")
    if controls:
        curves = collections.defaultdict(dict)
        for r in sweep + controls:
            if r["train_view"] != "fog_mask" or r["eval_view"] != "fog_mask" or r["arch"] == "attn_lstm":
                continue
            step = r["checkpoint_step"] if r["checkpoint_step"] is not None else r["training_target_timesteps"]
            run = r["run"].split("/")
            if r["role"] == "scratch":
                label, k = "scratch (15M)", step
            elif r["role"] == "finetune":
                label = run[2].split("__", 1)[1] if "__" in run[2] else "finetune (5M)"
                k = step - 10_000_000
            else:
                continue
            curves[(r["arch"], label)].setdefault(k / 1e6, {})[r["training_seed"]] = r["summary"]["mean_return"]
        ks = sorted({k for c in curves.values() for k in c})
        print("\n## E6 fog_mask return after k M steps on fog_mask (mean over seeds; per seed below)\n")
        print("| arch | run | " + " | ".join(f"{k:g}M" for k in ks) + " |\n|---|---|" + "---|" * len(ks))
        table = []
        for (arch, label), c in sorted(curves.items()):
            cells = []
            for k in ks:
                v = c.get(k)
                cells.append(f"{np.mean(list(v.values())):+.1f}" if v else "")
            print(f"| {arch} | {label} | " + " | ".join(cells) + " |")
            table.append({"arch": arch, "run": label,
                          "points": {f"{k:g}": [c[k].get(s) for s in (1, 2, 3)] for k in ks if k in c}})
        for row in table:
            print(f"- {row['arch']} {row['run']}: " + "; ".join(
                f"{k}M " + "/".join(fmt(v, 1) if v is not None else "–" for v in vals)
                for k, vals in row["points"].items()))
        report["controls"] = table
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, default=float) + "\n")


if __name__ == "__main__":
    main()
