"""Frozen evaluation of a whole sweep: every checkpoint on every view, spread over CPU worker processes.

Run from the project root, e.g. (the defaults are the cnn/lstm protocol in docs/analysis_plan.md):
    uv run python scripts/evaluate_sweep.py
    uv run python scripts/evaluate_sweep.py --policy greedy --roles scratch --views own   # greedy check
    uv run python scripts/evaluate_sweep.py --dry-run                                     # list the jobs

Checkpoints (under --root, default checkpoints/), for each --archs:
    scratch   scratch/<arch>/<view>/seed_<n>/final.pt
    finetune  finetune/<arch>/clean_<label>_to_<view>/seed_<n>/final.pt, label from --parent-checkpoint
    parent    scratch/<arch>/clean/seed_<n>/<--parent-checkpoint>, the fine-tunes' starting point
Only finished runs (final.pt present) are used, and folders set aside by run_all.sh (".aborted_") are skipped.
Before any game is played, every checkpoint is checked against its folder (architecture, training view,
scratch/fine-tune, and a fine-tune's recorded parent hash); any mismatch stops the run.

Each (checkpoint, view) job plays the same --episodes games with seeds --seed, --seed + 1, ... and the
result is appended as one JSON line to --output as soon as it finishes. Rerunning with the same output
skips the jobs already in it (same checkpoint bytes, view and protocol), so an interrupted run resumes.
If a checkpoint changes, its old rows stay in the file: analyses keep the row whose sha256 matches.

Greedy caveat: sticky actions are off, so a game's start depends only on the number of no-op actions at
reset (1-30). Seeds 10000-10029 give only 19 distinct no-op counts, so under --policy greedy several games
are exact repeats (effective n = 19, unevenly weighted). The stochastic policy is unaffected, because the
action generator differs per seed.
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROLES = ("scratch", "finetune", "parent")
# Bump when evaluate.py changes in a way that changes results, so old rows are not reused on resume.
PROTOCOL_VERSION = 1

# Worker processes inherit these; set before torch/OpenCV are imported anywhere in this process tree.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402

from pong_rl.evaluate import VIEWS, evaluate_checkpoint, software_versions, view_name  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parent_label(parent_checkpoint: str) -> str:
    """step_10.0M.pt -> 10.0M, as run_all.sh names fine-tune folders."""
    return parent_checkpoint.removeprefix("step_").removesuffix(".pt")


def discover(root: Path, archs, roles, parent_checkpoint: str) -> list[tuple[str, Path]]:
    """(role, checkpoint path) for every finished run, sorted for a stable job order."""
    found = []
    label = parent_label(parent_checkpoint)
    for arch in archs:
        if "scratch" in roles:
            found += [("scratch", p) for p in root.glob(f"scratch/{arch}/*/seed_*/final.pt")]
        if "finetune" in roles:
            found += [("finetune", p) for p in root.glob(f"finetune/{arch}/clean_{label}_to_*/seed_*/final.pt")]
        if "parent" in roles:
            found += [("parent", p) for p in root.glob(f"scratch/{arch}/clean/seed_*/{parent_checkpoint}")
                      if (p.parent / "final.pt").is_file()]  # only parents whose run finished
    return sorted((role, p) for role, p in found if not any(".aborted_" in part for part in p.parts))


def train_view_from_path(role: str, path: Path) -> str:
    folder = path.parent.parent.name  # scratch: <view>; finetune: <parent>_to_<view>
    return folder.split("_to_")[-1] if role == "finetune" else folder


def expected_parent(root: Path, path: Path, parent_checkpoint: str) -> Path:
    """finetune/<arch>/<...>/seed_<n>/final.pt -> scratch/<arch>/clean/seed_<n>/<parent_checkpoint>."""
    arch, seed_folder = path.relative_to(root).parts[1], path.parent.name
    return root / "scratch" / arch / "clean" / seed_folder / parent_checkpoint


def check_checkpoint(root: Path, role: str, path: Path, parent_checkpoint: str, hashes: dict) -> list[str]:
    """Problems with a checkpoint's metadata compared with the folder it sits in (empty when consistent)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    args = checkpoint.get("args", {})
    arch = path.relative_to(root).parts[1]
    problems = []
    saved_arch = checkpoint.get("arch", args.get("arch", "cnn"))
    if saved_arch != arch:
        problems.append(f"architecture {saved_arch}, folder says {arch}")
    saved_view = view_name(bool(args.get("use_fog")), bool(args.get("mask_enemy")))
    if saved_view != train_view_from_path(role, path):
        problems.append(f"training view {saved_view}, folder says {train_view_from_path(role, path)}")
    kind = checkpoint.get("kind", "scratch")
    if kind != ("finetune" if role == "finetune" else "scratch"):
        problems.append(f"kind {kind} in a {role} folder")
    if role == "finetune":
        parent = expected_parent(root, path, parent_checkpoint)
        recorded = (checkpoint.get("init_from") or {}).get("sha256")
        if not parent.is_file():
            problems.append(f"parent {parent} is missing")
        else:
            hashes.setdefault(str(parent), sha256(parent))
            if recorded != hashes[str(parent)]:
                problems.append(f"recorded parent hash does not match {parent}")
    return problems


def plan_jobs(checkpoints, views, root: Path) -> list[dict]:
    jobs = []
    for role, path in checkpoints:
        own = train_view_from_path(role, path)
        if own not in VIEWS:
            raise ValueError(f"cannot tell the training view of {path}")
        targets = []
        for view in views:
            target = own if view == "own" else view
            if target not in targets:
                targets.append(target)
        for target in targets:
            jobs.append({"role": role, "path": str(path), "run": str(path.parent.relative_to(root)),
                         "checkpoint_file": path.name, "eval_view": target})
    return jobs


def job_key(run: str, checkpoint_file: str, sha: str, view: str, protocol: dict) -> tuple:
    return (run, checkpoint_file, sha, view, protocol.get("version"), protocol.get("episodes"),
            protocol.get("seed"), protocol.get("policy"))


def repair_and_load_done(output: Path) -> set:
    """Keys of the jobs already in output. A last line cut short by an interruption is removed, so the next
    row starts on its own line; that job simply runs again."""
    done = set()
    if not output.exists():
        return done
    data = output.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        with output.open("r+b") as stream:
            stream.truncate(cut)
        print(f"Removed an incomplete last line from {output}", flush=True)
        data = data[:cut]
    for line in data.decode("utf-8").splitlines():
        try:
            row = json.loads(line)
            done.add(job_key(row["run"], row["checkpoint_file"], row["sha256"], row["eval_view"], row["protocol"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # not a result row: ignored
    return done


def run_job(job: dict, protocol: dict):
    start = time.time()
    try:
        result = evaluate_checkpoint(job["path"], episodes=protocol["episodes"], seed=protocol["seed"],
                                     policy=protocol["policy"], device="cpu", view=job["eval_view"], quiet=True)
    except Exception:  # reported by the parent; the other jobs continue
        return None, traceback.format_exc(), time.time() - start
    return result, None, time.time() - start


def init_worker():
    # The parent stops the workers itself on Ctrl-C / SIGTERM; no tracebacks from every worker.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass


def exit_on_signal(signum, _frame):
    raise SystemExit(128 + signum)  # runs the finally blocks, which stop the workers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--archs", nargs="+", default=["cnn", "lstm"])
    parser.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    parser.add_argument("--views", nargs="+", choices=("own", *VIEWS), default=list(VIEWS),
                        help="views to play each checkpoint on; own = its training view")
    parser.add_argument("--parent-checkpoint", default="step_10.0M.pt")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--policy", choices=("stochastic", "greedy"), default="stochastic")
    parser.add_argument("--workers", type=int, default=4, help="CPU processes; 0 runs in this process")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON lines file (default evaluations/sweep_<policy>.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="list the jobs and exit")
    args = parser.parse_args()
    if args.episodes <= 0 or args.seed < 0 or args.workers < 0:
        parser.error("--episodes must be positive, --seed and --workers nonnegative")
    output = args.output or Path("evaluations") / f"sweep_{args.policy}.jsonl"
    protocol = {"version": PROTOCOL_VERSION, "episodes": args.episodes, "seed": args.seed,
                "policy": args.policy, "device": "cpu"}

    checkpoints = discover(args.root, args.archs, args.roles, args.parent_checkpoint)
    if not checkpoints:
        parser.error(f"no finished checkpoints found under {args.root} for {' '.join(args.archs)}")
    hashes = {str(path): sha256(path) for _, path in checkpoints}
    problems = [f"  {path}: {problem}" for role, path in checkpoints
                for problem in check_checkpoint(args.root, role, path, args.parent_checkpoint, hashes)]
    if problems:
        parser.error("checkpoints that do not match their folders:\n" + "\n".join(problems))
    jobs = plan_jobs(checkpoints, args.views, args.root)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"another evaluate_sweep.py is writing {output}")
        done = repair_and_load_done(output)
        stream.seek(0, os.SEEK_END)
        todo = [job for job in jobs
                if job_key(job["run"], job["checkpoint_file"], hashes[job["path"]], job["eval_view"], protocol)
                not in done]
        counts = {role: sum(r == role for r, _ in checkpoints) for role in ROLES}
        print(f"{len(checkpoints)} checkpoints ({', '.join(f'{n} {r}' for r, n in counts.items() if n)}), "
              f"{len(jobs)} jobs, {len(jobs) - len(todo)} already in {output}, {len(todo)} to run "
              f"with {args.workers} workers", flush=True)
        if args.policy == "greedy":
            print("Note: under greedy play, games with the same number of reset no-ops are exact repeats "
                  "(see this script's docstring).", flush=True)
        if args.dry_run:
            for job in todo:
                print(f"  {job['run']}/{job['checkpoint_file']} on {job['eval_view']}")
            return
        if not todo:
            return
        failures = run_all(todo, protocol, hashes, args.workers, stream)
    print(f"Finished: {len(todo) - failures} written to {output}, {failures} failed", flush=True)
    if failures:
        sys.exit(1)


def run_all(todo, protocol, hashes, workers, stream) -> int:
    versions = software_versions()
    failures, written, started = 0, 0, time.time()

    def record(index, job, result, error, seconds):
        nonlocal failures, written
        label = f"[{index}/{len(todo)}] {job['run']}/{job['checkpoint_file']} on {job['eval_view']}"
        train_view = train_view_from_path(job["role"], Path(job["path"]))
        if error is None and result["sha256"] != hashes[job["path"]]:
            error = "checkpoint changed during the evaluation"
        if error is None and (result["eval_view"], result["train_view"]) != (job["eval_view"], train_view):
            error = (f"played {result['eval_view']} / trained on {result['train_view']}, "
                     f"expected {job['eval_view']} / {train_view}")
        if error is not None:
            failures += 1
            print(f"{label}: FAILED {error}", file=sys.stderr, flush=True)
            return
        # The runner's fields come last so the checkpoint's own metadata can never overwrite them;
        # the checkpoint's saved "run" (None for older checkpoints) is kept as checkpoint_run.
        row = {**{k: v for k, v in result.items() if k not in ("run", "eval_view", "train_view")},
               "checkpoint_run": result.get("run"),
               "role": job["role"], "run": job["run"], "checkpoint_file": job["checkpoint_file"],
               "eval_view": job["eval_view"], "train_view": train_view,
               "protocol": protocol, "software_versions": versions,
               "evaluated_at": datetime.now(timezone.utc).isoformat(), "seconds": round(seconds, 1)}
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        written += 1
        summary, win = result["summary"], result["summary"]["completed_win_rate"]
        eta = (time.time() - started) / index * (len(todo) - index)
        print(f"{label}: return {summary['mean_return']:+.1f}, win rate {'n/a' if win is None else f'{win:.0%}'}, "
              f"{seconds / 60:.1f} min (ETA {eta / 3600:.1f} h)", flush=True)

    if workers == 0:
        for index, job in enumerate(todo, 1):
            record(index, job, *run_job(job, protocol))
        return failures

    previous = {sig: signal.signal(sig, exit_on_signal) for sig in (signal.SIGTERM, signal.SIGHUP)}
    executor = concurrent.futures.ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker)
    futures = {}
    try:
        futures = {executor.submit(run_job, job, protocol): job for job in todo}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                result, error, seconds = future.result()
            except concurrent.futures.process.BrokenProcessPool:
                # A worker died (e.g. killed by the OOM killer): every job not yet written fails, so a rerun
                # resumes exactly them.
                lost = len(todo) - written - failures
                print(f"A worker process died; {lost} unfinished jobs are counted as failed. "
                      "Rerun the same command to resume them.", file=sys.stderr, flush=True)
                return failures + lost
            record(index, futures[future], result, error, seconds)
        return failures
    finally:
        for future in futures:
            future.cancel()
        # shutdown() alone waits for running jobs; stop the workers outright instead.
        for process in list(getattr(executor, "_processes", {}).values()):
            process.terminate()
        executor.shutdown(wait=True, cancel_futures=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
