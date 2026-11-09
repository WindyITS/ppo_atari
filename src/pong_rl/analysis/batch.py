"""Running a per-recording analysis over many recordings: parallel, appending each result as it finishes and
skipping recordings already in the output, so an interrupted run (a crash, a power cut) resumes where it left."""

import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys
from typing import Callable


def finished(output: Path, key: str) -> set[str]:
    """Values of `key` in the complete rows of a JSON-lines file (a torn last line is ignored)."""
    if not output.is_file():
        return set()
    done = set()
    for line in output.read_text().splitlines():
        try:
            done.add(json.loads(line)[key])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def run_all(function: Callable[[Path], dict], paths: list[Path], output: Path, key: str, workers: int,
            describe: Callable[[dict], str] = lambda row: "") -> int:
    """Apply function to every path whose str is not yet in output (rows identified by row[key]); returns
    the number of failures."""
    done = finished(output, key)
    todo = [p for p in paths if str(p) not in done]
    print(f"{len(paths) - len(todo)} already done, {len(todo)} to do", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size and not output.read_bytes().endswith(b"\n"):
        with output.open("a") as stream:  # close a torn last line so the next row starts on its own
            stream.write("\n")
    failures = 0
    with output.open("a", encoding="utf-8") as stream, concurrent.futures.ProcessPoolExecutor(
            max(1, workers), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(function, p): p for p in todo}
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            path = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # keep going; the failed recording is retried on the next run
                failures += 1
                print(f"[{count}/{len(todo)}] {path.stem}: FAILED {exc!r}", file=sys.stderr, flush=True)
                continue
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(f"[{count}/{len(todo)}] {path.stem}: {describe(row)}", flush=True)
    return failures
