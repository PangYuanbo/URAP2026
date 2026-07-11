from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "tools" / "score_predictionsgt_online_action_bank_motion.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run causal Action Memory scoring over independent sequence shards.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--homography-cache", type=Path, required=True)
    args, passthrough = parser.parse_known_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    shard_root = args.out_jsonl.parent / f"{args.out_jsonl.stem}_shards"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True)
    processes: list[tuple[int, subprocess.Popen[bytes], Path, Path]] = []
    for shard_index in range(args.workers):
        shard_jsonl = shard_root / f"shard_{shard_index:02d}.jsonl"
        shard_summary = shard_root / f"shard_{shard_index:02d}_summary.json"
        command = [
            sys.executable,
            str(WORKER),
            *passthrough,
            "--homography-cache",
            str(shard_root / f"homographies_{shard_index:02d}.pkl"),
            "--out-jsonl",
            str(shard_jsonl),
            "--out-summary",
            str(shard_summary),
            "--sequence-shard-index",
            str(shard_index),
            "--sequence-shard-count",
            str(args.workers),
        ]
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + str(REPO / "tools"), "PYTHONUNBUFFERED": "1"},
        )
        processes.append((shard_index, process, shard_jsonl, shard_summary))
        print(json.dumps({"kind": "action_memory_shard_started", "shard": shard_index, "pid": process.pid}), flush=True)

    completed: set[int] = set()
    while len(completed) < len(processes):
        for shard_index, process, _shard_jsonl, _shard_summary in processes:
            if shard_index in completed:
                continue
            return_code = process.poll()
            if return_code is None:
                continue
            if return_code:
                for other_index, other, _output, _summary in processes:
                    if other_index not in completed and other.poll() is None:
                        other.terminate()
                raise subprocess.CalledProcessError(return_code, process.args)
            completed.add(shard_index)
            print(json.dumps({"kind": "action_memory_shard_done", "shard": shard_index, "done": len(completed), "total": len(processes)}), flush=True)
        if len(completed) < len(processes):
            time.sleep(2.0)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("wb") as target:
        for _shard_index, _process, shard_jsonl, _shard_summary in processes:
            with shard_jsonl.open("rb") as source:
                shutil.copyfileobj(source, target)

    shard_summaries = [json.loads(path.read_text(encoding="utf-8")) for _index, _process, _jsonl, path in processes]
    sequences = sorted(
        [sequence for summary in shard_summaries for sequence in summary["sequences"]],
        key=lambda item: item["sequence"],
    )
    summary = {
        "kind": "online_action_bank_parallel_done",
        "workers": args.workers,
        "out_jsonl": str(args.out_jsonl),
        "frames": sum(int(item["frames"]) for item in shard_summaries),
        "candidates": sum(int(item["candidates"]) for item in shard_summaries),
        "frames_with_active_bank": sum(int(item["frames_with_active_bank"]) for item in shard_summaries),
        "sequences": sequences,
        "shard_summaries": [str(path) for _index, _process, _jsonl, path in processes],
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
