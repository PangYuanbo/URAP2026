from __future__ import annotations
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON = Path(sys.executable)
RUN = REPO / "artifacts" / "detached_nps_samurai_native_train_v10_sharded"
PROGRESS = RUN / "progress.json"
OUT = Path(r"D:\URAP_vatd_rank_results\nps_samurai_native_train_v9")
PREDICTIONS = Path(r"D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl")
FRAMES = Path(r"U:\URAP_datasets\TransVisDrone\NPS\AllFrames\train")
CACHE = Path(r"U:\URAP_datasets\TransVisDrone\NPS\SAMURAI\native_train_frames")
FPS = REPO / "data_templates" / "nps_sequence_fps.json"
SHARDS = 4


def sequence_name(image_id: str) -> str:
    return image_id.rsplit("_", 1)[0]


def report(stage: str, done: int, total: int, **extra) -> None:
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}
    RUN.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def main() -> int:
    with PREDICTIONS.open("rb") as handle:
        predictions = pickle.load(handle)
    lengths: dict[str, int] = {}
    for image_id in predictions:
        sequence = sequence_name(str(image_id))
        lengths[sequence] = lengths.get(sequence, 0) + 1
    groups: list[list[str]] = [[] for _ in range(SHARDS)]
    totals = [0] * SHARDS
    for sequence, frames in sorted(lengths.items(), key=lambda item: item[1], reverse=True):
        index = min(range(SHARDS), key=lambda shard: totals[shard])
        groups[index].append(sequence)
        totals[index] += frames
    OUT.mkdir(parents=True, exist_ok=True)
    shard_dir = OUT / "shards_v10"
    shard_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for index, sequences in enumerate(groups):
        command = [
            str(PYTHON), str(REPO / "tools" / "score_predictionsgt_samurai_native.py"),
            "--predictionsgt-pkl", str(PREDICTIONS), "--frame-root", str(FRAMES),
            "--output-jsonl", str(shard_dir / f"train_scores_{index}.jsonl"),
            "--output-summary", str(shard_dir / f"train_summary_{index}.json"),
            "--frame-cache", str(CACHE), "--progress-json", str(RUN / f"progress_{index}.json"),
            "--sequence-fps-json", str(FPS), "--sequences", *sequences,
            "--start-gate", "0.55", "--reset-gate", "0.70", "--reset-iou", "0.05", "--object-gate", "0.20",
            "--reset-policy", "any", "--reset-patience", "1", "--disagreement-reset-gate", "0.70",
        ]
        stdout = (RUN / "logs" / f"shard_{index}.out.txt").open("w", encoding="utf-8")
        stderr = (RUN / "logs" / f"shard_{index}.err.txt").open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1"}, stdout=stdout, stderr=stderr)
        processes.append((process, stdout, stderr, sequences, totals[index]))
    total_frames = sum(totals)
    while any(process.poll() is None for process, *_ in processes):
        completed_frames = 0
        shard_state = []
        for index, (process, *_rest) in enumerate(processes):
            progress_path = RUN / f"progress_{index}.json"
            payload = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
            shard_done = int(payload.get("done", 0)) if payload.get("stage") == "samurai_native_frames" else sum(lengths[name] for name in groups[index]) if process.poll() == 0 else 0
            completed_frames += min(totals[index], shard_done)
            shard_state.append({"shard": index, "pid": process.pid, "alive": process.poll() is None, "done": min(totals[index], shard_done), "total": totals[index], "last": payload})
        report("samurai_native_train_shards", completed_frames, total_frames, shards=shard_state)
        time.sleep(30)
    codes = []
    for process, stdout, stderr, *_ in processes:
        codes.append(process.wait())
        stdout.close(); stderr.close()
    if any(codes):
        raise RuntimeError(f"SAMURAI shard failures: {codes}")
    records = []
    for index in range(SHARDS):
        with (shard_dir / f"train_scores_{index}.jsonl").open("r", encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())
    records.sort(key=lambda item: (item["meta"]["seq"], item["meta"]["image_id"]))
    with (OUT / "train_scores.jsonl").open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, separators=(",", ":")) + "\n")
    summaries = [json.loads((shard_dir / f"train_summary_{index}.json").read_text(encoding="utf-8")) for index in range(SHARDS)]
    summary = {"kind": "samurai_native_sharded_train_done", "shards": SHARDS, "frames": total_frames, "groups": groups, "shard_frames": totals, "summaries": summaries, "output_jsonl": str(OUT / "train_scores.jsonl")}
    (OUT / "train_score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", total_frames, total_frames, summary=str(OUT / "train_score_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
