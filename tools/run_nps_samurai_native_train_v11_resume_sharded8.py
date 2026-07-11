from __future__ import annotations
import json
import os
import pickle
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON = Path(sys.executable)
RUN = REPO / "artifacts" / "detached_nps_samurai_native_train_v11_sharded8"
PROGRESS = RUN / "progress.json"
OUT = Path(r"D:\URAP_vatd_rank_results\nps_samurai_native_train_v9")
PREDICTIONS = Path(r"D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl")
FRAMES = Path(r"U:\URAP_datasets\TransVisDrone\NPS\AllFrames\train")
CACHE = Path(r"U:\URAP_datasets\TransVisDrone\NPS\SAMURAI\native_train_frames")
FPS = REPO / "data_templates" / "nps_sequence_fps.json"
OLD_SHARDS = OUT / "shards_v10"
NEW_SHARDS = OUT / "shards_v11_8"
PRESERVED = NEW_SHARDS / "preserved_complete.jsonl"
SHARDS = 8


def sequence_name(image_id: str) -> str:
    return image_id.rsplit("_", 1)[0]


def report(stage: str, done: int, total: int, **extra) -> None:
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}
    RUN.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def preserve_completed(lengths: dict[str, int]) -> tuple[set[str], int]:
    counts: Counter[str] = Counter()
    old_paths = sorted(OLD_SHARDS.glob("train_scores_*.jsonl"))
    for path in old_paths:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                item = json.loads(line)
                counts[str(item["meta"]["seq"])] += 1
    complete = {sequence for sequence, expected in lengths.items() if counts[sequence] == expected}
    NEW_SHARDS.mkdir(parents=True, exist_ok=True)
    preserved_frames = 0
    with PRESERVED.open("w", encoding="utf-8") as target:
        for path in old_paths:
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if str(item["meta"]["seq"]) in complete:
                        target.write(json.dumps(item, separators=(",", ":")) + "\n")
                        preserved_frames += 1
    return complete, preserved_frames


def main() -> int:
    with PREDICTIONS.open("rb") as handle:
        predictions = pickle.load(handle)
    lengths: dict[str, int] = {}
    for image_id in predictions:
        sequence = sequence_name(str(image_id))
        lengths[sequence] = lengths.get(sequence, 0) + 1
    complete, preserved_frames = preserve_completed(lengths)
    remaining = {sequence: frames for sequence, frames in lengths.items() if sequence not in complete}
    groups: list[list[str]] = [[] for _ in range(SHARDS)]
    totals = [0] * SHARDS
    for sequence, frames in sorted(remaining.items(), key=lambda item: item[1], reverse=True):
        index = min(range(SHARDS), key=lambda shard: totals[shard])
        groups[index].append(sequence)
        totals[index] += frames
    report("preserved_completed_sequences", preserved_frames, sum(lengths.values()), completed_sequences=sorted(complete), completed_sequence_count=len(complete), remaining_sequence_count=len(remaining), shard_totals=totals)
    processes = []
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    for index, sequences in enumerate(groups):
        if not sequences:
            continue
        command = [
            str(PYTHON), str(REPO / "tools" / "score_predictionsgt_samurai_native.py"),
            "--predictionsgt-pkl", str(PREDICTIONS), "--frame-root", str(FRAMES),
            "--output-jsonl", str(NEW_SHARDS / f"train_scores_{index}.jsonl"),
            "--output-summary", str(NEW_SHARDS / f"train_summary_{index}.json"),
            "--frame-cache", str(CACHE), "--progress-json", str(RUN / f"progress_{index}.json"),
            "--sequence-fps-json", str(FPS), "--sequences", *sequences,
            "--start-gate", "0.55", "--reset-gate", "0.70", "--reset-iou", "0.05", "--object-gate", "0.20",
            "--reset-policy", "any", "--reset-patience", "1", "--disagreement-reset-gate", "0.70",
        ]
        stdout = (RUN / "logs" / f"shard_{index}.out.txt").open("w", encoding="utf-8")
        stderr = (RUN / "logs" / f"shard_{index}.err.txt").open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1"}, stdout=stdout, stderr=stderr)
        processes.append((index, process, stdout, stderr, sequences, totals[index]))
    total_frames = sum(lengths.values())
    while any(process.poll() is None for _, process, *_ in processes):
        completed_frames = preserved_frames
        shard_state = []
        for index, process, _stdout, _stderr, sequences, shard_total in processes:
            progress_path = RUN / f"progress_{index}.json"
            payload = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
            shard_done = int(payload.get("done", 0)) if payload.get("stage") == "samurai_native_frames" else shard_total if process.poll() == 0 else 0
            completed_frames += min(shard_total, shard_done)
            shard_state.append({"shard": index, "pid": process.pid, "alive": process.poll() is None, "done": min(shard_total, shard_done), "total": shard_total, "last": payload})
        report("samurai_native_train_shards8", completed_frames, total_frames, preserved_frames=preserved_frames, shards=shard_state)
        time.sleep(30)
    codes = []
    for _index, process, stdout, stderr, *_ in processes:
        codes.append(process.wait())
        stdout.close(); stderr.close()
    if any(codes):
        raise RuntimeError(f"SAMURAI shard failures: {codes}")
    records = []
    with PRESERVED.open("r", encoding="utf-8") as source:
        records.extend(json.loads(line) for line in source if line.strip())
    for index, *_ in processes:
        with (NEW_SHARDS / f"train_scores_{index}.jsonl").open("r", encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())
    records.sort(key=lambda item: (item["meta"]["seq"], item["meta"]["image_id"]))
    if len(records) != total_frames:
        raise RuntimeError(f"merged frame count mismatch: {len(records)} != {total_frames}")
    with (OUT / "train_scores.jsonl").open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, separators=(",", ":")) + "\n")
    summaries = [json.loads((NEW_SHARDS / f"train_summary_{index}.json").read_text(encoding="utf-8")) for index, *_ in processes]
    summary = {"kind": "samurai_native_sharded8_resume_done", "sequences": len(lengths), "frames": total_frames, "preserved_sequences": sorted(complete), "preserved_frames": preserved_frames, "shards": SHARDS, "parts": summaries, "output_jsonl": str(OUT / "train_scores.jsonl")}
    (OUT / "train_score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", total_frames, total_frames, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
