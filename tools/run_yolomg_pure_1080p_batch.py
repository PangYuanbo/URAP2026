import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run full YOLOMG difference videos in a bounded parallel batch.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--first-sequence", type=int, default=5)
    parser.add_argument("--last-sequence", type=int, default=14)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return {}


def sequence_number(path):
    prefix = path.stem.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else None


def build_items(args):
    items = []
    for input_path in sorted(args.input_dir.glob("*.MP4")):
        sequence = sequence_number(input_path)
        if sequence is None or not args.first_sequence <= sequence <= args.last_sequence:
            continue
        output_path = args.output_dir / f"{input_path.stem}_yolomg_compensated_difference_1080p.mp4"
        job_dir = args.run_dir / "jobs" / input_path.stem
        items.append({
            "sequence": sequence,
            "input": str(input_path.resolve()),
            "output": str(output_path.resolve()),
            "manifest": str(output_path.with_suffix(".json").resolve()),
            "job_dir": str(job_dir.resolve()),
            "progress": str((job_dir / "progress.json").resolve()),
            "stdout": str((job_dir / "stdout.log").resolve()),
            "stderr": str((job_dir / "stderr.log").resolve()),
            "pid_file": str((job_dir / "pid.txt").resolve()),
            "state": "pending",
            "pid": None,
            "started_at": None,
            "completed_at": None,
            "return_code": None,
        })
    return items


def item_snapshot(item):
    progress = read_json(Path(item["progress"]))
    output = Path(item["output"])
    private_keys = {"process", "stdout_handle", "stderr_handle"}
    snapshot = {key: value for key, value in item.items() if key not in private_keys}
    snapshot.update({
        "done": progress.get("done", 0),
        "total": progress.get("total", 0),
        "last_output_timestamp": progress.get("last_output_timestamp"),
        "output_exists": output.exists(),
        "output_bytes": output.stat().st_size if output.exists() else 0,
    })
    return snapshot


def write_status(args, items, status, started_at):
    counts = {state: sum(item["state"] == state for item in items) for state in ("pending", "running", "completed", "failed")}
    payload = {
        "status": status,
        "started_at": started_at,
        "updated_at": utc_now(),
        "coordinator_pid": os.getpid(),
        "done": counts["completed"],
        "total": len(items),
        "pending": counts["pending"],
        "running": counts["running"],
        "failed": counts["failed"],
        "max_workers": args.max_workers,
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "run_dir": str(args.run_dir.resolve()),
        "items": [item_snapshot(item) for item in items],
    }
    write_json(args.run_dir / "status.json", payload)
    write_json(args.output_dir / "batch_manifest.json", payload)


def start_item(args, item):
    job_dir = Path(item["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    stdout_handle = Path(item["stdout"]).open("a", encoding="utf-8")
    stderr_handle = Path(item["stderr"]).open("a", encoding="utf-8")
    command = [str(args.python), str(args.worker_script), "--input", item["input"], "--output", item["output"], "--progress-json", item["progress"]]
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(command, cwd=str(args.worker_script.parent.parent), stdout=stdout_handle, stderr=stderr_handle, creationflags=creation_flags)
    item.update({
        "state": "running",
        "pid": process.pid,
        "started_at": utc_now(),
        "process": process,
        "stdout_handle": stdout_handle,
        "stderr_handle": stderr_handle,
    })
    Path(item["pid_file"]).write_text(str(process.pid), encoding="ascii")
    print(f"[START] {Path(item['input']).name} PID={process.pid}", flush=True)


def finish_item(item, return_code):
    item["return_code"] = return_code
    item["completed_at"] = utc_now()
    item["stdout_handle"].close()
    item["stderr_handle"].close()
    output_ok = Path(item["output"]).exists() and Path(item["manifest"]).exists()
    item["state"] = "completed" if return_code == 0 and output_ok else "failed"
    item.pop("process", None)
    item.pop("stdout_handle", None)
    item.pop("stderr_handle", None)
    print(f"[{item['state'].upper()}] {Path(item['input']).name} return_code={return_code}", flush=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")
    items = build_items(args)
    expected = args.last_sequence - args.first_sequence + 1
    if len(items) != expected:
        raise RuntimeError(f"Expected {expected} inputs, found {len(items)} in {args.input_dir}")
    started_at = utc_now()
    write_status(args, items, "running", started_at)
    while any(item["state"] in {"pending", "running"} for item in items):
        for item in items:
            if item["state"] == "running":
                return_code = item["process"].poll()
                if return_code is not None:
                    finish_item(item, return_code)
        available = args.max_workers - sum(item["state"] == "running" for item in items)
        for item in (entry for entry in items if entry["state"] == "pending"):
            if available <= 0:
                break
            start_item(args, item)
            available -= 1
        write_status(args, items, "running", started_at)
        time.sleep(5)
    final_status = "completed" if all(item["state"] == "completed" for item in items) else "failed"
    write_status(args, items, final_status, started_at)
    if final_status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
