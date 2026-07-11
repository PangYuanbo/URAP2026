from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON = Path(sys.executable)
SOURCE = Path(r"D:\URAP_vatd_rank_results\otb100_samurai_cmc_timebank_v2")
OUTPUT = Path(r"D:\URAP_vatd_rank_results\otb100_samurai_optflow_action_bank_v3")
RUN = REPO / "artifacts" / "detached_otb100_optflow_action_bank_follow_v3"
PROGRESS = RUN / "progress.json"
TOTAL = 100


def report(stage: str, done: int, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": TOTAL, "updated": datetime.now(timezone.utc).astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def counts() -> tuple[int, int]:
    available = len(list((SOURCE / "predictions").glob("*_raw_samurai.txt")))
    processed = len(list((OUTPUT / "sequence_results").glob("*.json")))
    return available, processed


def main() -> int:
    while True:
        available, processed = counts()
        if available > processed:
            command = [
                str(PYTHON), str(REPO / "tools" / "postprocess_otb100_action_bank_cmc.py"),
                "--source-dir", str(SOURCE), "--output-dir", str(OUTPUT), "--progress-json", str(RUN / "worker_progress.json"),
            ]
            process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + str(REPO / "tools"), "PYTHONUNBUFFERED": "1"})
            report("processing_available_sequences", processed, available=available, child_pid=process.pid, command=command)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
            available, processed = counts()
            report("available_sequences_processed", processed, available=available)
        source_summary = SOURCE / "summary.json"
        if source_summary.is_file() and processed >= TOTAL:
            summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
            report("done", processed, summary=summary)
            return 0
        report("waiting_for_samurai_sequences", processed, available=available)
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
