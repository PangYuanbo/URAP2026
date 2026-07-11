from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r'C:\Users\aaron\Desktop\URAP')
RUN = ROOT / 'artifacts' / 'detached_ard100_val_selection_pipeline_v2'
PROGRESS = RUN / 'progress.json'
OUT = Path(r'D:\URAP_vatd_rank_results\ard100_yolomg_generalization_v2')


def report(stage: str, done: int, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {'stage': stage, 'done': done, 'total': 3, 'updated': datetime.now().astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload), flush=True)


def execute(stage: str, done: int, script: str) -> None:
    command = [sys.executable, str(ROOT / 'tools' / script)]
    process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(ROOT)})
    report(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise RuntimeError(f'{stage} failed with exit code {code}')


def main() -> int:
    detector_result = OUT / 'yolomg_ard100_val_candidates' / 'results.txt'
    while not detector_result.is_file():
        report('waiting_for_corrected_val_detector', 0, detector_result=str(detector_result))
        time.sleep(20)
    execute('prepare_val_predictions', 1, 'run_ard100_yolomg_val_prepare.py')
    execute('select_on_val', 2, 'run_ard100_val_action_tuning.py')
    execute('evaluate_selected_on_test', 3, 'run_ard100_val_selected_test.py')
    summary = json.loads((OUT / 'action_bank_val_selected_summary.json').read_text(encoding='utf-8'))
    report('done', 3, summary=summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
