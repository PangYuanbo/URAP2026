from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def test_best_val_test_monitor_reports_baseline_gate() -> None:
    repo = Path(__file__).resolve().parents[1]
    run_id = "test_best_val_monitor_fixture"
    run_dir = repo / "artifacts" / "native_video_detector" / run_id
    output_root = run_dir / "best_val_test_watcher"
    logs_dir = output_root / "logs"
    script = repo / "tools" / "monitor_native_video_best_val_test_watcher.ps1"

    shutil.rmtree(run_dir, ignore_errors=True)
    try:
        logs_dir.mkdir(parents=True)
        summary_file = run_dir / "summary.json"
        best_json = run_dir / "continuous_val_watcher" / "best_val_checkpoint.json"
        result_json = output_root / "best_val_test_result.json"
        audit_json = run_dir / "mvp_audit.json"
        stdout_log = logs_dir / "watcher.out.log"
        stderr_log = logs_dir / "watcher.err.log"
        best_json.parent.mkdir(parents=True)

        summary_file.write_text("{}", encoding="utf-8")
        best_json.write_text(json.dumps({"weights": "dummy.pt"}), encoding="utf-8")
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        result_json.write_text(
            json.dumps(
                {
                    "baseline_status": "beat_baseline",
                    "baseline_primary_metric": "map50",
                    "baseline_primary_method": 0.95,
                    "baseline_primary_value": 0.9384170538,
                    "baseline_primary_delta": 0.0115829462,
                    "baseline_primary_beat": True,
                    "test_full_split": True,
                    "test_max_samples": 0,
                    "test_map50": 0.95,
                    "test_recall": 0.92,
                    "test_precision": 0.91,
                }
            ),
            encoding="utf-8",
        )
        audit_json.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "passed": True,
                    "primary_metric": "map50",
                    "failed_checks": [],
                }
            ),
            encoding="utf-8",
        )
        (output_root / "native_video_best_val_test_watcher.pid").write_text("999999\n", encoding="ascii")
        (output_root / "native_video_best_val_test_watcher.meta.json").write_text(
            json.dumps(
                {
                    "summary_file": str(summary_file),
                    "best_json": str(best_json),
                    "result_json": str(result_json),
                    "audit_json": str(audit_json),
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                    "runner_file": str(output_root / "native_video_best_val_test_watcher.runner.ps1"),
                }
            ),
            encoding="utf-8",
        )

        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-RunId",
                run_id,
                "-Tail",
                "0",
            ],
            cwd=repo,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        text = proc.stdout + proc.stderr
        assert proc.returncode == 0, text
        assert "Status: NOT RUNNING" in text
        assert "done/total: 5/5" in text
        assert "last completed unit: mvp_audit_done" in text
        assert f"audit_json: {audit_json}" in text
        assert "baseline gate: status=beat_baseline" in text
        assert "mvp audit: status=complete" in text
        assert "passed=True" in text
        assert "failed_checks=" in text
        assert "primary_metric=map50" in text
        assert "method=0.95" in text
        assert "baseline=0.9384170538" in text
        assert "delta=0.0115829462" in text
        assert "beat=True" in text
        assert "test_full_split=True" in text
        assert "test_max_samples=0" in text
        assert "test_map50=0.95" in text
        assert "test_recall=0.92" in text
        assert "test_precision=0.91" in text
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_best_val_test_monitor_script_parses() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "monitor_native_video_best_val_test_watcher.ps1"
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$tokens=$null; $errors=$null; "
                "[System.Management.Automation.Language.Parser]::ParseFile("
                f"'{script}', [ref]$tokens, [ref]$errors) > $null; "
                "if ($errors.Count) { $errors | Format-List *; exit 1 }"
            ),
        ],
        cwd=repo,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


if __name__ == "__main__":
    test_best_val_test_monitor_reports_baseline_gate()
    test_best_val_test_monitor_script_parses()
