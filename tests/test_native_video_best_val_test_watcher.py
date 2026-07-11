from __future__ import annotations

import subprocess
from pathlib import Path


def test_best_val_test_watcher_uses_validation_selected_thresholds() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "start_native_video_best_val_test_watcher_detached.ps1"
    text = script.read_text(encoding="utf-8")

    assert "`$selectedValEvalPath = [string]`$best.best_eval_json" in text
    assert "`$selectedScoreThreshold = [double]`$selectedValEval.score_threshold" in text
    assert "`$selectedTopK = [int]`$selectedValEval.top_k" in text
    assert "-ScoreThreshold `$selectedScoreThreshold" in text
    assert "-TopK `$selectedTopK" in text
    assert "-SweepScoreThresholds `$selectedScoreThreshold" in text
    assert "-SweepTopKs `$selectedTopK" in text
    assert "-RequireFullSplitBaseline 1" in text
    assert 'test_threshold_source = "validation"' in text
    assert "function Wait-StableFile" in text
    assert "timeout waiting for stable file" in text
    assert "weights_bytes = `$stableWeights.Length" in text
    assert 'throw "best eval json missing after test eval: `$bestEvalPath"' in text
    assert 'throw "baseline comparison json missing after test eval: `$comparePath"' in text
    assert "`$bestEval = Get-Content -LiteralPath `$bestEvalPath -Raw | ConvertFrom-Json" in text
    assert "`$comparison = Get-Content -LiteralPath `$comparePath -Raw | ConvertFrom-Json" in text
    assert "test_map50 = [double]`$bestEval.map50" in text
    assert "test_recall = [double]`$bestEval.recall" in text
    assert "baseline_status = [string]`$comparison.status" in text
    assert "baseline_primary_metric = [string]`$comparison.primary_metric" in text
    assert "baseline_primary_delta = [double]`$comparison.primary.delta" in text
    assert "baseline_primary_beat = [bool]`$comparison.primary.beat" in text
    assert "test_max_samples = $MaxSamples" in text
    assert "test_full_split = [bool]($MaxSamples -le 0)" in text
    assert '$AuditJson = Join-Path $RunDir "mvp_audit.json"' in text
    assert "audit_native_video_mvp_run.py" in text
    assert '--run-dir "$RunDir"' in text
    assert '--out-json "$AuditJson"' in text
    assert '--primary-metric "$PrimaryMetric"' in text
    assert 'throw "MVP audit failed with exit code `$auditExitCode. See: $AuditJson"' in text
    assert "mvp_audit_json = \"$AuditJson\"" in text
    assert "audit_json = $AuditJson" in text

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


def test_continuous_val_watcher_retries_failed_eval_and_tie_breaks() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "start_native_video_continuous_val_watcher_detached.ps1"
    text = script.read_text(encoding="utf-8")

    eval_call_idx = text.index("eval_native_video_checkpoint.ps1")
    seen_write_idx = text.index("`$seen[`$ckpt.FullName] = `$true")
    assert eval_call_idx < seen_write_idx
    assert 'throw "best eval json missing after eval: `$bestEvalPath"' in text
    assert "`$recall = [double]`$eval.recall" in text
    assert "`$precision = [double]`$eval.precision" in text
    assert "`$recall -gt [double]`$best.recall" in text
    assert "`$precision -gt [double]`$best.precision" in text
    assert "recall = `$recall" in text
    assert "precision = `$precision" in text
    assert "function Write-JsonAtomic" in text
    assert 'if (Test-Path "$SeenFile")' in text
    assert 'if (Test-Path "$BestJson")' in text
    assert '"restored_seen`":`$(`$seen.Count)' in text
    assert '"restored_best`":`$(`$null -ne `$best)' in text
    assert 'Write-JsonAtomic -Value `$best -Path "$BestJson" -Depth 5' in text
    assert 'Write-JsonAtomic -Value (`$seen.Keys | Sort-Object) -Path "$SeenFile" -Depth 3' in text

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
    test_best_val_test_watcher_uses_validation_selected_thresholds()
    test_continuous_val_watcher_retries_failed_eval_and_tie_breaks()
