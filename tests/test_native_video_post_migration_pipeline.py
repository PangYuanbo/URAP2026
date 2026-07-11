from __future__ import annotations

import subprocess
from pathlib import Path


def test_post_migration_pipeline_prepares_caches_before_training() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "start_native_video_post_migration_pipeline.ps1"
    text = script.read_text(encoding="utf-8")

    cache_check_idx = text.index("cache_check split=")
    train_start_idx = text.index("Starting native video detector training")
    assert cache_check_idx < train_start_idx
    assert "[bool]$UseFrameCache = $true" in text
    assert "[bool]$StartFrameCacheIfMissing = $true" in text
    assert "[int]$NumWorkers = 4" in text
    assert "[int]$PrefetchFactor = 4" in text
    assert "[bool]$PersistentWorkers = $true" in text
    assert "[int]$Nhead = 4" in text
    assert "function Test-FrameCacheReady" in text
    assert "function Test-FrameCacheBuilderRunning" in text
    assert 'Get-ChildItem -LiteralPath $SplitCacheDir -Filter "Clip_*_*.pt" -File' in text
    assert "$CachedFrameCount -ge $FrameCount" in text
    assert "cached_frame_count = $CachedFrameCount" in text
    assert "cached=$($Check.cached_frame_count)" in text
    assert "build_native_video_frame_cache.py" in text
    assert "start_native_video_frame_cache_detached.ps1" in text
    assert "Frame cache preparation is not complete; training was not started." in text
    assert "return" in text[text.index("Frame cache preparation is not complete; training was not started.") : train_start_idx]
    assert "train_cache_${ImageSize}" in text
    assert "val_cache_${ImageSize}" in text
    assert "test_cache_${ImageSize}" in text
    assert "-CacheDir $CacheDir" in text
    assert "-CacheDir $EvalCacheDir" in text
    assert "-CacheDir $FinalTestCacheDir" in text
    assert "-NumWorkers $NumWorkers" in text
    assert "-PrefetchFactor $PrefetchFactor" in text
    assert "-PersistentWorkers $PersistentWorkers" in text
    assert "-Nhead $Nhead" in text
    assert "MVP audit JSON:" in text
    assert "mvp_audit.json" in text


def test_post_migration_pipeline_dry_run_exits_before_side_effects() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "start_native_video_post_migration_pipeline.ps1"
    text = script.read_text(encoding="utf-8")

    dry_run_idx = text.index("if ($DryRun) {")
    readiness_idx = text.index("Checking native video dataset readiness")
    cache_start_idx = text.index("Starting frame cache builder")
    train_start_idx = text.index("Starting native video detector training")
    assert dry_run_idx < readiness_idx < cache_start_idx < train_start_idx
    dry_run_block = text[dry_run_idx:readiness_idx]
    assert "[switch]$DryRun" in text
    assert "DRY RUN: native video post-migration pipeline" in dry_run_block
    assert "No dataset readiness check, cache builder, training, or watcher process was started." in dry_run_block
    assert "Architecture: clip_len=$ClipLen future_len=$FutureLen num_queries=$NumQueries d_model=$DModel nhead=$Nhead" in dry_run_block
    assert "MVP audit JSON:" in dry_run_block
    assert "monitor_native_video_best_val_test_watcher.ps1" in dry_run_block
    assert "return" in dry_run_block


def test_train_detached_defaults_are_gpu_feed_friendly() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "start_native_video_detector_train_detached.ps1"
    text = script.read_text(encoding="utf-8")

    assert "[int]$NumWorkers = 4" in text
    assert "[int]$PrefetchFactor = 4" in text
    assert "[bool]$PersistentWorkers = $true" in text
    assert "[int]$Nhead = 4" in text
    assert '"--nhead", "$Nhead"' in text
    assert "nhead = $Nhead" in text
    assert '"--num-workers", "$NumWorkers"' in text
    assert '"--prefetch-factor", "$PrefetchFactor"' in text
    assert 'if ($PersistentWorkers) {' in text
    assert '$ArgsList += "--persistent-workers"' in text

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
    test_post_migration_pipeline_prepares_caches_before_training()
    test_train_detached_defaults_are_gpu_feed_friendly()
