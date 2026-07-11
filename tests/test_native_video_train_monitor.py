from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def test_train_monitor_reports_throughput_diagnosis() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "monitor_native_video_detector_train.ps1"
    text = script.read_text(encoding="utf-8")

    assert "$ProgressRows = @()" in text
    assert "$ProgressRows += $Obj" in text
    assert "$DataWaitRatio" in text
    assert "data_wait_ratio=" in text
    assert "avg_frames_per_second=" in text
    assert "data_loader_bottleneck" in text
    assert "compute_bound_or_balanced" in text
    assert "$DataWaitRatio -ge 0.35" in text
    assert "throughput diagnosis:" in text
    assert "dense_rank_weight" in text
    assert "dense_rank_positive_mode" in text
    assert "motion_score_mode" in text
    assert "motion_obj_weight" in text
    assert "proposal_mode" in text
    assert "quality_score_mode" in text
    assert "quality_loss_weight" in text
    assert "quality_warmup_steps" in text
    assert "quality_ramp_steps" in text
    assert "quality_loss_weight_effective" in text
    assert "seed:" in text

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


def test_train_monitor_executes_fixture_and_flags_data_bottleneck() -> None:
    repo = Path(__file__).resolve().parents[1]
    run_id = "test_monitor_fixture"
    out_dir = repo / "artifacts" / "native_video_detector" / run_id
    logs_dir = out_dir / "logs"
    stdout_log = logs_dir / "train_stdout.log"
    stderr_log = logs_dir / "train_stderr.log"
    script = repo / "tools" / "monitor_native_video_detector_train.ps1"

    shutil.rmtree(out_dir, ignore_errors=True)
    try:
        logs_dir.mkdir(parents=True)
        (out_dir / "train.pid").write_text("999999\n", encoding="ascii")
        stderr_log.write_text("", encoding="utf-8")
        rows = [
            {
                "kind": "native_video_train_progress",
                "epoch": 1,
                "batch": 1,
                "batches_total": 10,
                "global_step": 1,
                "loss": 2.0,
                "data_ms": 90.0,
                "step_ms": 30.0,
                "frames_per_second": 80.0,
            },
            {
                "kind": "native_video_train_progress",
                "epoch": 1,
                "batch": 2,
                "batches_total": 10,
                "global_step": 2,
                "loss": 1.9,
                "data_ms": 110.0,
                "step_ms": 40.0,
                "frames_per_second": 70.0,
                "quality_loss_weight_effective": 0.25,
            },
        ]
        stdout_log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        meta = {
            "epochs": 3,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "command": "python tools/train_native_video_detector.py",
            "ema": True,
            "ema_decay": 0.999,
            "memory_mode": "samurai",
            "memory_attention": "pooled_cross",
            "memory_slots": 64,
            "query_mode": "dense",
            "patch_stride": 4,
            "dense_obj_source": "conv",
            "dense_positive_radius": 0.012,
            "dense_positive_topk": 9,
            "dense_hard_negative_topk": 512,
            "dense_rank_weight": 1.0,
            "dense_rank_margin": 1.0,
            "dense_rank_negative_topk": 512,
            "dense_rank_positive_mode": "all",
            "motion_score_mode": "samurai",
            "motion_score_weight": 0.5,
            "proposal_mode": "heatmap",
            "quality_score_mode": "iou",
            "motion_obj_weight": 0.25,
            "dense_heatmap_weight": 0.75,
            "dense_heatmap_sigma": 0.015,
            "dense_heatmap_neg_weight": 0.01,
            "dense_heatmap_focal_gamma": 2.0,
            "quality_loss_weight": 0.5,
            "quality_warmup_steps": 5000,
            "quality_ramp_steps": 2000,
            "quality_positive_iou": 0.1,
            "quality_hard_negative_topk": 32,
            "quality_focal_gamma": 1.0,
            "seed": 123,
        }
        (out_dir / "train_meta.json").write_text(json.dumps(meta), encoding="utf-8")

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
        assert "done/total epochs: 0/3" in text
        assert "last completed unit: epoch=1 batch=2/10 global_step=2 loss=1.9" in text
        assert "throughput diagnosis: samples=2" in text
        assert "avg_data_ms=100" in text
        assert "avg_step_ms=35" in text
        assert "data_wait_ratio=0.741" in text
        assert "avg_frames_per_second=75" in text
        assert "diagnosis=data_loader_bottleneck" in text
        assert "dense_loss:" in text
        assert "dense_positive_radius=0.012" in text
        assert "dense_positive_topk=9" in text
        assert "dense_hard_negative_topk=512" in text
        assert "dense_rank_weight=1" in text
        assert "dense_rank_margin=1" in text
        assert "dense_rank_negative_topk=512" in text
        assert "dense_rank_positive_mode=all" in text
        assert "motion_score_mode=samurai" in text
        assert "motion_score_weight=0.5" in text
        assert "proposal_mode=heatmap" in text
        assert "quality_score_mode=iou" in text
        assert "motion_obj_weight=0.25" in text
        assert "dense_heatmap_weight=0.75" in text
        assert "dense_heatmap_sigma=0.015" in text
        assert "dense_heatmap_neg_weight=0.01" in text
        assert "dense_heatmap_focal_gamma=2" in text
        assert "quality_loss_weight=0.5" in text
        assert "quality_warmup_steps=5000" in text
        assert "quality_ramp_steps=2000" in text
        assert "quality_positive_iou=0.1" in text
        assert "quality_hard_negative_topk=32" in text
        assert "quality_focal_gamma=1" in text
        assert "quality_loss_weight_effective=0.25" in text
        assert "seed: 123" in text
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    test_train_monitor_reports_throughput_diagnosis()
    test_train_monitor_executes_fixture_and_flags_data_bottleneck()
