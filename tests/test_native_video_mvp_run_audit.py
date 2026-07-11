from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _write_run(run_dir: Path, *, full_split: bool = True, beat: bool = True, clip_len: int = 8) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "clip_len": clip_len,
                "future_len": 4,
                "output_chunk_len": 5,
                "num_queries": 32,
                "d_model": 192,
                "encoder_layers": 4,
                "decoder_layers": 2,
                "encoder_mode": "factorized",
                "parameter_count": {"trainable": 1234, "total": 1234},
                "architecture": {
                    "backbone": "small_conv_stem",
                    "output": "current_bbox_plus_4_future_bbox_chunk",
                },
                "loss_contract": {
                    "matching": "detr_hungarian_current_frame",
                    "bbox": ["l1", "giou"],
                    "objectness": "focal_bce",
                    "future_chunk": "smooth_l1",
                },
            }
        ),
        encoding="utf-8",
    )
    eval_dir = run_dir / "test_best_val_native_video_detector"
    eval_dir.mkdir()
    comparison = {
        "baseline_name": "TransVisDrone",
        "primary_metric": "map50",
        "require_full_split": True,
        "full_split": full_split,
        "max_samples": 0 if full_split else 100,
        "status": "beat_baseline" if beat else "below_baseline",
        "primary": {
            "method": 0.95 if beat else 0.1,
            "baseline": 0.9384170538,
            "delta": 0.011 if beat else -0.8,
            "beat": beat,
        },
    }
    comparison_path = eval_dir / "baseline_comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    watcher_dir = run_dir / "best_val_test_watcher"
    watcher_dir.mkdir()
    (watcher_dir / "best_val_test_result.json").write_text(
        json.dumps(
            {
                "test_full_split": full_split,
                "test_max_samples": 0 if full_split else 100,
                "test_threshold_source": "validation",
                "baseline_comparison_json": str(comparison_path),
            }
        ),
        encoding="utf-8",
    )


def _run_audit(repo: Path, run_dir: Path) -> tuple[int, dict]:
    out_json = run_dir / "audit.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "audit_native_video_mvp_run.py"),
            "--run-dir",
            str(run_dir),
            "--out-json",
            str(out_json),
        ],
        cwd=repo,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert out_json.exists()
    return proc.returncode, json.loads(out_json.read_text(encoding="utf-8"))


def test_native_video_mvp_run_audit_passes_complete_run() -> None:
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="native_video_audit_") as tmp:
        run_dir = Path(tmp) / "run"
        _write_run(run_dir)
        code, result = _run_audit(repo, run_dir)
        assert code == 0
        assert result["status"] == "complete"
        assert result["passed"] is True


def test_native_video_mvp_run_audit_rejects_subset_or_wrong_architecture() -> None:
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="native_video_audit_") as tmp:
        subset_run = Path(tmp) / "subset"
        _write_run(subset_run, full_split=False)
        subset_code, subset = _run_audit(repo, subset_run)
        assert subset_code == 1
        assert subset["status"] == "incomplete"
        assert any(check["name"] == "test_is_full_split" for check in subset["failed_checks"])

        wrong_arch_run = Path(tmp) / "wrong_arch"
        _write_run(wrong_arch_run, clip_len=4)
        wrong_code, wrong = _run_audit(repo, wrong_arch_run)
        assert wrong_code == 1
        assert any(check["name"] == "input_is_8_frame_clip" for check in wrong["failed_checks"])

        wrong_loss_run = Path(tmp) / "wrong_loss"
        _write_run(wrong_loss_run)
        summary = json.loads((wrong_loss_run / "summary.json").read_text(encoding="utf-8"))
        summary["loss_contract"]["future_chunk"] = "none"
        (wrong_loss_run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        wrong_loss_code, wrong_loss = _run_audit(repo, wrong_loss_run)
        assert wrong_loss_code == 1
        assert any(check["name"] == "uses_future_chunk_loss" for check in wrong_loss["failed_checks"])

        oversized_run = Path(tmp) / "oversized"
        _write_run(oversized_run)
        oversized_summary = json.loads((oversized_run / "summary.json").read_text(encoding="utf-8"))
        oversized_summary["d_model"] = 512
        oversized_summary["parameter_count"]["trainable"] = 50_000_000
        (oversized_run / "summary.json").write_text(json.dumps(oversized_summary), encoding="utf-8")
        oversized_code, oversized = _run_audit(repo, oversized_run)
        assert oversized_code == 1
        failed_names = {check["name"] for check in oversized["failed_checks"]}
        assert "d_model_is_mvp_sized" in failed_names
        assert "parameter_count_is_mvp_sized" in failed_names

        loose_compare_run = Path(tmp) / "loose_compare"
        _write_run(loose_compare_run)
        result_path = loose_compare_run / "best_val_test_watcher" / "best_val_test_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        comparison_path = Path(result["baseline_comparison_json"])
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        comparison["require_full_split"] = False
        comparison["full_split"] = None
        comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
        loose_code, loose = _run_audit(repo, loose_compare_run)
        assert loose_code == 1
        loose_failed = {check["name"] for check in loose["failed_checks"]}
        assert "baseline_comparison_requires_full_split" in loose_failed
        assert "baseline_comparison_is_full_split" in loose_failed


if __name__ == "__main__":
    test_native_video_mvp_run_audit_passes_complete_run()
    test_native_video_mvp_run_audit_rejects_subset_or_wrong_architecture()
