from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _check(condition: bool, name: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(condition),
        "details": details or {},
    }


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def audit_run(run_dir: Path, primary_metric: str = "map50") -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    best_val_test_path = run_dir / "best_val_test_watcher" / "best_val_test_result.json"

    summary = _read_json(summary_path)
    best_val_test = _read_json(best_val_test_path)
    comparison_path = (
        Path(str(best_val_test["baseline_comparison_json"]))
        if best_val_test.get("baseline_comparison_json")
        else None
    )
    comparison = _read_json(comparison_path) if comparison_path is not None else {}

    architecture = summary.get("architecture", {}) if isinstance(summary.get("architecture", {}), dict) else {}
    parameter_count = summary.get("parameter_count", {}) if isinstance(summary.get("parameter_count", {}), dict) else {}
    loss_contract = summary.get("loss_contract", {}) if isinstance(summary.get("loss_contract", {}), dict) else {}
    bbox_terms = loss_contract.get("bbox", [])
    if not isinstance(bbox_terms, list):
        bbox_terms = []
    num_queries = _positive_int(summary.get("num_queries"))
    d_model = _positive_int(summary.get("d_model"))
    encoder_layers = _positive_int(summary.get("encoder_layers"))
    decoder_layers = _positive_int(summary.get("decoder_layers"))
    trainable_params = _positive_int(parameter_count.get("trainable"))
    checks = [
        _check(summary_path.exists(), "summary_json_exists", {"path": str(summary_path)}),
        _check(int(summary.get("clip_len", -1)) == 8, "input_is_8_frame_clip", {"clip_len": summary.get("clip_len")}),
        _check(int(summary.get("future_len", -1)) == 4, "future_chunk_is_4_frames", {"future_len": summary.get("future_len")}),
        _check(int(summary.get("output_chunk_len", -1)) == 5, "output_chunk_len_is_current_plus_future", {"output_chunk_len": summary.get("output_chunk_len")}),
        _check(num_queries in {16, 32}, "query_count_is_16_or_32", {"num_queries": summary.get("num_queries")}),
        _check(d_model is not None and d_model <= 256, "d_model_is_mvp_sized", {"d_model": summary.get("d_model")}),
        _check(encoder_layers in {4, 5, 6}, "transformer_depth_is_4_to_6_layers", {"encoder_layers": summary.get("encoder_layers")}),
        _check(decoder_layers is not None and decoder_layers <= 3, "decoder_depth_is_mvp_sized", {"decoder_layers": summary.get("decoder_layers")}),
        _check(str(summary.get("encoder_mode", "")) == "factorized", "uses_temporal_factorized_encoder", {"encoder_mode": summary.get("encoder_mode")}),
        _check(str(architecture.get("backbone", "")) == "small_conv_stem", "uses_small_conv_stem", {"backbone": architecture.get("backbone")}),
        _check(str(architecture.get("output", "")) == "current_bbox_plus_4_future_bbox_chunk", "uses_bbox_chunk_head", {"output": architecture.get("output")}),
        _check(trainable_params is not None, "has_trainable_parameters", {"parameter_count": parameter_count}),
        _check(trainable_params is not None and trainable_params <= 25_000_000, "parameter_count_is_mvp_sized", {"parameter_count": parameter_count, "max_trainable": 25_000_000}),
        _check(str(loss_contract.get("matching", "")) == "detr_hungarian_current_frame", "uses_detr_matching_loss", {"matching": loss_contract.get("matching")}),
        _check({"l1", "giou"}.issubset(set(str(term) for term in bbox_terms)), "uses_l1_and_giou_bbox_loss", {"bbox": bbox_terms}),
        _check(str(loss_contract.get("objectness", "")) == "focal_bce", "uses_objectness_loss", {"objectness": loss_contract.get("objectness")}),
        _check(str(loss_contract.get("future_chunk", "")) == "smooth_l1", "uses_future_chunk_loss", {"future_chunk": loss_contract.get("future_chunk")}),
        _check(best_val_test_path.exists(), "best_val_test_result_exists", {"path": str(best_val_test_path)}),
        _check(bool(best_val_test.get("test_full_split")) is True, "test_is_full_split", {"test_full_split": best_val_test.get("test_full_split"), "test_max_samples": best_val_test.get("test_max_samples")}),
        _check(str(best_val_test.get("test_threshold_source", "")) == "validation", "test_threshold_selected_on_validation", {"test_threshold_source": best_val_test.get("test_threshold_source")}),
        _check(comparison_path.exists() if comparison_path is not None else False, "baseline_comparison_exists", {"path": str(comparison_path) if comparison_path is not None else ""}),
        _check(str(comparison.get("baseline_name", "")) == "TransVisDrone", "baseline_name_is_transvisdrone", {"baseline_name": comparison.get("baseline_name")}),
        _check(bool(comparison.get("require_full_split")) is True, "baseline_comparison_requires_full_split", {"require_full_split": comparison.get("require_full_split")}),
        _check(bool(comparison.get("full_split")) is True, "baseline_comparison_is_full_split", {"full_split": comparison.get("full_split")}),
        _check(int(comparison.get("max_samples", -1)) == 0, "baseline_comparison_uses_all_test_samples", {"max_samples": comparison.get("max_samples")}),
        _check(str(comparison.get("primary_metric", "")) == primary_metric, "baseline_primary_metric_matches", {"expected": primary_metric, "actual": comparison.get("primary_metric")}),
        _check(str(comparison.get("status", "")) == "beat_baseline", "baseline_status_beats_transvisdrone", {"status": comparison.get("status")}),
        _check(bool(comparison.get("primary", {}).get("beat")) is True, "baseline_primary_beat_true", {"primary": comparison.get("primary")}),
    ]
    failed = [check for check in checks if not check["passed"]]
    return {
        "run_dir": str(run_dir.resolve()),
        "summary_json": str(summary_path),
        "best_val_test_result_json": str(best_val_test_path),
        "baseline_comparison_json": str(comparison_path) if comparison_path is not None else "",
        "primary_metric": primary_metric,
        "status": "complete" if not failed else "incomplete",
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether a native-video run satisfies the MVP architecture and TransVisDrone baseline gate.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--primary-metric", default="map50")
    args = parser.parse_args()

    result = audit_run(args.run_dir, primary_metric=args.primary_metric)
    text = json.dumps(result, indent=2)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
