import json

from tools.collect_ego_adaptive_vatd_20pct_targets import collect


def test_collect_20pct_targets_passes_lower_is_better(tmp_path):
    comparison = tmp_path / "aot_comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "baseline": {"fppi": 0.25, "edr300": 0.90, "hfar": 100.0},
                "rows": [
                    {"run": "weak", "fppi": 0.22, "edr300": 0.90, "hfar": 100.0},
                    {"run": "strong", "fppi": 0.19, "edr300": 0.91, "hfar": 99.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = collect(
        comparison,
        primary_metric="fppi",
        direction="lower",
        min_relative_improvement=0.20,
        guards=[("edr300", "higher"), ("hfar", "lower")],
        tolerance=1e-12,
        method_name="Ego-Adaptive VATD",
        baseline_name="TransVisDrone",
    )

    assert result["status"] == "pass"
    assert result["wins"] == 1
    assert result["best_win"]["run"] == "strong"


def test_collect_20pct_targets_requires_guard_metrics(tmp_path):
    comparison = tmp_path / "nps_comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "baseline": {"map50": 0.50, "recall": 0.80},
                "rows": [
                    {"run": "primary_only", "map50": 0.65, "recall": 0.79},
                    {"run": "guarded", "map50": 0.61, "recall": 0.80},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = collect(
        comparison,
        primary_metric="map50",
        direction="higher",
        min_relative_improvement=0.20,
        guards=[("recall", "higher")],
        tolerance=1e-12,
        method_name="Ego-Adaptive VATD",
        baseline_name="YOLOMG",
    )

    assert result["status"] == "pass"
    assert result["wins"] == 1
    assert result["best_win"]["run"] == "guarded"
    failed = [row for row in result["evaluated_rows"] if row["run"] == "primary_only"][0]
    assert failed["target_guard_failures"] == ["recall:regressed"]
