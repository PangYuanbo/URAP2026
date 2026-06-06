from __future__ import annotations

import json

from tools.collect_vatd_nps_sweep_results import collect, write_outputs


def test_collect_vatd_nps_sweep_results_passes_only_strict_primary_win_with_guards(tmp_path):
    baseline = {
        "precision": 0.91,
        "recall": 0.90,
        "map50": 0.93,
        "map5095": 0.46,
        "f1": 0.905,
    }
    sweep = {
        "rows": [
            {"mode": "boost-only", "center": 0.1, "beta": 0.1, "precision": 0.92, "recall": 0.91, "map50": 0.94, "map5095": 0.461, "f1": 0.915},
            {"mode": "boost-only", "center": 0.2, "beta": 0.1, "precision": 0.91, "recall": 0.91, "map50": 0.929, "map5095": 0.461, "f1": 0.91},
        ]
    }
    baseline_path = tmp_path / "baseline.json"
    sweep_path = tmp_path / "sweep.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")

    result = collect(sweep_path, baseline_path)
    assert result["claim_gate"]["status"] == "pass"
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["best_win"]["map50"] == 0.94

    paths = write_outputs(result, tmp_path / "comparison.csv", tmp_path / "comparison.json")
    gate = json.loads((tmp_path / "comparison_claim_gate.json").read_text(encoding="utf-8"))
    assert paths["claim_gate_json"].endswith("comparison_claim_gate.json")
    assert gate["claim_gate"]["status"] == "pass"
    assert gate["best_win"]["claim_verdict"] == "win"


def test_collect_vatd_nps_sweep_results_rejects_recall_gain_when_map_drops(tmp_path):
    baseline = {
        "precision": 0.9161701278,
        "recall": 0.9013069500,
        "map50": 0.9384170538,
        "map5095": 0.4685363007,
        "f1": 0.9086777640,
    }
    sweep = {
        "best": {
            "mode": "boost-only",
            "center": 0.45,
            "beta": 0.005,
            "precision": 0.9153657360,
            "recall": 0.9021619641,
            "map50": 0.9383327601,
            "map5095": 0.4684091646,
            "f1": 0.9087158894,
        }
    }
    baseline_path = tmp_path / "baseline.json"
    sweep_path = tmp_path / "sweep.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")

    result = collect(sweep_path, baseline_path)
    assert result["claim_gate"]["status"] == "insufficient_evidence"
    assert result["wins"] == 0
    assert result["losses"] == 1
    assert result["rows"][0]["claim_verdict"] == "loss"
    assert result["rows"][0]["delta_recall_vs_baseline"] > 0
    assert result["rows"][0]["delta_map50_vs_baseline"] < 0
