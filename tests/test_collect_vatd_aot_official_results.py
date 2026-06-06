from __future__ import annotations

from tools.collect_vatd_aot_official_results import build_claim_gate, collect


def _summary(far: float, fppi: float, edr: float) -> dict:
    detected = int(round(edr * 1000))
    return {
        "far": far,
        "fppi": fppi,
        "Detection": {"Encounters": {"300": {"All": {"dr": edr, "detected": detected, "total": 1000}}}},
        "Tracking": {"Encounters": {"300": {"All": {"dr": edr, "detected": detected, "total": 1000}}}},
    }


def test_collect_vatd_aot_official_results_claim_gate_requires_strict_fppi_win(tmp_path):
    baseline = {"hfar": 89.476744, "fppi": 0.262318, "edr300": 0.925714}
    win_path = tmp_path / "route_b_official" / "vatd_win" / "official_eval" / "win.json"
    tie_path = tmp_path / "route_b_official" / "vatd_tie" / "official_eval" / "tie.json"
    loss_path = tmp_path / "route_b_official" / "vatd_loss" / "official_eval" / "loss.json"
    for path, summary in [
        (win_path, _summary(89.476744, 0.256119623, 0.925714)),
        (tie_path, _summary(89.476744, 0.262318, 0.925714)),
        (loss_path, _summary(89.476744, 0.250000000, 0.920000)),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(summary), encoding="utf-8")

    rows = collect([win_path, tie_path, loss_path], baseline)
    by_run = {row["run"]: row for row in rows}
    assert by_run["vatd_win"]["claim_verdict"] == "win"
    assert by_run["vatd_tie"]["claim_verdict"] == "tie"
    assert by_run["vatd_loss"]["claim_verdict"] == "loss"
    assert by_run["vatd_tie"]["beats_baseline"] is True

    gate = build_claim_gate(rows, "TransVisDrone", baseline)
    assert gate["status"] == "pass"
    assert gate["wins"] == 1
    assert gate["ties"] == 1
    assert gate["losses"] == 1
    assert gate["best_win"]["run"] == "vatd_win"


def test_collect_vatd_aot_official_results_claim_gate_rejects_tie_only(tmp_path):
    baseline = {"hfar": 89.476744, "fppi": 0.262318, "edr300": 0.925714}
    tie_path = tmp_path / "route_b_official" / "vatd_tie" / "official_eval" / "tie.json"
    tie_path.parent.mkdir(parents=True, exist_ok=True)
    tie_path.write_text(__import__("json").dumps(_summary(89.476744, 0.262318, 0.925714)), encoding="utf-8")

    rows = collect([tie_path], baseline)
    gate = build_claim_gate(rows, "TransVisDrone", baseline)
    assert gate["status"] == "insufficient_evidence"
    assert gate["wins"] == 0
    assert gate["ties"] == 1
    assert gate["best_win"] is None
