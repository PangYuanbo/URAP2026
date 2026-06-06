from __future__ import annotations

import json

from tools.collect_vatd_claim_summary import collect, write_markdown


def test_collect_vatd_claim_summary_requires_all_required_gates(tmp_path):
    aot_gate = tmp_path / "aot_gate.json"
    nps_gate = tmp_path / "nps_gate.json"
    aot_gate.write_text(
        json.dumps(
            {
                "claim_gate": {"status": "pass", "reason": "aot pass", "requires": "lower FPPI"},
                "wins": 3,
                "ties": 0,
                "losses": 2,
                "best_win": {"run": "aot_best"},
            }
        ),
        encoding="utf-8",
    )
    nps_gate.write_text(
        json.dumps(
            {
                "claim_gate": {"status": "insufficient_evidence", "reason": "nps not yet", "requires": "map50 win"},
                "wins": 0,
                "ties": 0,
                "losses": 9,
            }
        ),
        encoding="utf-8",
    )

    summary = collect([("aot", aot_gate), ("nps", nps_gate)], required=["aot", "nps"])
    assert summary["claim_gate"]["status"] == "insufficient_evidence"
    assert summary["failed_required"] == ["nps"]
    assert summary["gates"]["aot"]["status"] == "pass"
    assert summary["gates"]["nps"]["status"] == "insufficient_evidence"

    out_md = tmp_path / "summary.md"
    write_markdown(summary, out_md)
    text = out_md.read_text(encoding="utf-8")
    assert "- Overall status: insufficient_evidence" in text
    assert "| aot | True | pass | 3 | 0 | 2 | aot pass |" in text


def test_collect_vatd_claim_summary_passes_when_required_gates_pass(tmp_path):
    aot_gate = tmp_path / "aot_gate.json"
    nps_gate = tmp_path / "nps_gate.json"
    for path, reason in [(aot_gate, "aot pass"), (nps_gate, "nps pass")]:
        path.write_text(
            json.dumps({"claim_gate": {"status": "pass", "reason": reason, "requires": "strict win"}, "wins": 1}),
            encoding="utf-8",
        )

    summary = collect([("aot", aot_gate), ("nps", nps_gate)], required=["aot", "nps"])
    assert summary["claim_gate"]["status"] == "pass"
    assert summary["failed_required"] == []
    assert summary["missing_required"] == []
