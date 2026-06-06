from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect VATD AOT official summary JSON files into a comparison table.")
    parser.add_argument("--summary-glob", action="append", required=True, help="Glob for official AOT summary JSON files")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--baseline-name", default="TransVisDrone")
    parser.add_argument("--baseline-hfar", type=float, default=89.476744)
    parser.add_argument("--baseline-fppi", type=float, default=0.262318)
    parser.add_argument("--baseline-edr300", type=float, default=0.925714)
    return parser.parse_args()


def _run_name(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts):
        if part == "route_b_official" and index + 1 < len(parts):
            return parts[index + 1]
    return path.parent.name


def _encounter_all(summary: dict[str, Any], mode: str) -> dict[str, Any]:
    try:
        return dict(summary[mode]["Encounters"]["300"]["All"])
    except Exception:
        return {}


def _score_against_baseline(row: dict[str, Any], baseline: dict[str, float]) -> tuple[bool, str, str]:
    edr = row.get("edr300_detection")
    fppi = row.get("fppi")
    hfar = row.get("far")
    if edr is None or fppi is None or hfar is None:
        return False, "missing metric", "unavailable"
    if float(edr) + 1e-12 < baseline["edr300"]:
        return False, "lower EDR@300", "loss"
    if float(hfar) - baseline["hfar"] > 1e-3:
        return False, "higher HFAR", "loss"
    if float(fppi) - baseline["fppi"] > 1e-12:
        return False, "higher FPPI", "loss"
    if float(fppi) < baseline["fppi"] - 1e-12:
        return True, "beats FPPI at equal/better EDR and HFAR", "win"
    return True, "ties baseline", "tie"


def collect(paths: list[Path], baseline: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        summary = json.loads(path.read_text(encoding="utf-8-sig"))
        det_all = _encounter_all(summary, "Detection")
        trk_all = _encounter_all(summary, "Tracking")
        row: dict[str, Any] = {
            "run": _run_name(path),
            "summary_json": str(path),
            "min_det_score": summary.get("min_det_score"),
            "far": summary.get("far"),
            "fppi": summary.get("fppi"),
            "fl_dr_in_range": summary.get("fl_dr_in_range"),
            "fl_dr_above_area": summary.get("fl_dr_above_area"),
            "fl_dr_below_area": summary.get("fl_dr_below_area"),
            "edr300_detection": det_all.get("dr"),
            "edr300_detection_detected": det_all.get("detected"),
            "edr300_detection_total": det_all.get("total"),
            "edr300_tracking": trk_all.get("dr"),
            "edr300_tracking_detected": trk_all.get("detected"),
            "edr300_tracking_total": trk_all.get("total"),
        }
        beats, verdict, claim_verdict = _score_against_baseline(row, baseline)
        row["beats_baseline"] = beats
        row["verdict"] = verdict
        row["claim_verdict"] = claim_verdict
        if row.get("fppi") is not None:
            row["delta_fppi_vs_baseline"] = float(row["fppi"]) - baseline["fppi"]
        if row.get("far") is not None:
            row["delta_hfar_vs_baseline"] = float(row["far"]) - baseline["hfar"]
        if row.get("edr300_detection") is not None:
            row["delta_edr300_detection_vs_baseline"] = float(row["edr300_detection"]) - baseline["edr300"]
        rows.append(row)
    rows.sort(
        key=lambda item: (
            not bool(item.get("beats_baseline")),
            -(float(item.get("edr300_detection") or 0.0)),
            float(item.get("fppi") or 999.0),
            float(item.get("far") or 999999.0),
        )
    )
    return rows


def build_claim_gate(rows: list[dict[str, Any]], baseline_name: str, baseline: dict[str, float]) -> dict[str, Any]:
    wins = [row for row in rows if row.get("claim_verdict") == "win"]
    ties = [row for row in rows if row.get("claim_verdict") == "tie"]
    losses = [row for row in rows if row.get("claim_verdict") == "loss"]
    unavailable = [row for row in rows if row.get("claim_verdict") == "unavailable"]
    best_win = None
    if wins:
        best_win = min(
            wins,
            key=lambda row: (
                float(row.get("fppi") if row.get("fppi") is not None else float("inf")),
                -float(row.get("edr300_detection") if row.get("edr300_detection") is not None else 0.0),
                float(row.get("far") if row.get("far") is not None else float("inf")),
            ),
        )
    status = "pass" if best_win is not None else "insufficient_evidence"
    reason = (
        "at least one VATD row has lower FPPI at equal/better EDR@300 and HFAR"
        if best_win is not None
        else "no VATD row has strictly lower FPPI at equal/better EDR@300 and HFAR"
    )
    return {
        "status": status,
        "reason": reason,
        "requires": "strictly lower FPPI than baseline while keeping EDR@300 >= baseline and HFAR <= baseline",
        "baseline_method": baseline_name,
        "baseline": baseline,
        "rows": len(rows),
        "wins": len(wins),
        "ties": len(ties),
        "losses": len(losses),
        "unavailable": len(unavailable),
        "best_win": best_win,
    }


def main() -> None:
    args = parse_args()
    paths: list[Path] = []
    for pattern in args.summary_glob:
        paths.extend(Path(path) for path in glob.glob(pattern))
    baseline = {
        "hfar": float(args.baseline_hfar),
        "fppi": float(args.baseline_fppi),
        "edr300": float(args.baseline_edr300),
    }
    rows = collect(paths, baseline)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "beats_baseline",
        "verdict",
        "claim_verdict",
        "far",
        "fppi",
        "delta_fppi_vs_baseline",
        "edr300_detection",
        "delta_edr300_detection_vs_baseline",
        "edr300_tracking",
        "edr300_detection_detected",
        "edr300_detection_total",
        "delta_hfar_vs_baseline",
        "fl_dr_in_range",
        "fl_dr_above_area",
        "fl_dr_below_area",
        "min_det_score",
        "summary_json",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    claim_gate = build_claim_gate(rows, args.baseline_name, baseline)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps({"baseline_name": args.baseline_name, "baseline": baseline, "claim_gate": claim_gate, "rows": rows}, indent=2),
            encoding="utf-8",
        )
    claim_gate_json = args.out_csv.with_name(args.out_csv.stem + "_claim_gate.json")
    claim_gate_json.write_text(json.dumps(claim_gate, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "out_csv": str(args.out_csv),
                "out_json": str(args.out_json) if args.out_json else None,
                "claim_gate_json": str(claim_gate_json),
                "claim_gate": claim_gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
