from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect VATD NPS sweep rows and build a strict baseline claim gate.")
    parser.add_argument("--sweep-json", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--baseline-name", default="TransVisDrone")
    parser.add_argument("--method-name", default="VATD")
    parser.add_argument("--primary-metric", default="map50")
    parser.add_argument("--guard-metrics", nargs="*", default=["recall", "map5095"])
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_rows(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    rows = sweep.get("rows")
    if isinstance(rows, list) and rows:
        return [dict(row) for row in rows if isinstance(row, dict)]
    best = sweep.get("best")
    if isinstance(best, dict):
        return [dict(best)]
    return []


def _score_row(
    row: dict[str, Any],
    baseline: dict[str, Any],
    primary_metric: str,
    guard_metrics: list[str],
    tolerance: float,
) -> dict[str, Any]:
    out = dict(row)
    primary = _float_or_none(row.get(primary_metric))
    baseline_primary = _float_or_none(baseline.get(primary_metric))
    out[f"baseline_{primary_metric}"] = baseline_primary
    out[f"delta_{primary_metric}_vs_baseline"] = (
        None if primary is None or baseline_primary is None else primary - baseline_primary
    )
    missing: list[str] = []
    guard_losses: list[str] = []
    for metric in guard_metrics:
        value = _float_or_none(row.get(metric))
        baseline_value = _float_or_none(baseline.get(metric))
        out[f"baseline_{metric}"] = baseline_value
        out[f"delta_{metric}_vs_baseline"] = None if value is None or baseline_value is None else value - baseline_value
        if value is None or baseline_value is None:
            missing.append(metric)
        elif value + tolerance < baseline_value:
            guard_losses.append(metric)
    if primary is None or baseline_primary is None:
        verdict = "unavailable"
        reason = f"missing primary metric {primary_metric}"
    elif primary > baseline_primary + tolerance and not missing and not guard_losses:
        verdict = "win"
        reason = f"{primary_metric} is strictly higher and guard metrics do not regress"
    elif abs(primary - baseline_primary) <= tolerance and not missing and not guard_losses:
        verdict = "tie"
        reason = f"{primary_metric} ties baseline and guard metrics do not regress"
    else:
        verdict = "loss"
        if guard_losses:
            reason = "guard metric regression: " + ",".join(guard_losses)
        elif missing:
            reason = "missing guard metric: " + ",".join(missing)
        else:
            reason = f"{primary_metric} does not strictly beat baseline"
    out["claim_verdict"] = verdict
    out["claim_reason"] = reason
    return out


def collect(
    sweep_json: str | Path,
    baseline_json: str | Path,
    primary_metric: str = "map50",
    guard_metrics: list[str] | None = None,
    tolerance: float = 1e-12,
    baseline_name: str = "TransVisDrone",
    method_name: str = "VATD",
) -> dict[str, Any]:
    guard_metrics = guard_metrics if guard_metrics is not None else ["recall", "map5095"]
    sweep_path = Path(sweep_json)
    baseline_path = Path(baseline_json)
    sweep = json.loads(sweep_path.read_text(encoding="utf-8-sig"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    rows = [
        _score_row(row, baseline, primary_metric, guard_metrics, tolerance)
        for row in _load_rows(sweep)
    ]
    wins = [row for row in rows if row.get("claim_verdict") == "win"]
    ties = [row for row in rows if row.get("claim_verdict") == "tie"]
    losses = [row for row in rows if row.get("claim_verdict") == "loss"]
    unavailable = [row for row in rows if row.get("claim_verdict") == "unavailable"]
    best_win = None
    if wins:
        best_win = max(
            wins,
            key=lambda row: (
                _float_or_none(row.get(f"delta_{primary_metric}_vs_baseline")) or float("-inf"),
                _float_or_none(row.get(primary_metric)) or float("-inf"),
            ),
        )
    status = "pass" if best_win is not None else "insufficient_evidence"
    reason = (
        f"at least one {method_name} row strictly improves {primary_metric} without guard metric regression"
        if best_win is not None
        else f"no {method_name} row strictly improves {primary_metric} without guard metric regression"
    )
    return {
        "sweep_json": str(sweep_path),
        "baseline_json": str(baseline_path),
        "baseline_method": baseline_name,
        "method": method_name,
        "baseline": baseline,
        "primary_metric": primary_metric,
        "guard_metrics": guard_metrics,
        "tolerance": tolerance,
        "claim_gate": {
            "status": status,
            "reason": reason,
            "requires": f"{primary_metric} > baseline and each guard metric >= baseline: {','.join(guard_metrics)}",
        },
        "rows": rows,
        "wins": len(wins),
        "ties": len(ties),
        "losses": len(losses),
        "unavailable": len(unavailable),
        "best_win": best_win,
    }


def write_outputs(result: dict[str, Any], out_csv: str | Path, out_json: str | Path | None = None) -> dict[str, str | None]:
    csv_path = Path(out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "claim_verdict",
        "claim_reason",
        "mode",
        "center",
        "beta",
        "score_gate",
        "missing_score_behavior",
        "precision",
        "recall",
        "baseline_recall",
        "delta_recall_vs_baseline",
        "map50",
        "baseline_map50",
        "delta_map50_vs_baseline",
        "map5095",
        "baseline_map5095",
        "delta_map5095_vs_baseline",
        "f1",
        "detections",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({field: row.get(field) for field in fields})
    json_path = Path(out_json) if out_json is not None else csv_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    result_with_paths = dict(result)
    result_with_paths["comparison_csv"] = str(csv_path)
    result_with_paths["comparison_json"] = str(json_path)
    json_path.write_text(json.dumps(result_with_paths, indent=2), encoding="utf-8")
    gate_path = csv_path.with_name(csv_path.stem + "_claim_gate.json")
    gate = {
        "claim_gate": result["claim_gate"],
        "baseline_method": result["baseline_method"],
        "method": result["method"],
        "primary_metric": result["primary_metric"],
        "guard_metrics": result["guard_metrics"],
        "wins": result["wins"],
        "ties": result["ties"],
        "losses": result["losses"],
        "unavailable": result["unavailable"],
        "best_win": result["best_win"],
        "comparison_csv": str(csv_path),
        "comparison_json": str(json_path),
    }
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return {"comparison_csv": str(csv_path), "comparison_json": str(json_path), "claim_gate_json": str(gate_path)}


def main() -> None:
    args = parse_args()
    result = collect(
        args.sweep_json,
        args.baseline_json,
        primary_metric=args.primary_metric,
        guard_metrics=args.guard_metrics,
        tolerance=args.tolerance,
        baseline_name=args.baseline_name,
        method_name=args.method_name,
    )
    paths = write_outputs(result, args.out_csv, args.out_json)
    print(json.dumps({**paths, "claim_gate": result["claim_gate"], "wins": result["wins"], "rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
