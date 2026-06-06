from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict 20-percent improvement gate from VATD comparison JSON rows.")
    parser.add_argument("--comparison-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--method-name", default="Ego-Adaptive VATD")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--primary-metric", required=True)
    parser.add_argument("--direction", choices=["higher", "lower"], required=True)
    parser.add_argument("--min-relative-improvement", type=float, default=0.20)
    parser.add_argument("--guard", action="append", default=[], help="Guard as metric:direction where direction is higher or lower")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _baseline_value(payload: dict[str, Any], metric: str) -> float | None:
    baseline = payload.get("baseline")
    if isinstance(baseline, dict):
        value = _float_or_none(baseline.get(metric))
        if value is not None:
            return value
    key = f"baseline_{metric}"
    for row in payload.get("rows", []):
        if isinstance(row, dict):
            value = _float_or_none(row.get(key))
            if value is not None:
                return value
    return None


def _relative_improvement(value: float, baseline: float, direction: str) -> float | None:
    denom = abs(baseline)
    if denom <= 1e-12:
        return None
    if direction == "higher":
        return (value - baseline) / denom
    return (baseline - value) / denom


def _guard_pass(row: dict[str, Any], payload: dict[str, Any], guards: list[tuple[str, str]], tolerance: float) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for metric, direction in guards:
        value = _float_or_none(row.get(metric))
        baseline = _baseline_value(payload, metric)
        if value is None or baseline is None:
            failures.append(f"{metric}:missing")
            continue
        if direction == "higher" and value + tolerance < baseline:
            failures.append(f"{metric}:regressed")
        if direction == "lower" and value - tolerance > baseline:
            failures.append(f"{metric}:regressed")
    return not failures, failures


def collect(
    comparison_json: Path,
    primary_metric: str,
    direction: str,
    min_relative_improvement: float,
    guards: list[tuple[str, str]],
    tolerance: float,
    method_name: str,
    baseline_name: str,
) -> dict[str, Any]:
    payload = json.loads(comparison_json.read_text(encoding="utf-8-sig"))
    baseline_primary = _baseline_value(payload, primary_metric)
    rows: list[dict[str, Any]] = []
    wins: list[dict[str, Any]] = []
    for raw in payload.get("rows", []):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        value = _float_or_none(row.get(primary_metric))
        improvement = None if value is None or baseline_primary is None else _relative_improvement(value, baseline_primary, direction)
        guard_ok, guard_failures = _guard_pass(row, payload, guards, tolerance)
        row["target_primary_metric"] = primary_metric
        row["target_direction"] = direction
        row["target_baseline_value"] = baseline_primary
        row["target_value"] = value
        row["target_relative_improvement"] = improvement
        row["target_guard_failures"] = guard_failures
        row["target_pass"] = bool(improvement is not None and improvement + tolerance >= min_relative_improvement and guard_ok)
        if row["target_pass"]:
            wins.append(row)
        rows.append(row)
    best_win = None
    if wins:
        best_win = max(wins, key=lambda row: _float_or_none(row.get("target_relative_improvement")) or float("-inf"))
    status = "pass" if best_win is not None else "insufficient_evidence"
    reason = (
        f"at least one {method_name} row improves {primary_metric} by >= {min_relative_improvement:.3f}"
        if best_win is not None
        else f"no {method_name} row proves >= {min_relative_improvement:.3f} relative improvement on {primary_metric}"
    )
    return {
        "status": status,
        "reason": reason,
        "method": method_name,
        "baseline_method": baseline_name,
        "comparison_json": str(comparison_json),
        "primary_metric": primary_metric,
        "direction": direction,
        "baseline_value": baseline_primary,
        "min_relative_improvement": min_relative_improvement,
        "guards": [{"metric": metric, "direction": guard_direction} for metric, guard_direction in guards],
        "rows": len(rows),
        "wins": len(wins),
        "best_win": best_win,
        "evaluated_rows": rows,
    }


def main() -> None:
    args = parse_args()
    guards: list[tuple[str, str]] = []
    for item in args.guard:
        if ":" not in item:
            raise ValueError(f"guard must be metric:direction, got {item}")
        metric, direction = item.split(":", 1)
        if direction not in {"higher", "lower"}:
            raise ValueError(f"guard direction must be higher or lower, got {direction}")
        guards.append((metric, direction))
    result = collect(
        args.comparison_json,
        args.primary_metric,
        args.direction,
        args.min_relative_improvement,
        guards,
        args.tolerance,
        args.method_name,
        args.baseline_name,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "status": result["status"], "wins": result["wins"], "best_win": result["best_win"]}, indent=2))


if __name__ == "__main__":
    main()
