from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine VATD claim-gate JSON files into a paper-claim status summary.")
    parser.add_argument("--gate", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--required", nargs="*", default=None, help="Gate names required for overall pass; defaults to all gates")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    return parser.parse_args()


def _load_gate(path: str | Path) -> dict[str, Any]:
    gate_path = Path(path)
    data = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    gate = data.get("claim_gate") if isinstance(data.get("claim_gate"), dict) else data
    return {"path": str(gate_path), "raw": data, "gate": gate}


def collect(gates: list[tuple[str, str | Path]], required: list[str] | None = None) -> dict[str, Any]:
    if not gates:
        raise ValueError("At least one gate is required")
    required_names = set(required or [name for name, _ in gates])
    items: dict[str, Any] = {}
    for name, path in gates:
        loaded = _load_gate(path)
        status = str(loaded["gate"].get("status", "missing"))
        items[name] = {
            "path": loaded["path"],
            "required": name in required_names,
            "status": status,
            "reason": loaded["gate"].get("reason", ""),
            "requires": loaded["gate"].get("requires", ""),
            "wins": loaded["raw"].get("wins"),
            "ties": loaded["raw"].get("ties"),
            "losses": loaded["raw"].get("losses"),
            "best_win": loaded["raw"].get("best_win"),
        }
    missing_required = sorted(name for name in required_names if name not in items)
    failed_required = sorted(
        name
        for name, item in items.items()
        if item["required"] and item["status"] != "pass"
    )
    overall_status = "pass" if not missing_required and not failed_required else "insufficient_evidence"
    if missing_required:
        reason = "missing required gates: " + ",".join(missing_required)
    elif failed_required:
        reason = "required gates not passing: " + ",".join(failed_required)
    else:
        reason = "all required gates pass"
    return {
        "claim_gate": {
            "status": overall_status,
            "reason": reason,
            "requires": "all required protocol gates must pass",
        },
        "required": sorted(required_names),
        "missing_required": missing_required,
        "failed_required": failed_required,
        "gates": items,
    }


def write_markdown(summary: dict[str, Any], out_md: str | Path) -> None:
    lines = [
        "# VATD Claim Summary",
        "",
        f"- Overall status: {summary['claim_gate']['status']}",
        f"- Reason: {summary['claim_gate']['reason']}",
        "",
        "| gate | required | status | wins | ties | losses | reason |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for name, item in summary["gates"].items():
        lines.append(
            "| {name} | {required} | {status} | {wins} | {ties} | {losses} | {reason} |".format(
                name=name,
                required=str(bool(item.get("required"))),
                status=item.get("status", ""),
                wins="" if item.get("wins") is None else item.get("wins"),
                ties="" if item.get("ties") is None else item.get("ties"),
                losses="" if item.get("losses") is None else item.get("losses"),
                reason=str(item.get("reason", "")).replace("|", "\\|"),
            )
        )
    out_path = Path(out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = collect([(name, path) for name, path in args.gate], args.required)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.out_md is not None:
        write_markdown(summary, args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md) if args.out_md else None, **summary["claim_gate"]}, indent=2))


if __name__ == "__main__":
    main()
