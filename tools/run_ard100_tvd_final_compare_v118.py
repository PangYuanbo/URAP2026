from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
RUN = ROOT / "artifacts" / "detached_ard100_tvd_final_compare_v118"
OUT = Path(r"D:\URAP_vatd_rank_results\ard100_tvd_dense_final_v118.json")
ARD = Path(r"D:\URAP_vatd_rank_results\ard100_yolomg_generalization_v2\official_adapted_comparison.json")
V53 = Path(r"D:\URAP_vatd_rank_results\action_chunk_temporal_gate_v53\official_summary.json")
VATD_MAP50 = 0.93844
ROUTES = [
    {
        "method": "V116 dense Action Bank",
        "result": Path(r"D:\URAP_vatd_rank_results\tvd_dense_action_model_v116\official_summary.json"),
        "run": ROOT / "artifacts" / "detached_tvd_dense_action_model_v116",
        "metrics_key": "test",
    },
    {
        "method": "V117 dense Action Bank + temporal expert",
        "result": Path(r"D:\URAP_vatd_rank_results\tvd_dense_temporal_gate_v117\official_summary.json"),
        "run": ROOT / "artifacts" / "detached_tvd_dense_temporal_gate_v117",
        "metrics_key": "test_fixed",
    },
    {
        "method": "V119 domain-balanced dense Action Bank",
        "result": Path(r"D:\URAP_vatd_rank_results\tvd_domain_balanced_action_v119\official_summary.json"),
        "run": ROOT / "artifacts" / "detached_tvd_domain_balanced_action_v119",
        "metrics_key": "test_fixed",
    },
    {
        "method": "V120 OOF stacked Action Bank",
        "result": Path(r"D:\URAP_vatd_rank_results\tvd_oof_stack_v120\official_summary.json"),
        "run": ROOT / "artifacts" / "detached_tvd_oof_stack_v120",
        "metrics_key": "test_fixed",
    },
]


def report(stage: str, done: int, total: int = 2, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now().astimezone().isoformat(), **extra}
    (RUN / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def process_alive(pid: int) -> bool:
    command = f"if(Get-Process -Id {pid} -ErrorAction SilentlyContinue){{exit 0}}else{{exit 1}}"
    return subprocess.run(["powershell", "-NoProfile", "-Command", command], check=False).returncode == 0


def wait_route(route: dict[str, Any]) -> dict[str, Any]:
    result_path: Path = route["result"]
    run_dir: Path = route["run"]
    progress_path = run_dir / "progress.json"
    pid_path = run_dir / "pid.txt"
    while True:
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            return {"status": "completed", "payload": payload}
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None
        if progress and progress.get("stage") in {"stopped_by_gate", "failed"}:
            return {"status": "failed", "progress": progress}
        if pid_path.exists():
            worker_pid = int(pid_path.read_text().strip())
            if not process_alive(worker_pid):
                return {"status": "failed", "progress": progress, "reason": "process_not_running", "pid": worker_pid}
        time.sleep(30)


def tvd_row(name: str, metrics: dict[str, object]) -> dict[str, object]:
    map50 = float(metrics["map50"])
    gain = 100.0 * (map50 - VATD_MAP50)
    return {"method": name, "status": "completed", "map50": map50, "gain_over_vatd_points": gain, "target_3_to_5_met": 3.0 <= gain <= 5.0, "metrics": metrics}


def main() -> int:
    report("wait_results", 0)
    route_results = [(route, wait_route(route)) for route in ROUTES]
    ard = json.loads(ARD.read_text(encoding="utf-8"))
    v53 = json.loads(V53.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = [
        tvd_row("VATD baseline", {"map50": VATD_MAP50}),
        tvd_row("V53 Action Bank", v53["test_fixed"]),
    ]
    failures: list[dict[str, object]] = []
    for route, result in route_results:
        if result["status"] == "completed":
            metrics = result["payload"][route["metrics_key"]]
            rows.append(tvd_row(route["method"], metrics))
        else:
            failure = {"method": route["method"], **result}
            failures.append(failure)
            rows.append({"method": route["method"], "status": "failed", "detail": result})
    completed_rows = [row for row in rows[1:] if row.get("status") == "completed"]
    best = max(completed_rows, key=lambda row: float(row["map50"]))
    payload = {
        "ard100": ard,
        "tvd": {
            "vatd_map50": VATD_MAP50,
            "rows": rows,
            "failures": failures,
            "best": best,
            "target_3_to_5_met": bool(best["target_3_to_5_met"]),
        },
        "generalization": {
            "ard100_target_met": bool(ard.get("target_3_to_5_met")),
            "tvd_target_met": bool(best["target_3_to_5_met"]),
            "same_3_to_5_point_effect_generalized": bool(ard.get("target_3_to_5_met")) and bool(best["target_3_to_5_met"]),
            "failed_routes": [failure["method"] for failure in failures],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report("done", 2, output=str(OUT), summary=payload["generalization"], best_tvd=best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
