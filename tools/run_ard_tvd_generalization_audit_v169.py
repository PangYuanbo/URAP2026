from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
RUN = ROOT / "artifacts/detached_ard_tvd_generalization_audit_v169"
OUTPUT = Path(r"D:\URAP_vatd_rank_results\ard_tvd_generalization_audit_v169")
ARD = Path(r"D:\URAP_vatd_rank_results\ard100_yolomg_generalization_v2\official_comparison.json")
TVD_RUNS = {
    "V162 Action Bank": (
        ROOT / "artifacts/detached_tvd_track_supported_budget_v162",
        Path(r"D:\URAP_vatd_rank_results\tvd_track_supported_budget_v162\official_summary.json"),
    ),
    "V165 detector": (
        ROOT / "artifacts/detached_tvd_detector_hard_replay_v165_posteval",
        Path(r"D:\URAP_vatd_rank_results\tvd_detector_hard_replay_v165_posteval\official_summary.json"),
    ),
    "V166 detector + Action Bank": (
        ROOT / "artifacts/detached_tvd_v165_action_transfer_v166",
        Path(r"D:\URAP_vatd_rank_results\tvd_v165_action_transfer_v166\official_summary.json"),
    ),
    "V167 official + V165 detector fusion": (
        ROOT / "artifacts/detached_tvd_v165_detector_fusion_v167",
        Path(r"D:\URAP_vatd_rank_results\tvd_v165_detector_fusion_v167\official_summary.json"),
    ),
    "V168 detector fusion + Action Bank": (
        ROOT / "artifacts/detached_tvd_v167_action_transfer_v168",
        Path(r"D:\URAP_vatd_rank_results\tvd_v167_action_transfer_v168\official_summary.json"),
    ),
}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def report(stage: str, done: int, total: int = 2, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": now(), **extra}
    (RUN / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def process_command(pid: int) -> str | None:
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' -ErrorAction SilentlyContinue; if($p){{$p.CommandLine}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def wait_for_summary(name: str, run_dir: Path, summary_path: Path) -> dict[str, object]:
    while not summary_path.is_file():
        pid_file = run_dir / "pid.txt"
        if not pid_file.is_file():
            raise RuntimeError(f"{name}: missing PID and summary")
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        if process_command(pid) is None:
            progress_path = run_dir / "progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else None
            raise RuntimeError(f"{name}: NOT RUNNING before summary, progress={progress}")
        report("await_tvd_results", 0, waiting_for=name, pid=pid)
        time.sleep(30)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def validation_map50(name: str, summary: dict[str, object]) -> float:
    selection = summary["validation_selection"]
    if name == "V165 detector":
        mode = str(selection["selected_mode"])
        return float(selection[mode]["map50"])
    return float(selection["map50"])


def main() -> int:
    if not ARD.is_file():
        raise FileNotFoundError(ARD)
    ard = json.loads(ARD.read_text(encoding="utf-8"))
    tvd_results: list[dict[str, object]] = []
    for name, (run_dir, summary_path) in TVD_RUNS.items():
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else wait_for_summary(name, run_dir, summary_path)
        tvd_results.append(
            {
                "name": name,
                "validation_map50": validation_map50(name, summary),
                "test_fixed": summary["test_fixed"],
                "gain_over_vatd_points": float(summary["gain_over_vatd_points"]),
                "summary_path": str(summary_path),
            }
        )
    report("select_by_validation", 1, candidates=len(tvd_results))
    selected = max(tvd_results, key=lambda row: float(row["validation_map50"]))
    ard_gain = float(ard["action_over_vatd_points"])
    tvd_gain = float(selected["gain_over_vatd_points"])
    summary = {
        "selection_protocol": "TVD method family selected only by validation mAP@0.5; each test result was produced once with its validation-selected configuration",
        "ard100": {
            "dataset_images": int(ard["action_bank"]["images"]),
            "vatd_map50": float(ard["vatd"]["map50"]),
            "action_bank_map50": float(ard["action_bank"]["map50"]),
            "gain_over_vatd_points": ard_gain,
            "target_at_least_3_met": ard_gain >= 3.0,
            "target_3_to_5_met": 3.0 <= ard_gain <= 5.0,
            "source": str(ARD),
        },
        "tvd_candidates": sorted(tvd_results, key=lambda row: -float(row["validation_map50"])),
        "tvd_selected_by_validation": selected,
        "tvd_target_at_least_3_met": tvd_gain >= 3.0,
        "tvd_target_3_to_5_met": 3.0 <= tvd_gain <= 5.0,
        "generalization_target_met": ard_gain >= 3.0 and tvd_gain >= 3.0,
        "generalization_interpretation": "The Action Bank improvement generalizes across ARD100 and TVD only if both fixed-test gains exceed 3 percentage points over their VATD baselines.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT / "official_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", 2, summary=str(summary_path), generalization_target_met=summary["generalization_target_met"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        report("failed", 0, error=repr(error))
        raise
