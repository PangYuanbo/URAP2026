from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
TVD = Path(r"U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone")
TRAIN_RUN = ROOT / "artifacts/detached_tvd_detector_hard_replay_v165"
RUN = ROOT / "artifacts/detached_tvd_detector_hard_replay_v165_posteval"
TRAIN_OUTPUT = Path(r"D:\URAP_vatd_rank_results\tvd_detector_hard_replay_v165\hard8_replay12k_img1280_noval_e2")
OUTPUT = Path(r"D:\URAP_vatd_rank_results\tvd_detector_hard_replay_v165_posteval")
DATA = TVD / "data/NPS_URAP_D.yaml"
PYTHON = TVD / ".venv/Scripts/python.exe"
EVALUATOR = ROOT / "tools/eval_tvd_predictionsgt_pkl.py"
PROGRESS = RUN / "progress.json"
VATD_MAP50 = 0.93844


def now() -> str:
    return datetime.now().astimezone().isoformat()


def report(stage: str, done: int, total: int = 6, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(
        json.dumps({"stage": stage, "done": done, "total": total, "updated": now(), **extra}, indent=2),
        encoding="utf-8",
    )


def process_command(pid: int) -> str | None:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' -ErrorAction SilentlyContinue; if($p){{$p.CommandLine}}",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    value = result.stdout.strip()
    return value or None


def wait_for_training() -> list[Path]:
    pid_path = TRAIN_RUN / "pid.txt"
    meta_path = TRAIN_RUN / "meta.json"
    if not pid_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("V165 training PID/meta files are missing")
    training_pid = int(pid_path.read_text(encoding="utf-8").strip())
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    expected = str(meta.get("command_line", ""))
    command = process_command(training_pid)
    if command is not None and ("train.py" not in command or "hard8_replay12k_img1280_noval_e2" not in command):
        raise RuntimeError(f"PID {training_pid} does not match V165 training: {command}")
    report("await_training", 0, training_pid=training_pid, training_command=command or expected)
    while process_command(training_pid) is not None:
        time.sleep(30)
        report("await_training", 0, training_pid=training_pid)
    time.sleep(5)
    weights = [TRAIN_OUTPUT / "weights/epoch18.pt", TRAIN_OUTPUT / "weights/epoch19.pt"]
    missing = [str(path) for path in weights if not path.is_file() or path.stat().st_size < 100_000_000]
    if missing:
        raise RuntimeError(f"V165 training stopped without expected checkpoints: {missing}")
    return weights


def run_child(command: list[str], stage: str, done: int, log_name: str) -> None:
    logs = RUN / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{log_name}.stdout.log"
    stderr_path = logs / f"{log_name}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=TVD,
            stdout=stdout,
            stderr=stderr,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT), "WANDB_MODE": "disabled"},
        )
        (RUN / "child_pid.txt").write_text(str(process.pid), encoding="ascii")
        report(stage, done, child_pid=process.pid, command=command, stdout_log=str(stdout_path), stderr_log=str(stderr_path))
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def predictions_path(name: str) -> Path:
    return OUTPUT / name / "predictionsgt/predictionsgt_split_0.pkl"


def run_inference(weights: Path, task: str, name: str, augment: bool, done: int, batch_size: int) -> Path:
    command = [
        str(PYTHON), "val.py", "--task", task, "--data", str(DATA), "--weights", str(weights),
        "--img", "1280", "--batch-size", str(batch_size), "--half", "--num-frames", "5",
        "--conf-thres", "0.001", "--iou-thres", "0.6", "--save-json-gt",
        "--project", str(OUTPUT), "--name", name, "--exist-ok",
    ]
    if augment:
        command.append("--augment")
    run_child(command, f"{task}_{'tta' if augment else 'standard'}", done, name)
    predictions = predictions_path(name)
    if not predictions.is_file():
        raise FileNotFoundError(predictions)
    return predictions


def evaluate(predictions: Path, out_json: Path, stage: str, done: int) -> dict[str, object]:
    command = [
        str(PYTHON), str(EVALUATOR), "--tvd-root", str(TVD),
        "--predictionsgt-pkl", str(predictions), "--out-json", str(out_json),
    ]
    run_child(command, stage, done, out_json.stem)
    return json.loads(out_json.read_text(encoding="utf-8"))


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoints = wait_for_training()
    checkpoint_results: list[dict[str, object]] = []
    for index, weights in enumerate(checkpoints):
        checkpoint_name = weights.stem
        predictions = run_inference(weights, "val", f"val_{checkpoint_name}_standard", False, index, 8)
        result = evaluate(predictions, OUTPUT / f"val_{checkpoint_name}_standard_metrics.json", f"score_val_{checkpoint_name}", index + 1)
        checkpoint_results.append({"checkpoint": str(weights), **result})
    selected_checkpoint = max(checkpoint_results, key=lambda row: float(row["map50"]))
    weights = Path(str(selected_checkpoint["checkpoint"]))
    standard = {key: value for key, value in selected_checkpoint.items() if key != "checkpoint"}
    selected_standard_predictions = predictions_path(f"val_{weights.stem}_standard")
    standard_alias = predictions_path("val_standard")
    standard_alias.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_standard_predictions, standard_alias)
    tta_predictions = run_inference(weights, "val", "val_tta", True, 4, 4)
    tta = evaluate(tta_predictions, OUTPUT / "val_tta_metrics.json", "score_val_tta", 4)
    selected_augment = float(tta["map50"]) > float(standard["map50"])
    selected_mode = "tta" if selected_augment else "standard"
    selection = {
        "criterion": "maximum validation mAP@0.5",
        "checkpoint_candidates": checkpoint_results,
        "selected_checkpoint": str(weights),
        "selected_mode": selected_mode,
        "standard": standard,
        "tta": tta,
    }
    (OUTPUT / "validation_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    test_name = f"test_{selected_mode}"
    test_predictions = run_inference(weights, "test", test_name, selected_augment, 5, 4 if selected_augment else 8)
    test = evaluate(test_predictions, OUTPUT / "test_fixed_metrics.json", "score_fixed_test", 5)
    gain = (float(test["map50"]) - VATD_MAP50) * 100.0
    summary = {
        "method": "TransVisDrone V165 hard-clip + full-domain replay finetune",
        "weights": str(weights),
        "validation_selection": selection,
        "test_fixed": test,
        "vatd_baseline_map50": VATD_MAP50,
        "gain_over_vatd_points": gain,
        "target_3_to_5_met": 3.0 <= gain <= 5.0,
        "target_at_least_3_met": gain >= 3.0,
    }
    summary_path = OUTPUT / "official_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", 6, summary=str(summary_path), test_map50=test["map50"], gain_over_vatd_points=gain)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        report("failed", 0, error=repr(error))
        raise
