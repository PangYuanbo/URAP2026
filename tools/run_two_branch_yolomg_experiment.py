from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("RUN " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-windows-per-tracklet", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--center", type=float, default=0.20)
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--mode", choices=["additive", "suppress-only", "boost-only"], default="boost-only")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    status_path = out_dir / "status.json"
    train_tracklets = repo / "artifacts/yolomg_action/yolomg_train_lowconf_proposal_tracklets_20260605/proposal_tracklets.jsonl"
    test_tracklets = repo / "artifacts/yolomg_action/yolomg_test_lowconf_proposal_tracklets_20260605/proposal_tracklets.jsonl"
    images_list = Path("D:/URAP_datasets/ARD100_YOLOMG/test.txt")
    pred_label_dir = repo / "artifacts/yolomg_action/fulltest_lowconf_pred_export_20260605/pred_labels"
    scored_tracklets = out_dir / "test_tracklets_scored.jsonl"
    model_path = out_dir / "two_branch_motion_action.pt"
    train_summary = out_dir / "train_summary.json"
    out_label_dir = out_dir / "pred_labels"
    eval_dir = out_dir / "eval"
    score_field = "two_branch_motion_action_score"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    write_status(status_path, {"state": "running", "started": started, "step": "train_score", "done": 0, "total": 3})
    train_cmd = [
        sys.executable,
        "tools/train_two_branch_motion_action_scorer.py",
        "--train-tracklets",
        str(train_tracklets),
        "--test-tracklets",
        str(test_tracklets),
        "--out-test-tracklets",
        str(scored_tracklets),
        "--out-model",
        str(model_path),
        "--out-summary",
        str(train_summary),
        "--score-field",
        score_field,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--past-len",
        "8",
        "--future-len",
        "2",
        "--max-windows-per-tracklet",
        str(args.max_windows_per_tracklet),
    ]
    run_cmd(train_cmd, repo)

    write_status(status_path, {"state": "running", "started": started, "step": "rescore_labels", "done": 1, "total": 3})
    run_cmd(
        [
            sys.executable,
            "tools/yolomg_rescore_pred_labels_from_tracklets.py",
            "--images-list",
            str(images_list),
            "--pred-label-dir",
            str(pred_label_dir),
            "--tracklet-jsonl",
            str(scored_tracklets),
            "--out-label-dir",
            str(out_label_dir),
            "--score-field",
            score_field,
            "--center",
            str(args.center),
            "--beta",
            str(args.beta),
            "--mode",
            args.mode,
            "--missing-score-behavior",
            "keep",
            "--min-tracklet-rows",
            "3",
        ],
        repo,
    )

    write_status(status_path, {"state": "running", "started": started, "step": "eval", "done": 2, "total": 3})
    run_cmd(
        [
            sys.executable,
            "tools/yolomg_eval_pred_labels.py",
            "--images-list",
            str(images_list),
            "--pred-label-dir",
            str(out_label_dir),
            "--out-dir",
            str(eval_dir),
            "--conf-thres",
            "0.001",
            "--image-width",
            "1920",
            "--image-height",
            "1080",
        ],
        repo,
    )
    summary = {
        "state": "complete",
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": "YOLOMG train lowconf proposal tracklets -> test lowconf proposal tracklets -> pred-label rescore/eval",
        "train_summary": read_json(train_summary),
        "rescore_summary": read_json(out_dir / "pred_labels_rescore_summary.json"),
        "eval_manifest": read_json(eval_dir / "manifest.json"),
        "params": {"center": args.center, "beta": args.beta, "mode": args.mode},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_status(status_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
