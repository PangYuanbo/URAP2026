from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_compare(
    repo: Path,
    tmp_path: Path,
    map50: float,
    *,
    full_split: bool | None = None,
    require_full_split: bool = False,
) -> tuple[int, dict]:
    eval_json = tmp_path / "eval.json"
    out_json = tmp_path / f"compare_{map50}_{full_split}_{require_full_split}.json"
    eval_data = {
        "images": 10,
        "labels": 10,
        "detections": 20,
        "precision": map50,
        "recall": map50,
        "map50": map50,
        "map5095": map50,
        "f1": map50,
    }
    if full_split is not None:
        eval_data["full_split"] = full_split
        eval_data["max_samples"] = 0 if full_split else 5
    eval_json.write_text(
        json.dumps(eval_data),
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        str(repo / "tools" / "compare_native_video_eval_baseline.py"),
        "--eval-json",
        str(eval_json),
        "--out-json",
        str(out_json),
    ]
    if require_full_split:
        cmd.append("--require-full-split")
    proc = subprocess.run(cmd, cwd=repo, check=False)
    return proc.returncode, json.loads(out_json.read_text(encoding="utf-8"))


def test_native_video_eval_baseline_compare() -> None:
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="native_video_compare_") as tmp:
        below_code, below = _run_compare(repo, Path(tmp), 0.0)
        assert below_code == 1
        assert below["status"] == "below_baseline"
        beat_code, beat = _run_compare(repo, Path(tmp), 1.0)
        assert beat_code == 0
        assert beat["status"] == "beat_baseline"
        subset_code, subset = _run_compare(repo, Path(tmp), 1.0, full_split=False, require_full_split=True)
        assert subset_code == 1
        assert subset["status"] == "not_full_split"
        assert subset["full_split"] is False
        full_code, full = _run_compare(repo, Path(tmp), 1.0, full_split=True, require_full_split=True)
        assert full_code == 0
        assert full["status"] == "beat_baseline"


if __name__ == "__main__":
    test_native_video_eval_baseline_compare()
