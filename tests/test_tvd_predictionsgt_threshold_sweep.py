from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from pathlib import Path


def test_tvd_predictionsgt_threshold_sweep_selects_best(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    pkl_path = tmp_path / "predictionsgt.pkl"
    out_json = tmp_path / "sweep.json"
    out_csv = tmp_path / "sweep.csv"
    data = {
        "img1": {
            "labels": [{"bbox": [10.0, 10.0, 20.0, 20.0], "category_id": 0}],
            "detections": [
                {"bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.9, "category_id": 0},
                {"bbox": [50.0, 50.0, 60.0, 60.0], "score": 0.1, "category_id": 0},
            ],
        },
        "img2": {
            "labels": [{"bbox": [30.0, 30.0, 40.0, 40.0], "category_id": 0}],
            "detections": [
                {"bbox": [30.0, 30.0, 40.0, 40.0], "score": 0.8, "category_id": 0},
            ],
        },
    }
    with pkl_path.open("wb") as f:
        pickle.dump(data, f)

    subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "sweep_tvd_predictionsgt_thresholds.py"),
            "--predictionsgt-pkl",
            str(pkl_path),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--score-thresholds",
            "0.0",
            "0.2",
            "--top-ks",
            "1",
            "2",
            "--primary-metric",
            "f1",
        ],
        cwd=repo,
        check=True,
    )
    assert out_json.exists()
    assert out_csv.exists()
    text = out_json.read_text(encoding="utf-8")
    assert '"score_threshold": 0.2' in text
    assert '"top_k": 1' in text


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="tvd_sweep_") as tmp:
        test_tvd_predictionsgt_threshold_sweep_selects_best(Path(tmp))
