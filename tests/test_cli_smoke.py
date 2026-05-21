import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def test_cli_help_works():
    r = subprocess.run([sys.executable, "-m", "qstr_dronedet.cli", "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "motion-debug" in r.stdout


def test_motion_debug_synthetic_video(tmp_path: Path):
    video = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (96, 64))
    for i in range(4):
        frame = np.zeros((64, 96, 3), np.uint8)
        cv2.circle(frame, (20 + i * 4, 30), 2, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "qstr_dronedet.cli", "motion-debug", "--video", str(video), "--out", str(out), "--k-values", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "diagnostics.jsonl").exists()

