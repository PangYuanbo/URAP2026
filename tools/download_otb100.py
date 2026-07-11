from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import vot.dataset.otb as otb

ROOT = Path(r"D:\URAP_local_datasets\OTB100")
RUNNER = Path(r"C:\Users\aaron\Desktop\URAP\artifacts\detached_otb100_download")
PROGRESS = RUNNER / "progress.json"


def write(stage: str, **extra: object) -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps({"stage": stage, "updated": datetime.now(timezone.utc).astimezone().isoformat(), "root": str(ROOT), **extra}, indent=2), encoding="utf-8")


write("downloading", done=0, total=100)
otb._BASE_URL = "https://web.archive.org/web/20221118171918id_/http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/"
otb.download_otb100(str(ROOT))
sequences = sum(1 for path in ROOT.iterdir() if path.is_dir())
images = sum(1 for _ in ROOT.rglob("*.jpg"))
write("done", done=sequences, total=100, images=images)
print(json.dumps({"kind": "otb100_download_done", "sequences": sequences, "images": images, "root": str(ROOT)}), flush=True)


