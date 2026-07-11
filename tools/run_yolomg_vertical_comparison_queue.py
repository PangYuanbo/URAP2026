import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Wait for NPS full renders, then build vertical comparisons.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def atomic_write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    jobs = []
    for video in ("DJI_0619_W", "DJI_0621_W", "DJI_0623_W", "DJI_0625_W"):
        compact = video.lower().replace("_w", "").replace("_", "")
        if video == "DJI_0623_W":
            run_id = f"nps_tvl1_yolomg_{compact}_full"
            nps_root = args.repo / "artifacts" / "nps_tvl1_yolomg"
        else:
            run_id = f"nps_tvl1_cuda_yolomg_{compact}_full"
            nps_root = args.repo / "artifacts" / "nps_tvl1_cuda_yolomg"
        jobs.append({
            "video": video,
            "nps_manifest": nps_root / run_id / "manifest.json",
            "top": args.repo / "artifacts" / "yolomg_desktop_motion_diff" / "desktop_yolomg_motion_diff_full" / f"{video}_yolomg_motion_diff.avi",
            "bottom": nps_root / run_id / f"{video}_nps_tvl1_yolomg_diff.avi",
            "output": args.repo / "artifacts" / "yolomg_nps_vertical_comparison" / f"{video}_yolomg_vs_nps_vertical.avi",
        })
    while True:
        ready = []
        status_rows = []
        for job in jobs:
            is_ready = job["nps_manifest"].exists() and job["top"].exists() and job["bottom"].exists()
            ready.append(is_ready)
            status_rows.append({"video": job["video"], "nps_status": "completed" if is_ready else "waiting_for_manifest", "ready": is_ready})
        atomic_write(args.progress_json, {"status": "waiting", "done": sum(ready), "total": len(jobs), "videos": status_rows, "last_check_timestamp": time.time()})
        if all(ready):
            break
        time.sleep(args.poll_seconds)
    output_dir = jobs[0]["output"].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for job in jobs:
        child_progress = args.progress_json.parent / f"{job['video']}_merge_progress.json"
        atomic_write(args.progress_json, {"status": "merging", "done": completed, "total": len(jobs), "current_video": job["video"], "last_check_timestamp": time.time()})
        command = [str(args.python), str(args.repo / "tools" / "merge_yolomg_vertical_comparison.py"), "--top", str(job["top"]), "--bottom", str(job["bottom"]), "--output", str(job["output"]), "--progress-json", str(child_progress)]
        subprocess.run(command, cwd=args.repo, check=True)
        completed += 1
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps({"layout": "top=original YOLOMG, bottom=NPS Dual TV-L1 + YOLOMG", "outputs": [str(job["output"]) for job in jobs]}, indent=2), encoding="utf-8")
    atomic_write(args.progress_json, {"status": "completed", "done": completed, "total": len(jobs), "manifest": str(manifest), "last_check_timestamp": time.time()})


if __name__ == "__main__":
    main()