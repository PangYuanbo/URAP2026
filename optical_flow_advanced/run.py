import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "baseline-homography": ROOT / "tools" / "yolomg_motion_diff_video.py",
    "nps-tvl1": ROOT / "tools" / "nps_tvl1_yolomg_diff.py",
    "sea-raft": ROOT / "tools" / "sea_raft_video_flow_diff.py",
    "double-stage": ROOT / "tools" / "double_stage_motion_diff.py",
    "parallax-robust": ROOT / "tools" / "yolomg_parallax_robust_difference.py",
    "evaluate-parallax": ROOT / "tools" / "evaluate_parallax_robust_difference.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified launcher for the URAP advanced optical-flow pipelines."
    )
    parser.add_argument("command", choices=["list", "run"])
    parser.add_argument("method", nargs="?", choices=sorted(METHODS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "list":
        for name, script in METHODS.items():
            print(f"{name:20s} {script.relative_to(ROOT)}")
        return 0
    if not args.method:
        raise SystemExit("A method is required for the run command.")
    script = METHODS[args.method]
    if not script.exists():
        raise SystemExit(f"Method script not found: {script}")
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    return subprocess.run([sys.executable, str(script), *forwarded], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
