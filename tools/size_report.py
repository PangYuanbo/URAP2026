import argparse
import os
import shutil
from pathlib import Path


def dir_size_bytes(root: Path) -> int:
    if root.is_file():
        try:
            return root.stat().st_size
        except FileNotFoundError:
            return 0
    total = 0
    stack = [root]
    while stack:
        p = stack.pop()
        try:
            with os.scandir(p) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except FileNotFoundError:
                        # Expected when downloads are in progress.
                        continue
        except (FileNotFoundError, PermissionError):
            continue
    return total


def fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if x < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(x)} {unit}"
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", action="append", default=[], help="Path to measure (repeatable). Format: NAME=PATH or PATH")
    args = ap.parse_args()

    if not args.path:
        # Reasonable defaults for this workspace.
        args.path = [
            r"AOT_raw_part1=D:\URAP_datasets\AOT\part1",
            r"NPS_prepared=D:\URAP_datasets\TransVisDrone\NPS",
            r"AOT_yolo_fulltest_hardlinks_logical=D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest",
            r"Papers_PDFs=C:\Users\aaron\Desktop\URAP\doc",
            r"TransVisDrone_pretrained_weights=C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\pretrained\TransVisDrone_weights",
        ]

    print("Disk usage snapshot:")
    for drive in [r"D:\\", r"C:\\"]:
        try:
            du = shutil.disk_usage(drive)
            print(
                f"  {drive} used={fmt_bytes(du.used)} free={fmt_bytes(du.free)} total={fmt_bytes(du.total)}"
            )
        except Exception as e:
            print(f"  {drive} disk_usage failed: {e}")

    print("\nDirectory size snapshot (logical file sizes; hardlinks may double-count if you include both source and link trees):")
    for spec in args.path:
        if "=" in spec:
            name, p = spec.split("=", 1)
        else:
            name, p = spec, spec
        path = Path(p)
        if not path.exists():
            print(f"  {name}: MISSING ({path})")
            continue
        size_b = dir_size_bytes(path)
        print(f"  {name}: {fmt_bytes(size_b)} ({path})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
