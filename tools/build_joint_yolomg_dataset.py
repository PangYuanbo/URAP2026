from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_pairs(root: Path, split: str, skip_incomplete: bool = False) -> tuple[list[Path], list[Path], int]:
    images_dir = root / "images" / split
    motion_dir = root / "images2" / split
    labels_dir = root / "labels" / split
    if not images_dir.is_dir() or not motion_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"missing YOLOMG split directories under {root} for {split}")

    images = sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    paired_images: list[Path] = []
    motion: list[Path] = []
    missing: list[str] = []
    for image in images:
        relative = image.relative_to(images_dir)
        motion_path = motion_dir / relative
        label_path = (labels_dir / relative).with_suffix(".txt")
        if not motion_path.is_file() or not label_path.is_file():
            missing.append(str(relative))
            continue
        paired_images.append(image)
        motion.append(motion_path)
    if missing and not skip_incomplete:
        raise RuntimeError(f"{root.name}/{split}: {len(missing)} missing motion/label pairs; first={missing[:5]}")
    if not paired_images:
        raise RuntimeError(f"{root.name}/{split}: no images found")
    return paired_images, motion, len(missing)


def write_lines(path: Path, values: list[Path]) -> None:
    path.write_text("".join(f"{value.resolve().as_posix()}\n" for value in values), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build leak-free joint NPS+ARD100 YOLOMG lists")
    parser.add_argument("--nps-root", type=Path, default=Path(r"D:\URAP_local_datasets\NPS_YOLOMG"))
    parser.add_argument("--ard-root", type=Path, default=Path(r"D:\URAP_local_datasets\ARD100_YOLOMG"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\URAP_local_datasets\joint_yolomg"))
    parser.add_argument("--smoke-train-per-dataset", type=int, default=64)
    parser.add_argument("--smoke-val-per-dataset", type=int, default=32)
    parser.add_argument("--allow-missing-tests", action="store_true")
    parser.add_argument("--skip-incomplete-pairs", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {"datasets": {}, "policy": "train/val union; tests remain separate"}
    split_inputs: dict[str, tuple[list[Path], list[Path], list[Path], list[Path]]] = {}
    for split in ("train", "val"):
        nps_images, nps_motion, nps_skipped = collect_pairs(args.nps_root, split, args.skip_incomplete_pairs)
        ard_images, ard_motion, ard_skipped = collect_pairs(args.ard_root, split, args.skip_incomplete_pairs)
        split_inputs[split] = nps_images, nps_motion, ard_images, ard_motion
        images = nps_images + ard_images
        motion = nps_motion + ard_motion
        write_lines(args.output / f"{split}.txt", images)
        write_lines(args.output / f"{split}2.txt", motion)
        summary["datasets"][split] = {
            "nps": len(nps_images), "nps_skipped_incomplete": nps_skipped,
            "ard100": len(ard_images), "ard100_skipped_incomplete": ard_skipped,
            "total": len(images),
        }

    tests: dict[str, dict[str, int]] = {}
    for dataset_name, root in (("nps", args.nps_root), ("ard100", args.ard_root)):
        try:
            images, motion, skipped = collect_pairs(root, "test", args.skip_incomplete_pairs)
        except (FileNotFoundError, RuntimeError):
            if not args.allow_missing_tests:
                raise
            images, motion = split_inputs["val"][0:2] if dataset_name == "nps" else split_inputs["val"][2:4]
            skipped = 0
        write_lines(args.output / f"test_{dataset_name}.txt", images)
        write_lines(args.output / f"test_{dataset_name}2.txt", motion)
        tests[dataset_name] = {"total": len(images), "skipped_incomplete": skipped, "temporary_val_fallback": not (root / "images" / "test").is_dir()}
    summary["tests"] = tests

    yaml_text = "\n".join(
        [
            f"train: {(args.output / 'train.txt').resolve().as_posix()}",
            f"train2: {(args.output / 'train2.txt').resolve().as_posix()}",
            f"val: {(args.output / 'val.txt').resolve().as_posix()}",
            f"val2: {(args.output / 'val2.txt').resolve().as_posix()}",
            f"test: {(args.output / 'test_nps.txt').resolve().as_posix()}",
            f"test2: {(args.output / 'test_nps2.txt').resolve().as_posix()}",
            "nc: 1",
            "names: ['UAV']",
            "",
        ]
    )
    (args.output / "joint_nps_ard100.yaml").write_text(yaml_text, encoding="utf-8")

    smoke_root = args.output / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    for split, per_dataset in (("train", args.smoke_train_per_dataset), ("val", args.smoke_val_per_dataset)):
        nps_images, nps_motion, ard_images, ard_motion = split_inputs[split]
        write_lines(smoke_root / f"{split}.txt", nps_images[:per_dataset] + ard_images[:per_dataset])
        write_lines(smoke_root / f"{split}2.txt", nps_motion[:per_dataset] + ard_motion[:per_dataset])
    smoke_yaml = "\n".join(
        [
            f"train: {(smoke_root / 'train.txt').resolve().as_posix()}",
            f"train2: {(smoke_root / 'train2.txt').resolve().as_posix()}",
            f"val: {(smoke_root / 'val.txt').resolve().as_posix()}",
            f"val2: {(smoke_root / 'val2.txt').resolve().as_posix()}",
            f"test: {(args.output / 'test_nps.txt').resolve().as_posix()}",
            f"test2: {(args.output / 'test_nps2.txt').resolve().as_posix()}",
            "nc: 1",
            "names: ['UAV']",
            "",
        ]
    )
    (smoke_root / "joint_nps_ard100_smoke.yaml").write_text(smoke_yaml, encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
