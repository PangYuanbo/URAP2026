import os
import pickle
from pathlib import Path


SRC_ROOT = Path(r"D:\URAP_datasets\ARD100_YOLOMG")
OUT_ROOT = Path(r"D:\URAP_datasets\TransVisDrone\ARD100")


def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def parse_src_name(name: str):
    stem = Path(name).stem
    video_name, frame_str = stem.split("_")
    clip_id = int(video_name.replace("phantom", ""))
    frame_id = int(frame_str)
    return clip_id, frame_id


def convert_split(split: str):
    src_img_dir = SRC_ROOT / "images" / split
    src_lbl_dir = SRC_ROOT / "labels" / split
    out_img_dir = OUT_ROOT / "AllFrames" / split
    out_lbl_dir = OUT_ROOT / "Annotations" / split
    out_vid_dir = OUT_ROOT / "Videos" / split
    out_vid_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    img_files = sorted(src_img_dir.glob("*.jpg"))
    for src_img in img_files:
        clip_id, frame_id = parse_src_name(src_img.name)
        dst_name = f"Clip_{clip_id}_{frame_id:05d}.jpg"
        hardlink_or_copy(src_img, out_img_dir / dst_name)

        src_lbl = src_lbl_dir / f"{src_img.stem}.txt"
        dst_lbl = out_lbl_dir / f"Clip_{clip_id}_{frame_id:05d}.txt"
        if src_lbl.exists():
            hardlink_or_copy(src_lbl, dst_lbl)
        else:
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.touch()

        counts[clip_id] = max(counts.get(clip_id, 0), frame_id)

    with open(out_vid_dir / "video_length_dict.pkl", "wb") as f:
        pickle.dump(counts, f)

    return {
        "split": split,
        "num_images": len(img_files),
        "num_videos": len(counts),
        "video_length_pkl": str(out_vid_dir / "video_length_dict.pkl"),
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = [convert_split(split) for split in ("train", "val", "test")]

    yaml_text = f"""path: {OUT_ROOT.as_posix()}
train: {OUT_ROOT.as_posix()}/AllFrames/train
val: {OUT_ROOT.as_posix()}/AllFrames/val
test: {OUT_ROOT.as_posix()}/AllFrames/test
inference: {OUT_ROOT.as_posix()}/AllFrames/test
annotation_path: {OUT_ROOT.as_posix()}/Annotations
annotation_train: {OUT_ROOT.as_posix()}/Annotations/train
annotation_val: {OUT_ROOT.as_posix()}/Annotations/val
annotation_test: {OUT_ROOT.as_posix()}/Annotations/test
video_root_path: {OUT_ROOT.as_posix()}/Videos
video_root_path_train: {OUT_ROOT.as_posix()}/Videos/train
video_root_path_val: {OUT_ROOT.as_posix()}/Videos/val
video_root_path_test: {OUT_ROOT.as_posix()}/Videos/test
video_root_path_inference: {OUT_ROOT.as_posix()}/Videos/test
nc: 1
names: ['drone']
"""
    yaml_path = OUT_ROOT / "ARD100_TVD.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    lines = [f"yaml={yaml_path}"]
    for s in summaries:
        lines.append(f"{s['split']}: images={s['num_images']} videos={s['num_videos']} pkl={s['video_length_pkl']}")
    (OUT_ROOT / "prepare_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
