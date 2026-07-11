from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from qstr_dronedet.samurai_dataset import (
    associate_tracks,
    export_samurai_dataset,
    load_box_csv,
    select_tracks,
    validate_samurai_dataset,
)


class SamuraiDatasetTest(unittest.TestCase):
    def test_association_export_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frames = root / "frames"
            frames.mkdir()
            for frame_id in range(1, 7):
                Image.new("RGB", (64, 48), (frame_id, 0, 0)).save(frames / f"Clip_01_{frame_id:05d}.png")

            gt_path = root / "gt.csv"
            rows = []
            for frame_id in range(1, 7):
                rows.append(("Clip_01", frame_id, 5 + frame_id, 10, 9 + frame_id, 14, f"Clip_01/Clip_01_{frame_id:05d}.png"))
                if frame_id != 4:
                    rows.append(("Clip_01", frame_id, 40 - frame_id, 25, 45 - frame_id, 30, f"Clip_01/Clip_01_{frame_id:05d}.png"))
            with gt_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("seq", "frame_id", "x1", "y1", "x2", "y2", "video_path"))
                writer.writerows(rows)

            observations = load_box_csv(gt_path)
            tracks = associate_tracks(observations, max_gap=1)
            selected = select_tracks(tracks, min_visible_frames=5, min_visibility=0.8)
            self.assertEqual(2, len(selected))
            self.assertEqual([5, 6], sorted(track.visible_frames for track in selected))

            output = root / "samurai"
            manifest = export_samurai_dataset(
                selected,
                frames_root=frames,
                output_root=output,
                split="test",
                image_mode="hardlink",
                write_vos=True,
            )
            validation = validate_samurai_dataset(output, split="test")
            self.assertEqual(2, manifest["sequence_count"])
            self.assertEqual(12, validation["frames"])
            self.assertEqual(11, validation["visible_frames"])
            masks = list((output / "vos" / "Annotations").glob("*/*.png"))
            self.assertEqual(12, len(masks))


if __name__ == "__main__":
    unittest.main()
