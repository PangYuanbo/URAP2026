# Frame Annotation Tool

Static browser tool for continuing video frame annotation.

## Review Interpolated Frames

Use `review.html` to inspect interpolated boxes one frame at a time.

1. Start a local static server from the repo root if you want the built-in `522 interpolated` JSON button:
   `python3 -m http.server 8787`
2. Open `http://localhost:8787/annotation_tool/review.html`.
3. Click `Load 522 Video`, or choose the matching video file manually:
   `data/raw_videos/dji_fly_20260522_113924_10_1779475848691_hdrvideo.MP4`
4. Click `Load 522 JSON`, or choose an exported/interpolated annotation JSON manually.
5. Use arrow keys to move frame by frame. Press `Enter` for OK, `x` for wrong, `s` for skip, and `n` for next unreviewed frame.
6. Click `Export Review` to save a JSON containing bad frames and notes.

## Use

1. Open `annotator.html` in Chrome, Edge, or Safari.
2. Choose the video file.
3. Set `step` to `5`.
4. If assigning work, set `start frame`, `end frame`, and `annotator`.
5. Drag a box around the target. The tool auto-advances to the next target frame.
6. Click `Export JSON` when done.

The browser cannot silently save to the repository or Google Drive. Each annotator should export JSON and send it back.

## Collaboration

Recommended workflow:

- Split one video into non-overlapping frame ranges.
- Give each annotator the same video and their assigned start/end frame.
- Ask each annotator to export JSON with their name filled in.
- Merge returned JSON files later.

For `step = 5`, each 1000-frame assignment is about 200 boxes.
