# Frame Annotation Tool

Static browser tool for continuing video frame annotation.

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
