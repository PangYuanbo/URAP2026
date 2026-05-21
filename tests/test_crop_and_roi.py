import numpy as np
import torch

from qstr_dronedet.features.roi import crop_with_context, extract_temporal_tube, roi_align_multiscale


def test_crop_with_context_clamps_shape():
    frame = np.zeros((50, 60, 3), np.uint8)
    crop = crop_with_context(frame, (-5, -4, 4, 4), out_size=32)
    assert crop.shape == (32, 32, 3)


def test_temporal_tube_shape():
    frames = [np.zeros((50, 60, 3), np.uint8) for _ in range(2)]
    tube = extract_temporal_tube(frames, (10, 10, 12, 12), T=5, out_size=24)
    assert tube.shape == (5, 3, 24, 24)


def test_roi_fallback_shape():
    features = {"p2": torch.zeros(1, 8, 20, 30)}
    boxes = torch.tensor([[0.0, 0.0, 30.0, 20.0], [10.0, 10.0, 40.0, 30.0]])
    roi = roi_align_multiscale(features, boxes, image_size=(80, 120), output_size=7)
    assert roi.shape == (2, 8, 7, 7)

