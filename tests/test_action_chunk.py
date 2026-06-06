import numpy as np
import json

from qstr_dronedet.tracking.action_chunk import (
    action_reconstruction_error,
    actions_from_boxes,
    apply_box_action,
    attach_frame_priors_to_tracklets,
    box_action,
    build_frame_prior_index_from_heatmaps,
    build_action_chunk_samples_from_rows,
    export_action_chunk_dataset_from_tracklets,
    export_action_prior_heatmaps_from_sample_scores,
    gaussian_prior_heatmap,
    merge_action_chunk_datasets,
    reconstruct_boxes,
    split_action_chunk_dataset,
)


def test_box_action_roundtrip():
    prev = (10.0, 20.0, 18.0, 28.0)
    nxt = (13.0, 22.0, 21.0, 32.0)

    action = box_action(prev, nxt)
    out = apply_box_action(prev, action)

    assert np.allclose(out, nxt, atol=1e-5)


def test_apply_box_action_clamps_nonfinite_and_extreme_scale():
    out = apply_box_action((10.0, 20.0, 18.0, 28.0), (float("nan"), 1.0, 1000.0, float("-inf")))

    assert np.all(np.isfinite(out))
    assert out[2] > out[0]
    assert out[3] > out[1]


def test_actions_reconstruct_future_boxes():
    boxes = np.array(
        [
            [10.0, 20.0, 18.0, 28.0],
            [12.0, 21.0, 20.0, 29.0],
            [14.0, 23.0, 23.0, 33.0],
        ],
        dtype=np.float32,
    )

    actions = actions_from_boxes(boxes)
    reconstructed = reconstruct_boxes(boxes[0], actions)

    assert actions.shape == (2, 4)
    assert np.allclose(reconstructed, boxes[1:], atol=1e-5)
    assert action_reconstruction_error(boxes[0], actions, boxes[1:]) < 1e-5


def test_build_action_chunk_samples_from_rows_normalized():
    rows = [
        {"seq": "s1", "track_id": "t1", "frame_id": i, "bbox": [10 + i, 20, 18 + i, 28], "objectness": 0.5}
        for i in range(6)
    ]

    samples = build_action_chunk_samples_from_rows(rows, past_len=3, future_len=2, image_size=(100, 100))

    assert len(samples) == 2
    assert samples[0].seq == "s1"
    assert samples[0].track_id == "t1"
    assert samples[0].anchor_frame == 2
    assert samples[0].past_boxes.shape == (3, 4)
    assert samples[0].future_actions.shape == (2, 4)
    assert np.isclose(samples[0].past_boxes[0, 0], 0.10)


def test_build_action_chunk_samples_from_rows_normalized_by_row_image_size():
    rows = []
    for i in range(6):
        scale = 1 if i < 3 else 2
        rows.append(
            {
                "seq": "s1",
                "track_id": "t1",
                "frame_id": i,
                "bbox": [scale * (10 + i), scale * 20, scale * (18 + i), scale * 28],
                "objectness": 0.5,
                "image_width": scale * 100,
                "image_height": scale * 100,
            }
        )

    samples = build_action_chunk_samples_from_rows(rows, past_len=3, future_len=2, normalize_by_row_image_size=True)

    assert len(samples) == 2
    assert np.isclose(samples[0].past_boxes[0, 0], 0.10)
    assert np.isclose(samples[0].future_boxes[0, 0], 0.13)
    assert np.isclose(samples[0].future_actions[0, 0], 0.01)


def test_gaussian_prior_heatmap_peaks_near_box_center():
    heatmap = gaussian_prior_heatmap(np.array([[8.0, 8.0, 12.0, 12.0]], dtype=np.float32), image_size=(24, 24))

    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)

    assert heatmap.shape == (24, 24)
    assert heatmap.max() == 1.0
    assert abs(x - 10) <= 1
    assert abs(y - 10) <= 1


def test_export_action_chunk_dataset_from_tracklets(tmp_path):
    tracklets = tmp_path / "tracklets.jsonl"
    rows = [
        {"seq": "s1", "track_id": "t1", "frame_id": i, "bbox": [10 + i, 20, 18 + i, 28], "objectness": 0.5}
        for i in range(6)
    ]
    item = {
        "meta": {
            "seq": "s1",
            "track_id": "t1",
            "label": 1,
            "bucket": "hard_tiny_positive",
            "dataset_source": "nps",
        },
        "rows": rows,
    }
    tracklets.write_text(json.dumps(item) + "\n", encoding="utf-8")

    result = export_action_chunk_dataset_from_tracklets(
        tracklets,
        tmp_path / "action_chunks.jsonl",
        past_len=3,
        future_len=2,
        image_size=(100, 100),
    )
    exported = [json.loads(line) for line in result.jsonl_path.read_text(encoding="utf-8").splitlines()]

    assert result.summary["samples"] == 2
    assert result.summary["normalize_by_row_image_size"] is False
    assert result.summary["positive_samples"] == 2
    assert result.summary["bucket_counts"]["hard_tiny_positive"] == 2
    assert exported[0]["seq"] == "s1"
    assert exported[0]["track_id"] == "t1"
    assert exported[0]["label"] == 1
    assert len(exported[0]["past_boxes"]) == 3
    assert len(exported[0]["future_actions"]) == 2
    assert np.isclose(exported[0]["past_boxes"][0][0], 0.10)


def test_merge_action_chunk_datasets_writes_manifest(tmp_path):
    sample_a = {
        "seq": "nps_001",
        "track_id": "tube1",
        "anchor_frame": 7,
        "label": 1,
        "bucket": "hard_tiny_positive",
        "dataset_source": "old_name",
        "past_boxes": [[0.1, 0.1, 0.2, 0.2]],
        "past_scores": [0.8],
        "past_visible": [1.0],
        "future_actions": [[0.01, 0.0, 0.0, 0.0]],
        "future_boxes": [[0.11, 0.1, 0.21, 0.2]],
    }
    sample_b = {
        "seq": "aot_001",
        "track_id": "tube2",
        "anchor_frame": 8,
        "label": 0,
        "bucket": "hard_negative",
        "dataset_source": "old_name",
        "past_boxes": [[0.4, 0.4, 0.5, 0.5]],
        "past_scores": [0.2],
        "past_visible": [1.0],
        "future_actions": [[0.0, 0.01, 0.0, 0.0]],
        "future_boxes": [[0.4, 0.41, 0.5, 0.51]],
    }
    nps = tmp_path / "nps.jsonl"
    aot = tmp_path / "aot.jsonl"
    nps.write_text(json.dumps(sample_a) + "\n", encoding="utf-8")
    aot.write_text(json.dumps(sample_b) + "\n", encoding="utf-8")

    result = merge_action_chunk_datasets(
        [nps, aot],
        tmp_path / "merged.jsonl",
        source_names=["nps", "aot"],
        manifest_out=tmp_path / "manifest.json",
    )
    merged = [json.loads(line) for line in result.jsonl_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert result.summary["samples"] == 2
    assert result.summary["positive_samples"] == 1
    assert result.summary["negative_samples"] == 1
    assert result.summary["dataset_source_counts"] == {"nps": 1, "aot": 1}
    assert merged[0]["dataset_source"] == "nps"
    assert merged[1]["dataset_source"] == "aot"
    assert manifest["schema"]["future_actions"].startswith("[future_len, 4]")
    assert manifest["datasets"][0]["samples"] == 1


def test_split_action_chunk_dataset_is_source_and_sequence_aware(tmp_path):
    rows = []
    for source in ["nps", "aot"]:
        for seq_index in range(4):
            for sample_index in range(2):
                rows.append(
                    {
                        "seq": f"{source}_seq{seq_index}",
                        "track_id": f"tube{sample_index}",
                        "anchor_frame": sample_index,
                        "label": 1 if seq_index % 2 == 0 else 0,
                        "bucket": "positive" if seq_index % 2 == 0 else "hard_negative",
                        "dataset_source": source,
                        "past_boxes": [[0.1, 0.1, 0.2, 0.2]],
                        "past_scores": [0.8],
                        "past_visible": [1.0],
                        "future_actions": [[0.01, 0.0, 0.0, 0.0]],
                        "future_boxes": [[0.11, 0.1, 0.21, 0.2]],
                    }
                )
    dataset = tmp_path / "merged.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = split_action_chunk_dataset(
        dataset,
        tmp_path / "split",
        calib_fraction=0.25,
        test_fraction=0.25,
        seed=7,
    )
    train_rows = [json.loads(line) for line in (tmp_path / "split" / "train_action_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    calib_rows = [json.loads(line) for line in (tmp_path / "split" / "calib_action_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    test_rows = [json.loads(line) for line in (tmp_path / "split" / "test_action_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((tmp_path / "split" / "action_chunk_split_manifest.json").read_text(encoding="utf-8"))

    assert result.jsonl_path == tmp_path / "split" / "train_action_chunks.jsonl"
    assert result.summary["total_samples"] == 16
    assert result.summary["total_groups"] == 8
    assert manifest["sources"]["nps"]["train_groups"] == 2
    assert manifest["sources"]["nps"]["calib_groups"] == 1
    assert manifest["sources"]["nps"]["test_groups"] == 1
    assert len(train_rows) == 8
    assert len(calib_rows) == 4
    assert len(test_rows) == 4

    split_keys = {}
    for split_name, split_rows in [("train", train_rows), ("calib", calib_rows), ("test", test_rows)]:
        for row in split_rows:
            key = (row["dataset_source"], row["seq"])
            assert key not in split_keys or split_keys[key] == split_name
            split_keys[key] = split_name
            assert row["split"] == split_name


def test_export_action_prior_heatmaps_from_sample_scores(tmp_path):
    scores = tmp_path / "sample_scores.jsonl"
    row = {
        "seq": "s1",
        "track_id": "tube1",
        "anchor_frame": 3,
        "label": 1,
        "learned_boxes": [[8.0, 8.0, 12.0, 12.0], [9.0, 8.0, 13.0, 12.0]],
    }
    scores.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = export_action_prior_heatmaps_from_sample_scores(scores, tmp_path / "priors", image_size=(24, 24))
    manifest_rows = [json.loads(line) for line in result.jsonl_path.read_text(encoding="utf-8").splitlines()]
    heatmap = np.load(manifest_rows[0]["heatmap"])

    assert result.summary["exported"] == 1
    assert result.summary["skipped"] == 0
    assert manifest_rows[0]["seq"] == "s1"
    assert heatmap.shape == (24, 24)
    assert np.isclose(float(heatmap.max()), 1.0)

    split = export_action_prior_heatmaps_from_sample_scores(
        scores,
        tmp_path / "priors_split",
        image_size=(24, 24),
        split_horizon=True,
    )
    split_rows = [json.loads(line) for line in split.jsonl_path.read_text(encoding="utf-8").splitlines()]

    assert split.summary["exported"] == 2
    assert split.summary["split_horizon"] is True
    assert split_rows[0]["horizon_index"] == 0
    assert split_rows[0]["target_frame_id"] == 4
    assert split_rows[1]["horizon_index"] == 1
    assert split_rows[1]["target_frame_id"] == 5


def test_build_frame_prior_index_from_heatmaps(tmp_path):
    heatmap_dir = tmp_path / "heatmaps"
    heatmap_dir.mkdir()
    h1 = np.zeros((8, 8), dtype=np.float32)
    h2 = np.zeros((8, 8), dtype=np.float32)
    h1[2, 2] = 1.0
    h2[3, 3] = 1.0
    p1 = heatmap_dir / "p1.npy"
    p2 = heatmap_dir / "p2.npy"
    np.save(p1, h1)
    np.save(p2, h2)
    manifest = tmp_path / "action_prior_heatmaps.jsonl"
    rows = [
        {"seq": "s1", "track_id": "t1", "target_frame_id": 7, "heatmap": str(p1), "image_size": [8, 8]},
        {"seq": "s1", "track_id": "t2", "target_frame_id": 7, "heatmap": str(p2), "image_size": [8, 8]},
        {"seq": "s1", "track_id": "combined", "target_frame_id": None, "heatmap": str(p1), "image_size": [8, 8]},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = build_frame_prior_index_from_heatmaps(manifest, tmp_path / "frame_index")
    index_rows = [json.loads(line) for line in result.jsonl_path.read_text(encoding="utf-8").splitlines()]
    merged = np.load(index_rows[0]["prior"])

    assert result.summary["frames"] == 1
    assert result.summary["skipped_without_target_frame"] == 1
    assert index_rows[0]["seq"] == "s1"
    assert index_rows[0]["frame_id"] == 7
    assert index_rows[0]["num_tracklet_priors"] == 2
    assert set(index_rows[0]["track_ids"]) == {"t1", "t2"}
    assert np.isclose(float(merged[2, 2]), 1.0)
    assert np.isclose(float(merged[3, 3]), 1.0)


def test_attach_frame_priors_to_tracklets_supports_nested_and_flat_rows(tmp_path):
    prior_dir = tmp_path / "frame_priors"
    prior_dir.mkdir()
    heatmap7 = np.zeros((10, 10), dtype=np.float32)
    heatmap7[2:5, 2:5] = 0.8
    heatmap7[3, 3] = 1.0
    heatmap8 = np.zeros((10, 10), dtype=np.float32)
    heatmap8[6:9, 6:9] = 0.6
    np.save(prior_dir / "s1_000007.npy", heatmap7)
    np.save(prior_dir / "s1_000008.npy", heatmap8)
    frame_prior_index = tmp_path / "frame_prior_index.jsonl"
    frame_prior_index.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "seq": "s1",
                    "frame_id": 7,
                    "prior": str(prior_dir / "s1_000007.npy"),
                    "merge_mode": "max",
                    "num_tracklet_priors": 2,
                    "track_ids": ["t1", "t2"],
                    "image_size": [8, 8],
                },
                {
                    "seq": "s1",
                    "frame_id": 8,
                    "prior": str(prior_dir / "s1_000008.npy"),
                    "merge_mode": "max",
                    "num_tracklet_priors": 1,
                    "track_ids": ["t1"],
                    "image_size": [8, 8],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    nested = tmp_path / "tracklets.jsonl"
    nested.write_text(
        json.dumps(
            {
                "meta": {"seq": "s1", "track_id": "tube1"},
                "rows": [
                    {"frame_id": 7, "bbox": [2, 2, 5, 5]},
                    {"frame_id": 9, "bbox": [2, 2, 4, 4]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    nested_result = attach_frame_priors_to_tracklets(nested, frame_prior_index, tmp_path / "nested_attached.jsonl")
    nested_item = json.loads(nested_result.jsonl_path.read_text(encoding="utf-8").splitlines()[0])

    assert nested_result.summary["nested_tracklets"] == 1
    assert nested_result.summary["attached_rows"] == 1
    assert nested_result.summary["scored_rows"] == 1
    assert nested_result.summary["missing_rows"] == 1
    assert nested_item["meta"]["action_frame_prior_rows"] == 1
    assert nested_item["rows"][0]["action_frame_prior"] == str(prior_dir / "s1_000007.npy")
    assert nested_item["rows"][0]["action_frame_prior_num_tracklet_priors"] == 2
    assert nested_item["rows"][0]["action_frame_prior_score"] == 1.0
    assert nested_item["rows"][0]["action_frame_prior_bbox_max"] == 1.0
    assert "action_frame_prior" not in nested_item["rows"][1]

    flat = tmp_path / "flat.jsonl"
    flat.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"seq": "s1", "frame_id": 8, "bbox": [6, 6, 9, 9]},
                {"seq": "s2", "frame_id": 8, "bbox": [1, 1, 3, 3]},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    flat_result = attach_frame_priors_to_tracklets(flat, frame_prior_index, tmp_path / "flat_attached.jsonl")
    flat_rows = [json.loads(line) for line in flat_result.jsonl_path.read_text(encoding="utf-8").splitlines()]

    assert flat_result.summary["flat_rows"] == 2
    assert flat_result.summary["attached_rows"] == 1
    assert flat_result.summary["scored_rows"] == 1
    assert flat_rows[0]["action_frame_prior"] == str(prior_dir / "s1_000008.npy")
    assert flat_rows[0]["action_frame_prior_track_ids"] == ["t1"]
    assert np.isclose(flat_rows[0]["action_frame_prior_score"], 0.6)
    assert "action_frame_prior" not in flat_rows[1]
