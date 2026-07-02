import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

from tools.build_window_accuracy_dashboard import build_dashboard
from qstr_dronedet.evaluation.window_accuracy import (
    BoxRow,
    load_frame_manifest_antiuav_json,
    load_frame_lookup_image_dir,
    load_boxes_aot_groundtruth_json,
    load_boxes_aot_json,
    load_boxes_antiuav_json,
    load_boxes_csv,
    load_boxes_li_tetc_txt,
    load_boxes_tvd_pkl,
    load_boxes_xywh_file,
    load_boxes_yolo_dir,
    low_accuracy_segments,
    run_window_accuracy,
    sliding_window_metrics,
    write_metrics_csv,
    write_low_accuracy_segments_csv,
    write_plots,
    write_worst_windows_csv,
)


ROOT = Path(__file__).resolve().parents[1]


def test_sliding_window_metrics_uses_frame_local_matching():
    gt = [
        BoxRow("clip1", 0, (0, 0, 10, 10)),
        BoxRow("clip1", 3, (100, 100, 110, 110)),
    ]
    pred = [
        BoxRow("clip1", 0, (0, 0, 10, 10), 0.9),
        BoxRow("clip1", 1, (100, 100, 110, 110), 0.9),
    ]

    rows = sliding_window_metrics(gt, pred, fps=1.0, seconds=1.0, iou_threshold=0.5)
    by_frame = {r["frame_id"]: r for r in rows}

    assert by_frame[0]["tp"] == 1
    assert by_frame[0]["fp"] == 1
    assert by_frame[0]["fn"] == 0
    assert round(by_frame[0]["accuracy"], 3) == 0.5

    assert by_frame[2]["tp"] == 0
    assert by_frame[2]["fp"] == 1
    assert by_frame[2]["fn"] == 1
    assert by_frame[2]["accuracy"] == 0.0


def test_load_csv_and_write_outputs(tmp_path):
    gt_csv = tmp_path / "gt.csv"
    pred_csv = tmp_path / "pred.csv"
    for path, score in [(gt_csv, 1.0), (pred_csv, 0.8)]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "frame_id", "x1", "y1", "x2", "y2", "score", "label"])
            writer.writerow(["clipA", 5, 1, 1, 8, 8, score, "drone"])

    gt = load_boxes_csv(gt_csv)
    pred = load_boxes_csv(pred_csv, score_threshold=0.5, labels={"drone"})
    rows = sliding_window_metrics(gt, pred, fps=2.0, seconds=3.0)

    out_csv = tmp_path / "metrics.csv"
    plot_dir = tmp_path / "plots"
    write_metrics_csv(out_csv, rows)
    worst_csv = tmp_path / "worst.csv"
    write_worst_windows_csv(worst_csv, rows, per_video=3)
    segments_csv = tmp_path / "low_segments.csv"
    segments = write_low_accuracy_segments_csv(segments_csv, rows, fps=2.0, threshold=0.99)
    plot_paths = write_plots(plot_dir, rows)

    assert out_csv.exists()
    assert worst_csv.exists()
    assert segments_csv.exists()
    assert segments == low_accuracy_segments(rows, fps=2.0, threshold=0.99)
    assert plot_paths
    assert (plot_dir / "index.html").exists()


def test_frame_manifest_scores_every_listed_frame_including_empty_edges(tmp_path):
    gt_csv = tmp_path / "gt.csv"
    pred_csv = tmp_path / "pred.csv"
    manifest_csv = tmp_path / "frames.csv"
    gt_csv.write_text(
        "video,frame_id,x1,y1,x2,y2,score,label\nclipA,5,1,1,8,8,1,drone\n",
        encoding="utf-8",
    )
    pred_csv.write_text(
        "video,frame_id,x1,y1,x2,y2,score,label\nclipA,5,1,1,8,8,0.9,drone\n",
        encoding="utf-8",
    )
    manifest_csv.write_text("video,start_frame,end_frame\nclipA,0,10\n", encoding="utf-8")

    summary = run_window_accuracy(
        gt=gt_csv,
        pred=pred_csv,
        out_dir=tmp_path / "curves",
        fps=1.0,
        gt_format="csv",
        pred_format="csv",
        frame_manifest=manifest_csv,
        frame_manifest_format="csv",
        window_seconds=0.0,
        iou_threshold=0.5,
    )
    rows = list(csv.DictReader((tmp_path / "curves" / "per_frame_window_metrics.csv").open()))

    assert summary["frames"] == 11
    assert summary["frame_manifest_frames"] == 11
    assert rows[0]["frame_id"] == "0"
    assert rows[-1]["frame_id"] == "10"
    assert rows[0]["accuracy"] == "1.000000"
    assert rows[5]["accuracy"] == "1.000000"


def test_aot_predictions_use_image_dir_manifest_lookup_for_hash_names(tmp_path):
    frames = tmp_path / "frames" / "flightA"
    frames.mkdir(parents=True)
    (frames / "hash_b.png").write_bytes(b"")
    (frames / "hash_a.png").write_bytes(b"")
    gt_csv = tmp_path / "gt.csv"
    gt_csv.write_text("video,frame_id,x1,y1,x2,y2,score,label\n", encoding="utf-8")
    pred = tmp_path / "pred" / "flightA"
    pred.mkdir(parents=True)
    pred.joinpath("result.json").write_text(
        json.dumps(
            [
                {
                    "img_name": "hash_a.png",
                    "detections": [{"x": 50.0, "y": 50.0, "w": 20.0, "h": 20.0, "s": 0.9}],
                }
            ]
        ),
        encoding="utf-8",
    )

    lookup = load_frame_lookup_image_dir(tmp_path / "frames")
    summary = run_window_accuracy(
        gt=gt_csv,
        pred=tmp_path / "pred",
        out_dir=tmp_path / "curves_hash",
        fps=1.0,
        gt_format="csv",
        pred_format="aot-json",
        frame_manifest=tmp_path / "frames",
        frame_manifest_format="image-dir",
        window_seconds=0.0,
        score_threshold=0.25,
    )
    rows = list(csv.DictReader((tmp_path / "curves_hash" / "per_frame_window_metrics.csv").open()))

    assert lookup["hash_a.png"] == ("flightA", 0)
    assert lookup["hash_b.png"] == ("flightA", 1)
    assert summary["frames"] == 2
    assert rows[0]["frame_id"] == "0"
    assert rows[0]["fp"] == "1"
    assert rows[1]["frame_id"] == "1"
    assert rows[1]["fp"] == "0"


def test_load_yolo_dir_and_aot_json_formats(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "Clip_001_00000.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    results = tmp_path / "winner"
    (results / "Clip_001").mkdir(parents=True)
    (results / "Clip_001" / "result.json").write_text(
        json.dumps(
            [
                {
                    "img_name": "Clip_001_00001.png",
                    "detections": [{"x": 50.0, "y": 50.0, "w": 20.0, "h": 20.0, "s": 0.9}],
                }
            ]
        ),
        encoding="utf-8",
    )

    gt = load_boxes_yolo_dir(labels, frame_offset=1, img_size=(100.0, 100.0))
    pred = load_boxes_aot_json(results, score_threshold=0.5)
    rows = sliding_window_metrics(gt, pred, fps=1.0, seconds=0.0, iou_threshold=0.5)

    assert gt[0].video == "Clip_001"
    assert gt[0].frame_id == 1
    assert pred[0].video == "Clip_001"
    assert pred[0].frame_id == 1
    assert rows[0]["accuracy"] == 1.0


def test_load_aot_groundtruth_json_pairs_with_winner_result_json(tmp_path):
    gt_root = tmp_path / "aot" / "ImageSets"
    gt_root.mkdir(parents=True)
    gt_root.joinpath("groundtruth.json").write_text(
        json.dumps(
            {
                "metadata": {"fps": 10},
                "samples": {
                    "flightA": {
                        "metadata": {"resolution": {"width": 100, "height": 100}},
                        "entities": [
                            {
                                "blob": {"frame": 7},
                                "id": "object-1",
                                "bb": [40.0, 40.0, 20.0, 20.0],
                                "flight_id": "flightA",
                                "img_name": "1550844897919368155280dc81adbb3420cab502fb88d6abf84.png",
                            },
                            {
                                "blob": {"frame": 8},
                                "bb": [10.0, 10.0, 5.0, 5.0],
                                "flight_id": "flightA",
                                "img_name": "1550844897919368155280dc81adbb3420cab502fb88d6abf84.png",
                            },
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    pred_root = tmp_path / "results" / "flightA"
    pred_root.mkdir(parents=True)
    pred_root.joinpath("result.json").write_text(
        json.dumps(
            [
                {
                    "img_name": "1550844897919368155280dc81adbb3420cab502fb88d6abf84.png",
                    "frame": 7,
                    "detections": [{"x": 50.0, "y": 50.0, "w": 20.0, "h": 20.0, "s": 0.95}],
                }
            ]
        ),
        encoding="utf-8",
    )

    gt = load_boxes_aot_groundtruth_json(tmp_path / "aot")
    pred = load_boxes_aot_json(tmp_path / "results", score_threshold=0.5)
    rows = sliding_window_metrics(gt, pred, fps=10.0, seconds=0.0, iou_threshold=0.5)

    assert len(gt) == 1
    assert gt[0].video == "flightA"
    assert gt[0].frame_id == 7
    assert gt[0].bbox == (40.0, 40.0, 60.0, 60.0)
    assert pred[0].video == "flightA"
    assert pred[0].frame_id == 7
    assert rows[0]["accuracy"] == 1.0

    aggregate_pred = tmp_path / "aggregate_result.json"
    aggregate_pred.write_text(
        json.dumps(
            [
                {
                    "img_name": "1550844897919368155280dc81adbb3420cab502fb88d6abf84.png",
                    "detections": [{"x": 50.0, "y": 50.0, "w": 20.0, "h": 20.0, "s": 0.95}],
                }
            ]
        ),
        encoding="utf-8",
    )
    summary = run_window_accuracy(
        gt=tmp_path / "aot",
        pred=aggregate_pred,
        out_dir=tmp_path / "curves",
        fps=10.0,
        gt_format="aot-gt-json",
        pred_format="aot-json",
        window_seconds=0.0,
        iou_threshold=0.5,
    )
    assert summary["by_video"]["flightA"]["mean_accuracy"] == 1.0


def test_tracker_and_li_tetc_formats(tmp_path):
    anti_root = tmp_path / "antiuav"
    seq_dir = anti_root / "seqA"
    seq_dir.mkdir(parents=True)
    (anti_root / "list.txt").write_text("seqA\n", encoding="utf-8")
    (seq_dir / "IR_label.json").write_text(
        json.dumps({"gt_rect": [[10, 20, 5, 6], [0, 0, 0, 0], [11, 21, 5, 6]], "exist": [1, 0, 1]}),
        encoding="utf-8",
    )
    pred_dir = tmp_path / "edtc_results"
    pred_dir.mkdir()
    (pred_dir / "seqA.txt").write_text("10\t20\t5\t6\n0\t0\t0\t0\n11\t21\t5\t6\n", encoding="utf-8")

    gt = load_boxes_antiuav_json(anti_root)
    pred = load_boxes_xywh_file(pred_dir)
    rows = sliding_window_metrics(gt, pred, fps=1.0, seconds=0.0, iou_threshold=0.5)

    assert len(gt) == 2
    assert len(pred) == 2
    assert rows[0]["accuracy"] == 1.0

    li_gt = tmp_path / "Video_1_gt.txt"
    li_pred = tmp_path / "1_dt.txt"
    li_gt.write_text("time_layer: 12 detections: (181, 1229, 192, 1242), \n", encoding="utf-8")
    li_pred.write_text("time_layer: 12 detections: (181, 1229, 192, 1242), \n", encoding="utf-8")
    li_rows = sliding_window_metrics(
        load_boxes_li_tetc_txt(li_gt),
        load_boxes_li_tetc_txt(li_pred),
        fps=1.0,
        seconds=0.0,
        iou_threshold=0.5,
    )
    assert li_rows[0]["video"] == "Video_1"
    assert li_rows[0]["accuracy"] == 1.0


def test_antiuav_frame_manifest_includes_non_existing_frames(tmp_path):
    anti_root = tmp_path / "antiuav"
    seq_dir = anti_root / "seqA"
    seq_dir.mkdir(parents=True)
    (anti_root / "list.txt").write_text("seqA\n", encoding="utf-8")
    (seq_dir / "IR_label.json").write_text(
        json.dumps({"gt_rect": [[10, 20, 5, 6], [0, 0, 0, 0], [11, 21, 5, 6]], "exist": [1, 0, 1]}),
        encoding="utf-8",
    )

    frames = load_frame_manifest_antiuav_json(anti_root)

    assert frames["seqA"] == {1, 2, 3}


def test_transvisdrone_predictionsgt_pkl_format(tmp_path):
    pkl = tmp_path / "predictionsgt_split_0.pkl"
    with pkl.open("wb") as f:
        pickle.dump(
            {
                "Clip_041_00007": {
                    "labels": [{"bbox": [10, 20, 30, 40], "category_id": 0}],
                    "detections": [{"bbox": [10, 20, 30, 40], "score": 0.91, "category_id": 0}],
                },
                "Clip_041_00008": {
                    "labels": [{"bbox": [50, 60, 70, 80], "category_id": 0}],
                    "detections": [{"bbox": [1, 2, 3, 4], "score": 0.10, "category_id": 0}],
                },
            },
            f,
        )

    gt = load_boxes_tvd_pkl(pkl, kind="gt")
    pred = load_boxes_tvd_pkl(pkl, kind="pred", score_threshold=0.25)
    rows = sliding_window_metrics(gt, pred, fps=1.0, seconds=0.0, iou_threshold=0.5)
    by_frame = {row["frame_id"]: row for row in rows}

    assert gt[0].video == "Clip_041"
    assert gt[0].frame_id == 7
    assert len(gt) == 2
    assert len(pred) == 1
    assert by_frame[7]["accuracy"] == 1.0
    assert by_frame[8]["accuracy"] == 0.0


def test_batch_manifest_cli_generates_per_paper_outputs(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "Clip_001_00001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (pred_dir / "Clip_001_00001.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "out_root": "out",
                "defaults": {"fps": 1, "window_seconds": 0, "iou": 0.5, "score_threshold": 0.25},
                "runs": [
                    {
                        "name": "paper_fixture",
                        "method": "YOLO_fixture",
                        "gt": "gt",
                        "gt_format": "yolo-dir",
                        "pred": "pred",
                        "pred_format": "yolo-dir",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_paper_window_accuracy_batch.py"),
            "--manifest",
            str(manifest),
            "--base-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "complete: paper_fixture" in proc.stdout
    batch_summary = json.loads((tmp_path / "out" / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["complete"] == 1
    assert (tmp_path / "out" / "index.html").exists()
    assert (tmp_path / "out" / "dashboard.html").exists()
    run_summary = json.loads((tmp_path / "out" / "paper_fixture" / "summary.json").read_text(encoding="utf-8"))
    assert run_summary["by_video"]["Clip_001"]["mean_accuracy"] == 1.0
    assert (tmp_path / "out" / "paper_fixture" / "worst_windows.csv").exists()
    assert (tmp_path / "out" / "paper_fixture" / "low_accuracy_segments.csv").exists()
    assert (tmp_path / "out" / "paper_fixture" / "plots" / "Clip_001_window_metrics.svg").exists()


def test_pipeline_runs_ready_manifest_and_writes_reports(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "Clip_002_00001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (pred_dir / "Clip_002_00001.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "out_root": "curves",
                "defaults": {"fps": 1, "window_seconds": 0, "iou": 0.5},
                "runs": [
                    {
                        "name": "ready_fixture",
                        "method": "YOLO_fixture",
                        "gt": "gt",
                        "gt_format": "yolo-dir",
                        "pred": "pred",
                        "pred_format": "yolo-dir",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_paper_window_accuracy_pipeline.py"),
            "--skip-pull",
            "--manifest",
            str(manifest),
            "--base-dir",
            str(tmp_path),
            "--json",
            str(tmp_path / "pipeline.json"),
            "--markdown",
            str(tmp_path / "pipeline.md"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "complete_curves" in proc.stdout
    pipeline = json.loads((tmp_path / "pipeline.json").read_text(encoding="utf-8"))
    assert pipeline["steps"]["audit_before"]["counts"]["ready_to_run"] == 1
    assert pipeline["steps"]["audit_after"]["counts"]["complete_curves"] == 1
    assert (tmp_path / "curves" / "ready_fixture" / "plots" / "Clip_002_window_metrics.svg").exists()
    assert (tmp_path / "curves" / "ready_fixture" / "worst_windows.csv").exists()
    assert (tmp_path / "curves" / "ready_fixture" / "low_accuracy_segments.csv").exists()
    assert (tmp_path / "pipeline.md").exists()


def test_window_accuracy_dashboard_summarizes_worst_windows(tmp_path):
    run_dir = tmp_path / "curves" / "paper_run"
    plots = run_dir / "plots"
    plots.mkdir(parents=True)
    (plots / "Clip_001_window_metrics.svg").write_text("<svg></svg>\n", encoding="utf-8")
    (plots / "index.html").write_text("<html></html>\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "name": "paper_run",
                "method": "PaperX",
                "videos": 1,
                "frames": 3,
                "by_video": {
                    "Clip_001": {
                        "frames": 3,
                        "mean_accuracy": 0.4,
                        "min_accuracy": 0.1,
                        "mean_recall": 0.5,
                        "min_recall": 0.2,
                        "worst_frame_by_accuracy": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "worst_windows.csv").write_text(
        "video,rank,frame_id,window_start_frame,window_end_frame,gt,pred,tp,fp,fn,precision,recall,f1,accuracy\n"
        "Clip_001,1,2,0,4,10,3,1,2,9,0.333333,0.1,0.15,0.083333\n",
        encoding="utf-8",
    )
    (run_dir / "low_accuracy_segments.csv").write_text(
        "video,rank,start_frame,end_frame,start_time_sec,end_time_sec,center_frames,window_start_frame,window_end_frame,min_accuracy,mean_accuracy,mean_precision,mean_recall,mean_f1,worst_frame,gt,pred,tp,fp,fn\n"
        "Clip_001,1,2,4,0.066667,0.133333,3,0,6,0.083333,0.200000,0.333333,0.100000,0.150000,2,30,9,3,6,27\n",
        encoding="utf-8",
    )
    batch = {
        "out_root": str(tmp_path / "curves"),
        "runs": [
            {
                "name": "paper_run",
                "method": "PaperX",
                "status": "complete",
                "out": str(run_dir),
                "summary": str(run_dir / "summary.json"),
                "worst_windows_csv": str(run_dir / "worst_windows.csv"),
                "low_accuracy_segments_csv": str(run_dir / "low_accuracy_segments.csv"),
                "plot_index": str(plots / "index.html"),
            },
            {
                "name": "missing_run",
                "method": "PaperY",
                "status": "skipped_missing",
                "missing": ["gt", "pred"],
                "gt": "/missing/gt",
                "gt_format": "yolo-dir",
                "pred": "/missing/pred",
                "pred_format": "aot-json",
                "out": str(tmp_path / "curves" / "missing_run"),
            }
        ],
    }

    dashboard = build_dashboard(batch)
    html = dashboard.read_text(encoding="utf-8")

    assert "Worst +/-3s Windows" in html
    assert "Missing / Skipped Runs" in html
    assert "Continuous Low-Accuracy Segments" in html
    assert "PaperX" in html
    assert "PaperY" in html
    assert "missing_run" in html
    assert "Clip_001" in html
    assert "0.083" in html


def test_smoke_builder_generates_all_paper_curve_outputs(tmp_path):
    out_root = tmp_path / "smoke"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_paper_window_accuracy_smoke.py"),
            "--out-root",
            str(out_root),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "complete=6" in proc.stdout
    assert (out_root / "index.html").exists()
    batch_summary = json.loads((out_root / "curves" / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["complete"] == 6
    for run in batch_summary["runs"]:
        assert run["status"] == "complete"
        assert Path(run["plot_index"]).exists()
        assert Path(run["csv"]).exists()
        assert Path(run["worst_windows_csv"]).exists()
        assert Path(run["low_accuracy_segments_csv"]).exists()

    audit_json = tmp_path / "audit.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "audit_paper_window_accuracy_readiness.py"),
            "--manifest",
            str(out_root / "smoke_manifest.json"),
            "--base-dir",
            str(out_root),
            "--json",
            str(audit_json),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "complete_curves" in proc.stdout
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    assert audit["counts"]["complete_curves"] == 6
