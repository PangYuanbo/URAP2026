import subprocess
import sys
import json
import pickle
import zipfile
from pathlib import Path

from PIL import Image

from tools.download_aicrowd_lfs_weights import basic_auth_header, lfs_batch_payload
from tools.download_nps_videos import extract_clips, parse_clip_ids as parse_nps_clip_ids
from tools.inventory_aicrowd_lfs_weights import parse_lfs_pointer
from tools.inventory_external_window_accuracy_sources import extract_drive_items
from tools.prepare_aicrowd_nps_flight_dirs import clip_id_from_frame, prepare_flight_dirs
from tools.prepare_transvisdrone_nps import parse_clip_ids as parse_tvd_nps_clip_ids


ROOT = Path(__file__).resolve().parents[1]


def test_parse_aicrowd_lfs_pointer():
    pointer = "\n".join(
        [
            "version https://git-lfs.github.com/spec/v1",
            "oid sha256:3778eea4c618650c7089e10a9e55a1266087ae885bda247a85c0b1c41bd3ecc2",
            "size 85273509",
            "",
        ]
    )

    parsed = parse_lfs_pointer(pointer)

    assert parsed["oid_sha256"] == "3778eea4c618650c7089e10a9e55a1266087ae885bda247a85c0b1c41bd3ecc2"
    assert parsed["size_bytes"] == 85273509


def test_aicrowd_lfs_batch_helpers():
    payload = json.loads(
        lfs_batch_payload(
            [
                {
                    "oid_sha256": "3778eea4c618650c7089e10a9e55a1266087ae885bda247a85c0b1c41bd3ecc2",
                    "size_bytes": 85273509,
                }
            ]
        ).decode("utf-8")
    )

    assert payload["operation"] == "download"
    assert payload["objects"][0]["size"] == 85273509
    assert payload["objects"][0]["oid"] == "3778eea4c618650c7089e10a9e55a1266087ae885bda247a85c0b1c41bd3ecc2"
    assert basic_auth_header("oauth2", "token").startswith("Basic ")


def test_pull_paper_repos_dry_run_lists_public_and_private_repos():
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "pull_paper_repos.py"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "YOLOMG" in proc.stdout
    assert "TransVisDrone" in proc.stdout
    assert "Dogfight_Drone_Detection" in proc.stdout
    assert "would_snapshot: AICrowd_Winner_v022" in proc.stdout


def test_extract_drive_items_from_public_folder_html():
    html = '''
    <div data-id="1FxMl9zQsEGanzYMfj_yE7DpFQOHIa3Lc"><strong class="DNoYtb">UAVTrackEH.pth.tar</strong></div>
    <div>menu</div> null,[[null,"1FxMl9zQsEGanzYMfj_yE7DpFQOHIa3Lc"],0]
    <strong>validation.zip</strong></span></div>
    <div>menu</div> null,[[null,"1hPuGy2jCPqh4qoMeHBtwjYPDtEJ-18FH"],0]
    '''

    items = extract_drive_items(html)

    assert items == [
        {
            "name": "UAVTrackEH.pth.tar",
            "id": "1FxMl9zQsEGanzYMfj_yE7DpFQOHIa3Lc",
            "download_url": "https://drive.google.com/uc?export=download&id=1FxMl9zQsEGanzYMfj_yE7DpFQOHIa3Lc",
        },
        {
            "name": "validation.zip",
            "id": "1hPuGy2jCPqh4qoMeHBtwjYPDtEJ-18FH",
            "download_url": "https://drive.google.com/uc?export=download&id=1hPuGy2jCPqh4qoMeHBtwjYPDtEJ-18FH",
        },
    ]


def test_parse_nps_clip_ranges():
    assert parse_nps_clip_ids("37-40,45") == [37, 38, 39, 40, 45]
    assert parse_tvd_nps_clip_ids("37-38,40") == {37, 38, 40}


def test_extract_selected_nps_clips_from_zip(tmp_path):
    zip_path = tmp_path / "Videos.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("Videos/Clip_37.mov", b"clip37")
        archive.writestr("Videos/Clip_38.mov", b"clip38")
        archive.writestr("Videos/readme.txt", b"ignore")

    extracted = extract_clips(zip_path, tmp_path / "out", [38, 37])

    assert [item["clip_id"] for item in extracted] == [37, 38]
    assert (tmp_path / "out" / "Clip_37.mov").read_bytes() == b"clip37"
    assert (tmp_path / "out" / "Clip_38.mov").read_bytes() == b"clip38"


def test_readiness_audit_reports_missing_inputs_for_example_manifest(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "audit_paper_window_accuracy_readiness.py"),
            "--manifest",
            str(ROOT / "data_templates" / "paper_window_accuracy_runs.example.json"),
            "--json",
            str(tmp_path / "audit.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "missing_inputs" in proc.stdout
    assert "AICrowd_Winner_v022" in proc.stdout
    assert (tmp_path / "audit.json").exists()


def test_readiness_audit_accepts_aicrowd_api_snapshot_repo(tmp_path):
    repo = (
        tmp_path
        / "papers"
        / "AICrowd_AOT_Challenge_Winner"
        / "submission-v022"
        / "airborne-detection-starter-kit-submission-v022"
    )
    repo.mkdir(parents=True)
    (repo / ".urap_snapshot.json").write_text('{"ref": "submission-v022"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "out_root": "out",
                "defaults": {"fps": 30},
                "runs": [
                    {
                        "name": "aicrowd_fixture",
                        "method": "AICrowd_Winner_v022",
                        "gt": "missing_gt",
                        "gt_format": "yolo-dir",
                        "pred": "missing_pred",
                        "pred_format": "aot-json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "audit_paper_window_accuracy_readiness.py"),
            "--manifest",
            str(manifest),
            "--base-dir",
            str(tmp_path),
            "--json",
            str(tmp_path / "audit.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "missing_inputs: AICrowd_Winner_v022 / aicrowd_fixture missing=gt,pred" in proc.stdout
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    run = audit["runs"][0]
    assert run["repo"]["present"] is True
    assert run["repo"]["kind"] == "api_snapshot"
    assert run["status"] == "missing_inputs"


def test_readiness_audit_rejects_empty_yolo_label_directories(tmp_path):
    (tmp_path / "papers" / "YOLOMG" / ".git").mkdir(parents=True)
    (tmp_path / "gt").mkdir()
    (tmp_path / "pred").mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "out_root": "out",
                "defaults": {"fps": 30},
                "runs": [
                    {
                        "name": "empty_yolo_dirs",
                        "method": "YOLOMG",
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
            str(ROOT / "tools" / "audit_paper_window_accuracy_readiness.py"),
            "--manifest",
            str(manifest),
            "--base-dir",
            str(tmp_path),
            "--json",
            str(tmp_path / "audit.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "missing_inputs: YOLOMG / empty_yolo_dirs missing=gt,pred" in proc.stdout
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["counts"] == {"missing_inputs": 1}


def test_gap_report_lists_missing_inputs_and_candidate_paths(tmp_path):
    (tmp_path / "papers" / "YOLOMG" / ".git").mkdir(parents=True)
    candidate_gt = tmp_path / "datasets" / "ARD100_YOLOMG" / "labels" / "test"
    candidate_gt.mkdir(parents=True)
    (candidate_gt / "Clip_001_00001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    fixture_gt = tmp_path / "runs" / "window_accuracy" / "yolomg_test_images_dataset" / "labels"
    fixture_gt.mkdir(parents=True)
    (fixture_gt / "Clip_001_00001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "out_root": "out",
                "defaults": {"fps": 30},
                "runs": [
                    {
                        "name": "missing_yolo_run",
                        "method": "YOLOMG",
                        "gt": "D:/URAP_datasets/ARD100_YOLOMG/labels/test",
                        "gt_format": "yolo-dir",
                        "pred": "papers/YOLOMG/runs/val/missing/labels",
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
            str(ROOT / "tools" / "write_paper_window_accuracy_gap_report.py"),
            "--manifest",
            str(manifest),
            "--base-dir",
            str(tmp_path),
            "--json",
            str(tmp_path / "gap.json"),
            "--markdown",
            str(tmp_path / "gap.md"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "gaps=1" in proc.stdout
    report = json.loads((tmp_path / "gap.json").read_text(encoding="utf-8"))
    gap = report["gaps"][0]
    assert gap["name"] == "missing_yolo_run"
    assert "datasets/ARD100_YOLOMG/labels/test" in gap["gt"]["candidates"]
    assert "runs/window_accuracy/yolomg_test_images_dataset/labels" not in gap["gt"]["candidates"]
    assert gap["pred"]["candidates"] == []
    assert (tmp_path / "gap.md").exists()


def test_gap_report_includes_generation_commands_for_known_missing_run(tmp_path):
    (tmp_path / "papers" / "YOLOMG" / ".git").mkdir(parents=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "out_root": "out",
                "defaults": {"fps": 30},
                "runs": [
                    {
                        "name": "yolomg_ard100_test",
                        "method": "YOLOMG",
                        "gt": "D:/URAP_datasets/ARD100_YOLOMG/labels/test",
                        "gt_format": "yolo-dir",
                        "pred": "papers/YOLOMG/runs/val/missing/labels",
                        "pred_format": "yolo-dir",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "write_paper_window_accuracy_gap_report.py"),
            "--manifest",
            str(manifest),
            "--base-dir",
            str(tmp_path),
            "--json",
            str(tmp_path / "gap.json"),
            "--markdown",
            str(tmp_path / "gap.md"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads((tmp_path / "gap.json").read_text(encoding="utf-8"))
    commands = report["gaps"][0]["generation_commands"]
    assert any("start_yolo_eval_window_accuracy_detached.ps1" in command for command in commands)
    assert any("monitor_yolo_eval_window_accuracy.ps1" in command for command in commands)
    assert "### Generation commands" in (tmp_path / "gap.md").read_text(encoding="utf-8")


def test_prepare_aicrowd_nps_flight_dirs_groups_flat_clip_frames(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "Clip_001_00001.png").write_bytes(b"a")
    (frames / "Clip_001_00002.png").write_bytes(b"b")
    (frames / "Clip_040_00001.png").write_bytes(b"c")
    (frames / "not_a_clip.png").write_bytes(b"d")

    assert clip_id_from_frame(frames / "Clip_001_00001.png") == "Clip_001"
    summary = prepare_flight_dirs(frames, tmp_path / "prepared", mode="copy")

    assert summary["num_input_images"] == 4
    assert summary["num_clips"] == 2
    assert summary["num_unmatched"] == 1
    assert (tmp_path / "prepared" / "Clip_001" / "Clip_001_00001.png").is_file()
    assert (tmp_path / "prepared" / "Clip_040" / "Clip_040_00001.png").is_file()


def test_run_edtc_tracker_window_accuracy_dry_run_reports_paths(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_edtc_tracker_window_accuracy.py"),
            "--dataset-root",
            str(tmp_path / "AntiUAV600"),
            "--out",
            str(tmp_path / "curves"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["results_dir"].endswith("tracking_results/uavtrack_eh/urap_window_accuracy")
    assert report["generated_config"].endswith("experiments/uavtrack_eh/urap_window_accuracy.yaml")
    assert "tracking/test.py" in " ".join(report["tracker_command"])


def test_goal_audit_reports_incomplete_objective_with_generation_commands(tmp_path):
    (tmp_path / "papers" / "YOLOMG" / ".git").mkdir(parents=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "out_root": "out",
                "defaults": {"fps": 30},
                "runs": [
                    {
                        "name": "yolomg_ard100_test",
                        "method": "YOLOMG",
                        "gt": "D:/URAP_datasets/ARD100_YOLOMG/labels/test",
                        "gt_format": "yolo-dir",
                        "pred": "papers/YOLOMG/runs/val/missing/labels",
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
            str(ROOT / "tools" / "audit_paper_window_accuracy_goal.py"),
            "--manifest",
            str(manifest),
            "--base-dir",
            str(tmp_path),
            "--json",
            str(tmp_path / "goal.json"),
            "--markdown",
            str(tmp_path / "goal.md"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "status=incomplete" in proc.stdout
    report = json.loads((tmp_path / "goal.json").read_text(encoding="utf-8"))
    gates = {gate["name"]: gate["ok"] for gate in report["gates"]}
    assert gates["all_manifest_runs_complete"] is False
    assert gates["missing_runs_have_generation_commands"] is True
    assert report["methods"]["YOLOMG"]["generation_command_count"] == 2
    assert (tmp_path / "goal.md").exists()


def test_readiness_audit_accepts_single_li_tetc_txt_files(tmp_path):
    (tmp_path / "papers" / "Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking" / ".git").mkdir(parents=True)
    gt = tmp_path / "Video_14_gt.txt"
    pred = tmp_path / "14_dt.txt"
    gt.write_text("time_layer: 1 detections: (1, 2, 3, 4), \n", encoding="utf-8")
    pred.write_text("time_layer: 1 detections: (1, 2, 3, 4), \n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "out_root": "out",
                "defaults": {"fps": 29},
                "runs": [
                    {
                        "name": "li_single_file",
                        "method": "Li_TETC_NPS",
                        "gt": "Video_14_gt.txt",
                        "gt_format": "li-tetc-txt",
                        "pred": "14_dt.txt",
                        "pred_format": "li-tetc-txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "audit_paper_window_accuracy_readiness.py"),
            "--manifest",
            str(manifest),
            "--base-dir",
            str(tmp_path),
            "--json",
            str(tmp_path / "audit.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ready_to_run: Li_TETC_NPS / li_single_file" in proc.stdout
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["counts"] == {"ready_to_run": 1}


def test_discover_paper_runs_finds_available_outputs(tmp_path):
    gt = tmp_path / "datasets" / "ARD100_YOLOMG" / "labels" / "test"
    pred = tmp_path / "papers" / "YOLOMG" / "runs" / "val" / "run1" / "labels"
    gt.mkdir(parents=True)
    pred.mkdir(parents=True)
    (gt / "Clip_001_00001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (pred / "Clip_001_00001.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")

    manifest_out = tmp_path / "manifest.json"
    report_md = tmp_path / "report.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "discover_paper_window_accuracy_runs.py"),
            "--base-dir",
            str(tmp_path),
            "--manifest-out",
            str(manifest_out),
            "--report-json",
            str(tmp_path / "report.json"),
            "--report-md",
            str(report_md),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "runs=1" in proc.stdout
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifest["runs"][0]["method"] == "YOLOMG"
    assert manifest["runs"][0]["gt"] == "datasets/ARD100_YOLOMG/labels/test"
    assert manifest["runs"][0]["pred"] == "papers/YOLOMG/runs/val/run1/labels"
    assert report_md.exists()


def test_discover_paper_runs_pairs_li_tetc_prediction_with_matching_gt(tmp_path):
    gt = (
        tmp_path
        / "papers"
        / "Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking"
        / "Data"
        / "Annotation_update_180925"
        / "Video_14_gt.txt"
    )
    pred = (
        tmp_path
        / "papers"
        / "Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking"
        / "Experiment_Results"
        / "Final"
        / "txt"
        / "14_dt.txt"
    )
    gt_40 = gt.with_name("Video_40_gt.txt")
    pred_40 = pred.with_name("40_dt.txt")
    gt.parent.mkdir(parents=True)
    pred.parent.mkdir(parents=True)
    gt.write_text("time_layer: 1 detections: (1, 2, 3, 4), \n", encoding="utf-8")
    pred.write_text("time_layer: 1 detections: (1, 2, 3, 4), \n", encoding="utf-8")
    gt_40.write_text("time_layer: 1 detections: (1, 2, 3, 4), \n", encoding="utf-8")
    pred_40.write_text("time_layer: 1 detections: (1, 2, 3, 4), \n", encoding="utf-8")

    manifest_out = tmp_path / "manifest.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "discover_paper_window_accuracy_runs.py"),
            "--base-dir",
            str(tmp_path),
            "--manifest-out",
            str(manifest_out),
            "--report-json",
            str(tmp_path / "report.json"),
            "--report-md",
            str(tmp_path / "report.md"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "runs=2" in proc.stdout
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    runs = {run["name"]: run for run in manifest["runs"]}
    assert runs["li_tetc_video_14"]["gt"].endswith("Video_14_gt.txt")
    assert runs["li_tetc_video_14"]["pred"].endswith("14_dt.txt")
    assert runs["li_tetc_video_40"]["gt"].endswith("Video_40_gt.txt")
    assert runs["li_tetc_video_40"]["pred"].endswith("40_dt.txt")


def test_discover_paper_runs_finds_transvisdrone_predictionsgt_pkl(tmp_path):
    model_root = (
        tmp_path
        / "papers"
        / "TransVisDrone"
        / "runs"
        / "val"
        / "NPS"
        / "model_run"
    )
    pkl = (
        model_root
        / "best_augment_full_save"
        / "predictionsgt"
        / "predictionsgt_split_0.pkl"
    )
    pkl.parent.mkdir(parents=True)
    with pkl.open("wb") as f:
        pickle.dump(
            {
                "Clip_041_00001": {
                    "labels": [{"bbox": [10, 10, 20, 20], "category_id": 0}],
                    "detections": [{"bbox": [10, 10, 20, 20], "score": 0.9, "category_id": 0}],
                }
            },
            f,
        )
    extra_pred = model_root / "best_coco" / "best_predictions.pkl"
    extra_pred.parent.mkdir(parents=True)
    with extra_pred.open("wb") as f:
        pickle.dump([{"image_id": "Clip_041_00001", "category_id": 0, "bbox": [10, 10, 10, 10], "score": 0.9}], f)

    manifest_out = tmp_path / "manifest.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "discover_paper_window_accuracy_runs.py"),
            "--base-dir",
            str(tmp_path),
            "--manifest-out",
            str(manifest_out),
            "--report-json",
            str(tmp_path / "report.json"),
            "--report-md",
            str(tmp_path / "report.md"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "runs=2" in proc.stdout
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    runs = {run["name"]: run for run in manifest["runs"]}
    run = runs["transvisdrone_pkl_nps_best_augment_full_save"]
    assert run["method"] == "TransVisDrone"
    assert run["gt"] == run["pred"]
    assert run["gt_format"] == "tvd-pkl-gt"
    assert run["pred_format"] == "tvd-pkl-pred"
    extra = runs["transvisdrone_pkl_nps_best_coco"]
    assert extra["gt"] == run["gt"]
    assert extra["pred"].endswith("best_coco/best_predictions.pkl")


def test_discover_paper_runs_finds_esod_fixture_outputs(tmp_path):
    gt = tmp_path / "runs" / "window_accuracy" / "yolomg_test_images_dataset" / "labels"
    pred = tmp_path / "runs" / "window_accuracy" / "esod_test_images_eval" / "eval" / "official_test" / "labels"
    gt.mkdir(parents=True)
    pred.mkdir(parents=True)
    (gt / "Clip_001_00001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (pred / "Clip_001_00001.txt").write_text("2 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")

    manifest_out = tmp_path / "manifest.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "discover_paper_window_accuracy_runs.py"),
            "--base-dir",
            str(tmp_path),
            "--manifest-out",
            str(manifest_out),
            "--report-json",
            str(tmp_path / "report.json"),
            "--report-md",
            str(tmp_path / "report.md"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "runs=1" in proc.stdout
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    run = manifest["runs"][0]
    assert run["name"] == "esod_official_test"
    assert run["method"] == "ESOD"
    assert run["gt"] == "runs/window_accuracy/yolomg_test_images_dataset/labels"
    assert run["pred"] == "runs/window_accuracy/esod_test_images_eval/eval/official_test/labels"


def test_discover_paper_runs_finds_edtc_yolo_fixture_outputs(tmp_path):
    gt = tmp_path / "runs" / "window_accuracy" / "yolomg_test_images_dataset" / "labels"
    pred = tmp_path / "runs" / "window_accuracy" / "edtc_yolo_test_images_eval" / "eval" / "official_val" / "labels"
    gt.mkdir(parents=True)
    pred.mkdir(parents=True)
    (gt / "Clip_001_00001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (pred / "Clip_001_00001.txt").write_text("0 0.5 0.5 0.2 0.2 0.001\n", encoding="utf-8")

    manifest_out = tmp_path / "manifest.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "discover_paper_window_accuracy_runs.py"),
            "--base-dir",
            str(tmp_path),
            "--manifest-out",
            str(manifest_out),
            "--report-json",
            str(tmp_path / "report.json"),
            "--report-md",
            str(tmp_path / "report.md"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "runs=1" in proc.stdout
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    runs = {run["name"]: run for run in manifest["runs"]}
    run = runs["edtc_official_val"]
    assert run["method"] == "EDTC"
    assert run["gt"] == "runs/window_accuracy/yolomg_test_images_dataset/labels"
    assert run["pred"] == "runs/window_accuracy/edtc_yolo_test_images_eval/eval/official_val/labels"
    assert run["score_threshold"] == 0.001


def test_build_yolomg_test_images_dataset_creates_yolo_labels(tmp_path):
    from tools.build_yolomg_test_images_dataset import build_dataset

    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "mask").mkdir()
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(src / "images" / "frame_0001.jpg")
    mask = Image.new("L", (100, 80), color=0)
    for x in range(40, 46):
        for y in range(30, 36):
            mask.putpixel((x, y), 255)
    mask.save(src / "mask" / "frame_0001.jpg")

    report = build_dataset(src, tmp_path / "out")

    assert report["images"] == 1
    assert report["labeled_images"] == 1
    assert report["boxes"] >= 1
    label = (tmp_path / "out" / "labels" / "frame_0001.txt").read_text(encoding="utf-8").strip()
    assert label.startswith("0 ")
    assert (tmp_path / "out" / "yolomg_test_images.yaml").exists()


def test_li_tetc_compat_runner_installs_joblib_shim():
    python = ROOT / ".venv" / "paper-cv" / "bin" / "python"
    if not python.exists():
        return
    proc = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from tools.run_li_tetc_demo_compat import install_compat_shims;"
                "install_compat_shims();"
                "from sklearn.externals import joblib;"
                "print(hasattr(joblib, 'load'))"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "True" in proc.stdout


def test_li_tetc_compat_runner_patches_video_selection():
    from tools.run_li_tetc_demo_compat import patch_main_source

    source = "    test_ind = index[10*(ind-1):10*(ind-1)+1]\n"
    patched = patch_main_source(source, [40])

    assert patched.strip() == "test_ind = [39]"


def test_yolo_eval_window_accuracy_dry_run_uses_repo_specific_eval_args(tmp_path):
    common = [
        "--python",
        sys.executable,
        "--data",
        str(tmp_path / "data.yaml"),
        "--weights",
        str(tmp_path / "best.pt"),
        "--gt",
        str(tmp_path / "gt"),
        "--out",
        str(tmp_path / "out"),
        "--fps",
        "30",
        "--dry-run",
    ]

    tvd = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_yolo_eval_window_accuracy.py"),
            "--method",
            "transvisdrone",
            "--repo",
            str(tmp_path / "TransVisDrone"),
            "--num-frames",
            "7",
            *common,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    tvd_payload = json.loads(tvd.stdout)
    tvd_cmd = tvd_payload["command"]
    assert "--img" in tvd_cmd
    assert "--num-frames" in tvd_cmd
    assert tvd_cmd[tvd_cmd.index("--num-frames") + 1] == "7"
    assert "--save-txt" in tvd_cmd
    assert "--save-conf" in tvd_cmd

    esod = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_yolo_eval_window_accuracy.py"),
            "--method",
            "esod",
            "--repo",
            str(tmp_path / "ESOD"),
            *common,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    esod_cmd = json.loads(esod.stdout)["command"]
    assert "--img-size" in esod_cmd
    assert "--num-frames" not in esod_cmd

    edtc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_yolo_eval_window_accuracy.py"),
            "--method",
            "edtc",
            "--repo",
            str(tmp_path / "EDTC" / "yolov5"),
            *common,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    edtc_cmd = json.loads(edtc.stdout)["command"]
    assert "--img" in edtc_cmd
    assert "--num-frames" not in edtc_cmd


def test_yolo_eval_window_accuracy_preserves_python_symlink_path_in_dry_run(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_yolo_eval_window_accuracy.py"),
            "--method",
            "yolomg",
            "--repo",
            str(tmp_path / "YOLOMG"),
            "--python",
            ".venv/fake-python",
            "--data",
            str(tmp_path / "data.yaml"),
            "--weights",
            str(tmp_path / "best.pt"),
            "--gt",
            str(tmp_path / "gt"),
            "--out",
            str(tmp_path / "out"),
            "--fps",
            "30",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    command = json.loads(proc.stdout)["command"]
    assert command[0] == str(ROOT / ".venv" / "fake-python")


def test_yolo_eval_window_accuracy_absolutizes_eval_inputs_for_repo_cwd(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_yolo_eval_window_accuracy.py"),
            "--method",
            "yolomg",
            "--repo",
            str(tmp_path / "YOLOMG"),
            "--python",
            ".venv/fake-python",
            "--data",
            "relative_data.yaml",
            "--weights",
            "relative_weights.pt",
            "--gt",
            str(tmp_path / "gt"),
            "--out",
            str(tmp_path / "out"),
            "--fps",
            "30",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    command = json.loads(proc.stdout)["command"]
    assert command[command.index("--data") + 1] == str(ROOT / "relative_data.yaml")
    assert command[command.index("--weights") + 1] == str(ROOT / "relative_weights.pt")


def test_yolo_eval_window_accuracy_skip_eval_generates_curves_from_existing_labels(tmp_path):
    gt = tmp_path / "gt"
    pred = tmp_path / "pred"
    gt.mkdir()
    pred.mkdir()
    (gt / "Clip_003_00001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (pred / "Clip_003_00001.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")

    out = tmp_path / "curves"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_yolo_eval_window_accuracy.py"),
            "--method",
            "yolomg",
            "--skip-eval",
            "--pred-labels-dir",
            str(pred),
            "--gt",
            str(gt),
            "--gt-format",
            "yolo-dir",
            "--out",
            str(out),
            "--fps",
            "1",
            "--window-seconds",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "summary=" in proc.stdout
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["method"] == "YOLOMG"
    assert summary["eval_command"] is None
    assert summary["by_video"]["Clip_003"]["mean_accuracy"] == 1.0
    assert (out / "worst_windows.csv").exists()
    assert (out / "plots" / "Clip_003_window_metrics.svg").exists()
