from __future__ import annotations
import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
REPO=Path(r"C:\Users\aaron\Desktop\URAP"); TVD=Path(r"D:\urap_modal_stage\TransVisDrone"); TRAIN=Path(r"D:\URAP_nps_train_tvd"); VAL=Path(r"D:\URAP_nps_val_tvd"); INPUT=Path(r"D:\URAP_vatd_rank_inputs"); OUTPUT=Path(r"D:\URAP_vatd_rank_results\nps_train_val_to_test_cuda_rank_v3"); RUNNER=REPO/"artifacts/detached_nps_train_val_to_test_cuda_rank"; PROGRESS=RUNNER/"progress.json"; PYTHON=Path(sys.executable)
def progress(stage,done,total=6,**extra): RUNNER.mkdir(parents=True,exist_ok=True);PROGRESS.write_text(json.dumps({"stage":stage,"done":done,"total":total,"updated":datetime.now(timezone.utc).astimezone().isoformat(),**extra},indent=2),encoding="utf-8")
def run(command,cwd,stage,done):
 print(json.dumps({"kind":"pipeline_command","stage":stage,"command":command}),flush=True);p=subprocess.Popen(command,cwd=cwd,env={**os.environ,"PYTHONUNBUFFERED":"1","PYTHONPATH":str(REPO)});progress(stage,done,child_pid=p.pid,command=command);code=p.wait();
 if code: raise subprocess.CalledProcessError(code,command)
def main():
 prepare=REPO/"artifacts/detached_nps_train_video_prepare/progress.json"
 while True:
  state=json.loads(prepare.read_text(encoding="utf-8")) if prepare.exists() else {}
  frames_dir=TRAIN/"AllFrames/train"; count=int(state.get("frames",0) or 0)
  progress("waiting_for_video_prepare",0,prepare=state,frame_count=count)
  print(json.dumps({"kind":"waiting_for_video_prepare","frame_count":count,"prepare":state}),flush=True)
  if state.get("stage")=="done" and state.get("done")==38 and count==51951: break
  time.sleep(30)
 labels=TRAIN/"NPSvisdroneStyle/train/labels_official_dogfight"; videos=TRAIN/"Videos"; weights=VAL/"weights/best.pt"; OUTPUT.mkdir(parents=True,exist_ok=True)
 yaml_path=TRAIN/"NPS_official_train.yaml";yaml_path.write_text("\n".join([f"path: {str(TRAIN/'AllFrames').replace(chr(92),'/')}","train: train","val: train","test: train","inference: train",f"annotation_path: {str(TRAIN/'NPSvisdroneStyle').replace(chr(92),'/')}","annotation_train: train/labels_official_dogfight","annotation_val: train/labels_official_dogfight","annotation_test: train/labels_official_dogfight",f"video_root_path: {str(videos).replace(chr(92),'/')}","video_root_path_train: train","video_root_path_val: train","video_root_path_test: train","video_root_path_inference: train","nc: 1","names: ['drone']",""]),encoding="utf-8")
 inference=TRAIN/"runs/nps_train_rank_source"
 run([str(PYTHON),"val.py","--data",str(yaml_path),"--weights",str(weights),"--task","val","--img","1280","--num-frames","5","--save-json","--save-json-gt","--device","0","--batch-size","16","--half","--project",str(TRAIN/"runs"),"--name","nps_train_rank_source","--exist-ok"],TVD,"inference",1)
 predictions=inference/"predictionsgt/predictionsgt_split_0.pkl";route=TRAIN/"route_b_official"
 run([str(PYTHON),str(REPO/"tools/export_tvd_predictionsgt_to_route_b.py"),"--predictionsgt-pkl",str(predictions),"--out-run-root",str(route/"run"),"--out-gt-csv",str(route/"gt.csv"),"--out-summary",str(route/"export_summary.json"),"--frame-root",str(TRAIN/"AllFrames/train"),"--profile","hard_recovery","--diagnostics-name","diagnostics_raw.jsonl"],REPO,"export",2)
 run([str(PYTHON),"-m","qstr_dronedet.cli","build-proposal-tracklet-dataset","--run-roots",str(route/"run"),"--gt-csv",str(route/"gt.csv"),"--out",str(route/"tracklets"),"--profile","hard_recovery","--diagnostics-name","diagnostics_raw.jsonl","--max-gap","3","--base-radius","18","--radius-per-side","0.75","--min-iou","0.05","--min-score","0.0","--min-tracklet-rows","3","--iou-threshold","0.5","--center-threshold","24"],REPO,"tracklets",3)
 scored=OUTPUT/"nps_test_tracklets_train_val_rank_scored.jsonl"
 run([str(PYTHON),str(REPO/"tools/train_detection_row_score_head.py"),"--train-tracklets",str(route/"tracklets/proposal_tracklets.jsonl"),str(VAL/"route_b_official/tracklets/proposal_tracklets.jsonl"),"--train-gt-csv",str(route/"gt.csv"),str(VAL/"route_b_official/gt.csv"),"--test-tracklets",str(INPUT/"nps_tracklets_with_vatd.jsonl"),"--out-test-tracklets",str(scored),"--out-model",str(OUTPUT/"train_val_rank_model.pt"),"--out-summary",str(OUTPUT/"train_summary.json"),"--score-field","train_val_rank_score","--iou-threshold","0.5","--negative-min-score","0.005","--label-policy","unique-iou","--epochs","40","--batch-size","32768","--hidden","384","--lr","0.00035","--pairwise-weight","1.5","--pairwise-pairs","131072","--model-kind","unified-two-tower","--tracklet-aux-weight","0.25","--feature-groups","all"],REPO,"train",4)
 run([str(PYTHON),str(REPO/"tools/sweep_tvd_predictionsgt_two_score_fusion_fast.py"),"--tvd-root",str(TVD),"--predictionsgt-pkl",str(INPUT/"nps_predictionsgt_split_0.pkl"),"--meta-tracklet-jsonl",str(scored),"--meta-score-field","vatd_score","--row-tracklet-jsonl",str(scored),"--row-score-field","train_val_rank_score","--modes","logit-3mix","meta-logit-row-geom","meta-logit-row-suppress","meta-logit-row-boost","--alphas","0.00","0.01","0.02","0.04","0.06","0.08","0.10","0.14","0.20","--betas","0.005","0.01","0.02","0.04","0.06","0.08","0.10","0.12","0.16","0.20","0.24","0.32","0.40","--out-json",str(OUTPUT/"fusion_sweep_fast.json"),"--write-best-pkl",str(OUTPUT/"best_predictionsgt.pkl")],REPO,"evaluate",5)
 summary=json.loads((OUTPUT/"fusion_sweep_fast.json").read_text(encoding="utf-8"));progress("done",6,best=summary.get("best"),output=str(OUTPUT));print(json.dumps({"kind":"pipeline_done","best":summary.get("best"),"output":str(OUTPUT)}),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())


