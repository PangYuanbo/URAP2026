from __future__ import annotations
import gc,json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
import xgboost as xgb
ROOT=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.run_tvd_domain_balanced_action_v129 import TEST,FULL,NEIGHBOR,SIZE_MAP
from tools.run_tvd_oof_stack_v130 import metrics
from tools.run_tvd_trainval_refit_v160 import OUT,TVD,VATD,features,advanced_base,combine
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score
RUN=ROOT/'artifacts'/'detached_tvd_trainval_refit_v160_r2'

def report(stage,done,**extra):
 RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':3,'updated':datetime.now().astimezone().isoformat(),**extra};(RUN/'progress.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload),flush=True)

def main():
 selection=json.loads((OUT/'val_oof_sweep.json').read_text(encoding='utf-8'))['best'];summary129=json.loads(Path(r'D:\URAP_vatd_rank_results\tvd_domain_balanced_action_v129\official_summary.json').read_text(encoding='utf-8'));cfg=summary129['validation_selection'];selected_ratio=float(summary129['selected_ratio']);report('load_saved_model',0,validation_selection=selection);model=xgb.XGBClassifier();model.load_model(OUT/'trainval_refit.ubj');size_map=json.loads(SIZE_MAP.read_text(encoding='utf-8'));report('load_test_features',1);qx,_qy,_qg,qlocations,_qseq,_names=features(TEST,FULL/'test_forward.jsonl',FULL/'test_backward.jsonl',NEIGHBOR/'test_neighbor_scores.jsonl',False,size_map);learned=model.predict_proba(qx)[:,1].astype(np.float64);del qx,model;gc.collect();learned_map={(str(image_id),int(index)):float(score) for (image_id,index),score in zip(qlocations,learned)};qc,qp,qt,base_locations,qlabels,qbase=advanced_base('test',TEST);missing=[];aligned=np.empty(len(base_locations),np.float64)
 for row,(_seq,_fid,index,image_id,raw) in enumerate(base_locations):
  key=(str(image_id),int(index));value=learned_map.get(key)
  if value is None:missing.append(key);value=float(raw)
  aligned[row]=value
 if missing:raise RuntimeError(f'missing learned scores for {len(missing)} candidates; first={missing[:5]}')
 report('fixed_test',2,candidates=len(base_locations),mapped=len(aligned));refit=np.asarray([fuse_score(raw,float(pred),float(cfg['alpha']),str(cfg['mode'])) for raw,pred in zip((x[4] for x in base_locations),aligned)]);score=combine(qbase,refit,float(selection['alpha']),selection['mode']);test={**metrics(qc,score,qp,qt,TVD),'labels':qlabels,'detections':len(base_locations),'mapped_scores':len(aligned)};gain=100*(test['map50']-VATD);summary={'protocol':'OOF-selected fusion; final Action ranker refit on train+validation; untouched fixed test; exact candidate-key alignment','validation_selection':selection,'v129_selection':cfg,'selected_ratio':selected_ratio,'test_fixed':test,'vatd_map50':VATD,'gain_over_vatd_points':gain,'target_3_to_5_met':3<=gain<=5};(OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');report('done',3,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
