from __future__ import annotations
import gc,json,sys
from pathlib import Path
import numpy as np
R=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(R),str(R/'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_chunk_bidir_full import load_aux,arrays,hard_rows
from tools.run_tvd_forward_adapt_v84 import fit
from tools.train_action_bank_motion_token_listwise import write_score_jsonl
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores,clone_with_fused_scores
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data
VAL=Path(r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl');TEST=Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl');F=Path(r'D:\URAP_vatd_rank_results\action_chunk_full_dev_v36');V84=Path(r'D:\URAP_vatd_rank_results\tvd_forward_adapt_v84');O=Path(r'D:\URAP_vatd_rank_results\tvd_val_full_final_v99')
def main():
 O.mkdir(parents=True,exist_ok=True);vp=load_predictionsgt(VAL);va=load_aux(F/'val_forward.jsonl');vx,vy,vg,_,_=arrays(vp,va,va,True);del vp,va;gc.collect();tp=load_predictionsgt(TEST);ta=load_aux(F/'test_forward.jsonl');tx,_,_,tloc,_=arrays(tp,ta,ta,False);del ta;gc.collect();selected=hard_rows(vx,vy,vg,margin=.22,maxneg=18);model,pos=fit(vx[selected],vy[selected]);model.save_model(O/'val_full.ubj');scores=model.predict_proba(tx)[:,1].astype(np.float32);score_path=O/'test_scores.jsonl';write_score_jsonl(score_path,scores,tloc,'tvd_val_full_score');best=json.loads((V84/'val_sweep.json').read_text())['best'];score_map,_=load_row_scores(score_path,'tvd_val_full_score',1);fused=clone_with_fused_scores(tp,score_map,best['mode'],best['alpha'],'keep');metrics=evaluate_data(fused,Path(r'D:\urap_modal_stage\TransVisDrone'),O);summary={'protocol':'final model fitted on all official validation sequences; fusion fixed from V84 sequence OOF','validation_oof_config':best,'hard_rows':len(selected),'positive_rows':pos,'test':metrics,'gain_over_vatd_points':100*(metrics['map50']-.93844),'target_3_to_5_met':.03<=metrics['map50']-.93844<=.05};(O/'official_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
