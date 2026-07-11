from __future__ import annotations
import json,math,sys
from pathlib import Path
R=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(R),str(R/'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores,fuse_score
from tools.sweep_action_chunk_temporal_multiplicity import temporal_gate_map
VAL=Path(r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl');TEST=Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl');FPS=json.loads((R/'data_templates'/'nps_sequence_fps.json').read_text());O=Path(r'D:\URAP_vatd_rank_results\tvd_v53_track_memory_v89')
def build(data,b,e,t,mode,alpha):
 gate=temporal_gate_map(data,.3,3.,.75,FPS);out={}
 for iid,item in data.items():
  seq,fid,_=image_key(str(iid),0);rows=[]
  for i,row in enumerate(item.get('detections') or []):
   key=(seq,fid,i);raw=float(row.get('score',0));bs=max(1e-9,float(b.get(key,raw)));es=max(1e-9,float(e.get(key,bs)));aux=math.sqrt(bs*es) if gate.get(str(iid),False) else bs;v53=fuse_score(raw,aux,.4,'geom-mix');z=dict(row);z['score']=fuse_score(v53,float(t.get(key,v53)),alpha,mode);rows.append(z)
  out[iid]={'labels':item.get('labels',[]),'detections':rows}
 return out
def main():
 O.mkdir(parents=True,exist_ok=True);vb,_=load_row_scores(Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46\val_oof_scores.jsonl'),'action_chunk_neighbor_score',1);ve,_=load_row_scores(Path(r'D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52\val_expert_scores.jsonl'),'action_chunk_multi_expert_score',1);vt,_=load_row_scores(Path(r'D:\URAP_vatd_rank_results\tvd_track_memory_bank_v88\val_track_scores.jsonl'),'track_memory_score',1);vd=load_predictionsgt(VAL);rows=[]
 for mode in ('geom-mix','logit-mix','fp-suppress'):
  for alpha in (.005,.01,.02,.04,.06,.08,.1,.14,.2,.3):
   m=evaluate_data(build(vd,vb,ve,vt,mode,alpha),Path(r'D:\urap_modal_stage\TransVisDrone'),O);rows.append({'mode':mode,'alpha':alpha,**m});print(json.dumps(rows[-1]),flush=True)
 best=max(rows,key=lambda x:x['map50']);tb,_=load_row_scores(Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46\test_scores.jsonl'),'action_chunk_neighbor_score',1);te,_=load_row_scores(Path(r'D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52\test_expert_scores.jsonl'),'action_chunk_multi_expert_score',1);tt,_=load_row_scores(Path(r'D:\URAP_vatd_rank_results\tvd_track_memory_bank_v88\test_track_scores.jsonl'),'track_memory_score',1);test=evaluate_data(build(load_predictionsgt(TEST),tb,te,tt,best['mode'],best['alpha']),Path(r'D:\urap_modal_stage\TransVisDrone'),O);summary={'protocol':'V53 plus OOF track memory second-stage correction; val-selected fixed test','val_best':best,'test':test,'gain_over_vatd_points':100*(test['map50']-.93844),'target_3_to_5_met':.03<=test['map50']-.93844<=.05};(O/'official_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
