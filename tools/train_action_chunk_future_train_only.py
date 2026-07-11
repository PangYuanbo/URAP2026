from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
for candidate in (REPO,REPO/'tools'):
 if str(candidate) not in sys.path:sys.path.insert(0,str(candidate))
from tools.train_action_bank_motion_token_listwise import write_score_jsonl
from tools.train_action_chunk_chain_consistency import build_arrays
from tools.train_action_chunk_future_reliability import fit,valid_future_mask
VARIANTS={'full':np.arange(38),'action':np.arange(11,38),'history':np.arange(24,38),'compatibility':np.asarray(list(range(12,24))+list(range(24,38)),np.int64)}
def main():
 p=argparse.ArgumentParser()
 for name in ('train-pkl','train-chain','val-pkl','val-chain','test-pkl','test-chain','fps-json','out-dir','out-summary'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();tx,_,_,_,_,tseq,tfuture,ttime=build_arrays(a.train_pkl,a.train_chain,a.fps_json,True);vx,_,_,_,vloc,_,_,_=build_arrays(a.val_pkl,a.val_chain,a.fps_json,False);qx,_,_,_,qloc,_,_,_=build_arrays(a.test_pkl,a.test_chain,a.fps_json,False);valid=valid_future_mask(tseq,ttime);a.out_dir.mkdir(parents=True,exist_ok=True);records=[]
 for name,columns in VARIANTS.items():
  model,rows,positives=fit(tx[:,columns],tfuture,valid);val=model.predict_proba(vx[:,columns])[:,1].astype(np.float32);test=model.predict_proba(qx[:,columns])[:,1].astype(np.float32);write_score_jsonl(a.out_dir/f'val_{name}.jsonl',val,vloc,f'action_chunk_future_{name}');write_score_jsonl(a.out_dir/f'test_{name}.jsonl',test,qloc,f'action_chunk_future_{name}');path=a.out_dir/f'model_{name}.ubj';model.save_model(path);record={'variant':name,'features':len(columns),'rows':rows,'future_correct_rows':positives,'model':str(path)};records.append(record);print(json.dumps({'kind':'action_chunk_train_only_variant',**record}),flush=True)
 summary={'model':'pure Action Chunk future reliability trained only on NPS train sequences','validation_used_for_training':False,'variants':records,'train_rows':len(tx),'validation_rows':len(vx),'test_rows':len(qx)};a.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
