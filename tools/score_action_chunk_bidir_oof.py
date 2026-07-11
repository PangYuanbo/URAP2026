from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import xgboost as xgb
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/'tools'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_bank_motion_token_listwise import write_score_jsonl
from tools.train_action_chunk_bidir_full import arrays,load_aux

def main():
 parser=argparse.ArgumentParser(description='Generate strict held-video OOF scores from the four pure Action Chunk models.')
 for name in ('val-pkl','val-forward','val-backward','model-dir','out-scores','out-summary'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--score-field',default='action_chunk_bidir_oof_score');args=parser.parse_args();forward=load_aux(args.val_forward);backward=load_aux(args.val_backward);x,_,_,loc,seqs=arrays(load_predictionsgt(args.val_pkl),forward,backward,True);scores=np.zeros(len(x),np.float32);records=[]
 for held in sorted(set(seqs)):
  path=args.model_dir/f'action_chunk_without_{held}.ubj';booster=xgb.Booster();booster.load_model(path);booster.set_param({'device':'cpu'});mask=seqs==held;scores[mask]=booster.inplace_predict(x[mask]);records.append({'held_video':held,'rows':int(mask.sum()),'model':str(path)})
 write_score_jsonl(args.out_scores,scores,loc,args.score_field);summary={'protocol':'strict held-video OOF; each validation video uses the model that excluded it','features':x.shape[1],'rows':len(x),'models':records};args.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
