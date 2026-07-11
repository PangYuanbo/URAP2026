from __future__ import annotations
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/'tools'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key
from tools.sweep_tvd_predictionsgt_score_fusion import clone_with_fused_scores,load_row_scores

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--predictionsgt-pkl',type=Path,required=True);parser.add_argument('--scores',type=Path,required=True);parser.add_argument('--score-field',required=True);parser.add_argument('--mode',default='geom-mix');parser.add_argument('--alpha',type=float,required=True);parser.add_argument('--tvd-root',type=Path,required=True);parser.add_argument('--out-json',type=Path,required=True);args=parser.parse_args();raw=load_predictionsgt(args.predictionsgt_pkl);score_map,_=load_row_scores(args.scores,args.score_field,1);fused=clone_with_fused_scores(raw,score_map,args.mode,args.alpha,'keep');groups={}
 for image_id,item in raw.items():groups.setdefault(image_key(str(image_id),0)[0],[]).append(image_id)
 rows=[];tmp=args.out_json.parent/'per_clip_eval_tmp';tmp.mkdir(parents=True,exist_ok=True)
 for seq,ids in sorted(groups.items()):
  raw_metrics=evaluate_data({key:raw[key] for key in ids},args.tvd_root,tmp);fused_metrics=evaluate_data({key:fused[key] for key in ids},args.tvd_root,tmp);rows.append({'sequence':seq,'raw_map50':raw_metrics['map50'],'action_map50':fused_metrics['map50'],'delta':fused_metrics['map50']-raw_metrics['map50'],'precision':fused_metrics['precision'],'recall':fused_metrics['recall']})
 summary={'mode':args.mode,'alpha':args.alpha,'score_field':args.score_field,'per_sequence':rows};args.out_json.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
