from __future__ import annotations
import argparse,gc,json,pickle,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
for path in (REPO,REPO/'tools'):
 if str(path) not in sys.path:sys.path.insert(0,str(path))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.train_action_bank_motion_token_listwise import write_score_jsonl
from tools.train_action_chunk_bidir_full import load_aux
from tools.train_action_chunk_neighbor_full import load_neighbor,dataset_arrays,fit

def filter_sequences(data,allowed):
 return {image_id:item for image_id,item in data.items() if image_key(str(image_id),0)[0] in allowed}

def main():
 parser=argparse.ArgumentParser();
 for name in ('train-pkl','train-forward','train-backward','train-neighbor','test-pkl','test-forward','test-backward','test-neighbor','sequence-size-json','out-oof-pkl','out-oof-scores','out-test-scores','out-model-dir','out-summary'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--score-field',default='action_chunk_1920_score');args=parser.parse_args();sizes=json.loads(args.sequence_size_json.read_text(encoding='utf8'));allowed={seq for seq,size in sizes.items() if int(size[0])>=1900 and int(seq.split('_')[-1])<=36};train_data=filter_sequences(load_predictionsgt(args.train_pkl),allowed);args.out_oof_pkl.parent.mkdir(parents=True,exist_ok=True)
 with args.out_oof_pkl.open('wb') as output:pickle.dump(train_data,output,protocol=pickle.HIGHEST_PROTOCOL)
 forward,backward=load_aux(args.train_forward),load_aux(args.train_backward);neighbor,names=load_neighbor(args.train_neighbor);x,y,groups,locations,sequences=dataset_arrays(train_data,forward,backward,neighbor,True);del forward,backward,neighbor,train_data;gc.collect();test_forward,test_backward=load_aux(args.test_forward),load_aux(args.test_backward);test_neighbor,test_names=load_neighbor(args.test_neighbor);assert names==test_names;test_x,_,_,test_locations,test_sequences=dataset_arrays(load_predictionsgt(args.test_pkl),test_forward,test_backward,test_neighbor,False);del test_forward,test_backward,test_neighbor;gc.collect();unique=sorted(set(sequences.tolist()));folds=[unique[index::4] for index in range(4)];oof=np.zeros(len(x),np.float32);test_predictions=[];records=[];args.out_model_dir.mkdir(parents=True,exist_ok=True)
 for fold,held in enumerate(folds):
  validation_mask=np.isin(sequences,held);train_mask=~validation_mask;selected=np.flatnonzero(train_mask);mapping=np.full(len(x),-1,np.int64);mapping[selected]=np.arange(len(selected));fold_groups=[]
  for start,stop in groups:
   indices=np.arange(start,stop);kept=indices[train_mask[start:stop]]
   if len(kept):fold_groups.append((int(mapping[kept[0]]),int(mapping[kept[-1]])+1))
  model,count,pos=fit(x[selected],y[selected],fold_groups);oof[validation_mask]=model.predict_proba(x[validation_mask])[:,1];test_predictions.append(model.predict_proba(test_x)[:,1].astype(np.float32));model_path=args.out_model_dir/f'fold_{fold}.ubj';model.save_model(model_path);records.append({'fold':fold,'held_sequences':held,'hard_rows':count,'positive_rows':pos,'model':str(model_path)});print(json.dumps({'kind':'resolution_1920_fold',**records[-1]}),flush=True);del model,selected,mapping;gc.collect()
 write_score_jsonl(args.out_oof_scores,oof,locations,args.score_field);write_score_jsonl(args.out_test_scores,np.mean(np.stack(test_predictions),axis=0).astype(np.float32),test_locations,args.score_field);summary={'model':'1920x1080 corrected-label Action Bank specialist','allowed_sequences':sorted(allowed),'features':x.shape[1],'train_rows':len(x),'positive_candidates':int((y>=.5).sum()),'test_rows':len(test_x),'models':records,'score_field':args.score_field};args.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
