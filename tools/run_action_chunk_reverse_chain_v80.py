from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime
from pathlib import Path
REPO=Path(r'C:\Users\aaron\Desktop\URAP');RUN=REPO/'artifacts'/'detached_action_chunk_reverse_chain_v80';PROGRESS=RUN/'progress.json';OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_reverse_chain_v80');HOM=Path(r'D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies');FPS=REPO/'data_templates'/'nps_sequence_fps.json';FRAMES=REPO/'artifacts'/'cache_only_frames'
def report(stage,done,total=2,**extra):RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':total,'updated':datetime.now().astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(payload),encoding='utf8');print(json.dumps(payload),flush=True)
def execute(stage,done,pkl,split):
 command=[sys.executable,str(REPO/'tools'/'score_predictionsgt_action_chunk_bank.py'),'--predictionsgt-pkl',str(pkl),'--frame-root',str(FRAMES),'--homography-cache',str(HOM/f'{split}.pkl'),'--out-jsonl',str(OUT/f'{split}_reverse_chain.jsonl'),'--out-summary',str(OUT/f'{split}_summary.json'),'--sequence-fps-json',str(FPS),'--short-seconds','1','--long-seconds','3','--short-token-count','8','--long-token-count','16','--compact-chain-output','--reverse'];process=subprocess.Popen(command,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'});report(stage,done,child_pid=process.pid,command=command);code=process.wait()
 if code:raise RuntimeError(f'{stage} failed with {code}')
def main():
 OUT.mkdir(parents=True,exist_ok=True);execute('score_val_reverse_chain',0,Path(r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl'),'val');execute('score_test_reverse_chain',1,Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl'),'test');report('done',2,output=str(OUT));return 0
if __name__=='__main__':raise SystemExit(main())
