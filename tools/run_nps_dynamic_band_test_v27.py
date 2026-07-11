from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
REPO=Path(r'C:\Users\aaron\Desktop\URAP');PYTHON=Path(sys.executable);RUN=REPO/'artifacts'/'detached_nps_dynamic_band_test_v27';PROGRESS=RUN/'progress.json';OUT=Path(r'D:\URAP_vatd_rank_results\nps_dynamic_band_test_v27')
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);p={'stage':stage,'done':done,'total':2,'updated':datetime.now(timezone.utc).astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(p,indent=2),encoding='utf-8');print(json.dumps(p),flush=True)
def execute(stage,done,cmd):
 p=subprocess.Popen(cmd,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'});report(stage,done,child_pid=p.pid,command=cmd);code=p.wait()
 if code:raise subprocess.CalledProcessError(code,cmd)
def main():
 OUT.mkdir(parents=True,exist_ok=True);execute('derive_test',0,[str(PYTHON),str(REPO/'tools'/'derive_action_bank_dynamic_band.py'),'--input-jsonl',r'D:\URAP_vatd_rank_results\nps_online_action_bank_v14\test_scores.jsonl','--output-jsonl',str(OUT/'test_scores.jsonl'),'--center','.8','--width','.4']);execute('fixed_test',1,[str(PYTHON),str(REPO/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--tracklet-jsonl',str(OUT/'test_scores.jsonl'),'--per-row-score','--score-field','dynamic_band_score','--modes','geom-mix','--alphas','.02','--out-json',str(OUT/'test_fixed.json')]);test=json.loads((OUT/'test_fixed.json').read_text(encoding='utf-8'))['best'];summary={'protocol':'center/width/fusion fixed on Clips37-40; test Clips41-50','test_fixed':test,'target_map50':.97,'target_met':test['map50']>=.97};(OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');report('done',2,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
