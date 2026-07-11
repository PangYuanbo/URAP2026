from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime
from pathlib import Path

REPO=Path(r'C:\Users\aaron\Desktop\URAP')
RUN=REPO/'artifacts'/'detached_action_chunk_low_score_recovery_v70'
PROGRESS=RUN/'progress.json'
OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_low_score_recovery_v70')
BASE=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46')


def report(stage: str,done: int,total: int=2,**extra) -> None:
    RUN.mkdir(parents=True,exist_ok=True)
    payload={'stage':stage,'done':done,'total':total,'updated':datetime.now().astimezone().isoformat(),**extra}
    PROGRESS.write_text(json.dumps(payload),encoding='utf8')
    print(json.dumps(payload),flush=True)


def execute(stage: str,done: int,command: list[str]) -> None:
    process=subprocess.Popen(command,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'})
    report(stage,done,child_pid=process.pid,command=command)
    code=process.wait()
    if code:
        raise RuntimeError(f'{stage} failed with {code}')


def main() -> int:
    OUT.mkdir(parents=True,exist_ok=True)
    common=[sys.executable,str(REPO/'tools'/'sweep_tvd_predictionsgt_action_rescore.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--score-field','action_chunk_neighbor_score','--modes','gated-boost-low','--centers','.3 .5 .7 .85 .95','--betas','.005 .01 .02 .04 .08 .14','--score-gates','.02 .05 .1 .2 .3','--missing-score-behaviors','keep']
    execute('select_low_score_recovery_on_oof',0,common+['--predictionsgt-pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--tracklet-jsonl',str(BASE/'val_oof_scores.jsonl'),'--out-json',str(OUT/'val_sweep.json')])
    best=json.loads((OUT/'val_sweep.json').read_text(encoding='utf8'))['best']
    execute('fixed_test',1,[sys.executable,str(REPO/'tools'/'sweep_tvd_predictionsgt_action_rescore.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--tracklet-jsonl',str(BASE/'test_scores.jsonl'),'--score-field','action_chunk_neighbor_score','--modes','gated-boost-low','--centers',str(best['center']),'--betas',str(best['beta']),'--score-gates',str(best['score_gate']),'--missing-score-behaviors','keep','--out-json',str(OUT/'test_fixed.json')])
    test=json.loads((OUT/'test_fixed.json').read_text(encoding='utf8'))['best']
    summary={'protocol':'pure offline Action Chunk gated recovery of low-score candidates only; high raw-score candidates unchanged; OOF selection; fixed test','validation_selection':best,'test_fixed':test,'target_map50':.97,'target_met':test['map50']>=.97}
    (OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    report('done',2,summary=summary)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
