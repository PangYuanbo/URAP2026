from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime
from pathlib import Path

REPO=Path(r'C:\Users\aaron\Desktop\URAP')
RUN=REPO/'artifacts'/'detached_action_chunk_bounded_residual_v58'
PROGRESS=RUN/'progress.json'
OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_bounded_residual_v58')
BASE=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46')
PERSISTENT=Path(r'D:\URAP_vatd_rank_results\action_chunk_persistent_model_v55')


def report(stage: str,done: int,total: int=2,**extra) -> None:
    RUN.mkdir(parents=True,exist_ok=True)
    PROGRESS.write_text(json.dumps({'stage':stage,'done':done,'total':total,'updated':datetime.now().astimezone().isoformat(),**extra}),encoding='utf8')


def execute(stage: str,done: int,command: list[str]) -> None:
    process=subprocess.Popen(command,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'})
    report(stage,done,child_pid=process.pid,command=command)
    code=process.wait()
    if code:
        raise RuntimeError(f'{stage} failed with {code}')


def main() -> int:
    OUT.mkdir(parents=True,exist_ok=True)
    common=[sys.executable,str(REPO/'tools'/'sweep_action_chunk_bounded_residual.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--base-field','action_chunk_neighbor_score','--residual-field','action_chunk_neighbor_score']
    execute('select_bounded_residual_on_oof',0,common+['--predictionsgt-pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--base-jsonl',str(BASE/'val_oof_scores.jsonl'),'--residual-jsonl',str(PERSISTENT/'val_oof_scores.jsonl'),'--out-json',str(OUT/'val_sweep.json')])
    execute('fixed_test',1,common+['--predictionsgt-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--base-jsonl',str(BASE/'test_scores.jsonl'),'--residual-jsonl',str(PERSISTENT/'test_scores.jsonl'),'--fixed-config-json',str(OUT/'val_sweep.json'),'--out-json',str(OUT/'test_fixed.json')])
    validation=json.loads((OUT/'val_sweep.json').read_text(encoding='utf8'))['best']
    test=json.loads((OUT/'test_fixed.json').read_text(encoding='utf8'))['best']
    summary={'protocol':'pure Action Chunk Bank bounded persistent residual; OOF selection; fixed test','validation_selection':validation,'test_fixed':test,'target_map50':.97,'target_met':test['map50']>=.97}
    (OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    report('done',2,summary=summary)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
