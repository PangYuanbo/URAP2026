from __future__ import annotations
import json,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path

REPO=Path(r'C:\Users\aaron\Desktop\URAP')
RUN=REPO/'artifacts'/'detached_action_chunk_projected_heuristic_v64'
PROGRESS=RUN/'progress.json'
OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_projected_heuristic_v64')
PERSISTENT=Path(r'D:\URAP_vatd_rank_results\action_chunk_projected_persistent_v61')
PREREQUISITE=Path(r'D:\URAP_vatd_rank_results\action_chunk_projected_causal_residual_v63\official_summary.json')


def report(stage: str,done: int,total: int=1,**extra) -> None:
    RUN.mkdir(parents=True,exist_ok=True)
    payload={'stage':stage,'done':done,'total':total,'updated':datetime.now().astimezone().isoformat(),**extra}
    PROGRESS.write_text(json.dumps(payload),encoding='utf8')
    print(json.dumps(payload),flush=True)


def main() -> int:
    while not PREREQUISITE.is_file():
        report('waiting_for_projected_causal_residual_v63',0,prerequisite=str(PREREQUISITE))
        time.sleep(60)
    prior=json.loads(PREREQUISITE.read_text(encoding='utf8'))
    if prior.get('target_met'):
        report('skipped_target_already_met',1,prior_result=str(PREREQUISITE))
        return 0
    OUT.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(REPO/'tools'/'sweep_action_chunk_causal_heuristics.py'),'--val-pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--test-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--val-jsonl',str(PERSISTENT/'val_forward.jsonl'),'--test-jsonl',str(PERSISTENT/'test_forward.jsonl'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--out-json',str(OUT/'official_summary.json')]
    process=subprocess.Popen(command,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'})
    report('sweep_projected_causal_heuristic',0,child_pid=process.pid,command=command)
    code=process.wait()
    if code:
        raise RuntimeError(f'heuristic sweep failed with {code}')
    summary=json.loads((OUT/'official_summary.json').read_text(encoding='utf8'))
    report('done',1,summary=summary)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
