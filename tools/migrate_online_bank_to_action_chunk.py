from __future__ import annotations
import argparse
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);lines=0
 with a.input.open('r',encoding='utf-8-sig') as source,a.output.open('w',encoding='utf-8') as target:
  for line in source:
   target.write(line.replace('online_action_bank_','action_chunk_bank_').replace('online_action_bank_sequence','action_chunk_bank_sequence').replace('online_action_bank_done','action_chunk_bank_done'));lines+=1
 print(f'migrated_lines={lines} input={a.input} output={a.output}',flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
