from __future__ import annotations
import concurrent.futures
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

REPO=Path(r"C:\Users\aaron\Desktop\URAP")
RAW=Path(r"D:\URAP_nps_train_raw")
OUT=Path(r"D:\URAP_nps_train_tvd")
FRAMES=OUT/"AllFrames/train"
VIDEOS_META=OUT/"Videos/train"
RUNNER=REPO/"artifacts/detached_nps_train_video_prepare"
PROGRESS=RUNNER/"progress.json"
MODAL=Path(r"C:\Users\aaron\.local\bin\modal.exe")

def write(stage:str,done:int,total:int=38,**extra:object)->None:
 RUNNER.mkdir(parents=True,exist_ok=True)
 PROGRESS.write_text(json.dumps({"stage":stage,"done":done,"total":total,"updated":datetime.now(timezone.utc).astimezone().isoformat(),**extra},indent=2),encoding="utf-8")
 print(json.dumps({"kind":"prepare_progress","stage":stage,"done":done,"total":total,**extra}),flush=True)

def locate(clip:int)->Path:
 candidates=[RAW/"Videos"/f"Clip_{clip}.mov",RAW/f"Clip_{clip}.mov"]
 for path in candidates:
  if path.is_file(): return path
 raise FileNotFoundError(candidates)

def extract(clip:int)->tuple[int,int]:
 video=locate(clip);cap=cv2.VideoCapture(str(video))
 if not cap.isOpened(): raise RuntimeError(f"cannot open {video}")
 expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));count=0
 while True:
  ok,frame=cap.read()
  if not ok: break
  count+=1;path=FRAMES/f"Clip_{clip}_{count:05d}.png"
  if not cv2.imwrite(str(path),frame,[cv2.IMWRITE_PNG_COMPRESSION,1]): raise RuntimeError(f"write failed {path}")
 cap.release()
 if count!=expected: raise RuntimeError(f"clip {clip}: expected {expected}, got {count}")
 return clip,count

def main()->int:
 RAW.mkdir(parents=True,exist_ok=True);FRAMES.mkdir(parents=True,exist_ok=True);VIDEOS_META.mkdir(parents=True,exist_ok=True)
 write("download_videos",0)
 command=[str(Path(sys.executable)),str(REPO/"tools/download_nps_zip_curl_ranges.py"),"--url","https://engineering.purdue.edu/~bouman/UAV_Dataset/Videos.zip","--zip",str(RAW/"Videos.zip"),"--out-dir",str(RAW/"Videos"),"--clips","1-36","--workers","4","--chunk-mib","10","--json",str(RUNNER/"official_download.json")]
 process=subprocess.Popen(command,cwd=REPO,env={**os.environ,"PYTHONUTF8":"1","PYTHONIOENCODING":"utf-8","TERM":"dumb"})
 write("download_videos",0,child_pid=process.pid,command=command)
 code=process.wait()
 if code: raise subprocess.CalledProcessError(code,command)
 sizes=sum(locate(c).stat().st_size for c in range(1,37));write("download_complete",1,video_bytes=sizes)
 lengths:dict[int,int]={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(extract,c):c for c in range(1,37)}
  completed=0
  for future in concurrent.futures.as_completed(futures):
   clip,count=future.result();lengths[clip]=count;completed+=1
   with (VIDEOS_META/"video_length_dict.pkl").open("wb") as handle: pickle.dump(dict(sorted(lengths.items())),handle)
   write("extract_frames",1+completed,last_clip=clip,last_clip_frames=count,frames=sum(lengths.values()))
 total_frames=sum(lengths.values())
 if total_frames!=51951: raise RuntimeError(f"expected 51951 frames, got {total_frames}")
 write("done",38,frames=total_frames,clips=len(lengths),video_bytes=sizes)
 return 0
if __name__=="__main__":raise SystemExit(main())




