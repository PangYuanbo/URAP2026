from __future__ import annotations
import argparse, concurrent.futures, json, shutil, subprocess, urllib.request, zipfile
from pathlib import Path

def content_length(url:str)->int:
 req=urllib.request.Request(url,method='HEAD',headers={'User-Agent':'URAP2026-curl-range/1.0'})
 with urllib.request.urlopen(req,timeout=60) as r:return int(r.headers['Content-Length'])
def fetch(url:str,path:Path,start:int,end:int)->dict:
 expected=end-start+1
 if path.is_file() and path.stat().st_size==expected:return {'part':path.name,'status':'present','bytes':expected}
 path.unlink(missing_ok=True)
 cmd=['curl.exe','-L','--fail','--retry','5','--retry-delay','2','--range',f'{start}-{end}','--output',str(path),url]
 subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 actual=path.stat().st_size
 if actual!=expected:raise RuntimeError(f'{path}: expected {expected}, got {actual}')
 return {'part':path.name,'status':'downloaded','bytes':actual}
def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--zip',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--chunk-mib',type=int,default=32);p.add_argument('--clips',default='1-36');p.add_argument('--json',type=Path,required=True);a=p.parse_args()
 total=content_length(a.url);a.zip.parent.mkdir(parents=True,exist_ok=True);chunk=a.chunk_mib*1024*1024;ranges=[]
 for i,start in enumerate(range(0,total,chunk)):
  end=min(total-1,start+chunk-1);ranges.append((i,start,end,a.zip.with_suffix(a.zip.suffix+f'.part{i:03d}')))
 reports=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(fetch,a.url,path,start,end):i for i,start,end,path in ranges}
  for f in concurrent.futures.as_completed(futures):
   report=f.result();reports.append(report);print(json.dumps({'kind':'curl_range_done','done':len(reports),'total':len(ranges),**report}),flush=True)
 tmp=a.zip.with_suffix('.zip.tmp')
 with tmp.open('wb') as out:
  for _,_,_,path in ranges:
   with path.open('rb') as src:shutil.copyfileobj(src,out,16*1024*1024)
 if tmp.stat().st_size!=total:raise RuntimeError(f'zip expected {total}, got {tmp.stat().st_size}')
 tmp.replace(a.zip)
 for *_,path in ranges:path.unlink(missing_ok=True)
 clips=[]
 for token in a.clips.split(','):
  if '-' in token:
   s,e=map(int,token.split('-',1));clips.extend(range(s,e+1))
  else:clips.append(int(token))
 a.out_dir.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.zip) as z:
  members={Path(info.filename).name:info for info in z.infolist() if not info.is_dir()}
  extracted=[]
  for clip in clips:
   name=f'Clip_{clip}.mov';info=members[name];target=a.out_dir/name
   with z.open(info) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,16*1024*1024)
   extracted.append({'clip':clip,'bytes':target.stat().st_size})
 report={'url':a.url,'zip':str(a.zip),'bytes':total,'workers':len(ranges),'out_dir':str(a.out_dir),'clips':extracted};a.json.parent.mkdir(parents=True,exist_ok=True);a.json.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'kind':'curl_download_done',**report}),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())

