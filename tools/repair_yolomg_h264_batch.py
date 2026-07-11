import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--ffmpeg',type=Path,required=True);p.add_argument('--ffprobe',type=Path,required=True)
    p.add_argument('--generation-status',type=Path,required=True);p.add_argument('--generation-pid',type=int,required=True)
    p.add_argument('--old-dir',type=Path,required=True);p.add_argument('--result-dir',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--backup-dir',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--max-workers',type=int,default=2)
    return p.parse_args()


def now(): return datetime.now(timezone.utc).isoformat()

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,path)

def read_json(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError,PermissionError):return {}

def pid_alive(pid):
    import ctypes
    handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
    if not handle:return False
    code=ctypes.c_ulong();ok=ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(handle)
    return bool(ok and code.value==259)

def source_prefix_map(source_dir):
    result={}
    for path in source_dir.glob('*.MP4'):
        parts=path.stem.split('_',3)
        if len(parts)==4:result[parts[3]]=path.stem
    return result

def build_items(a):
    prefixes=source_prefix_map(a.source_dir);items=[]
    candidates=list(a.old_dir.glob('*.mp4'))+list(a.result_dir.glob('*.mp4'))
    for path in candidates:
        if '.partial.' in path.name:continue
        sequence=None;target_name=path.name
        head=path.name.split('_',1)[0]
        if head.isdigit():sequence=int(head)
        else:
            original=path.name.split('_yolomg_',1)[0]
            organized=prefixes.get(original)
            if organized:
                sequence=int(organized.split('_',1)[0]);target_name=f'{organized}_yolomg_compensated_difference_1080p.mp4'
        if sequence is None or not 1<=sequence<=28:continue
        target=a.result_dir/target_name
        job=a.run_dir/'jobs'/f'{sequence:03d}'
        items.append({'sequence':sequence,'source':str(path.resolve()),'target':str(target.resolve()),'temp':str((target.parent/(target.stem+'.h264.partial.mp4')).resolve()),'backup':str((a.backup_dir/path.name).resolve()),'stdout':str((job/'ffmpeg_progress.log').resolve()),'stderr':str((job/'ffmpeg_stderr.log').resolve()),'state':'pending','pid':None,'started_at':None,'completed_at':None,'duration':0.0,'out_time_seconds':0.0,'return_code':None})
    unique={item['sequence']:item for item in items}
    result=[unique[n] for n in sorted(unique)]
    if len(result)!=28:raise RuntimeError(f'Expected 28 source outputs, found {len(result)} sequences={sorted(unique)}')
    return result

def probe(ffprobe,path):
    command=[str(ffprobe),'-v','error','-show_entries','format=duration:stream=codec_name,profile,pix_fmt,width,height','-of','json',str(path)]
    return json.loads(subprocess.check_output(command,text=True,encoding='utf-8'))

def duration_of(ffprobe,path):return float(probe(ffprobe,path)['format']['duration'])

def parse_progress(path):
    try:
        values={}
        for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
            if '=' in line:
                key,value=line.split('=',1);values[key]=value
        return float(values.get('out_time_us','0'))/1_000_000
    except (FileNotFoundError,ValueError,PermissionError):return 0.0

def snapshot(item):
    data={k:v for k,v in item.items() if k not in {'process','stdout_handle','stderr_handle'}}
    data['out_time_seconds']=parse_progress(Path(item['stdout'])) if item['state']=='running' else item.get('out_time_seconds',0.0)
    target=Path(item['target']);data['target_exists']=target.exists();data['target_bytes']=target.stat().st_size if target.exists() else 0
    return data

def status_write(a,items,status,started,error=None):
    counts={s:sum(i['state']==s for i in items) for s in ('pending','running','completed','failed')}
    data={'status':status,'started_at':started,'updated_at':now(),'coordinator_pid':os.getpid(),'done':counts['completed'],'total':len(items),'pending':counts['pending'],'running':counts['running'],'failed':counts['failed'],'error':error,'items':[snapshot(i) for i in items]}
    write_json(a.run_dir/'status.json',data);write_json(a.result_dir/'h264_repair_manifest.json',data)

def start_item(a,item):
    Path(item['stdout']).parent.mkdir(parents=True,exist_ok=True);Path(item['temp']).unlink(missing_ok=True)
    item['duration']=duration_of(a.ffprobe,Path(item['source']))
    out=Path(item['stdout']).open('w',encoding='utf-8');err=Path(item['stderr']).open('w',encoding='utf-8')
    command=[str(a.ffmpeg),'-hide_banner','-y','-i',item['source'],'-map','0:v:0','-c:v','h264_nvenc','-preset','p5','-tune','hq','-rc','vbr','-cq','20','-b:v','12M','-maxrate','20M','-bufsize','40M','-profile:v','high','-level:v','4.2','-pix_fmt','yuv420p','-tag:v','avc1','-movflags','+faststart','-an','-progress','pipe:1','-nostats',item['temp']]
    proc=subprocess.Popen(command,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    item.update(state='running',pid=proc.pid,started_at=now(),process=proc,stdout_handle=out,stderr_handle=err)
    print(f"[START] {item['sequence']:03d} PID={proc.pid}",flush=True)

def finish_item(a,item,return_code):
    item['stdout_handle'].close();item['stderr_handle'].close();item['return_code']=return_code;item['completed_at']=now();item['out_time_seconds']=parse_progress(Path(item['stdout']))
    good=False
    if return_code==0 and Path(item['temp']).exists():
        data=probe(a.ffprobe,Path(item['temp']));stream=data['streams'][0];new_duration=float(data['format']['duration'])
        good=stream.get('codec_name')=='h264' and stream.get('profile')=='High' and stream.get('pix_fmt')=='yuv420p' and stream.get('width')==1920 and stream.get('height')==1080 and abs(new_duration-item['duration'])<0.25
    if good:
        backup=Path(item['backup']);backup.parent.mkdir(parents=True,exist_ok=True)
        source=Path(item['source']);target=Path(item['target'])
        if backup.exists():backup.unlink()
        shutil.move(str(source),str(backup))
        if target.exists() and target!=source:target.unlink()
        os.replace(item['temp'],target);item['state']='completed'
        write_json(target.with_suffix('.playback.json'),{'codec':'h264','profile':'High','pixel_format':'yuv420p','encoder':'h264_nvenc','movflags':'faststart','source_backup':str(backup),'verified_at':now()})
    else:item['state']='failed'
    for key in ('process','stdout_handle','stderr_handle'):item.pop(key,None)
    print(f"[{item['state'].upper()}] {item['sequence']:03d} return_code={return_code}",flush=True)

def main():
    a=args_parser();a.run_dir.mkdir(parents=True,exist_ok=True);a.result_dir.mkdir(parents=True,exist_ok=True);a.backup_dir.mkdir(parents=True,exist_ok=True);started=now()
    while True:
        generation=read_json(a.generation_status)
        if generation.get('status')=='completed' and generation.get('done')==generation.get('total') and generation.get('failed')==0:break
        if not pid_alive(a.generation_pid):
            status_write(a,[],'failed',started,'Generation coordinator stopped before successful completion');raise SystemExit(1)
        status_write(a,[],'waiting_for_generation',started);time.sleep(10)
    items=build_items(a);status_write(a,items,'running',started)
    while any(i['state'] in {'pending','running'} for i in items):
        for item in items:
            if item['state']=='running':
                code=item['process'].poll()
                if code is not None:finish_item(a,item,code)
        slots=a.max_workers-sum(i['state']=='running' for i in items)
        for item in (i for i in items if i['state']=='pending'):
            if slots<=0:break
            start_item(a,item);slots-=1
        status_write(a,items,'running',started);time.sleep(3)
    final='completed' if all(i['state']=='completed' for i in items) else 'failed';status_write(a,items,final,started)
    if final!='completed':raise SystemExit(1)

if __name__=='__main__':main()
