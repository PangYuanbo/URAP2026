from __future__ import annotations
import json
from pathlib import Path
import modal
app=modal.App('urap-nps-video-size-audit-v1')
raw=modal.Volume.from_name('nps-dataset')
@app.function(volumes={'/raw':raw},timeout=1800)
def audit():
 root=Path('/raw/Videos'); rows=[]
 for clip in range(1,37):
  p=root/f'Clip_{clip}.mov'; rows.append({'clip':clip,'exists':p.exists(),'size':p.stat().st_size if p.exists() else 0})
 return {'count':sum(r['exists'] for r in rows),'bytes':sum(r['size'] for r in rows),'rows':rows}
@app.local_entrypoint()
def main(): print(json.dumps(audit.remote(),indent=2))
