#!/usr/bin/env python3
import json, os, re, tarfile
from pathlib import Path
import zstandard as zstd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

OUT=Path('codetrace_artifact_probe');OUT.mkdir(exist_ok=True)
ds=load_dataset('NJU-LINK/CodeTraceBench',split='verified')

def shape(x,d=0):
    if d>5:return '<depth>'
    if isinstance(x,dict):
        return {str(k):shape(v,d+1) for k,v in list(x.items())[:40]}
    if isinstance(x,list):
        return {'__list_len__':len(x),'first':shape(x[0],d+1) if x else None,'last':shape(x[-1],d+1) if x else None}
    s=str(x)
    s=re.sub(r'[0-9a-f]{20,}','<id>',s,flags=re.I)
    return s[:500]

chosen=[];seen=set()
for i,r in enumerate(ds):
    a=str(r.get('agent'))
    if a not in seen:
        chosen.append((i,r));seen.add(a)
    if len(seen)>=4:break
rows=[]
for i,r in chosen:
    ap=r.get('artifact_path');fp=hf_hub_download('NJU-LINK/CodeTraceBench',ap,repo_type='dataset')
    info={'index':i,'traj_id':r.get('traj_id'),'agent':r.get('agent'),'step_count':r.get('step_count'),'artifact_path':ap,'size':os.path.getsize(fp),'members':[]}
    with open(fp,'rb') as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as reader:
            with tarfile.open(fileobj=reader,mode='r|') as tf:
                for j,m in enumerate(tf):
                    if j>=120:break
                    item={'name':m.name,'size':m.size,'type':str(m.type)}
                    if m.isfile() and m.size<20_000_000 and any(m.name.lower().endswith(x) for x in ('.json','.jsonl')):
                        f=tf.extractfile(m)
                        if f:
                            b=f.read();s=b.decode('utf-8','ignore')
                            try:
                                obj=json.loads(s)
                                item['json_shape']=shape(obj)
                            except Exception:
                                lines=[]
                                for line in s.splitlines()[:3]:
                                    try:lines.append(shape(json.loads(line)))
                                    except Exception:lines.append(line[:500])
                                item['jsonl_shape']=lines
                    info['members'].append(item)
    rows.append(info)
(OUT/'probe.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps([{'traj_id':x['traj_id'],'agent':x['agent'],'step_count':x['step_count'],'artifact_path':x['artifact_path'],'members':[{'name':m['name'],'size':m['size'],'keys':list(m.get('json_shape',{}))[:20] if isinstance(m.get('json_shape'),dict) else []} for m in x['members'][:30]]} for x in rows],indent=2,ensure_ascii=False))
