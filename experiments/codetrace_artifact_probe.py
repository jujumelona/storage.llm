#!/usr/bin/env python3
import io, json, os, re, tarfile
from pathlib import Path
import zstandard as zstd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

OUT=Path('codetrace_artifact_probe');OUT.mkdir(exist_ok=True)
ds=load_dataset('NJU-LINK/CodeTraceBench',split='verified')
rows=[]
for i in range(min(4,len(ds))):
    r=ds[i];ap=r.get('artifact_path')
    fp=hf_hub_download('NJU-LINK/CodeTraceBench',ap,repo_type='dataset')
    info={'index':i,'traj_id':r.get('traj_id'),'agent':r.get('agent'),'step_count':r.get('step_count'),'artifact_path':ap,'size':os.path.getsize(fp),'members':[]}
    with open(fp,'rb') as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as reader:
            with tarfile.open(fileobj=reader,mode='r|') as tf:
                for j,m in enumerate(tf):
                    if j>=100:break
                    item={'name':m.name,'size':m.size,'type':str(m.type)}
                    if m.isfile() and m.size<2_000_000 and any(m.name.lower().endswith(x) for x in ('.json','.jsonl','.yaml','.yml','.txt','.md')):
                        f=tf.extractfile(m)
                        if f:
                            s=f.read(min(m.size,12000)).decode('utf-8','ignore')
                            s=re.sub(r'[0-9a-f]{20,}','<id>',s,flags=re.I)
                            item['sample']=s[:4000]
                    info['members'].append(item)
    rows.append(info)
(OUT/'probe.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps([{'traj_id':x['traj_id'],'agent':x['agent'],'step_count':x['step_count'],'artifact_path':x['artifact_path'],'members':[m['name'] for m in x['members'][:30]]} for x in rows],indent=2,ensure_ascii=False))
