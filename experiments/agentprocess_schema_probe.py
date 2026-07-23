#!/usr/bin/env python3
import json, os, re
from pathlib import Path
from huggingface_hub import snapshot_download

OUT=Path(os.environ.get('OUT','agentprocess_schema_probe'))
OUT.mkdir(parents=True,exist_ok=True)
ROOT=Path(snapshot_download(repo_id='LulaCola/AgentProcessBench',repo_type='dataset',local_dir='/tmp/AgentProcessBench'))
files=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()]
manifest={'root':str(ROOT),'files':files,'samples':{}}

def clean(x,d=0):
    if d>7:return '<depth>'
    if isinstance(x,dict):return {str(k):clean(v,d+1) for k,v in x.items()}
    if isinstance(x,list):return [clean(v,d+1) for v in x[:12]]
    s=str(x);s=re.sub(r'[0-9a-f]{16,}','<id>',s,flags=re.I);return s[:2000]

for p in [p for p in ROOT.rglob('*') if p.is_file()]:
    rel=str(p.relative_to(ROOT))
    try:
        if p.suffix in {'.json','.jsonl'}:
            if p.suffix=='.jsonl':
                rows=[]
                with p.open(encoding='utf-8',errors='ignore') as f:
                    for _,line in zip(range(3),f):rows.append(json.loads(line))
            else:
                obj=json.load(p.open(encoding='utf-8',errors='ignore'))
                rows=obj[:3] if isinstance(obj,list) else obj
            manifest['samples'][rel]=clean(rows)
        elif p.suffix=='.parquet':
            import pyarrow.parquet as pq
            t=pq.read_table(p)
            manifest['samples'][rel]={'schema':str(t.schema),'rows':t.num_rows,'sample':clean(t.slice(0,min(3,t.num_rows)).to_pylist())}
    except Exception as e:
        manifest['samples'][rel]={'error':repr(e)}

(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'files':files,'sample_keys':list(manifest['samples'])},indent=2,ensure_ascii=False))
