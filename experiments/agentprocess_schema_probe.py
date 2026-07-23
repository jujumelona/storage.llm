#!/usr/bin/env python3
import json, os, re
from pathlib import Path
from datasets import load_dataset

OUT=Path(os.environ.get('OUT','agentprocess_schema_probe'))
OUT.mkdir(parents=True,exist_ok=True)

ds=load_dataset('LulaCola/AgentProcessBench')
manifest={}
for split,table in ds.items():
    manifest[split]={'rows':len(table),'features':str(table.features)}
    samples=[]
    for i in range(min(3,len(table))):
        row=table[i]
        def clean(x,d=0):
            if d>7:return '<depth>'
            if isinstance(x,dict):return {str(k):clean(v,d+1) for k,v in x.items()}
            if isinstance(x,list):return [clean(v,d+1) for v in x[:12]]
            s=str(x)
            s=re.sub(r'[0-9a-f]{16,}','<id>',s,flags=re.I)
            return s[:2000]
        samples.append(clean(row))
    (OUT/f'{split}_samples.json').write_text(json.dumps(samples,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(manifest,indent=2,ensure_ascii=False))
