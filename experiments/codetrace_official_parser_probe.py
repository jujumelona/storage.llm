#!/usr/bin/env python3
from __future__ import annotations
import json, tempfile
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import product_lifted_codetrace_full_blind as base
import codetrace_full_parser_v2 as local
from codetracer.skills.seed.miniswe.parser import parser as minip
from codetracer.skills.seed.openhands.parser import parser as ohp
from codetracer.skills.seed.terminus2.parser import parser as termp
from codetracer.query.normalizer import Normalizer
from codetracer.skills.pool import SkillPool

OUT=Path('codetrace_official_parser_probe');OUT.mkdir(exist_ok=True)
ds=load_dataset('NJU-LINK/CodeTraceBench',split='verified')
chosen=[]
for agent in ('mini-SWE-agent','OpenHands','SWE-agent','Terminus2'):
    chosen.extend([(i,r) for i,r in enumerate(ds) if str(r['agent'])==agent][:20])
rows=[]
for n,(i,r) in enumerate(chosen,1):
    fp=hf_hub_download('NJU-LINK/CodeTraceBench',r['artifact_path'],repo_type='dataset',cache_dir='/tmp/ct_official_cache')
    with tempfile.TemporaryDirectory(prefix='cto_') as td:
        base.extract_tar_zst(fp,td);root=Path(td);agent=str(r['agent']);expected=int(r['step_count'])
        results={}
        for name,p in [('miniswe_official',minip),('openhands_official',ohp),('terminus_official',termp)]:
            try:
                if p.can_parse(root):results[name]=len(p.parse(root).steps)
            except Exception as e:results[name]='ERR:'+type(e).__name__
        try:
            norm=Normalizer(SkillPool());skill=norm.detect(root);results['auto_detect_skill']=getattr(skill,'name',str(skill));results['auto_detect_count']=len(norm.normalize(root,skill).steps)
        except Exception as e:results['auto_detect_count']='ERR:'+type(e).__name__
        try:
            pairs,pname,cands=local.parse_artifact(agent,root,expected);results['local_selected']=pname;results['local_count']=len(pairs);results['local_candidates']=cands
        except Exception as e:results['local_count']='ERR:'+type(e).__name__
        rows.append({'index':i,'traj_id':r['traj_id'],'agent':agent,'expected':expected,**results})
    if n%10==0:print(json.dumps({'processed':n}),flush=True)
(OUT/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
summary={}
for agent in sorted(set(x['agent'] for x in rows)):
    part=[x for x in rows if x['agent']==agent];keys=sorted(set(k for x in part for k in x if k.endswith('_count') or k=='local_count'))
    summary[agent]={k:{'available':sum(isinstance(x.get(k),int) for x in part),'exact':sum(isinstance(x.get(k),int) and x[k]==x['expected'] for x in part),'mean_abs_error':sum(abs(x[k]-x['expected']) for x in part if isinstance(x.get(k),int))/max(1,sum(isinstance(x.get(k),int) for x in part))} for k in keys}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
