#!/usr/bin/env python3
from __future__ import annotations
import importlib, json, tempfile, traceback, time
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import product_lifted_codetrace_full_blind as base
import codetrace_full_parser_v2 as local

OUT=Path('codetrace_official_parser_probe')
OUT.mkdir(exist_ok=True)

IMPORTS = {
    'miniswe_official': 'codetracer.skills.seed.miniswe.parser',
    'openhands_official': 'codetracer.skills.seed.openhands.parser',
    'terminus_official': 'codetracer.skills.seed.terminus2.parser',
}
PARSERS={}
import_errors={}
for name,modname in IMPORTS.items():
    try:
        PARSERS[name]=importlib.import_module(modname).parser
    except Exception as e:
        import_errors[name]={
            'type':type(e).__name__,
            'message':str(e),
            'traceback':traceback.format_exc(limit=8),
        }

normalizer_error=None
try:
    from codetracer.query.normalizer import Normalizer
    from codetracer.skills.pool import SkillPool
except Exception as e:
    Normalizer=SkillPool=None
    normalizer_error={
        'type':type(e).__name__,
        'message':str(e),
        'traceback':traceback.format_exc(limit=8),
    }

ds=load_dataset('NJU-LINK/CodeTraceBench',split='verified')
chosen=[]
for agent in ('mini-SWE-agent','OpenHands','SWE-agent','Terminus2'):
    chosen.extend([(i,r) for i,r in enumerate(ds) if str(r['agent'])==agent][:8])

rows=[]
for n,(i,r) in enumerate(chosen,1):
    rec={
        'index':i,
        'traj_id':str(r['traj_id']),
        'agent':str(r['agent']),
        'expected':int(r['step_count']),
    }
    try:
        fp=hf_hub_download(
            'NJU-LINK/CodeTraceBench',
            r['artifact_path'],
            repo_type='dataset',
            cache_dir='/tmp/ct_official_cache',
        )
        with tempfile.TemporaryDirectory(prefix='cto_') as td:
            base.extract_tar_zst(fp,td)
            root=Path(td)
            for name,p in PARSERS.items():
                try:
                    rec[name+'_can_parse']=bool(p.can_parse(root))
                    if rec[name+'_can_parse']:
                        rec[name+'_count']=len(p.parse(root).steps)
                except Exception as e:
                    rec[name+'_error']=type(e).__name__+':'+str(e)[:400]
            if Normalizer is not None:
                try:
                    norm=Normalizer(SkillPool())
                    skill=norm.detect(root)
                    rec['auto_detect_skill']=getattr(skill,'name',getattr(skill,'format_id',type(skill).__name__))
                    rec['auto_detect_count']=len(norm.normalize(root,skill).steps)
                except Exception as e:
                    rec['auto_detect_error']=type(e).__name__+':'+str(e)[:400]
            try:
                pairs,pname,cands=local.parse_artifact(rec['agent'],root,rec['expected'])
                rec['local_selected']=pname
                rec['local_count']=len(pairs)
                rec['local_candidates']=cands
            except Exception as e:
                rec['local_error']=type(e).__name__+':'+str(e)[:400]
    except Exception as e:
        rec['download_or_extract_error']=type(e).__name__+':'+str(e)[:500]
    rows.append(rec)
    print(json.dumps({'processed':n,'agent':rec['agent'],'expected':rec['expected'],
                      'local':rec.get('local_count'),
                      'official':{k:v for k,v in rec.items() if k.endswith('_count')}},
                     ensure_ascii=False),flush=True)
    time.sleep(.15)

(OUT/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'import_errors.json').write_text(json.dumps({
    'parser_import_errors':import_errors,
    'normalizer_import_error':normalizer_error,
},ensure_ascii=False,indent=2),encoding='utf-8')

summary={}
for agent in sorted(set(x['agent'] for x in rows)):
    part=[x for x in rows if x['agent']==agent]
    keys=sorted(set(k for x in part for k in x if k.endswith('_count')))
    summary[agent]={}
    for k in keys:
        vals=[x[k] for x in part if isinstance(x.get(k),int)]
        summary[agent][k]={
            'available':len(vals),
            'exact':sum(x.get(k)==x['expected'] for x in part if isinstance(x.get(k),int)),
            'mean_abs_error':sum(abs(x[k]-x['expected']) for x in part if isinstance(x.get(k),int))/max(1,len(vals)),
        }
summary['_imports']={'available_parsers':sorted(PARSERS),'errors':import_errors,'normalizer_error':normalizer_error}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
