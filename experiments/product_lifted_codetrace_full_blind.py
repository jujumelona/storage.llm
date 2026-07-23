#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, re, shutil, tarfile, tempfile
from pathlib import Path
import zstandard as zstd

import product_lifted_agentprocess as core
import product_lifted_agentprocess_v2 as v2

SEED=20260723
random.seed(SEED)
KINDS=("current","ueta","graph","product_no_topology","product")
CFG={"hist":2,"future":0,"bad_precision":.35,"support":2,"hard_bad":4,"topk":4,"hard_weight":.55,"transition_weight":.45,"unavoidable_bonus":1.3,"distance_bonus":.7,"local_weight":.15,"route_weight":.15,"boundary_mode":"early"}
PATH_RE=re.compile(r"(?:/[A-Za-z0-9_.-]+)+|https?://\S+|[A-Za-z0-9_.-]+\.(?:py|js|ts|json|yaml|yml|txt|md|csv|rs|go|java|cpp|c|h)",re.I)
TEST_RE=re.compile(r"\b(test|pytest|unittest|check|verify|validate|assert|compare|diff|lint|mypy|compile|cargo test|go test)\b",re.I)
MUTATE_RE=re.compile(r"\b(edit|write|patch|replace|create|delete|remove|modify|update|apply_patch|str_replace|sed|cat\s*>)\b",re.I)
READ_RE=re.compile(r"\b(read|view|open|cat|grep|find|search|list|ls|inspect|head|tail|show|pwd|stat)\b",re.I)
EXEC_RE=re.compile(r"\b(run|execute|bash|shell|python|node|npm|make|cmake|cargo|java|terminal)\b",re.I)
FINAL_RE=re.compile(r"\b(final|submit|done|completed|solution|answer)\b",re.I)
BASH_RE=re.compile(r"```(?:bash|sh)\s*\n(.*?)\n?```",re.I|re.S)

def nested_text(x):
    if x is None:return ""
    if isinstance(x,str):return x
    if isinstance(x,list):return "\n".join(nested_text(v) for v in x)
    if isinstance(x,dict):
        for k in ("text","content","command","keystrokes","action","observation","output","message"):
            if k in x and x[k] is not None:
                z=nested_text(x[k])
                if z:return z
        return json.dumps(x,ensure_ascii=False,sort_keys=True)
    return str(x)

def parse_jsonish(x):
    if isinstance(x,(dict,list)):return x
    if isinstance(x,str):
        try:return json.loads(x)
        except Exception:return x
    return x

def operation(action):
    t=action.lower()
    if FINAL_RE.search(t):return "FINAL"
    if TEST_RE.search(t):return "VERIFY"
    if MUTATE_RE.search(t):return "MUTATE"
    if EXEC_RE.search(t):return "EXECUTE"
    if READ_RE.search(t):return "READ"
    if "?" in action or "plan" in t or "thought" in t:return "PLAN"
    return "TOOL_OTHER"

def resources(text,ablate=False):
    if ablate:return ("none",)
    out=set()
    for s in PATH_RE.findall(text):
        if s.startswith("http"):out.add("url")
        elif "/" in s:out.add("path")
        else:out.add("file")
    low=text.lower()
    for k in ("repo","file","path","url","query","package","test","module","function","class"):
        if k in low:out.add(k)
    return tuple(sorted(out)[:5]) or ("none",)

def status(action,obs):
    z=f"{action}\n{obs}"
    if core.ERROR_RE.search(z):return "ERROR"
    if core.PARTIAL_RE.search(z):return "PARTIAL"
    if obs.strip():return "SUCCESS"
    return "NO_RESULT"

def extract_tar_zst(path,out):
    with open(path,'rb') as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as reader:
            with tarfile.open(fileobj=reader,mode='r|') as tf:
                tf.extractall(out)

def parse_miniswe(root):
    files=sorted(root.rglob('*.traj.json'),key=lambda p:p.stat().st_size,reverse=True)
    if not files:return []
    try:obj=json.loads(files[0].read_text(encoding='utf-8',errors='ignore'))
    except Exception:return []
    msgs=obj.get('messages') if isinstance(obj,dict) else None
    if not isinstance(msgs,list):return []
    out=[]
    for i,m in enumerate(msgs):
        if not isinstance(m,dict) or m.get('role')!='assistant':continue
        content=nested_text(m.get('content'));mm=BASH_RE.search(content)
        if not mm:continue
        action=mm.group(1).strip();obs=''
        for z in msgs[i+1:]:
            if not isinstance(z,dict):continue
            if z.get('role')=='assistant':break
            c=nested_text(z.get('content'))
            if '<returncode>' in c or '<output>' in c:obs=c;break
        out.append((action,obs))
    return out

def parse_sweagent(root):
    files=sorted((p for p in root.rglob('*.traj') if p.is_file()),key=lambda p:p.stat().st_size,reverse=True)
    if not files:return []
    try:obj=json.loads(files[0].read_text(encoding='utf-8',errors='ignore'))
    except Exception:return []
    seq=None
    if isinstance(obj,list):seq=obj
    elif isinstance(obj,dict):
        for k in ('trajectory','history','steps','traj'):
            if isinstance(obj.get(k),list):seq=obj[k];break
    if not isinstance(seq,list):return []
    out=[]
    for x in seq:
        if not isinstance(x,dict):continue
        action=nested_text(x.get('action'))
        if not action:
            response=nested_text(x.get('response'))
            m=BASH_RE.search(response);action=m.group(1).strip() if m else response
        obs=nested_text(x.get('observation',x.get('output','')))
        if action.strip():out.append((action,obs))
    return out

def response_action(obj):
    if not isinstance(obj,dict):return ''
    r=obj.get('response') or {};choices=r.get('choices') if isinstance(r,dict) else None
    if not isinstance(choices,list) or not choices:return ''
    msg=choices[0].get('message') if isinstance(choices[0],dict) else None
    if not isinstance(msg,dict):return ''
    calls=msg.get('tool_calls') or []
    parts=[]
    for tc in calls:
        if not isinstance(tc,dict):continue
        f=tc.get('function') or {};parts.append(str(f.get('name',''))+' '+nested_text(f.get('arguments','')))
    if parts:return '\n'.join(parts)
    return nested_text(msg.get('content'))

def request_last_observation(obj):
    msgs=obj.get('messages') if isinstance(obj,dict) else None
    if not isinstance(msgs,list) or not msgs:return ''
    for m in reversed(msgs):
        if not isinstance(m,dict):continue
        if str(m.get('role','')).lower() in {'tool','function'}:return nested_text(m.get('content'))
        if str(m.get('role','')).lower() in {'assistant','user'}:break
    return ''

def parse_openhands(root):
    files=[]
    for p in root.rglob('*.json'):
        low=p.name.lower()
        if any(x in low for x in ('report','output','context','result')):continue
        try:obj=json.loads(p.read_text(encoding='utf-8',errors='ignore'))
        except Exception:continue
        if isinstance(obj,dict) and isinstance(obj.get('response'),dict) and isinstance(obj.get('messages'),list):files.append((p,obj))
    files.sort(key=lambda z:z[0].name)
    out=[]
    for i,(p,obj) in enumerate(files):
        action=response_action(obj)
        if not action.strip():continue
        obs=request_last_observation(files[i+1][1]) if i+1<len(files) else ''
        out.append((action,obs))
    return out

def response_txt_action(s):
    try:o=json.loads(s)
    except Exception:o=None
    if isinstance(o,dict):
        cmds=o.get('commands') or []
        if isinstance(cmds,list) and cmds:
            parts=[]
            for c in cmds:
                if isinstance(c,dict):parts.append(nested_text(c.get('keystrokes','')))
                else:parts.append(nested_text(c))
            z='\n'.join(x for x in parts if x)
            if z:return z
        return '\n'.join(x for x in (nested_text(o.get('analysis')),nested_text(o.get('plan'))) if x)
    return s.strip()

def parse_terminus(root):
    logs=next((p for p in root.rglob('agent-logs') if p.is_dir()),None)
    if logs is None:return []
    eps=[]
    for p in logs.iterdir():
        if p.is_dir() and p.name.startswith('episode-'):
            try:eps.append(int(p.name.split('-',1)[1]))
            except Exception:pass
    if not eps:return []
    out=[]
    for sid in range(1,max(eps)+2):
        ap=logs/f'episode-{sid-1}'/'response.txt';op=logs/f'episode-{sid}'/'prompt.txt'
        if not ap.exists() and not op.exists():continue
        action=response_txt_action(ap.read_text(encoding='utf-8',errors='ignore')) if ap.exists() else ''
        obs=op.read_text(encoding='utf-8',errors='ignore') if op.exists() else ''
        marker='New Terminal Output:'
        if marker in obs:obs=obs.split(marker,1)[1].strip()
        if action.strip():out.append((action,obs))
    return out

def parse_artifact(agent,root):
    if agent=='mini-SWE-agent':return parse_miniswe(root)
    if agent=='SWE-agent':return parse_sweagent(root)
    if agent=='OpenHands':return parse_openhands(root)
    if agent=='Terminus2':return parse_terminus(root)
    return []

def compile_pairs(tid,pairs,ablate=False):
    events=[];mask=0;last_op='START';last_res=('none',);last_status='NONE';repeats=0
    for pos,(action,obs) in enumerate(pairs):
        op=operation(action);res=resources(action+'\n'+obs,ablate);st=status(action,obs);hard=0
        if st=='SUCCESS':mask&=~core.UNRESOLVED_ERROR
        if st=='ERROR':mask|=core.UNRESOLVED_ERROR;hard+=1
        if st=='PARTIAL':mask|=core.PARTIAL_EVIDENCE;hard+=1
        if op in {'READ','VERIFY'} and st=='SUCCESS':
            mask&=~core.PARTIAL_EVIDENCE
            if op=='VERIFY':mask&=~core.NEED_VERIFY
        if op=='MUTATE':mask|=core.NEED_VERIFY
        if op==last_op and res==last_res and st in {'ERROR','NO_RESULT'} and last_status in {'ERROR','NO_RESULT'}:
            repeats+=1;mask|=core.REPEAT_NO_PROGRESS;hard+=2+min(2,repeats)
        else:repeats=0;mask&=~core.REPEAT_NO_PROGRESS
        if op=='FINAL':
            if mask&core.UNRESOLVED_ERROR:hard+=5
            if mask&core.PARTIAL_EVIDENCE:hard+=3
            if mask&core.NEED_VERIFY:hard+=2
        atoms=(f'OP:{op}',f'ST:{st}',f'MASK:{mask}',f'RES:{"+".join(res)}',f'TLEN:{core.bucket(len(action))}',f'OLEN:{core.bucket(len(obs))}',f'PREV:{last_op}',f'PREVST:{last_status}',f'HARD:{min(hard,6)}')
        events.append({'step_id':pos+1,'op':op,'status':st,'resources':list(res),'atoms':list(atoms),'hard':hard,'mask':mask,'text_len':len(action),'obs_len':len(obs)})
        last_op,last_res,last_status=op,res,st
    return events

def incorrect_ids(row):
    x=row.get('incorrect_stages');
    if isinstance(x,str):
        try:x=json.loads(x)
        except Exception:x=[]
    out=set()
    for s in x or []:
        if not isinstance(s,dict):continue
        for q in s.get('incorrect_step_ids') or []:
            try:out.add(int(q))
            except Exception:pass
        for st in s.get('steps') or []:
            if not isinstance(st,dict):continue
            labs=st.get('labels') or []
            if isinstance(labs,str):
                try:labs=json.loads(labs)
                except Exception:labs=[labs]
            if any(str(v).lower()=='incorrect' for v in labs):
                try:out.add(int(st['step_id']))
                except Exception:pass
    return sorted(out)

def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    ds=load_dataset('NJU-LINK/CodeTraceBench',split='verified');out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    blind=open(out/'blind_events.jsonl','w',encoding='utf-8');labels=[];stats=[]
    cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
    for idx,row in enumerate(ds):
        tid=str(row['traj_id']);agent=str(row['agent']);bad=incorrect_ids(row);ap=row.get('artifact_path');pairs=[];err=''
        try:
            fp=hf_hub_download('NJU-LINK/CodeTraceBench',ap,repo_type='dataset',cache_dir=str(cache))
            with tempfile.TemporaryDirectory(prefix='ctb_') as td:
                extract_tar_zst(fp,td);pairs=parse_artifact(agent,Path(td))
        except Exception as e:err=type(e).__name__+':'+str(e)[:300]
        ev=compile_pairs(tid,pairs,False);eva=compile_pairs(tid,pairs,True)
        if ev:
            blind.write(json.dumps({'instance_id':tid,'agent':agent,'expected_step_count':int(row['step_count']),'events':ev,'events_ablation':eva},ensure_ascii=False)+'\n')
        if bad:labels.append({'instance_id':tid,'agent':agent,'gold_step':min(bad),'incorrect_steps':bad,'expected_step_count':int(row['step_count'])})
        stats.append({'instance_id':tid,'agent':agent,'expected':int(row['step_count']),'parsed':len(ev),'has_gold':bool(bad),'max_gold':max(bad) if bad else None,'parse_error':err})
        if (idx+1)%25==0:print(json.dumps({'processed':idx+1,'parsed':sum(x['parsed']>0 for x in stats),'exact_count':sum(x['parsed']==x['expected'] for x in stats)},ensure_ascii=False),flush=True)
    blind.close();(out/'labels_sealed.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf-8');(out/'parse_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    parsed=sum(x['parsed']>0 for x in stats);exact=sum(x['parsed']==x['expected'] for x in stats);cover=sum(x['parsed']>=x['max_gold'] for x in stats if x['has_gold'] and x['parsed']>0);goldn=sum(x['has_gold'] for x in stats)
    manifest={'rows':len(ds),'parsed':parsed,'exact_step_count':exact,'gold_covered':cover,'gold_rows':goldn,'blind_sha256':hashlib.sha256((out/'blind_events.jsonl').read_bytes()).hexdigest(),'labels_sha256':hashlib.sha256((out/'labels_sealed.json').read_bytes()).hexdigest()}
    (out/'prepare_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))

def train_source(root):
    tr=core.load_all(root);return tr,core.train_model(tr,CFG)

def event_from(tid,pos,x):
    return core.Event(tid,'codetrace_full',pos,int(x['step_id']),'0',x['op'],x['status'],tuple(x['resources']),tuple(x['atoms']),int(x['hard']),int(x['mask']),int(x['text_len']),0,False)

def predict(args):
    if 'label' in Path(args.blind).name.lower():raise RuntimeError('label-like predictor input')
    source,model=train_source(Path(args.source));rows=[]
    for line in Path(args.blind).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        x=json.loads(line);es=[event_from(x['instance_id'],i,z) for i,z in enumerate(x['events'])];ea=[event_from(x['instance_id'],i,z) for i,z in enumerate(x['events_ablation'])]
        tr={'tid':x['instance_id'],'dataset':'codetrace_full','events':es,'has_error':False,'first_error_msg':None};ta={**tr,'events':ea};item={'instance_id':x['instance_id'],'agent':x['agent'],'n_events':len(es),'expected_step_count':x['expected_step_count'],'predictions':{},'ablation':{}}
        for k in KINDS:
            p=core.predict(tr,model,k);q=core.predict(ta,model,k);item['predictions'][k]=p+1 if p>=0 else -1;item['ablation'][k]=q+1 if q>=0 else -1
        item['predictions']['first_step']=1;item['ablation']['first_step']=1;rows.append(item)
    payload={'protocol':'Frozen AgentProcessBench Product-Lifted model on full CodeTrace raw trajectories before labels','source_config':CFG,'source_training_trajectories':len(source),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'blind_sha256':hashlib.sha256(Path(args.blind).read_bytes()).hexdigest(),'predictor_had_labels':False,'predictions':rows}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8');Path(str(out)+'.sha256').write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'\n',encoding='utf-8');print(json.dumps({'predictions':len(rows),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))

def metric(rows,k,field='predictions'):
    if not rows:return {'n':0,'exact':0.,'near1':0.,'near3':0.,'mae':0.}
    d=[abs(int(r[field][k])-int(r['gold_step'])) for r in rows];n=len(d);return {'n':n,'exact':sum(x==0 for x in d)/n,'near1':sum(x<=1 for x in d)/n,'near3':sum(x<=3 for x in d)/n,'mae':sum(d)/n}

def paired(rows,a='product',b='current',nboot=10000):
    vals=[];im=de=0
    for r in rows:
        ca=int(r['predictions'][a]==r['gold_step']);cb=int(r['predictions'][b]==r['gold_step']);vals.append(ca-cb);im+=ca>cb;de+=cb>ca
    rng=random.Random(SEED);boots=[]
    for _ in range(nboot):boots.append(sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals))
    boots.sort();return {'n':len(vals),'gain':sum(vals)/len(vals),'bootstrap_95_ci':[boots[int(.025*nboot)],boots[int(.975*nboot)]],'improved':im,'degraded':de}

def evaluate(args):
    pp=Path(args.predictions);payload=json.loads(pp.read_text(encoding='utf-8'));expected=Path(str(pp)+'.sha256').read_text().strip();actual=hashlib.sha256(pp.read_bytes()).hexdigest()
    if expected!=actual:raise RuntimeError('predictions changed after sealing')
    lab={x['instance_id']:x for x in json.loads(Path(args.labels).read_text(encoding='utf-8'))};rows=[]
    for p in payload['predictions']:
        g=lab.get(p['instance_id'])
        if g and p['n_events']>=g['gold_step']:
            z=dict(p);z.update(g);rows.append(z)
    allk=('first_step',)+KINDS;metrics={k:metric(rows,k) for k in allk};abl={k:metric(rows,k,'ablation') for k in allk};groups={}
    for a in sorted(set(r['agent'] for r in rows)):
        part=[r for r in rows if r['agent']==a];groups[a]={k:metric(part,k) for k in allk}
    pc=paired(rows,'product','current');pu=paired(rows,'product','ueta');pt=paired(rows,'product','product_no_topology');deltas=[groups[a]['product']['exact']-groups[a]['current']['exact'] for a in groups if groups[a]['product']['n']>=20]
    pm=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    strict={'hash_verified':expected==actual and payload.get('predictor_had_labels') is False,'parsed_at_least_800':pm['parsed']>=800,'exact_step_count_rate_80pct':pm['exact_step_count']/pm['rows']>=.80,'gold_covered_at_least_700':len(rows)>=700,'product_plus_5pp_over_current':metrics['product']['exact']>=metrics['current']['exact']+.05,'bootstrap_lower_positive':pc['bootstrap_95_ci'][0]>0,'product_beats_ueta':metrics['product']['exact']>metrics['ueta']['exact'],'topology_ablation_positive':metrics['product']['exact']>metrics['product_no_topology']['exact'],'all_large_agent_groups_nonnegative':all(x>=-1e-12 for x in deltas) if deltas else False,'identity_ablation_product_not_below_current':abl['product']['exact']>=abl['current']['exact']}
    summary={'benchmark':'CodeTraceBench verified full-trajectory first incorrect step blind transfer','prepare_manifest':pm,'evaluated':len(rows),'metrics':metrics,'identity_ablation':abl,'agent_groups':groups,'paired_product_minus_current':pc,'paired_product_minus_ueta':pu,'paired_product_minus_no_topology':pt,'prediction_sha256':actual,'code_sha256':payload['code_sha256'],'strict':strict,'overall_pass':all(strict.values())}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['instance_id','agent','n_events','expected_step_count','gold_step']+[f'{k}_step' for k in allk];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            z={q:r[q] for q in ('instance_id','agent','n_events','expected_step_count','gold_step')};z.update({f'{k}_step':r['predictions'][k] for k in allk});w.writerow(z)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='mode',required=True)
    p=sp.add_parser('prepare');p.add_argument('--out',required=True);p.add_argument('--cache',required=True)
    p=sp.add_parser('predict');p.add_argument('--source',required=True);p.add_argument('--blind',required=True);p.add_argument('--out',required=True)
    p=sp.add_parser('evaluate');p.add_argument('--predictions',required=True);p.add_argument('--labels',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    a=ap.parse_args();{'prepare':prepare,'predict':predict,'evaluate':evaluate}[a.mode](a)
if __name__=='__main__':main()
