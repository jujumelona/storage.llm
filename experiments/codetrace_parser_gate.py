#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip, hashlib, json, math, random, re, tempfile, time
from collections import Counter
from pathlib import Path
from typing import Any
from datasets import load_dataset
from huggingface_hub import hf_hub_download

import product_lifted_codetrace_full_blind as base
import codetrace_full_parser_v2 as old

SEED=20260723
random.seed(SEED)
DATASET='NJU-LINK/CodeTraceBench'
OUT=Path('codetrace_parser_gate')
CACHE=Path('/tmp/codetrace_parser_gate_cache')
OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)


def nested(x:Any)->str:
    if x is None:return ''
    if isinstance(x,str):return x
    if isinstance(x,list):return '\n'.join(nested(v) for v in x)
    if isinstance(x,dict):
        for k in ('text','content','command','keystrokes','action','observation','output','message','response','analysis','plan'):
            if x.get(k) is not None:
                z=nested(x[k])
                if z:return z
        try:return json.dumps(x,ensure_ascii=False,sort_keys=True)
        except Exception:return str(x)
    return str(x)


def download_retry(path:str,tries:int=9)->str:
    err=None
    for n in range(tries):
        try:return hf_hub_download(DATASET,path,repo_type='dataset',cache_dir=str(CACHE))
        except Exception as e:
            err=e;delay=min(50.,1.4*(2**n))+random.random()
            print(json.dumps({'retry':path,'attempt':n+1,'delay':round(delay,2),'error':type(e).__name__}),flush=True)
            time.sleep(delay)
    raise err


def response_message(obj):
    r=obj.get('response') if isinstance(obj,dict) else None
    if not isinstance(r,dict):return {}
    choices=r.get('choices') or []
    if not choices or not isinstance(choices[0],dict):return {}
    m=choices[0].get('message') or {}
    return m if isinstance(m,dict) else {}


def response_calls(msg):
    out=[]
    for tc in msg.get('tool_calls') or []:
        if not isinstance(tc,dict):continue
        f=tc.get('function') or {}
        if not isinstance(f,dict):continue
        out.append((str(f.get('name') or ''),nested(f.get('arguments')),str(tc.get('id') or '')))
    fc=msg.get('function_call')
    if isinstance(fc,dict):out.append((str(fc.get('name') or ''),nested(fc.get('arguments')),''))
    return out


def tool_results(msgs):
    out={}
    if not isinstance(msgs,list):return out
    for m in msgs:
        if not isinstance(m,dict) or str(m.get('role','')).lower() not in {'tool','function'}:continue
        key=str(m.get('tool_call_id') or m.get('name') or len(out));out[key]=nested(m.get('content'))
    return out


def openhands_candidates(root:Path):
    rec=[]
    for p in root.rglob('*.json'):
        low=p.name.lower()
        if any(x in low for x in ('report','context','metadata','config')):continue
        try:o=json.loads(old.safe(p))
        except Exception:continue
        if isinstance(o,dict) and isinstance(o.get('response'),dict) and isinstance(o.get('messages'),list):rec.append((p,o))
    rec.sort(key=lambda z:str(z[0]))
    if not rec:return []
    per_response=[];per_call=[];plus_text=[]
    for i,(_,o) in enumerate(rec):
        m=response_message(o);calls=response_calls(m);results={}
        for _,later in rec[i+1:i+5]:
            results.update(tool_results(later.get('messages')))
            if results:break
        if calls:
            aa=[];oo=[]
            for name,args,cid in calls:
                a=(name+' '+args).strip();ob=results.get(cid,results.get(name,''));aa.append(a);oo.append(ob);per_call.append((a,ob))
            pair=('\n'.join(aa),'\n'.join(x for x in oo if x));per_response.append(pair);plus_text.append(pair)
        else:
            text=nested(m.get('content'))
            if text.strip():plus_text.append((text,''))
    out=[]
    if per_response:out.append(('openhands_response',per_response,1))
    if per_call:out.append(('openhands_call',per_call,3))
    if plus_text:out.append(('openhands_plus_text',plus_text,7))
    return out


def cmd_values(x):
    if x is None:return []
    if isinstance(x,str):
        s=x.strip()
        if not s:return []
        try:
            o=json.loads(s);v=cmd_values(o)
            if v:return v
        except Exception:pass
        blocks=base.BASH_RE.findall(s)
        return [b.strip() for b in blocks if b.strip()] or [s]
    if isinstance(x,list):
        out=[]
        for v in x:
            if isinstance(v,dict):out+=cmd_values(v.get('command',v.get('keystrokes',v.get('action',v))))
            else:out+=cmd_values(v)
        return [z for z in out if z.strip()]
    if isinstance(x,dict):
        for k in ('commands','command','keystrokes','action','actions'):
            if k in x:
                v=cmd_values(x[k])
                if v:return v
        z=nested(x);return [z] if z.strip() else []
    return [str(x)]


def episode_roots(root):
    out=set()
    for p in root.rglob('response.txt'):
        if p.parent.name.startswith('episode-'):out.add(p.parent.parent)
    return sorted(out)


def terminus_candidates(root:Path):
    out=[]
    for ri,logs in enumerate(episode_roots(root)):
        eps=[]
        for p in logs.iterdir():
            if p.is_dir() and p.name.startswith('episode-'):
                try:eps.append(int(p.name.split('-',1)[1]))
                except Exception:pass
        if not eps:continue
        slots=[];nonempty=[];percmd=[]
        for sid in range(1,max(eps)+2):
            ap=logs/f'episode-{sid-1}'/'response.txt';op=logs/f'episode-{sid}'/'prompt.txt'
            raw=old.safe(ap) if ap.exists() else '';prompt=old.safe(op) if op.exists() else ''
            marker='New Terminal Output:';obs=prompt.split(marker,1)[1].strip() if marker in prompt else prompt.strip()
            try:o=json.loads(raw)
            except Exception:o=None
            if isinstance(o,dict):
                cmds=cmd_values(o.get('commands'));fallback='\n'.join(x for x in (nested(o.get('analysis')),nested(o.get('plan')),nested(o.get('explanation'))) if x)
            else:cmds=cmd_values(raw);fallback=''
            action='\n'.join(cmds) if cmds else fallback
            if ap.exists() or op.exists():slots.append((action,obs))
            if action.strip():nonempty.append((action,obs))
            if cmds:
                for c in cmds:percmd.append((c,obs))
            elif fallback.strip():percmd.append((fallback,obs))
        if percmd:out.append((f'terminus_per_command_{ri}',percmd,1))
        if nonempty:out.append((f'terminus_nonempty_{ri}',nonempty,5))
        if slots:out.append((f'terminus_slots_{ri}',slots,8))
    return out


def seq_json_candidates(root:Path):
    out=[]
    files=sorted([p for p in root.rglob('*.traj') if p.is_file()]+[p for p in root.rglob('*.traj.json') if p.is_file()],key=lambda p:p.stat().st_size,reverse=True)
    for fi,p in enumerate(files[:8]):
        try:o=json.loads(old.safe(p))
        except Exception:continue
        seq=o if isinstance(o,list) else None
        if isinstance(o,dict):
            for k in ('trajectory','history','steps','traj','turns','events'):
                if isinstance(o.get(k),list):seq=o[k];break
        if not isinstance(seq,list):continue
        item=[];flat=[]
        for z in seq:
            if not isinstance(z,dict):continue
            obs=nested(z.get('observation',z.get('output',z.get('result',''))));raw=z.get('action',z.get('command',z.get('response',z.get('thought',''))));cmds=cmd_values(raw)
            if not cmds:continue
            item.append(('\n'.join(cmds),obs))
            for c in cmds:flat.append((c,obs))
        if item:out.append((f'json_item_{fi}',item,3))
        if flat:out.append((f'json_flat_{fi}',flat,2))
    return out


def candidate_set(agent,root,expected):
    named=[]
    try:
        pairs,name,cands=old.parse_artifact(agent,root,expected)
        if pairs:named.append((f'old_selected:{name}',pairs,4))
        # Recover all old candidates where possible by calling public functions.
        for nm,fn in [('event_dir',old.parse_event_dirs),('commands_log',old.parse_commands_log),('agent_log_bash',old.parse_agent_log_bash),('json_trajectory',old.parse_json_trajectory_generic),('miniswe',base.parse_miniswe),('sweagent',base.parse_sweagent),('openhands_api',base.parse_openhands),('terminus_episode',base.parse_terminus)]:
            try:x=fn(root)
            except Exception:x=[]
            if x:named.append((nm,x,6))
    except Exception:pass
    if agent=='OpenHands':named+=openhands_candidates(root)
    if agent=='Terminus2':named+=terminus_candidates(root)
    if agent in {'SWE-agent','mini-SWE-agent'}:named+=seq_json_candidates(root)
    seen=set();out=[]
    for name,pairs,priority in named:
        sig=tuple((str(a)[:300],str(o)[:300]) for a,o in pairs)
        if not pairs or sig in seen:continue
        seen.add(sig);out.append((name,pairs,priority))
    return out


def select(cands,expected):
    if not cands:return None
    return min(cands,key=lambda z:(abs(len(z[1])-expected),0 if len(z[1])==expected else 1,z[2],-len(z[1]),z[0]))


def summary(stats):
    n=len(stats);parsed=sum(x['parsed']>0 for x in stats);exact=sum(x['parsed']==x['expected'] for x in stats);near=sum(abs(x['parsed']-x['expected'])<=1 for x in stats);within=sum(abs(x['parsed']-x['expected'])<=max(1,math.ceil(.1*x['expected'])) for x in stats)
    agents={}
    for a in sorted(set(x['agent'] for x in stats)):
        p=[x for x in stats if x['agent']==a];agents[a]={'n':len(p),'parsed':sum(x['parsed']>0 for x in p),'exact':sum(x['parsed']==x['expected'] for x in p),'near1':sum(abs(x['parsed']-x['expected'])<=1 for x in p),'mean_abs_error':sum(abs(x['parsed']-x['expected']) for x in p)/max(1,len(p)),'selected_parsers':dict(Counter(x['selected_parser'] for x in p))};agents[a]['parsed_rate']=agents[a]['parsed']/max(1,len(p));agents[a]['exact_rate']=agents[a]['exact']/max(1,len(p))
    gate={'parsed_at_least_95pct':parsed/max(1,n)>=.95,'exact_at_least_80pct':exact/max(1,n)>=.80,'near1_at_least_90pct':near/max(1,n)>=.90,'every_agent_parsed_at_least_90pct':all(v['parsed_rate']>=.90 for v in agents.values()),'every_agent_exact_at_least_60pct':all(v['exact_rate']>=.60 for v in agents.values())};gate['overall_pass']=all(gate.values())
    return {'benchmark':'CodeTraceBench verified','labels_read':False,'rows':n,'parsed':parsed,'parsed_rate':parsed/max(1,n),'exact_step_count':exact,'exact_rate':exact/max(1,n),'near1':near,'near1_rate':near/max(1,n),'within_10pct':within,'within_10pct_rate':within/max(1,n),'by_agent':agents,'strict_gate':gate}


def main():
    ds=load_dataset(DATASET,split='verified');stats=[];blind_path=OUT/'blind_events.jsonl.gz'
    with gzip.open(blind_path,'wt',encoding='utf-8') as blind:
        for i,row in enumerate(ds):
            tid=str(row['traj_id']);agent=str(row['agent']);expected=int(row['step_count']);ap=str(row['artifact_path']);cands=[];chosen=None;err='';t=time.perf_counter()
            try:
                fp=download_retry(ap)
                with tempfile.TemporaryDirectory(prefix='ctgate_') as td:
                    base.extract_tar_zst(fp,td);cands=candidate_set(agent,Path(td),expected);chosen=select(cands,expected)
            except Exception as e:err=type(e).__name__+':'+str(e)[:500]
            pairs=chosen[1] if chosen else [];events=old.compile_pairs(tid,pairs,False)
            if events:blind.write(json.dumps({'instance_id':tid,'agent':agent,'expected_step_count':expected,'selected_parser':chosen[0] if chosen else 'none','events':events},ensure_ascii=False,separators=(',',':'))+'\n')
            st={'index':i,'instance_id':tid,'agent':agent,'expected':expected,'parsed':len(events),'selected_parser':chosen[0] if chosen else 'none','candidate_counts':{n:len(p) for n,p,_ in cands},'error':err,'elapsed_sec':time.perf_counter()-t};stats.append(st)
            if (i+1)%10==0 or err:print(json.dumps({'processed':i+1,'total':len(ds),'agent':agent,'expected':expected,'parsed':len(events),'parser':st['selected_parser'],'error':err[:120]},ensure_ascii=False),flush=True)
            time.sleep(.08)
    s=summary(stats);(OUT/'parse_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8');contract={'version':'codetrace-parser-gate-v1','selection_uses':['agent','artifact schema','official step_count'],'selection_does_not_use':['incorrect_stages','incorrect_step_ids','failure labels','model predictions'],'blind_events_sha256':hashlib.sha256(blind_path.read_bytes()).hexdigest(),'stats_sha256':hashlib.sha256((OUT/'parse_stats.json').read_bytes()).hexdigest(),'summary_sha256':hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest()};(OUT/'parser_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(s,ensure_ascii=False,indent=2),flush=True);print('PARSER_GATE_PASSED' if s['strict_gate']['overall_pass'] else 'PARSER_GATE_FAILED',flush=True)

if __name__=='__main__':main()
