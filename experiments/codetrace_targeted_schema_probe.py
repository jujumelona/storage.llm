#!/usr/bin/env python3
from __future__ import annotations
import ast, json, re, tempfile, time
from collections import Counter, defaultdict
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import product_lifted_codetrace_full_blind as base
import codetrace_full_parser_v2 as old

OUT=Path('codetrace_targeted_schema_probe');OUT.mkdir(exist_ok=True)
CACHE=Path('/tmp/codetrace_targeted_schema_cache');CACHE.mkdir(exist_ok=True)
INDICES=[591,564,495,489,607,656,734,706,736,709,728,478,240,911,906,966,839]
HEX=re.compile(r'\b[0-9a-f]{16,}\b',re.I)
LONGNUM=re.compile(r'\b\d{8,}\b')

def safe_snip(x,n=180):
    if x is None:return ''
    if not isinstance(x,str):
        try:x=json.dumps(x,ensure_ascii=False,sort_keys=True)
        except Exception:x=str(x)
    x=HEX.sub('<id>',x);x=LONGNUM.sub('<num>',x);x=re.sub(r'\s+',' ',x).strip()
    return x[:n]

def download(path,tries=8):
    err=None
    for i in range(tries):
        try:return hf_hub_download('NJU-LINK/CodeTraceBench',path,repo_type='dataset',cache_dir=str(CACHE))
        except Exception as e:
            err=e;time.sleep(min(30,1.5*(2**i)))
    raise err

def json_obj(p):
    try:return json.loads(p.read_text(encoding='utf-8',errors='ignore'))
    except Exception:return None

def event_schema(root):
    files=[];actions=Counter();observations=Counter();keys=Counter();causes=0;ids=0;action_events=[];obs_causes=Counter();api=0
    for p in root.rglob('*.json'):
        o=json_obj(p)
        if isinstance(o,dict):
            files.append(p);keys.update(o.keys())
            if o.get('id') is not None:ids+=1
            if o.get('cause') is not None:causes+=1
            if 'action' in o:
                a=str(o.get('action'));actions[a]+=1
                if a and a.lower() not in {'none','null'}:action_events.append({'file':p.name,'action':a,'keys':sorted(o.keys()),'args_keys':sorted((o.get('args') or {}).keys()) if isinstance(o.get('args'),dict) else [],'has_tool_meta':isinstance(o.get('tool_call_metadata'),dict),'sample':safe_snip(o.get('args') or o.get('message') or '')})
            if 'observation' in o:observations[str(o.get('observation'))]+=1
            if isinstance(o.get('cause'),int):obs_causes[str(o.get('observation'))]+=1
            if isinstance(o.get('response'),dict) and isinstance(o.get('messages'),list):api+=1
    action_ids=set();obs_linked=set()
    for p in files:
        o=json_obj(p)
        if not isinstance(o,dict):continue
        if isinstance(o.get('id'),int) and o.get('action') not in (None,''):action_ids.add(o['id'])
        if isinstance(o.get('cause'),int) and ('observation' in o or 'content' in o):obs_linked.add(o['cause'])
    return {'json_dict_files':len(files),'key_counts':dict(keys.most_common(30)),'actions':dict(actions),'observations':dict(observations),'cause_count':causes,'id_count':ids,'api_response_files':api,'action_event_count':len(action_events),'action_with_linked_observation':len(action_ids&obs_linked),'action_without_linked_observation':len(action_ids-obs_linked),'action_event_examples':action_events[:30]}

def text_schema(root):
    out={'commands_txt':[],'agent_logs':[],'episode_roots':[],'traj_files':[]}
    for p in root.rglob('commands.txt'):
        lines=p.read_text(encoding='utf-8',errors='ignore').splitlines();parsed=[]
        for line in lines:
            try:v=ast.literal_eval(line)
            except Exception:v=None
            if isinstance(v,str):parsed.append(v)
            elif isinstance(v,list):parsed.append(''.join(str(x) for x in v if x!='Enter'))
        out['commands_txt'].append({'path':str(p.relative_to(root)),'lines':len(lines),'parsed_nonempty':sum(bool(x.strip()) for x in parsed),'unique':len(set(x for x in parsed if x.strip())),'examples':[safe_snip(x) for x in parsed[:8]]})
    for p in root.rglob('agent.log'):
        s=p.read_text(encoding='utf-8',errors='ignore');out['agent_logs'].append({'path':str(p.relative_to(root)),'lines':len(s.splitlines()),'bash_fences':s.count('```bash'),'returncodes':s.count('<returncode>'),'thought_markers':len(re.findall(r'(?m)^THOUGHT:',s)),'promptish':len(re.findall(r'(?m)^[^\s].*#\s',s))})
    for logs in root.rglob('agent-logs'):
        if not logs.is_dir():continue
        eps=[p for p in logs.iterdir() if p.is_dir() and p.name.startswith('episode-')]
        if eps:
            cmds=[]
            for ep in eps:
                rp=ep/'response.txt'
                if not rp.exists():continue
                s=rp.read_text(encoding='utf-8',errors='ignore')
                try:o=json.loads(s)
                except Exception:o=None
                c=(o.get('commands') or []) if isinstance(o,dict) else []
                cmds.append(len(c) if isinstance(c,list) else 0)
            out['episode_roots'].append({'path':str(logs.relative_to(root)),'episodes':len(eps),'responses':sum((ep/'response.txt').exists() for ep in eps),'prompts':sum((ep/'prompt.txt').exists() for ep in eps),'command_count_sum':sum(cmds),'command_count_hist':dict(Counter(cmds))})
    for p in list(root.rglob('*.traj'))+list(root.rglob('*.traj.json')):
        o=json_obj(p);rec={'path':str(p.relative_to(root)),'size':p.stat().st_size}
        if isinstance(o,dict):
            rec['top_keys']=sorted(o.keys())
            seq=None;seqkey=None
            for k in ('trajectory','history','steps','traj','turns','events','messages'):
                if isinstance(o.get(k),list):seq=o[k];seqkey=k;break
            if seq is not None:
                rec['seq_key']=seqkey;rec['seq_len']=len(seq);rec['entry_key_hist']=dict(Counter(tuple(sorted(x.keys())) for x in seq if isinstance(x,dict)))
                rec['nonempty_action']=sum(bool(old.nested(x.get('action',x.get('command',''))).strip()) for x in seq if isinstance(x,dict))
                rec['nonempty_response']=sum(bool(old.nested(x.get('response','')).strip()) for x in seq if isinstance(x,dict))
                rec['nonempty_observation']=sum(bool(old.nested(x.get('observation',x.get('output',''))).strip()) for x in seq if isinstance(x,dict))
                rec['role_hist']=dict(Counter(str(x.get('role')) for x in seq if isinstance(x,dict)))
        elif isinstance(o,list):rec['list_len']=len(o);rec['entry_key_hist']=dict(Counter(tuple(sorted(x.keys())) for x in o if isinstance(x,dict)))
        out['traj_files'].append(rec)
    return out

def parser_counts(agent,root,expected):
    counts={}
    fns={'event_dir':old.parse_event_dirs,'commands_log':old.parse_commands_log,'agent_log_bash':old.parse_agent_log_bash,'json_generic':old.parse_json_trajectory_generic,'miniswe':base.parse_miniswe,'sweagent':base.parse_sweagent,'openhands':base.parse_openhands,'terminus':base.parse_terminus}
    for n,fn in fns.items():
        try:counts[n]=len(fn(root))
        except Exception as e:counts[n]='ERR:'+type(e).__name__
    try:
        p,n,c=old.parse_artifact(agent,root,expected);counts['selected_name']=n;counts['selected_count']=len(p);counts['candidate_counts']=c
    except Exception as e:counts['selected_error']=type(e).__name__+':'+str(e)[:200]
    return counts

def main():
    ds=load_dataset('NJU-LINK/CodeTraceBench',split='verified');rows=[]
    for n,idx in enumerate(INDICES,1):
        r=ds[idx];rec={'index':idx,'traj_id':str(r['traj_id']),'agent':str(r['agent']),'expected_step_count':int(r['step_count']),'artifact_path':str(r['artifact_path'])}
        try:
            fp=download(rec['artifact_path'])
            with tempfile.TemporaryDirectory(prefix='ctschema_') as td:
                base.extract_tar_zst(fp,td);root=Path(td)
                rec['member_count']=sum(1 for p in root.rglob('*') if p.is_file())
                rec['suffix_hist']=dict(Counter(p.suffix.lower() or '<none>' for p in root.rglob('*') if p.is_file()))
                rec['basename_hist']=dict(Counter(p.name for p in root.rglob('*') if p.is_file()).most_common(30))
                rec['parser_counts']=parser_counts(rec['agent'],root,rec['expected_step_count'])
                rec['event_schema']=event_schema(root)
                rec['text_schema']=text_schema(root)
        except Exception as e:rec['error']=type(e).__name__+':'+str(e)[:500]
        rows.append(rec);print(json.dumps({'processed':n,'index':idx,'agent':rec['agent'],'expected':rec['expected_step_count'],'selected':rec.get('parser_counts',{}).get('selected_count'),'actions':rec.get('event_schema',{}).get('actions'),'error':rec.get('error','')},ensure_ascii=False),flush=True)
    (OUT/'probe.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    agg={}
    for agent in sorted(set(x['agent'] for x in rows)):
        p=[x for x in rows if x['agent']==agent];agg[agent]={'n':len(p),'expected_sum':sum(x['expected_step_count'] for x in p),'selected_sum':sum(x.get('parser_counts',{}).get('selected_count',0) for x in p if isinstance(x.get('parser_counts',{}).get('selected_count',0),int)),'actions':dict(sum((Counter(x.get('event_schema',{}).get('actions',{})) for x in p),Counter()))}
    (OUT/'summary.json').write_text(json.dumps({'labels_read':False,'indices':INDICES,'aggregate':agg},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(agg,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
