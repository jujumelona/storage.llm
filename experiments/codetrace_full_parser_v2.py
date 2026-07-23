#!/usr/bin/env python3
from __future__ import annotations
import ast, json, re
from pathlib import Path
import product_lifted_codetrace_full_blind as base
import product_lifted_agentprocess as core

ANSI_RE=re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

def safe(path):return path.read_text(encoding='utf-8',errors='ignore') if path.exists() else ''

def parse_event_dirs(root):
    candidates=[]
    for evdir in root.rglob('events'):
        if not evdir.is_dir():continue
        evs=[]
        for p in evdir.glob('*.json'):
            try:o=json.loads(safe(p))
            except Exception:continue
            if isinstance(o,dict):evs.append(o)
        def eid(x):
            try:return int(x.get('id',10**9))
            except Exception:return 10**9
        evs.sort(key=eid);obs={}
        for e in evs:
            c=e.get('cause')
            if isinstance(c,int) and ('observation' in e or e.get('observation')=='run'):obs.setdefault(c,e)
        out=[]
        for e in evs:
            if e.get('action') not in ('run','run_ipython'):continue
            args=e.get('args') or {};cmd=''
            if isinstance(args,dict):cmd=str(args.get('command') or args.get('code') or '')
            if not cmd:
                t=e.get('tool_call_metadata') or {}
                if isinstance(t,dict) and isinstance(t.get('args'),dict):cmd=str(t['args'].get('command') or '')
            if not cmd:continue
            oe=obs.get(e.get('id'));ob=str((oe or {}).get('content') or (oe or {}).get('observation') or '')
            out.append((cmd,ob))
        if out:candidates.append(out)
    return max(candidates,key=len) if candidates else []

def command_lines(path):
    out=[]
    if not path.exists():return out
    for line in safe(path).splitlines():
        try:v=ast.literal_eval(line.strip())
        except Exception:continue
        if isinstance(v,str):cmd=v.replace('\r\n','\n').replace('\r','\n').rstrip('\n')
        elif isinstance(v,list):
            parts=[str(x) for x in v if str(x) not in {'Enter','ENTER'}];cmd=''.join(parts).strip()
        else:continue
        if cmd and cmd!='C-d':out.append(cmd)
    return out

def parse_commands_log(root):
    best=[]
    for cp in root.rglob('commands.txt'):
        cmds=command_lines(cp)
        if not cmds:continue
        logs=list(cp.parent.rglob('agent.log'))
        if not logs:
            cand=[(c,'') for c in cmds]
        else:
            lines=[ANSI_RE.sub('',x) for x in safe(logs[0]).splitlines()];pos=0;cand=[]
            for cmd in cmds:
                hit=-1
                for i in range(pos,len(lines)):
                    if cmd in lines[i]:hit=i;break
                if hit<0:cand.append((cmd,''));continue
                end=hit+1
                while end<len(lines) and not re.match(r'^[^\s].*#\s',lines[end]):end+=1
                cand.append((cmd,'\n'.join(lines[hit+1:end])[:12000]));pos=hit+1
        if len(cand)>len(best):best=cand
    return best

def parse_agent_log_bash(root):
    best=[]
    for p in root.rglob('agent.log'):
        lines=safe(p).splitlines();out=[];i=0
        while i<len(lines):
            if lines[i].strip() in {'```bash','```sh'}:
                j=i+1
                while j<len(lines) and lines[j].strip()!='```':j+=1
                action='\n'.join(lines[i+1:j]).strip();k=j+1
                while k<len(lines) and '<returncode>' not in lines[k] and lines[k].strip() not in {'```bash','```sh'}:k+=1
                obs='';
                if k<len(lines) and '<returncode>' in lines[k]:
                    q=k
                    while q<len(lines) and not (q>k and lines[q].strip() in {'```bash','```sh'}):q+=1
                    obs='\n'.join(lines[k:q])[:12000]
                if action:out.append((action,obs));i=max(j,k)
            i+=1
        if len(out)>len(best):best=out
    return best

def parse_json_trajectory_generic(root):
    best=[]
    for p in list(root.rglob('*.traj'))+list(root.rglob('*.traj.json')):
        try:o=json.loads(safe(p))
        except Exception:continue
        seq=o if isinstance(o,list) else None
        if isinstance(o,dict):
            for k in ('trajectory','history','steps','traj'):
                if isinstance(o.get(k),list):seq=o[k];break
            if seq is None and isinstance(o.get('messages'),list):
                cand=base.parse_miniswe(root)
                if len(cand)>len(best):best=cand
                continue
        if not isinstance(seq,list):continue
        out=[]
        for z in seq:
            if not isinstance(z,dict):continue
            a=base.nested_text(z.get('action')) or base.nested_text(z.get('command'))
            if not a:
                r=base.nested_text(z.get('response'));m=base.BASH_RE.search(r);a=m.group(1).strip() if m else r
            ob=base.nested_text(z.get('observation',z.get('output','')))
            if a.strip():out.append((a,ob))
        if len(out)>len(best):best=out
    return best

def parse_artifact(agent,root,expected):
    named=[]
    funcs=[('event_dir',parse_event_dirs),('commands_log',parse_commands_log),('agent_log_bash',parse_agent_log_bash),('json_trajectory',parse_json_trajectory_generic),('miniswe',base.parse_miniswe),('sweagent',base.parse_sweagent),('openhands_api',base.parse_openhands),('terminus_episode',base.parse_terminus)]
    for name,fn in funcs:
        try:x=fn(root)
        except Exception:x=[]
        if x:named.append((name,x))
    if not named:return [],'none',[]
    named.sort(key=lambda z:(abs(len(z[1])-expected),-len(z[1]),z[0]))
    return named[0][1],named[0][0],[{'parser':n,'count':len(x)} for n,x in named]

def obs_status(action,obs):
    z=obs or ''
    if core.ERROR_RE.search(z):return 'ERROR'
    if core.PARTIAL_RE.search(z):return 'PARTIAL'
    if z.strip():return 'SUCCESS'
    return 'NO_RESULT'

def compile_pairs(tid,pairs,ablate=False):
    old=base.status;base.status=obs_status
    try:return base.compile_pairs(tid,pairs,ablate)
    finally:base.status=old
