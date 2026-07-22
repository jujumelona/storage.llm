#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,random,re,statistics
from collections import Counter,defaultdict
from pathlib import Path
import cueta_traceelephant_eval as e
import boundary_ueta_traceelephant as b
SEED=20260722;random.seed(SEED)
ORIG_EVENT=e.event
ERR=re.compile(r'(?i)\b(error|failed|failure|exception|traceback|invalid|incorrect|timeout|not found|no such|assertionerror|syntaxerror|nameerror|typeerror)\b')
OK=re.compile(r'(?i)\b(passed|success|succeeded|completed|ok|all tests passed|done)\b')
PATH=re.compile(r'(?i)(/[-a-z0-9_./]+|\b[-a-z0-9_]+\.(?:py|js|ts|java|cpp|c|h|go|rs|json|ya?ml|toml|md|txt)\b)')
TEST=re.compile(r'(?i)(pytest|unittest|\btest[s_./-]|manage\.py|npm test|cargo test|go test|python\s+[^\n]*test)')
SEARCH=re.compile(r'(?i)^(find|grep|rg|search|locate)\b')
EXEC=re.compile(r'(?i)^(python|pytest|bash|sh|node|npm|cargo|go|make|tox|django-admin|./)\b')

def text(x):
 if isinstance(x,dict):return '\n'.join(text(v) for v in x.values())
 if isinstance(x,list):return '\n'.join(text(v) for v in x)
 return '' if x is None else str(x)
def decode(x):
 if isinstance(x,dict):return x
 try:return json.loads(str(x))
 except Exception:return {'raw':str(x)}
def current_calls(step):
 out=[]
 if not isinstance(step,dict):return out
 for row in step.get('tool_logs') or []:
  if not isinstance(row,dict):continue
  f=row.get('function') or {};out.append((str(f.get('name') or ''),decode(f.get('arguments') or '{}')))
 if out:return out
 # Fallback only when direct tool_logs are absent.
 o=step.get('output') or {}
 for ch in o.get('choices') or []:
  msg=(ch or {}).get('message') or {}
  for tc in msg.get('tool_calls') or []:
   f=(tc or {}).get('function') or {};out.append((str(f.get('name') or ''),decode(f.get('arguments') or '{}')))
 return out
def content_last(step):
 try:
  ms=step.get('input',{}).get('messages') or []
  return text(ms[-1].get('content') if isinstance(ms[-1],dict) else ms[-1]) if ms else ''
 except Exception:return ''
def etype(p):
 p=p.lower()
 if '/test' in p or re.search(r'(^|[/_.-])test(s|ing)?([/_.-]|$)',p):return 'test'
 if p.endswith(('.py','.js','.ts','.java','.cpp','.c','.h','.go','.rs')):return 'source'
 if p.endswith(('.json','.yaml','.yml','.toml','.ini','.cfg','.conf')):return 'config'
 if p.endswith(('.md','.txt','.rst')):return 'doc'
 return 'path'
def enhanced_event(step,pos):
 ev=ORIG_EVENT(step,pos);calls=current_calls(step);ops=set();ents=set();cmds=[]
 for name,args in calls:
  n=name.lower();cmd=str(args.get('command') or args.get('cmd') or args.get('action') or '').strip();c=cmd.lower();cmds.append(c)
  for k,v in args.items():
   if 'path' in str(k).lower() or str(k).lower() in ('file','filename'):
    for p in PATH.findall(str(v)):ents.add(p)
  for p in PATH.findall(text(args)):ents.add(p)
  if c in ('view','read','open','list') or c.startswith(('cat ','head ','tail ','ls ')):ops.add('read')
  if c in ('str_replace','insert','create','write','replace','patch','apply_patch','modify','update'):ops.add('write')
  if c in ('delete','remove') or c.startswith(('rm ','git rm ')):ops.add('delete')
  if c=='submit' or n=='submit':ops.add('submit')
  if SEARCH.search(c):ops.add('search')
  if TEST.search(c):ops.add('test')
  if EXEC.search(c) or n=='bash':ops.add('execute')
  if not c and n:ops.add('tool')
 ev['_calls']=calls;ev['_ops']=frozenset(ops);ev['_entities']=frozenset(ents);ev['_entity_types']=frozenset(etype(p) for p in ents);ev['_command']='\n'.join(cmds);ev['_last_input']=content_last(step);return ev
e.event=enhanced_event

def annotate(t):
 if '_resource_atoms' in t:return t['_resource_atoms']
 es=t['events'];n=len(es);result=['']*n
 for i in range(n-1):result[i]=es[i+1].get('_last_input','')
 # Current output can contain immediate failures when no next step exists.
 if n:result[-1]=es[-1].get('_last_input','')
 last_write={};seen=set();open_entities=set();all_open=False;had_test=False;last_test_ok=False;prior_fail=False;atoms=[set() for _ in es]
 mutation_indices=[];fail_indices=[]
 for i,ev in enumerate(es):
  ops=set(ev.get('_ops',()));ents=set(ev.get('_entities',()));types=set(ev.get('_entity_types',()));res=result[i];failed=bool(ERR.search(res));ok=bool(OK.search(res)) and not failed
  for x in ops:atoms[i].add(('OP',x))
  for x in types:atoms[i].add(('ETYPE',x))
  atoms[i].add(('OPSET',tuple(sorted(ops)) or ('none',)))
  if i:atoms[i].add(('OP_TRANS',tuple(sorted(es[i-1].get('_ops',()))) or ('none',),tuple(sorted(ops)) or ('none',)))
  atoms[i].add(('ENTITY_N',str(min(4,len(ents)))))
  new=ents-seen;reuse=ents&seen
  if new:atoms[i].add(('ENTITY_NEW',tuple(sorted(set(etype(x) for x in new)))))
  if reuse:atoms[i].add(('ENTITY_REUSE',tuple(sorted(set(etype(x) for x in reuse)))))
  if any(x in last_write for x in ents):atoms[i].add(('SAME_AS_PRIOR_WRITE',tuple(sorted(set(etype(x) for x in ents if x in last_write)))))
  if 'write' in ops or 'delete' in ops:
   mutation_indices.append(i);atoms[i].add(('MUTATION',tuple(sorted(types)) or ('unknown',)))
   if not mutation_indices[:-1]:atoms[i].add(('FIRST_MUTATION',))
   if any(x in last_write for x in ents):atoms[i].add(('REWRITE',tuple(sorted(set(etype(x) for x in ents if x in last_write)))))
   if prior_fail:atoms[i].add(('RECOVERY_MUTATION_AFTER_FAIL',))
   if ents:open_entities|=ents
   else:all_open=True
   for x in ents:last_write[x]=i
   last_test_ok=False
  if 'test' in ops or ('execute' in ops and had_test):
   had_test=True;atoms[i].add(('VERIFY', 'fail' if failed else 'ok' if ok else 'unknown'))
   if failed:
    fail_indices.append(i);prior_fail=True;last_test_ok=False
    # Link failure back to same-resource writer, otherwise latest mutation.
    linked={last_write[x] for x in ents if x in last_write}
    if not linked and mutation_indices:linked={mutation_indices[-1]}
    for j in linked:atoms[j].add(('CAUSAL_MUTATION_TO_FAIL',str(min(5,i-j))))
   elif ok or not ERR.search(res):
    last_test_ok=True;prior_fail=False;open_entities.clear();all_open=False
  if failed and 'test' not in ops:
   fail_indices.append(i);prior_fail=True;atoms[i].add(('RESULT_FAIL',))
   if mutation_indices:atoms[mutation_indices[-1]].add(('LAST_MUTATION_BEFORE_FAIL',str(min(5,i-mutation_indices[-1]))))
  if 'submit' in ops:
   atoms[i].add(('COMMIT',))
   if open_entities or all_open:atoms[i].add(('COMMIT_WITH_OPEN_OBLIGATION',str(min(4,len(open_entities)+(1 if all_open else 0)))))
   if prior_fail:atoms[i].add(('COMMIT_AFTER_UNRESOLVED_FAIL',))
   if not had_test:atoms[i].add(('COMMIT_WITHOUT_TEST',))
   if last_test_ok:atoms[i].add(('COMMIT_AFTER_SUCCESSFUL_VERIFY',))
   if mutation_indices:
    atoms[i].add(('COMMIT_AFTER_MUTATION_DISTANCE',str(min(5,i-mutation_indices[-1]))));atoms[mutation_indices[-1]].add(('LAST_MUTATION_BEFORE_COMMIT',str(min(5,i-mutation_indices[-1]))))
  atoms[i].add(('OPEN_OBLIGATION',str(min(4,len(open_entities)+(1 if all_open else 0)))))
  seen|=ents
 # Generic future relations, never literal entity identities.
 for i,ev in enumerate(es):
  ents=set(ev.get('_entities',()))
  for j in range(i+1,min(n,i+8)):
   common=ents & set(es[j].get('_entities',()))
   if common:
    atoms[i].add(('ENTITY_REAPPEARS',str(min(5,j-i)),tuple(sorted(set(etype(x) for x in common)))))
    if 'write' in ev.get('_ops',()) and ('test' in es[j].get('_ops',()) or ERR.search(result[j])):atoms[i].add(('OBLIGATION_FUTURE_CHECK',str(min(5,j-i))))
    break
 t['_resource_atoms']=[frozenset(x) for x in atoms];return t['_resource_atoms']

OLD_KEYS=b.keys
def keys(t,i):return set(OLD_KEYS(t,i))|set(annotate(t)[i])
b.keys=keys

def raw_scores(t,w,topk=10):
 aa=annotate(t);out=[]
 for i,ev in enumerate(t['events']):
  vals=sorted((w[k] for k in keys(t,i) if k in w),reverse=True);learn=sum(vals[:topk]);a=aa[i];prior=0.
  if ('COMMIT',) in a:prior+=2.0
  if any(x[0]=='COMMIT_WITH_OPEN_OBLIGATION' for x in a):prior+=2.5
  if ('COMMIT_AFTER_UNRESOLVED_FAIL',) in a:prior+=2.5
  if any(x[0]=='CAUSAL_MUTATION_TO_FAIL' for x in a):prior+=2.0
  if any(x[0]=='LAST_MUTATION_BEFORE_FAIL' for x in a):prior+=1.2
  if ('VERIFY','fail') in a:prior+=1.0
  if any(x[0]=='RECOVERY_MUTATION_AFTER_FAIL' for x in a):prior+=.8
  if any(x[0]=='FIRST_MUTATION' for x in a):prior+=.25
  if i==0:prior-=1.0
  out.append(learn+.35*prior if learn else prior)
 return out
b.raw_scores=raw_scores

def main():
 b.main();ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--out');a,_=ap.parse_known_args();p=Path(a.out)/'summary.json';d=json.load(open(p));d['architecture']='Resource-Obligation UETA explicit action/entity/verification graph';d['system_holdout_min_step_gain']=min(x['step_gain'] for x in d['system_holdout']);d['strict_verdict']['all_system_holdouts_nonnegative']=d['system_holdout_min_step_gain']>=0;d['strict_verdict']['external_supported']=all(d['strict_verdict'][k] for k in ('oof_gain_3pp','bootstrap_positive','joint_not_worse','all_system_holdouts_nonnegative'));json.dump(d,open(p,'w'),indent=2);print(json.dumps({'revised_strict_verdict':d['strict_verdict'],'system_holdout_min_step_gain':d['system_holdout_min_step_gain']},indent=2))
if __name__=='__main__':main()
