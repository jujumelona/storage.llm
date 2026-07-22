#!/usr/bin/env python3
import argparse,json,re,sys
from pathlib import Path
import boundary_ueta_traceelephant as b
OPS={
 'read':re.compile(r'(?i)\b(read|open|view|load|inspect|cat|head|tail)\b'),
 'search':re.compile(r'(?i)\b(search|find|query|browse|grep|locate)\b'),
 'write':re.compile(r'(?i)\b(write|edit|patch|modify|update|replace|create|delete|remove|cancel|commit|submit|apply_patch)\b'),
 'execute':re.compile(r'(?i)\b(run|execute|command|shell|bash|python|invoke|call)\b'),
 'test':re.compile(r'(?i)\b(test|verify|check|assert|lint|pytest|validation)\b'),
 'plan':re.compile(r'(?i)\b(plan|reason|think|strategy|todo)\b'),
 'final':re.compile(r'(?i)\b(final|answer|finish|complete|terminate)\b')}
ERR=re.compile(r'(?i)\b(error|failed|failure|exception|traceback|invalid|wrong|incorrect|timeout)\b')
PATH=re.compile(r'(?i)(?:\b[a-z0-9_.-]+/[a-z0-9_./-]+|\b[a-z0-9_.-]+\.(?:py|js|ts|java|cpp|go|rs|json|yaml|yml|toml|md)\b)')
_old_keys=b.keys;_old_raw=b.raw_scores
def op(ev):
 t=ev.get('evidence','');return frozenset(k for k,p in OPS.items() if p.search(t))
def keys(t,i):
 out=set(_old_keys(t,i));cur=op(t['events'][i]);prev=op(t['events'][i-1]) if i else frozenset();nxt=op(t['events'][i+1]) if i+1<len(t['events']) else frozenset()
 for x in cur:out.add(('OP',x));out.add(('OP_TRANS',tuple(sorted(prev)) or ('start',),x));out.add(('OP_NEXT',x,tuple(sorted(nxt)) or ('end',)))
 for x in cur-prev:out.add(('OP_ADD',x,tuple(sorted(prev)) or ('start',)))
 if PATH.search(t['events'][i].get('evidence','')):out.add(('ENTITY','path'))
 if i and PATH.search(t['events'][i-1].get('evidence','')) and PATH.search(t['events'][i].get('evidence','')):out.add(('ENTITY_REL','path_continuity'))
 return out
def raw_scores(t,w,topk=10):
 base=[];seen=set()
 for i,ev in enumerate(t['events']):
  vals=sorted((w[k] for k in keys(t,i) if k in w),reverse=True);learn=sum(vals[:topk]);ops=op(ev);prior=0.
  if 'write' in ops:prior+=2.2
  if 'test' in ops:prior+=.9
  if 'execute' in ops:prior+=.45
  if 'final' in ops:prior+=.2
  if ERR.search(ev.get('evidence','')):prior+=1.8
  if 'write' in ops and 'test' not in seen:prior+=.5
  if i==0:prior-=.8
  if ops-seen:prior+=.2
  seen|=ops;base.append(learn+.08*prior if learn>0 else prior)
 return base
b.keys=keys;b.raw_scores=raw_scores
def main():
 b.main();ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--out');a,_=ap.parse_known_args();p=Path(a.out)/'summary.json';d=json.load(open(p));d['architecture']='Relation-Boundary-UETA explicit operation/event hazard engine';d['system_holdout_min_step_gain']=min(x['step_gain'] for x in d['system_holdout']);d['strict_verdict']['all_system_holdouts_nonnegative']=d['system_holdout_min_step_gain']>=0;d['strict_verdict']['external_supported']=all(d['strict_verdict'][k] for k in ('oof_gain_3pp','bootstrap_positive','joint_not_worse','all_system_holdouts_nonnegative'));json.dump(d,open(p,'w'),indent=2);print(json.dumps({'revised_strict_verdict':d['strict_verdict'],'system_holdout_min_step_gain':d['system_holdout_min_step_gain']},indent=2))
if __name__=='__main__':main()
