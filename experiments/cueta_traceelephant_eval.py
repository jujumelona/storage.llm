#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,hashlib,itertools,json,math,random,re,statistics,tarfile,time,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import trail_nondense_v4_runner as base
import ueta_causal_generalization as cu
import ueta_trail_eval as ue
SEED=20260722;random.seed(SEED)
ERR=re.compile(r'(?i)\b(error|exception|failed|failure|timeout|invalid|denied|refused|unavailable|not found|traceback|wrong|incorrect)\b')
OK=re.compile(r'(?i)\b(success|succeeded|completed|done|passed|resolved|ok)\b')
EXC=re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b');HTTP=re.compile(r'\b([45]\d\d)\b');WORD=re.compile(r'[A-Za-z][A-Za-z0-9_./:-]{2,40}')
BAN=('L:','B:','T:','N:','PN:','P:')
def norm(x,n=80):
 s=str(x or '').strip().lower();s=re.sub(r'[0-9a-f]{8,}','<id>',s);s=re.sub(r'\d+','<n>',s);s=re.sub(r'\s+','_',s);return re.sub(r'[^a-z0-9_./:<>=+-]','',s)[:n] or 'empty'
def b(n,c=(0,1,2,4,8,16,32,64,128)):return str(sum(n>x for x in c))
def flat(x,p='',d=0):
 if d>9:return
 if isinstance(x,dict):
  for k,v in x.items():yield from flat(v,(p+'.' if p else '')+str(k),d+1)
 elif isinstance(x,list):
  for v in x:yield from flat(v,p+'[]',d+1)
 else:yield p.lower(),x
def pint(x):
 try:return int(x)
 except Exception:
  m=re.search(r'-?\d+',str(x));return int(m.group()) if m else None
def unpack(root):
 for p in list(root.rglob('*')):
  if not p.is_file():continue
  try:
   if p.name.lower().endswith('.zip'):
    q=p.parent/(p.stem+'_unpacked');q.mkdir(exist_ok=True);zipfile.ZipFile(p).extractall(q)
   elif p.name.lower().endswith(('.tar','.tar.gz','.tgz')):
    q=p.parent/(p.name.split('.')[0]+'_unpacked');q.mkdir(exist_ok=True);tarfile.open(p).extractall(q)
  except Exception:pass
def agent(step):
 keys=('agent_name','agent','role','sender','source','speaker','owner','executor')
 if isinstance(step,dict):
  d={str(k).lower():v for k,v in step.items()}
  for k in keys:
   if k in d and isinstance(d[k],(str,int)) and len(str(d[k]))<120:return str(d[k])
 for p,v in flat(step):
  if p.split('.')[-1].replace('[]','') in keys and isinstance(v,(str,int)) and len(str(v))<120:return str(v)
 return 'unknown'
def aclass(a):
 a=norm(a)
 for x in ('user','assistant','system','tool','orchestrator','planner','coder','reviewer','surfer','executor'):
  if x in a:return x
 return 'agent' if a not in ('empty','unknown') else 'unknown'
def event(step,pos):
 fv=list(flat(step));txt='\n'.join(str(v) for _,v in fv if isinstance(v,(str,int,float,bool)));lo=txt.lower();a=agent(step);fs={'K:event','A:'+aclass(a),'D:'+b(max([p.count('.') for p,_ in fv] or [0])),'Z:'+b(len(txt)),'C:'+b(len(fv))}
 keys={p.split('.')[-1].replace('[]','') for p,_ in fv}
 for k in keys:
  if any(x in k for x in ('input','prompt','request','question')):fs.add('Q:input')
  if any(x in k for x in ('output','response','result','observation')):fs.add('Q:output')
  if 'message' in k or 'content' in k:fs.add('Q:message')
  if 'tool' in k or 'function' in k:fs.add('Q:tool')
  if 'code' in k or 'patch' in k or 'command' in k:fs.add('Q:code')
  if 'error' in k or 'exception' in k:fs.add('Q:error_field')
 fs.add('S:error' if ERR.search(txt) else ('S:success' if OK.search(txt) else 'S:neutral'))
 for x in set(EXC.findall(txt)):fs.add('E:'+norm(x))
 for x in set(HTTP.findall(txt)):fs.add('H:'+x[0])
 tools=[];names=[]
 for p,v in fv:
  if not isinstance(v,(str,int)) or len(str(v))>100:continue
  leaf=p.split('.')[-1].replace('[]','')
  if any(x in p for x in ('tool','function','command','action')) and leaf in ('name','tool','tool_name','function','command','action'):tools.append(norm(v))
  elif leaf in ('name','type','event_type','kind'):names.append(norm(v))
 for x in tools[:3]:fs.add('T:'+x)
 for x in names[:3]:fs.add('N:'+x)
 seen=[]
 for x in WORD.findall(lo):
  x=norm(x,36)
  if x not in seen and x not in ('the','and','that','this','with','from','agent','step','message','content'):seen.append(x)
 for x in seen[:16]:fs.add('L:'+x)
 sid=pos+1
 if isinstance(step,dict):
  d={str(k).lower():v for k,v in step.items()}
  for k in ('step_number','step','index','step_id','round','turn'):
   if k in d and pint(d[k]) is not None:sid=pint(d[k]);break
 return {'sid':str(sid),'agent':a,'aclass':aclass(a),'tool':tools[:1],'features':fs,'evidence':txt[:500]}
def compile_seq(tid,steps):
 es=[event(x,i) for i,x in enumerate(steps)];pa=Counter();pt=Counter();pe=0;rows=[]
 for i,e in enumerate(es):
  fs=set(e['features']);fs.add('R:'+str(min(i,5)))
  if i==0:fs.add('Q:first')
  else:
   p=es[i-1];fs.add('PN:'+p['aclass']);fs.add('Q:same_agent' if norm(p['agent'])==norm(e['agent']) else 'Q:agent_switch')
   if 'S:error' in p['features']:fs.add('Q:after_error')
   if p['tool'] and e['tool'] and p['tool'][0]==e['tool'][0]:fs.add('Q:same_tool_again')
  if pe:fs.add('Q:prior_error')
  if pa[norm(e['agent'])]:fs.add('Q:agent_repeat')
  if e['tool'] and pt[e['tool'][0]]:fs.add('Q:tool_repeat')
  pa[norm(e['agent'])]+=1
  if e['tool']:pt[e['tool'][0]]+=1
  if 'S:error' in fs:pe+=1
  if i==len(es)-1:fs.add('Q:last')
  rows.append({'tid':tid,'sid':e['sid'],'features':frozenset(fs),'evidence':e['evidence']})
 return rows,es
def resolve(es,g):
 g=pint(g)
 if g is None:return None
 q=[i for i,e in enumerate(es) if pint(e['sid'])==g]
 if len(q)==1:return q[0]
 if 1<=g<=len(es):return g-1
 if 0<=g<len(es):return g
 return None
def load(root):
 unpack(root);unpack(root);tr=[];skip=Counter()
 for mp in root.rglob('trace_metadata.json'):
  sp=mp.parent/'step_records.json'
  if not sp.exists():continue
  try:m=json.load(open(mp));steps=json.load(open(sp))
  except Exception:skip['json']+=1;continue
  if not isinstance(steps,list) or not steps or m.get('mistake_step') is None or m.get('mistake_agent') is None:skip['schema']+=1;continue
  tid=str(m.get('task_id') or m.get('trajectory_id') or mp.parent.name);rows,es=compile_seq(tid,steps);gi=resolve(es,m['mistake_step'])
  if gi is None:skip['step']+=1;continue
  gold={(rows[gi]['sid'],'Failure')};tr.append({'tid':tid,'system':str(m.get('system_name') or mp.parents[1].name),'records':rows,'events':es,'gold':gold,'gi':gi,'gold_agent':str(m['mistake_agent'])})
 return tr,dict(skip)
def stripped(records):
 out=[]
 for r in records:q=dict(r);q['features']=frozenset(x for x in r['features'] if not x.startswith(BAN));out.append(q)
 return out
def flatten_traces(ts,clean=True):return [r for t in ts for r in (stripped(t['records']) if clean else t['records'])]
def goldmap(ts):return {t['tid']:t['gold'] for t in ts}
def top1(scores,ts):
 out={}
 for t in ts:
  arr=scores.get(t['tid'],[]);best=max(arr,key=lambda x:(x[0],-next((i for i,e in enumerate(t['events']) if e['sid']==x[1]),999999)),default=None)
  out[t['tid']]=best[1] if best else t['records'][0]['sid']
 return out
def match(a,g):
 a=norm(a);g=norm(g);return a==g or (a not in ('empty','unknown') and g not in ('empty','unknown') and (a in g or g in a))
def metrics(pred,ts):
 rows=[]
 for t in ts:
  sid=pred[t['tid']];pi=next((i for i,e in enumerate(t['events']) if e['sid']==sid),0);ac=match(t['events'][pi]['agent'],t['gold_agent'])
  rows.append({'tid':t['tid'],'system':t['system'],'gold_index':t['gi'],'pred_index':pi,'gold_step':next(iter(t['gold']))[0],'pred_step':sid,'gold_agent':t['gold_agent'],'pred_agent':t['events'][pi]['agent'],'step_exact':pi==t['gi'],'step_tol1':abs(pi-t['gi'])<=1,'agent_correct':ac,'joint_exact':pi==t['gi'] and ac})
 n=len(rows);m={k:sum(x[k] for x in rows)/n for k in ('step_exact','step_tol1','agent_correct','joint_exact')};m['n']=n;return m,rows
def split(ts,prefix='v'):
 v=[t for t in ts if int(hashlib.sha256((prefix+t['tid']).encode()).hexdigest()[:8],16)%5==0];c=[t for t in ts if t not in v]
 if len(v)<2:q=sorted(ts,key=lambda x:x['tid']);v=q[:max(1,len(q)//5)];c=q[len(v):]
 return c,v
def cal_top(scores,ts):return metrics(top1(scores,ts),ts)[0]['step_exact']
def fit_u(core,val,horizon):
 cr=flatten_traces(core);vr=flatten_traces(val);g=goldmap(core);best=(-1,None)
 for depth in (1,2,3):
  for ms in (2,3,5):
   routes=cu.discover_routes(cr,{t['tid'] for t in core},depth,horizon,ms)
   for p in (.03,.06,.12,.2):
    cs=cu.map_routes(cr,{t['tid'] for t in core},g,routes,depth,horizon,2,p)
    if not cs:continue
    a=cal_top(cu.score(vr,{t['tid'] for t in val},cs,depth,horizon),val)
    if a>best[0]:best=(a,{'depth':depth,'min_support':ms,'min_precision':p,'routes':len(routes),'contracts':len(cs)})
 return best[1] or {'depth':1,'min_support':2,'min_precision':.03}
def train_u(ts,cfg,h):
 r=flatten_traces(ts);ids={t['tid'] for t in ts};routes=cu.discover_routes(r,ids,cfg['depth'],h,cfg['min_support']);cs=cu.map_routes(r,ids,goldmap(ts),routes,cfg['depth'],h,2,cfg['min_precision']);return routes,cs
def fit_b(core,val):
 cr=flatten_traces(core);vr=flatten_traces(val);best=(-1,None)
 for mp in (2,3,5):
  rules=base.learn_rules(cr,{t['tid'] for t in core},goldmap(core),mp);a=cal_top(base.score_records(vr,{t['tid'] for t in val},rules,True),val)
  if a>best[0]:best=(a,{'min_positive':mp})
 return best[1]
def train_b(ts,cfg):return base.learn_rules(flatten_traces(ts),{t['tid'] for t in ts},goldmap(ts),cfg['min_positive'])
def outer(ts,folds=5):
 br=[];ur=[];fi=[]
 for f in range(folds):
  test=[t for t in ts if int(hashlib.sha256(('o'+t['tid']).encode()).hexdigest()[:8],16)%folds==f];train=[t for t in ts if t not in test]
  if not test:continue
  core,val=split(train);bc=fit_b(core,val);best=(-1,None)
  for h in (0,1):
   uc=fit_u(core,val,h);_,cs=train_u(core,uc,h);a=cal_top(cu.score(flatten_traces(val),{t['tid'] for t in val},cs,uc['depth'],h),val)
   if a>best[0]:best=(a,(h,uc))
  h,uc=best[1];brr=train_b(train,bc);routes,cs=train_u(train,uc,h);bp=top1(base.score_records(flatten_traces(test),{t['tid'] for t in test},brr,True),test);up=top1(cu.score(flatten_traces(test),{t['tid'] for t in test},cs,uc['depth'],h),test);bm,brows=metrics(bp,test);um,urows=metrics(up,test);br+=brows;ur+=urows;fi.append({'fold':f,'train':len(train),'test':len(test),'horizon':h,'baseline':bm,'cueta':um,'baseline_rules':len(brr),'routes':len(routes),'contracts':len(cs),'baseline_cfg':bc,'cueta_cfg':uc})
 def agg(r):
  n=len(r);return {'n':n,**{k:sum(x[k] for x in r)/n for k in ('step_exact','step_tol1','agent_correct','joint_exact')}}
 return agg(br),agg(ur),br,ur,fi
def holdout(ts):
 out=[]
 for s,n in Counter(t['system'] for t in ts).items():
  te=[t for t in ts if t['system']==s];tr=[t for t in ts if t['system']!=s]
  if len(te)<5 or len(tr)<15:continue
  c,v=split(tr,'x');bc=fit_b(c,v);best=(-1,None)
  for h in (0,1):
   uc=fit_u(c,v,h);_,cs=train_u(c,uc,h);a=cal_top(cu.score(flatten_traces(v),{t['tid'] for t in v},cs,uc['depth'],h),v)
   if a>best[0]:best=(a,(h,uc))
  h,uc=best[1];bp=top1(base.score_records(flatten_traces(te),{t['tid'] for t in te},train_b(tr,bc),True),te);_,cs=train_u(tr,uc,h);up=top1(cu.score(flatten_traces(te),{t['tid'] for t in te},cs,uc['depth'],h),te);bm,_=metrics(bp,te);um,_=metrics(up,te);out.append({'held_out_system':s,'train':len(tr),'test':len(te),'baseline':bm,'cueta':um,'step_gain':um['step_exact']-bm['step_exact'],'joint_gain':um['joint_exact']-bm['joint_exact']})
 return out
def bootstrap(b,u,n=2000):
 B={x['tid']:x['step_exact'] for x in b};U={x['tid']:x['step_exact'] for x in u};ids=sorted(B);r=random.Random(SEED);g=[]
 for _ in range(n):
  s=[r.choice(ids) for _ in ids];g.append(statistics.mean(U[x]-B[x] for x in s))
 g.sort();return {'mean':statistics.mean(g),'ci95_low':g[int(.025*n)],'ci95_high':g[int(.975*n)]}
def writecsv(p,r):
 if not r:Path(p).write_text('');return
 k=list(dict.fromkeys(x for z in r for x in z));
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=k);w.writeheader();w.writerows(r)
def main():
 a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--out',required=True);x=a.parse_args();out=Path(x.out);out.mkdir(parents=True,exist_ok=True);ts,skip=load(Path(x.data))
 if len(ts)<30:raise RuntimeError('parsed only '+str(len(ts))+' '+str(skip))
 bm,um,br,ur,fi=outer(ts);ho=holdout(ts);boot=bootstrap(br,ur);c,v=split(ts,'final');best=(-1,None)
 for h in (0,1):
  uc=fit_u(c,v,h);_,cs=train_u(c,uc,h);q=cal_top(cu.score(flatten_traces(v),{t['tid'] for t in v},cs,uc['depth'],h),v)
  if q>best[0]:best=(q,(h,uc))
 h,cfg=best[1];routes,cs=train_u(ts,cfg,h);ck={'architecture':'C-UETA sparse external trace attribution','mode':'structure_only','horizon':h,'config':cfg,'contracts':cs,'forbidden_dense_components_used':[]}
 with gzip.open(out/'CUETA_TRACE_SINGLE_CHECKPOINT.json.gz','wt') as f:json.dump(ck,f,default=list)
 sample=flatten_traces(ts)[:500];ids={r['tid'] for r in sample};sub=cs[:5000];t=time.perf_counter();A=cu.score(sample,ids,sub,cfg['depth'],h);na=time.perf_counter()-t;t=time.perf_counter();B=cu.score(sample,ids,sub,cfg['depth'],h);ix=time.perf_counter()-t
 hg=statistics.mean(z['step_gain'] for z in ho) if ho else None;ver={'oof_gain_3pp':um['step_exact']-bm['step_exact']>=.03,'bootstrap_positive':boot['ci95_low']>0,'heldout_system_gain':hg is not None and hg>0,'joint_not_worse':um['joint_exact']>=bm['joint_exact'],'checkpoint_created':(out/'CUETA_TRACE_SINGLE_CHECKPOINT.json.gz').exists()};ver['external_supported']=all(ver.values())
 summary={'architecture':'C-UETA external sparse trace attribution','dataset':'TraceElephant','forbidden_dense_components_used':[],'traces':len(ts),'events':sum(len(t['events']) for t in ts),'systems':Counter(t['system'] for t in ts),'skipped':skip,'baseline_oof':bm,'cueta_oof':um,'step_gain':um['step_exact']-bm['step_exact'],'joint_gain':um['joint_exact']-bm['joint_exact'],'bootstrap':boot,'folds':fi,'system_holdout':ho,'system_holdout_mean_step_gain':hg,'checkpoint':{'horizon':h,'config':cfg,'routes':len(routes),'contracts':len(cs),'bytes':(out/'CUETA_TRACE_SINGLE_CHECKPOINT.json.gz').stat().st_size},'runtime_replay':{'events':len(sample),'contracts':len(sub),'answers_identical':A==B,'first_seconds':na,'second_seconds':ix},'strict_verdict':ver,'scope_note':'No embedding, dense latent state, pretrained encoder, LLM call, hidden-state method, TF-IDF/SVD, nearest-vector retrieval, code generation, or external solver.'};json.dump(summary,open(out/'summary.json','w'),indent=2,default=dict);writecsv(out/'baseline_predictions.csv',br);writecsv(out/'cueta_predictions.csv',ur);writecsv(out/'system_holdout.csv',ho);print(json.dumps(summary,indent=2,default=dict))
if __name__=='__main__':main()
