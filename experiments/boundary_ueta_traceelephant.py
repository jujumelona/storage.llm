#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,random,statistics
from collections import Counter
from pathlib import Path
import cueta_traceelephant_eval as e
import trail_nondense_v4_runner as base
SEED=20260722;random.seed(SEED)
_orig=e.load
def load(root):
 ts,sk=_orig(root);seen=Counter()
 for t in ts:
  k=(t['system'],t['tid']);seen[k]+=1;z=f"{t['system']}::{t['tid']}::{seen[k]}";t['tid']=z
  for r in t['records']:r['tid']=z
 return ts,sk
def role(x):
 a=x['aclass']
 return a if a in ('user','system','tool') else 'worker'
def atoms(ev):
 fs=ev['features'];a={'ROLE:'+role(ev)}
 for old,new in [('S:error','STATUS:error'),('S:success','STATUS:success'),('S:neutral','STATUS:neutral'),('Q:tool','KIND:tool'),('Q:code','KIND:code'),('Q:input','KIND:input'),('Q:output','KIND:output'),('Q:message','KIND:message'),('Q:error_field','KIND:error_field'),('Q:agent_switch','REL:agent_switch'),('Q:same_agent','REL:same_agent'),('Q:after_error','REL:after_error'),('Q:prior_error','REL:prior_error'),('Q:agent_repeat','REL:agent_repeat'),('Q:tool_repeat','REL:tool_repeat'),('Q:same_tool_again','REL:same_tool_again'),('Q:first','BOUND:first'),('Q:last','BOUND:last')]:
  if old in fs:a.add(new)
 return frozenset(a)
def sig(ev):
 a=atoms(ev);kind=next((x for x in ('STATUS:error','KIND:tool','KIND:code','KIND:output','KIND:input','KIND:message','STATUS:success','STATUS:neutral') if x in a),'KIND:other');return (next(x for x in a if x.startswith('ROLE:')),kind)
def phase(i,n):return 'PHASE:'+str(min(4,int(5*i/max(1,n))))
def keys(t,i):
 es=t['events'];n=len(es);cur=atoms(es[i]);cs=sig(es[i]);prev=sig(es[i-1]) if i else ('START','START');nxt=sig(es[i+1]) if i+1<n else ('END','END');out={('CUR',x) for x in cur};out|={('TRANS',prev,cs),('NEXT',cs,nxt),('TRI',prev,cs,nxt),('PHASE',phase(i,n),cs)}
 if i:
  pa=atoms(es[i-1]);
  for x in sorted(cur-pa):out.add(('ADD',x,prev,cs))
  for x in sorted(pa-cur):out.add(('DROP',x,prev,cs))
 if i+1<n:
  na=atoms(es[i+1])
  for x in sorted(na-cur):out.add(('FUTURE_ADD',x,cs,nxt))
 pref=es[:i];out.add(('PREFIX_ERROR',str(min(3,sum('S:error' in q['features'] for q in pref))),cs));out.add(('PREFIX_SWITCH',str(min(3,sum('Q:agent_switch' in q['features'] for q in pref))),cs))
 repeated=sum(sig(q)==cs for q in es[:i]);out.add(('REPEAT_SIG',str(min(3,repeated)),cs))
 return out
def discover(ts,minsup):
 c=Counter()
 for t in ts:
  for i in range(len(t['events'])):c.update(keys(t,i))
 return {k:n for k,n in c.items() if n>=minsup}
def fit(ts,minsup=3,minpos=2,minprec=.03):
 routes=discover(ts,minsup);tot=Counter();pos=Counter();N=sum(len(t['events']) for t in ts);B=(len(ts)+1)/(N+2);rs=set(routes)
 for t in ts:
  for i in range(len(t['events'])):
   m=keys(t,i)&rs;tot.update(m)
   if i==t['gi']:pos.update(m)
 w={}
 for k,pn in pos.items():
  if pn<minpos:continue
  pr=(pn+1)/(tot[k]+2);lift=math.log(max(1e-12,pr/B))
  if pr>=minprec and lift>0:w[k]=lift+.2*math.log1p(pn)+.03*math.log1p(routes[k])
 return routes,w
def raw_scores(t,w,topk=10):
 out=[]
 for i in range(len(t['events'])):
  v=sorted((w[k] for k in keys(t,i) if k in w),reverse=True);prior=.02*('S:error' in t['events'][i]['features'])+.01*('Q:after_error' in t['events'][i]['features']);out.append(sum(v[:topk])+prior)
 return out
def transform(r,mode,alpha):
 z=[];mx=0.
 for i,x in enumerate(r):
  p=r[i-1] if i else 0.
  if mode=='delta':y=x-alpha*p
  elif mode=='cumulative':y=x-alpha*mx
  else:y=x
  z.append(y);mx=max(mx,x)
 return z
def predict(ts,w,topk,mode,alpha):
 out={}
 for t in ts:
  s=transform(raw_scores(t,w,topk),mode,alpha);out[t['tid']]=max(range(len(s)),key=lambda i:(s[i],-i))
 return out
def metrics(pred,ts):
 rows=[]
 for t in ts:
  pi=pred[t['tid']];ac=e.match(t['events'][pi]['agent'],t['gold_agent']);rows.append({'tid':t['tid'],'system':t['system'],'gold_index':t['gi'],'pred_index':pi,'gold_step':next(iter(t['gold']))[0],'pred_step':t['events'][pi]['sid'],'gold_agent':t['gold_agent'],'pred_agent':t['events'][pi]['agent'],'step_exact':pi==t['gi'],'step_tol1':abs(pi-t['gi'])<=1,'agent_correct':ac,'joint_exact':pi==t['gi'] and ac})
 n=len(rows);return {'n':n,**{k:sum(x[k] for x in rows)/n for k in ('step_exact','step_tol1','agent_correct','joint_exact')}},rows
def split(ts,p='v'):
 v=[t for t in ts if int(hashlib.sha256((p+t['tid']).encode()).hexdigest()[:8],16)%5==0];c=[t for t in ts if t not in v]
 if len(v)<2:q=sorted(ts,key=lambda x:x['tid']);v=q[:max(1,len(q)//5)];c=q[len(v):]
 return c,v
def tune(c,v):
 best=(-1,None)
 for ms in (2,3,5):
  _,w=fit(c,ms,2,.03)
  for top in (5,10,20):
   for mode in ('raw','delta','cumulative'):
    for a in ((0.,) if mode=='raw' else (.25,.5,.75,1.0)):
     m,_=metrics(predict(v,w,top,mode,a),v);key=(m['step_exact'],m['step_tol1'],m['joint_exact'])
     if best[1] is None or key>best[0]:best=(key,{'min_support':ms,'topk':top,'mode':mode,'alpha':a})
 return best[1]
def fit_base(c,v):
 cr=e.flatten_traces(c);vr=e.flatten_traces(v);best=(-1,2)
 for m in (2,3,5):
  r=base.learn_rules(cr,{t['tid'] for t in c},e.goldmap(c),m);p=e.top1(base.score_records(vr,{t['tid'] for t in v},r,True),v);q=e.metrics(p,v)[0]['step_exact']
  if q>best[0]:best=(q,m)
 return best[1]
def run_once(train,test):
 c,v=split(train,'inner');cfg=tune(c,v);_,w=fit(train,cfg['min_support'],2,.03);pm,pr=metrics(predict(test,w,cfg['topk'],cfg['mode'],cfg['alpha']),test);bc=fit_base(c,v);br=base.learn_rules(e.flatten_traces(train),{t['tid'] for t in train},e.goldmap(train),bc);bm,brows=e.metrics(e.top1(base.score_records(e.flatten_traces(test),{t['tid'] for t in test},br,True),test),test);return bm,pm,brows,pr,cfg,len(w)
def oof(ts):
 B=[];U=[];fold=[]
 for f in range(5):
  te=[t for t in ts if int(hashlib.sha256(('outer'+t['tid']).encode()).hexdigest()[:8],16)%5==f];tr=[t for t in ts if t not in te]
  if not te:continue
  bm,um,br,ur,cfg,n=run_once(tr,te);B+=br;U+=ur;fold.append({'fold':f,'train':len(tr),'test':len(te),'baseline':bm,'boundary_ueta':um,'config':cfg,'contracts':n})
 def agg(r):
  n=len(r);return {'n':n,**{k:sum(x[k] for x in r)/n for k in ('step_exact','step_tol1','agent_correct','joint_exact')}}
 return agg(B),agg(U),B,U,fold
def holdout(ts):
 out=[]
 for s in sorted(set(t['system'] for t in ts)):
  te=[t for t in ts if t['system']==s];tr=[t for t in ts if t['system']!=s]
  if len(te)<5:continue
  bm,um,_,_,cfg,n=run_once(tr,te);out.append({'held_out_system':s,'train':len(tr),'test':len(te),'baseline':bm,'boundary_ueta':um,'step_gain':um['step_exact']-bm['step_exact'],'joint_gain':um['joint_exact']-bm['joint_exact'],'config':cfg,'contracts':n})
 return out
def boot(B,U,n=2000):
 b={x['tid']:x['step_exact'] for x in B};u={x['tid']:x['step_exact'] for x in U};ids=sorted(b);r=random.Random(SEED);g=[]
 for _ in range(n):
  q=[r.choice(ids) for _ in ids];g.append(statistics.mean(u[x]-b[x] for x in q))
 g.sort();return {'mean':statistics.mean(g),'ci95_low':g[int(.025*n)],'ci95_high':g[int(.975*n)]}
def writecsv(p,r):
 if not r:Path(p).write_text('');return
 k=list(dict.fromkeys(x for z in r for x in z));
 with open(p,'w',newline='') as f:csv.DictWriter(f,fieldnames=k).writeheader();w=csv.DictWriter(f,fieldnames=k);w.writerows(r)
def main():
 a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--out',required=True);x=a.parse_args();out=Path(x.out);out.mkdir(parents=True,exist_ok=True);ts,sk=load(Path(x.data));bm,um,B,U,fold=oof(ts);ho=holdout(ts);bs=boot(B,U);c,v=split(ts,'final');cfg=tune(c,v);routes,w=fit(ts,cfg['min_support'],2,.03);ck={'architecture':'Boundary-UETA explicit transition hazard engine','config':cfg,'weights':[[list(k),z] for k,z in w.items()],'forbidden_dense_components_used':[]};gzip.open(out/'BOUNDARY_UETA_CHECKPOINT.json.gz','wt').write(json.dumps(ck));hg=statistics.mean(z['step_gain'] for z in ho);ver={'oof_gain_3pp':um['step_exact']-bm['step_exact']>=.03,'bootstrap_positive':bs['ci95_low']>0,'heldout_system_gain_positive':hg>0,'joint_not_worse':um['joint_exact']>=bm['joint_exact']};ver['external_supported']=all(ver.values());summary={'architecture':'Boundary-UETA explicit transition hazard engine','dataset':'TraceElephant','traces':len(ts),'events':sum(len(t['events']) for t in ts),'systems':Counter(t['system'] for t in ts),'skipped':sk,'baseline_oof':bm,'boundary_ueta_oof':um,'step_gain':um['step_exact']-bm['step_exact'],'joint_gain':um['joint_exact']-bm['joint_exact'],'bootstrap':bs,'folds':fold,'system_holdout':ho,'system_holdout_mean_step_gain':hg,'checkpoint':{'config':cfg,'routes':len(routes),'weights':len(w),'bytes':(out/'BOUNDARY_UETA_CHECKPOINT.json.gz').stat().st_size},'strict_verdict':ver,'forbidden_dense_components_used':[]};json.dump(summary,open(out/'summary.json','w'),indent=2,default=dict);writecsv(out/'baseline_predictions.csv',B);writecsv(out/'boundary_predictions.csv',U);print(json.dumps(summary,indent=2,default=dict))
if __name__=='__main__':main()
