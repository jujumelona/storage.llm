#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,glob,hashlib,itertools,json,math,os,random,statistics,time
from collections import Counter,defaultdict
from pathlib import Path
import trail_nondense_v4_runner as b

SEED=20260721
random.seed(SEED)

def prf(tp,fp,fn):
 p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.;return p,r,2*p*r/(p+r) if p+r else 0.
def micro(pred,tids,gold):
 gt={(t,s,c) for t in tids for s,c in gold.get(t,set())};pp={(t,s,c) for t in tids for s,c in pred.get(t,set())};tp=len(gt&pp);p,r,f=prf(tp,len(pp-gt),len(gt-pp));return {'precision':p,'recall':r,'f1':f,'gold':len(gt),'predicted':len(pp),'tp':tp}
def group(records):
 g=defaultdict(list)
 for r in records:g[r['tid']].append(r)
 return g

def sfeatures(fs):
 out=[]
 for pref in ('P:','E:','H:','F:','Q:','R:','T:','K:','S:','D:','PN:','N:','L:'):
  x=sorted(z for z in fs if z.startswith(pref));out.extend(x[:2] if pref=='L:' else x[:1] if pref in {'N:','PN:'} else x)
 return out[:12]
def sig(r):
 fs=r['features']
 for pref in ('P:','E:','H:'):
  x=sorted(z for z in fs if z.startswith(pref))
  if x:return x[0]
 for x in ('F:error','F:final'):
  if x in fs:return x
 x=sorted(z for z in fs if z.startswith('T:') and z!='T:<empty>')
 if x:return x[0]
 x=sorted(z for z in fs if z.startswith('K:'))
 return x[0] if x else 'EMPTY'
def atoms(r):return tuple(sfeatures(r['features'])[:10])
def context(seq,i,depth=3,horizon=4):
 hs=[tuple(sig(seq[j]) for j in range(i-d+1,i+1)) for d in range(1,depth+1) if i-d+1>=0];gs=sfeatures(seq[i]['features'])[:9];obs=[];future=seq[i+1:i+1+horizon]
 for off,r in enumerate(future,1):obs.append(('AT'+str(off),sig(r)))
 seen=set()
 for r in future:
  for a in atoms(r):
   if a not in seen:seen.add(a);obs.append(('EV',a))
   if len(seen)>=8:break
  if len(seen)>=8:break
 if len(future)>=2:obs.append(('SEQ',sig(future[0]),sig(future[1])))
 return hs,gs,obs or [('END',)]

def learn(records,tids,gold):
 gs=group([r for r in records if r['tid'] in tids]);at=Counter();ap=defaultdict(Counter);ct=Counter();cp=defaultdict(Counter);cats=Counter();n=0
 for tid,seq in gs.items():
  gb=defaultdict(set)
  for s,c in gold.get(tid,set()):gb[s].add(c)
  for i,r in enumerate(seq):
   n+=1;cs=gb.get(r['sid'],set());cats.update(cs);hs,guards,obs=context(seq,i)
   for h in hs:
    for gd in guards:
     a=(h,gd);at[a]+=1
     for c in cs:ap[c][a]+=1
     for ob in obs:
      k=(h,gd,ob);ct[k]+=1
      for c in cs:cp[c][k]+=1
 out=[]
 for c,pc in cp.items():
  base=(cats[c]+1)/(n+2)
  for k,pn in pc.items():
   if pn<2:continue
   h,gd,ob=k;an=ap[c][(h,gd)]
   if an<2:continue
   cov=pn/an;prec=(pn+1)/(ct[k]+2);sc=math.log(max(1e-9,prec/base))+.22*math.log1p(pn)+.35*cov+.08*len(h)
   if prec>=.08 and sc>0:out.append({'cat':c,'history':h,'guard':gd,'obligation':ob,'score':sc,'support':pn,'precision':prec,'coverage':cov})
 out.sort(key=lambda x:(-x['score'],-x['support'],x['cat']));kept=[];cnt=Counter()
 for c in out:
  k=(c['cat'],c['history'][-1],c['guard'])
  if cnt[k]>=20:continue
  cnt[k]+=1;kept.append(c)
  if len(kept)>=40000:break
 return kept
def filt(cs,d,cov,prec):return [x for x in cs if len(x['history'])<=d and x['coverage']>=cov and x['precision']>=prec]
def score(records,tids,cs):
 idx=defaultdict(list)
 for j,c in enumerate(cs):idx[(c['history'],c['guard'],c['obligation'])].append(j)
 out=defaultdict(list)
 for tid,seq in group([r for r in records if r['tid'] in tids]).items():
  for i,r in enumerate(seq):
   hs,guards,obs=context(seq,i);best={}
   for h in hs:
    for gd in guards:
     for ob in obs:
      for j in idx.get((h,gd,ob),()):
       c=cs[j]
       if c['cat'] not in best or c['score']>best[c['cat']][0]:best[c['cat']]=(c['score'],j)
   for cat,(sc,j) in best.items():out[tid].append((sc,r['sid'],cat,j,r.get('evidence','')))
 return out
def select(scores,tids,k,th):
 out={}
 for tid in tids:
  arr=sorted(scores.get(tid,[]),key=lambda x:(-x[0],x[1],x[2]));keep=[];seen=set()
  for x in arr:
   z=(x[1],x[2])
   if x[0]<th or z in seen:continue
   seen.add(z);keep.append(z)
   if len(keep)>=k:break
  out[tid]=set(keep)
 return out
def cal(scores,tids,gold):
 vals=sorted({x[0] for t in tids for x in scores.get(t,[])});ths=[.2,.5,.8,1.1,1.4,1.8,2.2,2.8]
 if vals:ths += [vals[min(len(vals)-1,int(len(vals)*q))] for q in (.2,.4,.6,.75,.85)]
 best=(-1,5,.5)
 for k in range(3,11):
  for th in ths:
   f=micro(select(scores,tids,k,th),tids,gold)['f1']
   if f>best[0]:best=(f,k,th)
 return {'f1':best[0],'k':best[1],'threshold':best[2]}
def split(tids,fold):
 test={t for t in tids if int(hashlib.sha256(t.encode()).hexdigest()[:8],16)%2==fold};train=set(tids)-test;val={t for t in train if int(hashlib.sha256(('v'+t).encode()).hexdigest()[:8],16)%4==0};core=train-val
 return core,val,test,train

def oof(records,tids,gold):
 bp,up,cp={},{},{};folds=[];saved=[]
 for fold in (0,1):
  core,val,test,train=split(tids,fold)
  br0=b.learn_rules(records,core,gold);bv=b.score_records(records,val,br0,True);bc=cal(bv,val,gold)
  all0=learn(records,core,gold);best=(-1,None,None,None)
  for d in (1,2,3):
   for cov in (.35,.5,.65,.8):
    for prec in (.1,.16,.24,.34):
     cc=filt(all0,d,cov,prec)
     if not cc:continue
     uv=score(records,val,cc);uc=cal(uv,val,gold)
     if uc['f1']>best[0]:best=(uc['f1'],(d,cov,prec),uc,cc)
  br=b.learn_rules(records,train,gold);bs=b.score_records(records,test,br,True);allc=learn(records,train,gold);cfg=best[1];cs=filt(allc,*cfg);us=score(records,test,cs)
  bpred=select(bs,test,bc['k'],bc['threshold']);upred=select(us,test,best[2]['k'],best[2]['threshold'])
  uv=score(records,val,best[3]);unionv=defaultdict(list)
  for t in val:unionv[t]=list(bv.get(t,[]))+list(uv.get(t,[]))
  cc=cal(unionv,val,gold);unions=defaultdict(list)
  for t in test:unions[t]=list(bs.get(t,[]))+list(us.get(t,[]))
  cpred=select(unions,test,cc['k'],cc['threshold'])
  bp.update(bpred);up.update(upred);cp.update(cpred);saved.append({'fold':fold,'train':train,'test':test,'br':br,'cs':cs,'bc':bc,'uc':best[2]})
  folds.append({'fold':fold,'core':len(core),'val':len(val),'test':len(test),'baseline_rules':len(br),'ueta_contracts':len(cs),'depth':cfg[0],'coverage':cfg[1],'precision':cfg[2],'baseline_val_f1':bc['f1'],'ueta_val_f1':best[0],'union_val_f1':cc['f1']})
 return bp,up,cp,folds,saved

def learn_valid(records,tids):
 ac=Counter();oc=Counter()
 for seq in group([r for r in records if r['tid'] in tids]).values():
  for i in range(len(seq)):
   hs,_,obs=context(seq,i,2,4)
   for h in hs:
    ac[h]+=1
    for ob in set(obs):oc[(h,ob)]+=1
 out=[]
 for (h,ob),n in oc.items():
  cov=n/ac[h]
  if n>=3 and cov>=.7:out.append((h,ob,cov,n))
 return sorted(out,key=lambda x:(-x[2],-x[3]))[:20000]
def valid_score(seq,cs):
 by=defaultdict(list)
 for h,ob,w,n in cs:by[h].append((ob,w))
 sat=tot=0.
 for i in range(len(seq)):
  hs,_,obs=context(seq,i);obs=set(obs)
  for h in hs:
   for ob,w in by.get(h,()):tot+=w;sat+=w if ob in obs else 0
 return sat/tot if tot else .5
def bigram(records,tids):
 c=Counter();a=Counter()
 for seq in group([r for r in records if r['tid'] in tids]).values():
  ss=[sig(r) for r in seq]
  for x,y in zip(ss,ss[1:]):c[(x,y)]+=1;a[x]+=1
 return c,a
def bscore(seq,m):
 c,a=m;ss=[sig(r) for r in seq];v=[(c[(x,y)]+1)/(a[x]+20) for x,y in zip(ss,ss[1:])];return sum(v)/len(v) if v else .5
def perturb(seq,kind):
 x=[dict(r) for r in seq]
 if len(x)<4:return x
 if kind=='delete':del x[max(1,len(x)//3)]
 else:j=max(1,min(len(x)-2,len(x)//2));x[j],x[j+1]=x[j+1],x[j]
 for i,r in enumerate(x):r['i']=i
 return x
def auc(y,s):
 p=[z for a,z in zip(y,s) if a];n=[z for a,z in zip(y,s) if not a];return sum(1 if x>z else .5 if x==z else 0 for x in p for z in n)/(len(p)*len(n))
def counterfactual(records,tids):
 y=[];u=[];base=[];rows=[]
 for fold in (0,1):
  _,_,test,train=split(tids,fold);vc=learn_valid(records,train);bm=bigram(records,train)
  for tid,seq in group([r for r in records if r['tid'] in test]).items():
   for kind,x,label in [('original',seq,1),('delete',perturb(seq,'delete'),0),('swap',perturb(seq,'swap'),0)]:
    a=valid_score(x,vc);z=bscore(x,bm);y.append(label);u.append(a);base.append(z);rows.append({'fold':fold,'trace_id':tid,'variant':kind,'original':label,'ueta':a,'bigram':z})
 return {'ueta_auc':auc(y,u),'bigram_auc':auc(y,base),'rows':rows}
def corrupt(records,relevant):
 rel=sorted(relevant);out=[]
 for r in records:
  rng=random.Random(int(hashlib.sha256((r['tid']+r['sid']).encode()).hexdigest()[:16],16));actual=set(r['features']);univ=set(actual)
  if rel:
   st=rng.randrange(len(rel));univ.update(rel[(st+j)%len(rel)] for j in range(min(24,len(rel))))
  new=set()
  for f in univ:
   yes=no=0;truth=f in actual
   for _ in range(31):
    if rng.random()<.4:continue
    obs=(not truth) if rng.random()<.3 else truth;yes+=obs;no+=not obs
   if yes>no:new.add(f)
  q=dict(r);q['features']=frozenset(new);out.append(q)
 return out
def noise_eval(records,tids,gold,saved,cleanb,cleanu):
 bp={};up={}
 for x in saved:
  rel=set(z for r in x['br'] for z in r['need'])|{c['guard'] for c in x['cs']};noisy=corrupt([r for r in records if r['tid'] in x['test']],rel);bs=b.score_records(noisy,x['test'],x['br'],True);us=score(noisy,x['test'],x['cs']);bp.update(select(bs,x['test'],x['bc']['k'],x['bc']['threshold']));up.update(select(us,x['test'],x['uc']['k'],x['uc']['threshold']))
 bm=micro(bp,tids,gold);um=micro(up,tids,gold);cb=micro(cleanb,tids,gold);cu=micro(cleanu,tids,gold);return {'baseline':bm,'ueta':um,'baseline_retention':bm['f1']/cb['f1'] if cb['f1'] else 0,'ueta_retention':um['f1']/cu['f1'] if cu['f1'] else 0}
def trie(paths):
 root={};st=1;tr=0
 for path in paths:
  n=root
  for tok in path:
   if tok not in n:n[tok]={};st+=1;tr+=1
   n=n[tok]
 return root,st,tr
def storage(records,gold,saved):
 explicit=[];cs=[]
 for x in saved:
  for tid,seq in group([r for r in records if r['tid'] in x['train']]).items():
   gb=defaultdict(set)
   for sid,c in gold.get(tid,set()):gb[sid].add(c)
   for i,r in enumerate(seq):
    for c in gb.get(r['sid'],()):hs,gs,ob=context(seq,i);explicit.append((c,hs,gs,ob))
  cs.extend((c['cat'],c['history'],c['guard'],c['obligation'],round(c['score'],5)) for c in x['cs'])
 ht,hs,htr=trie([x[1] for x in cs]);ft,fs,ftr=trie([x[3] for x in cs]);eb=len(json.dumps(explicit,separators=(',',':')).encode());cb=len(json.dumps(cs,separators=(',',':')).encode());ab=len(json.dumps({'h':ht,'f':ft,'e':[(x[0],x[2],x[4]) for x in cs]},separators=(',',':')).encode());return {'explicit_bytes':eb,'contracts_bytes':cb,'automaton_bytes':ab,'automaton_ratio':ab/eb if eb else 0,'contracts':len(cs),'history_states':hs,'future_states':fs,'transitions':htr+ftr}
def runtime(records,saved):
 x=max(saved,key=lambda z:len(z['cs']));cs=x['cs'];sample=[r for r in records if r['tid'] in x['test']][:500];tids={r['tid'] for r in sample};t=time.perf_counter();ix=score(sample,tids,cs);a=time.perf_counter()-t;t=time.perf_counter();nv=defaultdict(list)
 for tid,seq in group(sample).items():
  for i,r in enumerate(seq):
   hs,gs,ob=context(seq,i);H=set(hs);G=set(gs);O=set(ob);best={}
   for j,c in enumerate(cs):
    if c['history'] in H and c['guard'] in G and c['obligation'] in O and (c['cat'] not in best or c['score']>best[c['cat']][0]):best[c['cat']]=(c['score'],j)
   for cat,(sc,j) in best.items():nv[tid].append((sc,r['sid'],cat,j,''))
 n=time.perf_counter()-t;norm=lambda d:{k:sorted((round(z[0],9),z[1],z[2]) for z in v) for k,v in d.items()};return {'sample_records':len(sample),'contracts':len(cs),'identical':norm(ix)==norm(nv),'indexed_seconds':a,'naive_seconds':n,'speedup':n/a if a else 0}
def bootstrap(base,ueta,tids,gold,n=1000):
 rng=random.Random(SEED);ts=sorted(tids);d=[]
 for _ in range(n):
  samp=[rng.choice(ts) for _ in ts];B={};U={};G={};ids=set()
  for j,t in enumerate(samp):z=f'{j}:{t}';ids.add(z);B[z]=base.get(t,set());U[z]=ueta.get(t,set());G[z]=gold.get(t,set())
  d.append(micro(U,ids,G)['f1']-micro(B,ids,G)['f1'])
 d.sort();return {'mean':statistics.mean(d),'ci95_low':d[int(.025*n)],'ci95_high':d[int(.975*n)]}
def writecsv(path,rows):
 if not rows:Path(path).write_text('');return
 keys=list(dict.fromkeys(k for r in rows for k in r));f=open(path,'w',newline='',encoding='utf-8');w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows);f.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('--trail',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);data=glob.glob(a.trail+'/benchmarking/data/GAIA/*.json')+glob.glob(a.trail+'/benchmarking/data/SWE Bench/*.json');anns=glob.glob(a.trail+'/benchmarking/processed_annotations_gaia/*.json')+glob.glob(a.trail+'/benchmarking/processed_annotations_swe_bench/*.json');gold=b.load_gold(anns);records=[];tids=set()
 for z in data:tid,rs=b.trace_records(z);tids.add(tid);records.extend(rs)
 print('loaded',len(tids),len(records),flush=True);bp,up,cp,folds,saved=oof(records,tids,gold);bm=micro(bp,tids,gold);um=micro(up,tids,gold);cm=micro(cp,tids,gold);boot=bootstrap(bp,up,tids,gold);cf=counterfactual(records,tids);ne=noise_eval(records,tids,gold,saved,bp,up);st=storage(records,gold,saved);rt=runtime(records,saved);ver={'ueta_f1_improves':um['f1']>bm['f1']+.005,'bootstrap_ci_positive':boot['ci95_low']>0,'counterfactual_pass':cf['ueta_auc']>=.7 and cf['ueta_auc']>=cf['bigram_auc']+.03,'storage_pass':st['automaton_ratio']<1,'noise_pass':ne['ueta_retention']>=.9,'runtime_pass':rt['identical'] and rt['speedup']>1};ver['overall_supported']=all(ver.values());summary={'architecture':'UETA underapproximate history-conditioned future obligation automata','forbidden_dense_components_used':[],'traces':len(tids),'records':len(records),'baseline':bm,'ueta':um,'union':cm,'ueta_minus_baseline_f1':um['f1']-bm['f1'],'bootstrap':boot,'counterfactual':{k:v for k,v in cf.items() if k!='rows'},'noise':ne,'storage':st,'runtime':rt,'folds':folds,'strict_verdict':ver};json.dump(summary,open(out/'summary.json','w'),indent=2);writecsv(out/'folds.csv',folds);writecsv(out/'counterfactual.csv',cf['rows']);rows=[]
 for t in tids:
  for m,pred in [('baseline',bp),('ueta',up),('union',cp)]:
   for s,c in pred.get(t,set()):rows.append({'trace_id':t,'method':m,'span_id':s,'category':c,'gold':int((s,c) in gold.get(t,set()))})
 writecsv(out/'predictions.csv',rows);cr=[]
 for x in saved:
  for c in x['cs'][:10000]:cr.append({'fold':x['fold'],'category':c['cat'],'history':' > '.join(c['history']),'guard':c['guard'],'obligation':' > '.join(c['obligation']),'score':c['score'],'support':c['support'],'precision':c['precision'],'coverage':c['coverage']})
 writecsv(out/'contracts.csv',cr);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
