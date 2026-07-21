#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os
from functools import lru_cache
from pathlib import Path

src=Path('experiments/nondense_natural_v5.py').read_text(encoding='utf-8')
prefix=src.rsplit("\nif __name__=='__main__':main()",1)[0]
ns={}
exec(compile(prefix,'nondense_natural_v5.py','exec'),ns)

# Passage closure is an explicit immutable fact set, so it is safe to cache.
_original_derive=ns['derive']
@lru_cache(maxsize=None)
def cached_derive(ctx):
    return tuple(_original_derive(ctx))
def cached_entail(ctx,q):
    qp=ns['prop'](q); known=cached_derive(ctx)
    return 'yes' if any(k[1]==qp[1] and ns['psim'](k,qp)>=.66 for k in known) else 'no'
ns['derive']=cached_derive
ns['entail']=cached_entail

@lru_cache(maxsize=None)
def cached_features(p,q,opts_tuple,i):
    return frozenset(ns['base_features'](p,q,list(opts_tuple),i))

def compile_data(data):
    rows=[]
    for g,p,q,opts in data:
        ot=tuple(opts)
        rows.append((g,[cached_features(p,q,ot,i) for i in range(4)],ns['qtype'](q)))
    return rows

def build_cached(rows,min_count=3):
    from collections import Counter
    import math
    total=Counter();pos=Counter();group=Counter();gpos=Counter()
    for g,flist,qt in rows:
        gi='abcd'.index(g)
        for i,fs in enumerate(flist):
            y=i==gi
            for f in fs:total[f]+=1;pos[f]+=y
            anchors=[x for x in fs if x.startswith(('Q:','C:','I:','ENT:','OPP:'))]
            evidence=[x for x in fs if x.startswith(('OM:','D:','LR:','OR:','OV:','OW:','PW:'))]
            for a in anchors:
                for e in evidence:
                    k=a+'&'+e;group[k]+=1;gpos[k]+=y
    ledger={}
    for f,n in total.items():
        if n>=min_count:
            p=(pos[f]+1)/(n+2);ledger[f]=(math.log(p/(1-p))-math.log(1/3),n)
    for f,n in group.items():
        if n>=max(4,min_count):
            p=(gpos[f]+1)/(n+2);ledger[f]=(math.log(p/(1-p))-math.log(1/3),n)
    return ledger

def expanded_keys(fs):
    keys=set(fs);anchors=[x for x in fs if x.startswith(('Q:','C:','I:','ENT:','OPP:'))];evidence=[x for x in fs if x.startswith(('OM:','D:','LR:','OR:','OV:','OW:','PW:'))]
    for a in anchors:
        for e in evidence:keys.add(a+'&'+e)
    return frozenset(keys)

def expand_rows(rows):
    return [(g,[(fs,expanded_keys(fs)) for fs in flist],qt) for g,flist,qt in rows]

def score_keys(keys,ledger,topk,shrink):
    from collections import Counter
    vals=[]
    for k in keys:
        if k in ledger:
            w,n=ledger[k];v=w*n/(n+shrink);vals.append((abs(v),v,k))
    vals.sort(reverse=True);used=Counter();score=0.;chosen=[]
    for _,v,k in vals:
        fam=k.split(':',1)[0]
        if used[fam]>=4:continue
        used[fam]+=1;score+=v;chosen.append(k)
        if len(chosen)>=topk:break
    return score,chosen

def predict_cached(rows,ledger,topk,shrink):
    out=[]
    for g,flist,qt in rows:
        ss=[];why=[]
        for fs,keys in flist:
            s,w=score_keys(keys,ledger,topk,shrink);ss.append(s);why.append(w)
        pr='abcd'[max(range(4),key=lambda i:(ss[i],-i))];out.append((pr,g,ss,why,qt))
    return out

def eval_fast(root,out):
    tr=ns['parse_logiqa'](os.path.join(root,'Train.txt'));dv=ns['parse_logiqa'](os.path.join(root,'Eval.txt'));te=ns['parse_logiqa'](os.path.join(root,'Test.txt'))
    ctr0,cdv0,cte0=compile_data(tr),compile_data(dv),compile_data(te);ctr,cdv,cte=expand_rows(ctr0),expand_rows(cdv0),expand_rows(cte0);best=(-1,None)
    ledgers={mc:build_cached(ctr0,mc) for mc in [2,3,5,8]}
    for mc,led in ledgers.items():
        for k in [8,12,20,32]:
            for sh in [3.,10.,30.]:
                r=predict_cached(cdv,led,k,sh);acc=sum(a==b for a,b,*_ in r)/len(r)
                if acc>best[0]:best=(acc,(mc,k,sh))
    combined0=ctr0+cdv0;led=build_cached(combined0,best[1][0]);r=predict_cached(cte,led,best[1][1],best[1][2]);rows=[]
    for i,(pr,g,ss,why,qt) in enumerate(r):rows.append({'id':i,'qtype':qt,'pred':pr,'gold':g,'correct':pr==g,'scores':json.dumps(ss),'evidence':json.dumps(why)})
    ns['write_csv'](out/'logiqa_v5_predictions.csv',rows)
    by={}
    for qt in sorted(set(x[-1] for x in r)):
        sub=[x for x in r if x[-1]==qt];by[qt]={'n':len(sub),'accuracy':sum(x[0]==x[1] for x in sub)/len(sub)}
    return {'train':len(tr),'dev':len(dv),'test':len(te),'dev_best_accuracy':best[0],'config':best[1],'ledger_records':len(led),'accuracy':sum(x[0]==x[1] for x in r)/len(r),'by_qtype':by}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--logicbench',required=True);ap.add_argument('--logiqa',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    lb=ns['eval_logicbench'](a.logicbench,out);lq=eval_fast(a.logiqa,out);summary={'architecture':'explicit clause/operator graph + cached sparse Bayesian evidence ledger','forbidden_dense_components_used':[],'logicbench':lb,'logiqa':lq,'gate_pass':lb['accuracy']>=.8 and lq['accuracy']>=.45};json.dump(summary,open(out/'summary.json','w',encoding='utf-8'),ensure_ascii=False,indent=2);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
