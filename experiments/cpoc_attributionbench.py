#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, itertools, json, random, re, statistics, time
from collections import defaultdict
from pathlib import Path

SEED=20260724
WORD=re.compile(r"[A-Za-z][A-Za-z0-9_'-]{1,40}")
NUM=re.compile(r"[-+]?\d+(?:\.\d+)?%?")
SENT=re.compile(r"(?<=[.!?])\s+|\n+")
NEG=re.compile(r"\b(no|not|never|none|neither|without|cannot|can't|isn't|aren't|didn't|doesn't|won't)\b",re.I)
COMP=re.compile(r"\b(more|less|fewer|greater|smaller|larger|twice|half|at least|at most|before|after|higher|lower)\b",re.I)
STOP=set('a an the this that these those is are was were be been being of to in on at for from with by and or as it its their they them he she his her there here what which who whom whose how why when where do does did have has had can could may might will would should must very more most less many much some any all one two according also however therefore so seems about almost approximately around said says say based claim claims people person local locals'.split())

def norm(s):return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9.%+/_-]+',' ',str(s).lower())).strip()
def toks(s):return [w.lower() for w in WORD.findall(str(s)) if len(w)>=2 and w.lower() not in STOP]
def atoms(s):
    text=str(s);t=toks(text);out={'W:'+x for x in t};out|={'N:'+x.rstrip('%') for x in NUM.findall(norm(text))}
    for i in range(len(t)-1):out.add('B:'+t[i]+'_'+t[i+1])
    if NEG.search(text):out.add('POL:NEG')
    if COMP.search(text):out.add('REL:COMP')
    return out

def split_sentences(s):
    xs=[x.strip() for x in SENT.split(str(s)) if x.strip()]
    return xs or ([str(s).strip()] if str(s).strip() else [])
def num_values(a):
    out=[]
    for x in a:
        if x.startswith('N:'):
            try:out.append(float(x[2:]))
            except:pass
    return out

def relation(ca,ea):
    cw={x for x in ca if x.startswith(('W:','B:'))};ew={x for x in ea if x.startswith(('W:','B:'))}
    cn={x for x in ca if x.startswith('N:')};en={x for x in ea if x.startswith('N:')}
    overlap=len(cw&ew)/max(1,len(cw));num_ok=(not cn) or cn<=en;pol_same=('POL:NEG' in ca)==('POL:NEG' in ea)
    return {'overlap':overlap,'exact':overlap>=.82 and num_ok and pol_same,'compression':overlap>=.48 and num_ok and pol_same,'contradiction':overlap>=.34 and ((cn and en and not cn<=en) or not pol_same)}

def pair_inference(ca,a,b):
    union=a|b;cw={x for x in ca if x.startswith(('W:','B:'))};uw={x for x in union if x.startswith(('W:','B:'))};cover=len(cw&uw)/max(1,len(cw));cn=num_values(ca);vals=num_values(union);num=False
    for x,y in itertools.combinations(vals,2):
        candidates=(x+y,x-y,y-x,x*y,x/y if y else 1e300,y/x if x else 1e300)
        if any(any(abs(v-z)<=1e-6*max(1,abs(z)) for v in candidates) for z in cn):num=True
    pol=('POL:NEG' in ca)==('POL:NEG' in union)
    return cover>=.62 and pol and ((not cn) or num or {f'N:{z:g}' for z in cn}<=union)

def compile_item(row,cfg,refs_override=None):
    claims=split_sentences(row.get('claim') or '');refs=refs_override if refs_override is not None else (row.get('references') or []);ev=[]
    for rid,r in enumerate(refs):
        for sid,s in enumerate(split_sentences(r)):ev.append((rid,sid,s,atoms(s)))
    clauses=[];edges=[];contra=0;lexvals=[]
    for ci,c in enumerate(claims):
        ca=atoms(c);alts=[];lexvals.append(max((relation(ca,e[3])['overlap'] for e in ev),default=0))
        for ei,e in enumerate(ev):
            rel=relation(ca,e[3])
            if rel['contradiction']:contra+=1;edges.append((ci,ei,'CONTRADICTION'))
            if rel['exact']:alts.append(frozenset([ei]));edges.append((ci,ei,'QUOTATION'))
            elif rel['compression']:alts.append(frozenset([ei]));edges.append((ci,ei,'COMPRESSION'))
        if cfg['pairs'] and len(ev)<=cfg['max_evidence_for_pairs']:
            for i,j in itertools.combinations(range(len(ev)),2):
                if pair_inference(ca,ev[i][3],ev[j][3]):alts.append(frozenset([i,j]));edges.extend(((ci,i,'INFERENCE'),(ci,j,'INFERENCE')))
        uq=sorted(set(alts),key=lambda x:(len(x),tuple(x)));clauses.append([a for a in uq if not any(b<a for b in uq)])
    coverage=sum(bool(x) for x in clauses)/max(1,len(clauses));all_supported=bool(clauses) and all(bool(x) for x in clauses);score=coverage+cfg['all_bonus']*all_supported-cfg['contra_penalty']*min(contra,3);gates=[];selected=set()
    for ci,alts in enumerate(clauses):
        for a in alts:gates.append(('AND',ci,tuple(sorted(a))))
        gates.append(('OR',ci,tuple(sorted(tuple(sorted(a)) for a in alts))))
        if alts:
            m=min(len(a) for a in alts)
            for a in alts:
                if len(a)==m:selected|=set(a)
    gates.append(('ROOT_AND',tuple(range(len(clauses)))))
    return {'lex_score':sum(lexvals)/max(1,len(lexvals)),'score':score,'clauses':[[sorted(a) for a in x] for x in clauses],'edges':edges,'gates':gates,'support_evidence':sorted(selected),'n_evidence':len(ev)}

def metrics(y,s,t):
    p=[int(x>=t) for x in s];tp=sum(a==1 and b==1 for a,b in zip(y,p));tn=sum(a==0 and b==0 for a,b in zip(y,p));fp=sum(a==0 and b==1 for a,b in zip(y,p));fn=sum(a==1 and b==0 for a,b in zip(y,p));pr=tp/max(1,tp+fp);rc=tp/max(1,tp+fn);sp=tn/max(1,tn+fp)
    return {'n':len(y),'accuracy':(tp+tn)/max(1,len(y)),'precision':pr,'recall':rc,'specificity':sp,'balanced_accuracy':(rc+sp)/2,'f1':2*pr*rc/max(1e-12,pr+rc),'tp':tp,'tn':tn,'fp':fp,'fn':fn}
def bootstrap(y,a,b,ta,tb,n=5000):
    rng=random.Random(SEED);v=[]
    for _ in range(n):
        ix=[rng.randrange(len(y)) for _ in y];v.append(sum(int((a[i]>=ta)==y[i])-int((b[i]>=tb)==y[i]) for i in ix)/len(ix))
    v.sort();return {'mean':statistics.mean(v),'lo':v[int(.025*n)],'hi':v[int(.975*n)]}
def load_rows():
    from datasets import load_dataset
    kw=dict(path='osunlp/AttributionBench',name='subset_balanced');return [[dict(x) for x in load_dataset(**kw,split=s)] for s in ('train','test','test_ood')]
def tune(train):
    va=[r for r in train if int(hashlib.sha256(str(r['id']).encode()).hexdigest()[:8],16)%5==0];cfgs=[{'pairs':p,'all_bonus':a,'contra_penalty':c,'max_evidence_for_pairs':20} for p in (False,True) for a in (0,.15,.30) for c in (.05,.12,.20)];best=None
    for cfg in cfgs:
        z=[compile_item(r,cfg) for r in va];y=[int(r['attribution_label']=='attributable') for r in va]
        for th in [i/20 for i in range(-5,31)]:
            m=metrics(y,[x['score'] for x in z],th);key=(m['balanced_accuracy'],m['accuracy'])
            if best is None or key>best[0]:best=(key,cfg,th,m)
    base=[compile_item(r,{'pairs':False,'all_bonus':0,'contra_penalty':0,'max_evidence_for_pairs':0}) for r in va];y=[int(r['attribution_label']=='attributable') for r in va];lb=None
    for th in [i/100 for i in range(101)]:
        m=metrics(y,[x['lex_score'] for x in base],th);key=(m['balanced_accuracy'],m['accuracy'])
        if lb is None or key>lb[0]:lb=(key,th,m)
    return best[1],best[2],lb[1],{'rows':len(va),'product':best[3],'lexical':lb[2]}
def evaluate(rows,cfg,th,lth,seed):
    refs=[r.get('references') or [] for r in rows];rng=random.Random(seed);shuf=refs[:];rng.shuffle(shuf);out=[];rew=[];gates=set();raw=0;t0=time.perf_counter()
    for r in rows:
        x=compile_item(r,cfg);out.append(x);gates.update(repr(g) for g in x['gates']);raw+=len(json.dumps(r.get('references') or [],ensure_ascii=False).encode())
    runtime=time.perf_counter()-t0
    for r,q in zip(rows,shuf):rew.append(compile_item(r,cfg,q))
    y=[int(r['attribution_label']=='attributable') for r in rows];s=[x['score'] for x in out];ls=[x['lex_score'] for x in out];rs=[x['score'] for x in rew];td=[];rd=[]
    for k,(r,x) in enumerate(zip(rows,out)):
        if not y[k] or not r.get('references'):continue
        evmap=[]
        for rid,ref in enumerate(r['references']):
            for _ in split_sentences(ref):evmap.append(rid)
        rids={evmap[e] for e in x['support_evidence'] if e<len(evmap)} or {0};top=compile_item(r,cfg,[v for j,v in enumerate(r['references']) if j not in rids])['score'];j=int(hashlib.sha256(str(r['id']).encode()).hexdigest()[:8],16)%len(r['references']);ran=compile_item(r,cfg,[v for q,v in enumerate(r['references']) if q!=j])['score'];td.append(x['score']-top);rd.append(x['score']-ran)
    return {'metrics':metrics(y,s,th),'lexical':metrics(y,ls,lth),'rewired':metrics(y,rs,th),'bootstrap':bootstrap(y,s,ls,th,lth),'deletion':{'n':len(td),'top_mean_drop':statistics.mean(td) if td else 0,'random_mean_drop':statistics.mean(rd) if rd else 0,'top_beats_random_rate':sum(a>b for a,b in zip(td,rd))/max(1,len(td))},'storage':{'raw_reference_bytes':raw,'shared_gate_bytes':len(json.dumps(sorted(gates)).encode()),'ratio':len(json.dumps(sorted(gates)).encode())/max(1,raw),'unique_gates':len(gates)},'runtime_seconds':runtime,'predictions':[{'id':r['id'],'label':y[i],'score':s[i],'lexical':ls[i],'rewired':rs[i],'clauses':out[i]['clauses'],'edges':out[i]['edges']} for i,r in enumerate(rows)]}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);train,test,ood=load_rows();cfg,th,lth,tuning=tune(train);te=evaluate(test,cfg,th,lth,SEED);oo=evaluate(ood,cfg,th,lth,SEED+1);strict={'test_gain_5pp':te['metrics']['accuracy']>=te['lexical']['accuracy']+.05,'test_ci_positive':te['bootstrap']['lo']>0,'ood_nonworse':oo['metrics']['accuracy']>=oo['lexical']['accuracy'],'rewire_drop_5pp':te['metrics']['accuracy']>=te['rewired']['accuracy']+.05,'deletion_beats_random':te['deletion']['top_mean_drop']>te['deletion']['random_mean_drop'] and te['deletion']['top_beats_random_rate']>.55,'storage_below_raw':te['storage']['ratio']<1};summary={'benchmark':'AttributionBench subset_balanced','structure':'claim-clause AND × evidence-alternative OR explicit provenance circuit','config':cfg,'threshold':th,'lexical_threshold':lth,'tuning':tuning,'test':{k:v for k,v in te.items() if k!='predictions'},'test_ood':{k:v for k,v in oo.items() if k!='predictions'},'strict':strict,'overall_pass':all(strict.values()),'excluded':['embeddings','dense latent vectors','pretrained encoders','LLM calls','TF-IDF/SVD','nearest-vector','code generation','external solver']};(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'test_predictions.jsonl').write_text('\n'.join(json.dumps(x) for x in te['predictions']));(out/'ood_predictions.jsonl').write_text('\n'.join(json.dumps(x) for x in oo['predictions']));(out/'checkpoint.json').write_text(json.dumps({'config':cfg,'threshold':th,'lexical_threshold':lth},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
