#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, random, re, statistics, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SEED=20260725
STOP=set('''a an the this that these those is are was were be been being to of in on at for from with by and or then as it its we you they he she them their his her our your which what who whom whose where when why how can could would should may might must do does did have has had use using used tool tools result results answer question image information know known now based find get need first next last there here about into more some any all each one two three four five than also only directly specific please'''.split())
TOKEN_RE=re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}|\d+(?:\.\d+)?")
CAP_RE=re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}|[A-Z]{2,}(?:\s+[A-Z]{2,})*)\b")
QUOTE_RE=re.compile(r"['\"]([^'\"]{2,80})['\"]")
NUM_RE=re.compile(r"\b\d{1,4}(?:[.,]\d+)*(?:%|st|nd|rd|th)?\b")
DATE_RE=re.compile(r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",re.I)
READ_WORDS=('description','ocr','search','detect','recognition','classification','caption','retrieve','lookup','read','identify','localize','segment')
CALC_WORDS=('calculator','calculate','count','measure','math','wolfram','area','distance')

def canon_tool(name:str)->str:
    n=re.sub(r'[^a-z0-9]+','_',str(name).lower()).strip('_')
    if n in {'question','answer','final'}:return 'CLAIM'
    if any(w in n for w in CALC_WORDS):return 'CALC'
    if any(w in n for w in READ_WORDS):return 'READ'
    return n.upper()[:40] or 'TOOL'

def norm(s:str)->str:return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9]+',' ',str(s).lower())).strip()

def values(s:str)->list[tuple[str,str]]:
    s=str(s or '');xs=[]
    for q in QUOTE_RE.findall(s):
        q=norm(q)
        if len(q)>=3:xs.append(('ENTITY',q))
    for c in CAP_RE.findall(s):
        c=norm(c)
        if c and c not in STOP and len(c)>=3:xs.append(('ENTITY',c))
    for d in DATE_RE.findall(s):
        d=norm(d)
        if d:xs.append(('DATE',d))
    for n in NUM_RE.findall(s):
        n=norm(n)
        if n:xs.append(('NUM',n))
    for t in TOKEN_RE.findall(s):
        t=norm(t)
        if t and t not in STOP and len(t)>=3 and not t.isdigit():xs.append(('TOKEN',t))
    out=[];seen=set()
    for x in xs:
        if x not in seen:seen.add(x);out.append(x)
    return out

def flatten(x:Any)->str:
    if x is None:return ''
    if isinstance(x,str):return x
    if isinstance(x,(int,float,bool)):return str(x)
    if isinstance(x,dict):return ' '.join(f'{k} {flatten(v)}' for k,v in sorted(x.items()))
    if isinstance(x,list):return ' '.join(flatten(v) for v in x)
    return str(x)

def aid(step:int,kind:str,value:str)->str:return f'{step}:{kind}:{hashlib.sha256(value.encode()).hexdigest()[:20]}'

def load(path:Path)->list[dict]:
    out=[]
    for line in path.read_text(encoding='utf-8',errors='ignore').splitlines():
        try:
            x=json.loads(line)
            if isinstance(x,dict):out.append(x)
        except Exception:pass
    return out

def split_row(row:dict,idx:int):
    ctx=row.get('context') or row.get('contexts') or []
    if isinstance(ctx,str):
        try:ctx=json.loads(ctx)
        except Exception:ctx=[]
    if not isinstance(ctx,list):return None,None
    q=str(row.get('better_ques') or row.get('question') or '')
    ans=str(row.get('answer') or row.get('correct_answer') or '')
    tools=[]
    for x in ctx:
        if not isinstance(x,dict):continue
        if canon_tool(x.get('name',''))=='CLAIM':
            q=str(x.get('question') or x.get('ori_question') or q);ans=str(x.get('answer') or ans)
        else:tools.append(x)
    if len(tools)<2 or not ans:return None,None
    tid=str(row.get('id') or row.get('image_path') or f'row:{idx}')
    blind_steps=[];out_atoms=[];ann_by_step=[]
    for i,x in enumerate(tools):
        ot=flatten(x.get('output') or x.get('result') or x.get('observation') or '')
        it=flatten(x.get('input') or '')
        oa=[{'id':aid(i,k,v),'kind':k,'value':v} for k,v in values(ot)]
        ia=[{'kind':k,'value':v} for k,v in values(it)]
        ann=' '.join(str(x.get(k) or '') for k in ('thought_query','thought','thought_choose'))
        blind_steps.append({'idx':i,'tool':canon_tool(x.get('name','')),'output_atoms':oa,'input_atoms':ia,'output_len':len(ot.encode()),'input_len':len(it.encode())})
        out_atoms.append(oa);ann_by_step.append(values(ann))
    claim_atoms=[{'kind':k,'value':v} for k,v in values(ans)]
    blind={'tid':tid,'steps':blind_steps,'claim_atoms':claim_atoms,'claim_len':len((q+ans).encode())}
    critical=set()
    for i,oa in enumerate(out_atoms):
        for a in oa:
            strong=a['kind'] in {'ENTITY','DATE','NUM'} or (a['kind']=='TOKEN' and len(a['value'])>=5)
            if not strong:continue
            for j in range(i+1,len(ann_by_step)):
                if any(a['value']==v and (a['kind']==k or a['kind']=='TOKEN' or k=='TOKEN') for k,v in ann_by_step[j]):
                    critical.add(a['id']);break
    all_ids=[a['id'] for xs in out_atoms for a in xs]
    atom_meta={a['id']:(i,a['kind'],a['value']) for i,xs in enumerate(out_atoms) for a in xs}
    pairs=[]
    for cid in sorted(critical):
        si,sk,_=atom_meta[cid]
        cand=[x for x in all_ids if x not in critical and atom_meta[x][0]==si and atom_meta[x][1]==sk]
        if not cand:cand=[x for x in all_ids if x not in critical and atom_meta[x][1]==sk]
        if not cand:continue
        nid=sorted(cand,key=lambda x:hashlib.sha256((tid+cid+x).encode()).hexdigest())[0]
        pairs.append({'critical_id':cid,'noncritical_id':nid})
    label={'tid':tid,'pairs':pairs,'critical_ids':sorted(critical),'annotation_fields_used':['thought_query','thought','thought_choose']} if pairs else None
    return blind,label

@dataclass(frozen=True)
class Atom:id:str;kind:str;value:str
@dataclass
class Step:idx:int;tool:str;outs:tuple[Atom,...];ins:tuple[Atom,...]
@dataclass
class Trace:tid:str;steps:list[Step];claim:tuple[Atom,...];raw_bytes:int
@dataclass(frozen=True)
class Edge:src:int;dst:int;atom:Atom
@dataclass
class Circuit:tr:Trace;edges:list[Edge];incoming:dict[int,list[Edge]];used:set[str];bytes:int

class Contracts:
    def __init__(self):self.n=collections.Counter();self.p=collections.Counter();self.keep=set()
    def add(self,s,d,k,p):self.n[(s,d,k)]+=1;self.p[(s,d,k)]+=int(p)
    def final(self):self.keep={k for k,n in self.n.items() if n>=3 and (self.p[k]+1)/(n+2)>=.20}
    def ok(self,s,d,k):return (s,d,k) in self.keep

def to_trace(x:dict)->Trace:
    ss=[]
    for s in x['steps']:
        outs=tuple(Atom(a['id'],a['kind'],a['value']) for a in s['output_atoms'])
        ins=tuple(Atom('',a['kind'],a['value']) for a in s['input_atoms'])
        ss.append(Step(s['idx'],s['tool'],outs,ins))
    claim=tuple(Atom('',a['kind'],a['value']) for a in x['claim_atoms'])
    raw=sum(s['output_len']+s['input_len'] for s in x['steps'])+x['claim_len']
    return Trace(x['tid'],ss,claim,raw)

def match(a:Atom,b:Atom)->bool:return a.value==b.value and (a.kind==b.kind or a.kind=='TOKEN' or b.kind=='TOKEN')
def flows(s:Step,d:Iterable[Atom]):return [(a,b) for a in s.outs for b in d if match(a,b)]

def train_contracts(ts:list[Trace])->Contracts:
    c=Contracts()
    for tr in ts:
        for j in range(1,len(tr.steps)):
            d=tr.steps[j];ks={a.kind for a in d.ins}
            for i in range(j):
                s=tr.steps[i];pos={b.kind if b.kind!='TOKEN' else a.kind for a,b in flows(s,d.ins)}
                for k in ks:c.add(s.tool,d.tool,k,k in pos)
        ks={a.kind for a in tr.claim}
        for s in tr.steps:
            pos={b.kind if b.kind!='TOKEN' else a.kind for a,b in flows(s,tr.claim)}
            for k in ks:c.add(s.tool,'CLAIM',k,k in pos)
    c.final();return c

def circuit(tr:Trace,c:Contracts,rewire:int|None=None)->Circuit:
    es=[];n=len(tr.steps)
    for j in range(1,n):
        d=tr.steps[j]
        for i in range(j):
            s=tr.steps[i]
            for a,b in flows(s,d.ins):
                k=b.kind if b.kind!='TOKEN' else a.kind
                if c.ok(s.tool,d.tool,k):es.append(Edge(i,j,a))
    for i,s in enumerate(tr.steps):
        for a,b in flows(s,tr.claim):
            k=b.kind if b.kind!='TOKEN' else a.kind
            if c.ok(s.tool,'CLAIM',k):es.append(Edge(i,n,a))
    if not any(e.dst==n for e in es):es.append(Edge(n-1,n,Atom('weak','WEAK','weak')))
    if rewire is not None:
        r=random.Random(rewire^int(hashlib.sha256(tr.tid.encode()).hexdigest()[:12],16));ne=[]
        for e in es:
            pool=[a for a in tr.steps[e.src].outs if a.id!=e.atom.id and a.kind==e.atom.kind] or [a for a in tr.steps[e.src].outs if a.id!=e.atom.id]
            ne.append(Edge(e.src,e.dst,r.choice(pool) if pool else e.atom))
        es=ne
    inc=collections.defaultdict(list)
    for e in es:inc[e.dst].append(e)
    return Circuit(tr,es,inc,{e.atom.id for e in es if e.atom.kind!='WEAK'},4*n+13*len(es))

def anomaly(c:Circuit,present:set[str])->float:
    n=len(c.tr.steps);reach=[False]*(n+1);reach[0]=True;b=t=0
    for j in range(1,n+1):
        es=c.incoming.get(j,[])
        if not es:
            if j<n:reach[j]=True
            continue
        groups=collections.defaultdict(list)
        for e in es:groups[(e.atom.kind,e.atom.value)].append(e)
        okall=True
        for gs in groups.values():
            t+=1;ok=False
            for e in gs:
                if (e.atom.kind=='WEAK' or e.atom.id in present) and (e.src==0 or reach[e.src]):ok=True;break
            if not ok:b+=1;okall=False
        reach[j]=okall
    return (0 if reach[n] else 1)+b/max(1,t)

def citation(c:Circuit,present:set[str])->float:
    es=c.incoming.get(len(c.tr.steps),[])
    return 0 if any(e.atom.kind=='WEAK' or e.atom.id in present for e in es) else 1

def prepare(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);blinds=[];labels=[]
    for i,r in enumerate(load(Path(args.input))):
        b,l=split_row(r,i)
        if b:blinds.append(b)
        if l:labels.append(l)
    bp=out/'blind.jsonl';lp=out/'labels_sealed.json'
    with bp.open('w',encoding='utf-8') as f:
        for x in blinds:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    lp.write_text(json.dumps(labels,ensure_ascii=False),encoding='utf-8')
    m={'blind_rows':len(blinds),'labeled_rows':len(labels),'pairs':sum(len(x['pairs']) for x in labels),'blind_sha256':hashlib.sha256(bp.read_bytes()).hexdigest(),'labels_sha256':hashlib.sha256(lp.read_bytes()).hexdigest()}
    (out/'prepare.json').write_text(json.dumps(m,indent=2),encoding='utf-8');print(json.dumps(m,indent=2))

def read_blind(path:Path):return [to_trace(json.loads(x)) for x in path.read_text().splitlines() if x.strip()]

def predict(args):
    if Path(args.labels_guard).exists():raise RuntimeError('sealed labels visible to predictor')
    train=read_blind(Path(args.train));test=read_blind(Path(args.test));c=train_contracts(train)
    out=[];raw=cb=0;t0=time.perf_counter();na=[];idx=[]
    for tr in test:
        cc=circuit(tr,c);raw+=tr.raw_bytes;cb+=cc.bytes
        ids=sorted(a.id for s in tr.steps for a in s.outs);present=set(ids)
        scores={x:anomaly(cc,present-{x}) for x in ids};cs={x:citation(cc,present-{x}) for x in ids};rew={}
        for sd in range(5):
            rc=circuit(tr,c,SEED+sd);rew[str(sd)]={x:anomaly(rc,present-{x}) for x in ids}
        out.append({'tid':tr.tid,'scores':scores,'citation':cs,'rewired':rew,'circuit_used':sorted(cc.used)})
        for x in ids[:8]:
            ncc=circuit(tr,c);na.append(anomaly(ncc,present-{x}));idx.append(anomaly(cc,present-{x}))
    t1=time.perf_counter()
    payload={'predictor_had_labels':False,'contracts':len(c.keep),'raw_bytes':raw,'circuit_bytes':cb,'storage_ratio':cb/max(1,raw),'runtime_seconds':t1-t0,'outputs_identical':na==idx,'predictions':out}
    p=Path(args.out);p.write_text(json.dumps(payload,separators=(',',':')),encoding='utf-8');Path(str(p)+'.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
    print(json.dumps({k:v for k,v in payload.items() if k!='predictions'},indent=2))

def auc(a,b):return sum(1 if x>y else .5 if x==y else 0 for x,y in zip(a,b))/len(a) if a else .5
def boot(d,n=5000):
    r=random.Random(SEED);z=[]
    for _ in range(n):z.append(sum(d[r.randrange(len(d))] for __ in d)/len(d))
    z.sort();return [z[int(.025*n)],z[int(.975*n)]] if d else [0,0]

def evaluate(args):
    pp=Path(args.predictions);assert hashlib.sha256(pp.read_bytes()).hexdigest()==Path(str(pp)+'.sha256').read_text().strip()
    payload=json.loads(pp.read_text());pr={x['tid']:x for x in payload['predictions']};labs=json.loads(Path(args.labels).read_text())
    cp=[];cn=[];bp=[];bn=[];hit=[];bhit=[];rew=[[] for _ in range(5)];rewn=[[] for _ in range(5)];usable=0
    for l in labs:
        p=pr.get(l['tid'])
        if not p:continue
        crit=set(l['critical_ids']);cands=set(p['scores'])
        if not crit&cands:continue
        pred=max(sorted(cands),key=lambda x:(p['scores'][x],x));hit.append(int(pred in crit))
        bpred=max(sorted(cands),key=lambda x:(p['citation'][x],x));bhit.append(int(bpred in crit))
        for pair in l['pairs']:
            a=pair['critical_id'];b=pair['noncritical_id']
            if a not in p['scores'] or b not in p['scores']:continue
            cp.append(p['scores'][a]);cn.append(p['scores'][b]);bp.append(p['citation'][a]);bn.append(p['citation'][b])
            for sd in range(5):rew[sd].append(p['rewired'][str(sd)][a]);rewn[sd].append(p['rewired'][str(sd)][b])
        usable+=1
    ca=auc(cp,cn);ba=auc(bp,bn);ra=[auc(x,y) for x,y in zip(rew,rewn)]
    pairdiff=[(1 if x>y else .5 if x==y else 0)-(1 if u>v else .5 if u==v else 0) for x,y,u,v in zip(cp,cn,bp,bn)]
    result={'usable_traces':usable,'n_pairs':len(cp),'cpoc_auc':ca,'citation_auc':ba,'gain':ca-ba,'bootstrap_gain_95_ci':boot(pairdiff),'cut_hit1':statistics.mean(hit) if hit else 0,'citation_cut_hit1':statistics.mean(bhit) if bhit else 0,'rewire_auc':ra,'rewire_auc_mean':statistics.mean(ra) if ra else .5,'rewire_drop':ca-(statistics.mean(ra) if ra else .5),'storage_ratio':payload['storage_ratio'],'runtime_outputs_identical':payload['outputs_identical']}
    strict={'enough_pairs':len(cp)>=300,'cpoc_auc':ca>=.80,'gain':ca-ba>=.10 and result['bootstrap_gain_95_ci'][0]>0,'cut_hit':result['cut_hit1']>=.70,'rewire_drop':result['rewire_drop']>=.10,'storage':result['storage_ratio']<=.50,'exact':result['runtime_outputs_identical']}
    result['strict']=strict;result['overall_pass']=all(strict.values());Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

def main():
    a=argparse.ArgumentParser();sp=a.add_subparsers(dest='mode',required=True)
    p=sp.add_parser('prepare');p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p=sp.add_parser('predict');p.add_argument('--train',required=True);p.add_argument('--test',required=True);p.add_argument('--labels-guard',required=True);p.add_argument('--out',required=True)
    p=sp.add_parser('evaluate');p.add_argument('--predictions',required=True);p.add_argument('--labels',required=True);p.add_argument('--out',required=True)
    x=a.parse_args();{'prepare':prepare,'predict':predict,'evaluate':evaluate}[x.mode](x)
if __name__=='__main__':main()
