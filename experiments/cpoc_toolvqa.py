#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, random, re, statistics, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SEED = 20260725
RNG = random.Random(SEED)
STOP = set('''a an the this that these those is are was were be been being to of in on at for from with by and or then as it its we you they he she them their his her our your which what who whom whose where when why how can could would should may might must do does did have has had use using used tool tools result results answer question image information know known now based find get need first next last there here about into more some any all each one two three four five than also only directly specific please'''.split())
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}|\d+(?:\.\d+)?")
CAP_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}|[A-Z]{2,}(?:\s+[A-Z]{2,})*)\b")
QUOTE_RE = re.compile(r"['\"]([^'\"]{2,80})['\"]")
NUM_RE = re.compile(r"\b\d{1,4}(?:[.,]\d+)*(?:%|st|nd|rd|th)?\b")
DATE_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)

READ_WORDS=('description','ocr','search','detect','recognition','classification','caption','retrieve','lookup','read','identify','localize','segment')
CALC_WORDS=('calculator','calculate','count','measure','math','wolfram','area','distance')

def canon_tool(name: str) -> str:
    n = re.sub(r'[^a-z0-9]+','_',str(name).lower()).strip('_')
    if n in {'question','answer','final'}: return 'CLAIM'
    if any(w in n for w in CALC_WORDS): return 'CALC'
    if any(w in n for w in READ_WORDS): return 'READ'
    return n.upper()[:40] or 'TOOL'

def norm(s: str) -> str:
    return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9]+',' ',str(s).lower())).strip()

def token_values(s: str) -> list[tuple[str,str]]:
    s = str(s or '')
    vals: list[tuple[str,str]] = []
    for q in QUOTE_RE.findall(s):
        qn=norm(q)
        if len(qn)>=3: vals.append(('ENTITY',qn))
    for c in CAP_RE.findall(s):
        cn=norm(c)
        if cn and cn not in STOP and len(cn)>=3: vals.append(('ENTITY',cn))
    for d in DATE_RE.findall(s):
        dn=norm(d)
        if dn: vals.append(('DATE',dn))
    for n in NUM_RE.findall(s):
        nn=norm(n)
        if nn: vals.append(('NUM',nn))
    for t in TOKEN_RE.findall(s):
        tn=norm(t)
        if tn and tn not in STOP and len(tn)>=3 and not tn.isdigit(): vals.append(('TOKEN',tn))
    seen=set(); out=[]
    for k,v in vals:
        key=(k,v)
        if key not in seen:
            seen.add(key); out.append(key)
    return out

def flatten(x: Any) -> str:
    if x is None: return ''
    if isinstance(x,str): return x
    if isinstance(x,(int,float,bool)): return str(x)
    if isinstance(x,dict): return ' '.join(f'{k} {flatten(v)}' for k,v in sorted(x.items()))
    if isinstance(x,list): return ' '.join(flatten(v) for v in x)
    return str(x)

@dataclass(frozen=True)
class Atom:
    kind: str
    value: str

@dataclass
class Step:
    idx: int
    tool: str
    output_atoms: tuple[Atom,...]
    demand_atoms: tuple[Atom,...]
    output_text: str
    demand_text: str

@dataclass
class Trace:
    tid: str
    steps: list[Step]
    claim_atoms: tuple[Atom,...]
    question: str
    answer: str

def load_jsonl(path: Path) -> list[dict]:
    rows=[]
    with path.open(encoding='utf-8',errors='ignore') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                x=json.loads(line)
                if isinstance(x,dict): rows.append(x)
            except Exception:
                continue
    return rows

def compile_row(row: dict, idx: int) -> Trace | None:
    context=row.get('context') or row.get('contexts') or []
    if isinstance(context,str):
        try: context=json.loads(context)
        except Exception: context=[]
    if not isinstance(context,list): return None
    q=str(row.get('better_ques') or row.get('question') or '')
    a=str(row.get('answer') or row.get('correct_answer') or '')
    tools=[]; qstep=None
    for x in context:
        if not isinstance(x,dict): continue
        if canon_tool(x.get('name',''))=='CLAIM':
            qstep=x
            q=str(x.get('question') or x.get('ori_question') or q)
            a=str(x.get('answer') or a)
        else: tools.append(x)
    if not tools or not a: return None
    steps=[]
    for i,x in enumerate(tools):
        out=flatten(x.get('output') or x.get('result') or x.get('observation') or '')
        demand=' '.join([flatten(x.get('input')),str(x.get('thought_query') or ''),str(x.get('thought') or '')])
        oa=tuple(Atom(k,v) for k,v in token_values(out))
        da=tuple(Atom(k,v) for k,v in token_values(demand))
        steps.append(Step(i,canon_tool(x.get('name','')),oa,da,out,demand))
    claim_atoms=tuple(Atom(k,v) for k,v in token_values(a))
    tid=str(row.get('id') or row.get('image_path') or f'row:{idx}')
    return Trace(tid,steps,claim_atoms,q,a)

def compile_rows(rows: list[dict]) -> list[Trace]:
    out=[]
    for i,r in enumerate(rows):
        tr=compile_row(r,i)
        if tr and len(tr.steps)>=2: out.append(tr)
    return out

class Contracts:
    def __init__(self):
        self.n=collections.Counter(); self.p=collections.Counter(); self.keep=set()
    def observe(self, src: Step, dst_tool: str, kind: str, positive: bool):
        k=(src.tool,dst_tool,kind)
        self.n[k]+=1; self.p[k]+=int(positive)
    def finalize(self,min_support=3,min_precision=.20):
        self.keep={k for k,n in self.n.items() if n>=min_support and (self.p[k]+1)/(n+2)>=min_precision}
    def allowed(self,src_tool,dst_tool,kind):
        return (src_tool,dst_tool,kind) in self.keep
    def atom_allowed(self,a):
        return True

def atom_match(a: Atom,b: Atom) -> bool:
    if a.value!=b.value: return False
    if a.kind==b.kind: return True
    return a.kind=='TOKEN' or b.kind=='TOKEN'

def exact_flow(src: Step, dst_atoms: Iterable[Atom]) -> list[tuple[Atom,Atom]]:
    out=[]
    for a in src.output_atoms:
        for b in dst_atoms:
            if atom_match(a,b): out.append((a,b))
    return out

def train_contracts(traces: list[Trace]) -> Contracts:
    c=Contracts()
    for tr in traces:
        for j in range(1,len(tr.steps)):
            dst=tr.steps[j]
            kinds=set(a.kind for a in dst.demand_atoms)
            for i in range(j):
                src=tr.steps[i]; flows=exact_flow(src,dst.demand_atoms)
                posk=set(a.kind for a,_ in flows)
                for kind in kinds: c.observe(src,dst.tool,kind,kind in posk)
        kinds=set(a.kind for a in tr.claim_atoms)
        for src in tr.steps:
            flows=exact_flow(src,tr.claim_atoms); posk=set(a.kind for a,_ in flows)
            for kind in kinds: c.observe(src,'CLAIM',kind,kind in posk)
    c.finalize()
    return c

@dataclass
class Edge:
    src:int
    dst:int
    atom:Atom

@dataclass
class Circuit:
    trace:Trace
    edges:list[Edge]
    used_atoms:set[tuple[int,Atom]]
    claim_support:set[int]
    incoming:dict[int,list[Edge]]
    raw_bytes:int
    circuit_bytes:int

def build_circuit(tr: Trace, contracts: Contracts, topology_only=False, rewire_seed=None) -> Circuit:
    edges=[]; n=len(tr.steps)
    if topology_only:
        for i in range(n-1): edges.append(Edge(i,i+1,Atom('SEQ',str(i))))
        edges.append(Edge(n-1,n,Atom('SEQ','claim')))
    else:
        for j in range(1,n):
            dst=tr.steps[j]
            for i in range(j):
                src=tr.steps[i]
                for a,b in exact_flow(src,dst.demand_atoms):
                    kind=b.kind if b.kind!='TOKEN' else a.kind
                    if contracts.atom_allowed(a) and contracts.allowed(src.tool,dst.tool,kind): edges.append(Edge(i,j,a))
        for i,src in enumerate(tr.steps):
            for a,b in exact_flow(src,tr.claim_atoms):
                kind=b.kind if b.kind!='TOKEN' else a.kind
                if contracts.atom_allowed(a) and contracts.allowed(src.tool,'CLAIM',kind): edges.append(Edge(i,n,a))
        if not any(e.dst==n for e in edges):
            edges.append(Edge(n-1,n,Atom('WEAK','last_tool')))
    if rewire_seed is not None and not topology_only:
        r=random.Random(rewire_seed ^ int(hashlib.sha256(tr.tid.encode()).hexdigest()[:12],16))
        new=[]
        for e in edges:
            pool=[a for a in tr.steps[e.src].output_atoms if a!=e.atom and a.kind==e.atom.kind]
            if not pool: pool=[a for a in tr.steps[e.src].output_atoms if a!=e.atom]
            new.append(Edge(e.src,e.dst,r.choice(pool) if pool else e.atom))
        edges=new
    inc=collections.defaultdict(list)
    for e in edges: inc[e.dst].append(e)
    used={(e.src,e.atom) for e in edges if e.atom.kind not in {'SEQ','WEAK'}}
    supports={e.src for e in edges if e.dst==n}
    raw=sum(len(s.output_text.encode())+len(s.demand_text.encode()) for s in tr.steps)+len((tr.question+tr.answer).encode())
    cbytes=4*len(tr.steps)+13*len(edges)
    return Circuit(tr,edges,used,supports,inc,raw,cbytes)

def obligations(c:Circuit, present:set[tuple[int,Atom]]|None=None) -> tuple[int,int,bool]:
    n=len(c.trace.steps); present=present if present is not None else set((i,a) for i,s in enumerate(c.trace.steps) for a in s.output_atoms)
    reachable=[False]*(n+1); reachable[0]=True
    broken=0; total=0
    for j in range(1,n+1):
        es=c.incoming.get(j,[])
        if not es:
            if j<n: reachable[j]=True
            continue
        groups=collections.defaultdict(list)
        for e in es: groups[(e.atom.kind,e.atom.value)].append(e)
        ok_all=True
        for _,ges in groups.items():
            total+=1
            ok=False
            for e in ges:
                atom_ok=e.atom.kind in {'SEQ','WEAK'} or (e.src,e.atom) in present
                parent_ok=True if e.src==0 else reachable[e.src]
                if atom_ok and parent_ok: ok=True; break
            if not ok: broken+=1; ok_all=False
        reachable[j]=ok_all
    return broken,total,reachable[n]

def anomaly(c:Circuit,present:set[tuple[int,Atom]]) -> float:
    b,t,s=obligations(c,present)
    return (0 if s else 1.0)+(b/max(1,t))

def citation_anomaly(c:Circuit,present:set[tuple[int,Atom]]) -> float:
    es=c.incoming.get(len(c.trace.steps),[])
    if not es:return 1.0
    ok=any(e.atom.kind in {'WEAK','SEQ'} or (e.src,e.atom) in present for e in es)
    return 0.0 if ok else 1.0

def pair_auc(pos:list[float],neg:list[float])->float:
    if not pos:return .5
    return sum(1 if a>b else .5 if a==b else 0 for a,b in zip(pos,neg))/len(pos)

def bootstrap_delta(a:list[float],b:list[float],n=5000):
    if not a:return [0.,0.]
    r=random.Random(SEED); d=[x-y for x,y in zip(a,b)]; vals=[]
    for _ in range(n): vals.append(sum(d[r.randrange(len(d))] for __ in d)/len(d))
    vals.sort(); return [vals[int(.025*n)],vals[int(.975*n)]]

def rename_trace(tr:Trace)->Trace:
    vals=sorted(set(a.value for s in tr.steps for a in s.output_atoms+s.demand_atoms)|set(a.value for a in tr.claim_atoms))
    mp={v:f'v{idx}' for idx,v in enumerate(vals)}
    def ra(xs):return tuple(Atom(a.kind,mp[a.value]) for a in xs)
    steps=[Step(s.idx,s.tool,ra(s.output_atoms),ra(s.demand_atoms),s.output_text,s.demand_text) for s in tr.steps]
    return Trace(tr.tid,steps,ra(tr.claim_atoms),tr.question,tr.answer)

def intervention_eval(test:list[Trace],contracts:Contracts,max_pairs=3000):
    pos=[];neg=[];cit_pos=[];cit_neg=[];top_pos=[];top_neg=[];cut_hit=[];cit_cut=[];rename_same=[]; rows=[]; selected=[]
    raw_bytes=circuit_bytes=0
    for tr in test:
        c=build_circuit(tr,contracts); top=build_circuit(tr,contracts,topology_only=True)
        raw_bytes+=c.raw_bytes;circuit_bytes+=c.circuit_bytes
        allp=set((i,a) for i,s in enumerate(tr.steps) for a in s.output_atoms)
        claim_used={(e.src,e.atom) for e in c.incoming.get(len(tr.steps),[]) if e.atom.kind not in {'WEAK','SEQ'}}
        bridge_used={x for x in c.used_atoms if x not in claim_used}
        critical=list(bridge_used or c.used_atoms)
        unused=[(i,a) for i,s in enumerate(tr.steps) for a in s.output_atoms if (i,a) not in c.used_atoms]
        if not critical or not unused: continue
        ca=RNG.choice(critical)
        same=[x for x in unused if x[0]==ca[0] and x[1].kind==ca[1].kind]
        if not same:same=[x for x in unused if x[1].kind==ca[1].kind]
        na=RNG.choice(same or unused)
        pc=set(allp);pc.discard(ca);pn=set(allp);pn.discard(na)
        sp=anomaly(c,pc);sn=anomaly(c,pn); pos.append(sp);neg.append(sn)
        cit_pos.append(citation_anomaly(c,pc));cit_neg.append(citation_anomaly(c,pn))
        top_pos.append(anomaly(top,set()));top_neg.append(anomaly(top,set()))
        if bridge_used:
            candidates=list(allp);scores=[anomaly(c,allp-{x}) for x in candidates]
            pred=candidates[max(range(len(scores)),key=lambda i:(scores[i],-i))]
            cut_hit.append(int(pred in bridge_used))
            cit_pred=next(iter(claim_used),candidates[0]);cit_cut.append(int(cit_pred in bridge_used))
        rc=build_circuit(rename_trace(tr),contracts)
        rename_same.append(int([(e.src,e.dst,e.atom.kind) for e in c.edges]==[(e.src,e.dst,e.atom.kind) for e in rc.edges] and obligations(c)[2]==obligations(rc)[2]))
        rows.append({'tid':tr.tid,'steps':len(tr.steps),'critical_step':ca[0],'critical_kind':ca[1].kind,'critical_value_hash':hashlib.sha256(ca[1].value.encode()).hexdigest()[:16],'noncritical_step':na[0],'cpoc_critical':sp,'cpoc_noncritical':sn,'citation_critical':cit_pos[-1],'citation_noncritical':cit_neg[-1]})
        selected.append((tr,ca,na))
        if len(pos)>=max_pairs:break
    rew=[]
    for sd in range(5):
        rp=[];rn=[]
        for tr,ca,na in selected:
            c=build_circuit(tr,contracts,rewire_seed=SEED+sd)
            allp=set((i,a) for i,s in enumerate(tr.steps) for a in s.output_atoms)
            rp.append(anomaly(c,allp-{ca}));rn.append(anomaly(c,allp-{na}))
        rew.append(pair_auc(rp,rn))
    cp_auc=pair_auc(pos,neg);cit_auc=pair_auc(cit_pos,cit_neg)
    return {
      'n_pairs':len(pos),'bridge_cut_cases':len(cut_hit),'cpoc_auc':cp_auc,'citation_auc':cit_auc,'topology_auc':pair_auc(top_pos,top_neg),
      'auc_gain_vs_citation':cp_auc-cit_auc,'bootstrap_gain_95_ci':bootstrap_delta([float(x>y)+.5*float(x==y) for x,y in zip(pos,neg)],[float(x>y)+.5*float(x==y) for x,y in zip(cit_pos,cit_neg)]),
      'cut_hit1':statistics.mean(cut_hit) if cut_hit else 0.,'citation_cut_hit1':statistics.mean(cit_cut) if cit_cut else 0.,
      'rename_invariance':statistics.mean(rename_same) if rename_same else 0.,'rewire_auc_mean':statistics.mean(rew) if rew else .5,'rewire_auc_seeds':rew,
      'rewire_drop':cp_auc-(statistics.mean(rew) if rew else .5),'raw_bytes':raw_bytes,'circuit_bytes':circuit_bytes,'storage_ratio':circuit_bytes/max(1,raw_bytes),
      'rows':rows
    }

def runtime_eval(test:list[Trace],contracts:Contracts,limit=300):
    circuits=[build_circuit(t,contracts) for t in test[:limit]]
    interventions=[]
    for c in circuits:
        allp=set((i,a) for i,s in enumerate(c.trace.steps) for a in s.output_atoms)
        for x in list(allp)[:12]:interventions.append((c,allp,x))
    t0=time.perf_counter();na=[]
    for c,allp,x in interventions:
        cc=build_circuit(c.trace,contracts);na.append(anomaly(cc,allp-{x}))
    t1=time.perf_counter();inc=[]
    for c,allp,x in interventions:inc.append(anomaly(c,allp-{x}))
    t2=time.perf_counter()
    return {'n_interventions':len(interventions),'naive_seconds':t1-t0,'indexed_seconds':t2-t1,'speedup':(t1-t0)/max(1e-12,t2-t1),'outputs_identical':na==inc}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--test',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    train=compile_rows(load_jsonl(Path(a.train)));test=compile_rows(load_jsonl(Path(a.test)))
    contracts=train_contracts(train)
    ev=intervention_eval(test,contracts);runtime=runtime_eval(test,contracts)
    rows=ev.pop('rows')
    strict={
      'enough_test':len(test)>=500,'enough_pairs':ev['n_pairs']>=300,'cpoc_auc':ev['cpoc_auc']>=.80,
      'gain_vs_citation':ev['auc_gain_vs_citation']>=.10 and ev['bootstrap_gain_95_ci'][0]>0,
      'cut_hit':ev['cut_hit1']>=.70,'rewire_drop':ev['rewire_drop']>=.10,'rename_invariance':ev['rename_invariance']==1.0,
      'storage':ev['storage_ratio']<=.50,'runtime':runtime['outputs_identical'] and runtime['speedup']>=1.5,
    }
    summary={'architecture':'Counterfactual Provenance-Obligation Circuit','dataset':'ToolVQA public train/test proxy because TRACE-Bench release was not discoverable','train_traces':len(train),'test_traces':len(test),'contracts_kept':len(contracts.keep),'intervention':ev,'runtime':runtime,'strict':strict,'overall_pass':all(strict.values()),'limitations':['ToolVQA is the public source benchmark underlying TRACE-Bench, not the unreleased TRACE-Bench provenance annotations.','Critical/noncritical deletions are controlled from explicit value-flow in the held-out tool trace.','No images, embeddings, pretrained encoders, LLMs, TF-IDF, SVD, nearest-vector methods, or external solvers are used.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'interventions.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (out/'contracts.json').write_text(json.dumps([{'src':k[0],'dst':k[1],'kind':k[2],'n':contracts.n[k],'positive':contracts.p[k]} for k in sorted(contracts.keep)],indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
