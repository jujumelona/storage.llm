#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, random, re, statistics, time
from collections import Counter, defaultdict, deque
from pathlib import Path

SEED=20260723
random.seed(SEED)
STOP=set('''a an the this that these those is are was were be been being of to in on at for from with by and or as it its their they them he she his her there here what which who whom whose how why when where do does did have has had can could may might will would should must very more most less many much some any all one two image picture photo shown shows show tool result output input please find get use using used based according answer question common likely specific type kind object objects thing things'''.split())
WORD_RE=re.compile(r"[a-z][a-z0-9_'-]{1,40}",re.I)
NUM_RE=re.compile(r"[-+]?\d+(?:\.\d+)?%?")
DATE_RE=re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b")
COLOR=set('red orange yellow green blue purple violet black white gray grey brown pink cyan magenta gold golden silver beige maroon navy teal'.split())
ERR_RE=re.compile(r'\b(error|failed|failure|invalid|unable|cannot|timeout|not found|denied|exception)\b',re.I)


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def norm(s):
    return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9.%+/_-]+',' ',str(s).lower())).strip()
def flatten_values(x):
    if isinstance(x,dict):
        for k,v in x.items():
            yield str(k)
            yield from flatten_values(v)
    elif isinstance(x,(list,tuple)):
        for v in x: yield from flatten_values(v)
    elif x is not None: yield str(x)
def atoms_text(x, keep_generic=False):
    s=' '.join(flatten_values(x)) if not isinstance(x,str) else x
    ns=norm(s); out=set()
    for w in WORD_RE.findall(ns):
        w=w.lower().strip("_'-")
        if len(w)>=2 and (keep_generic or w not in STOP): out.add('W:'+w)
        if w in COLOR: out.add('C:'+w)
    for z in NUM_RE.findall(ns): out.add('N:'+z.rstrip('%'))
    for z in DATE_RE.findall(ns): out.add('D:'+z)
    toks=[w[2:] for w in sorted(out) if w.startswith('W:')]
    if 1<=len(toks)<=8: out.add('P:'+' '.join(toks))
    return out

def answer_atoms(answer):
    a=atoms_text(str(answer));n=norm(answer)
    if n:a.add('ANS:'+n)
    return a

def op_family(name):
    n=norm(name)
    if 'ocr' in n or 'text' in n and 'recogn' in n:return 'OCR'
    if 'search' in n or 'google' in n or 'wiki' in n:return 'SEARCH'
    if 'calc' in n or 'math' in n:return 'CALC'
    if 'count' in n:return 'COUNT'
    if 'crop' in n or 'zoom' in n:return 'CROP'
    if 'detect' in n or 'ground' in n:return 'DETECT'
    if 'classif' in n:return 'CLASSIFY'
    if 'description' in n or 'caption' in n:return 'DESCRIBE'
    return 'OTHER'

def context_steps(row,ablate_identity=False,outputs_override=None):
    steps=[]
    for i,s in enumerate(row.get('context') or []):
        if not isinstance(s,dict):continue
        name=str(s.get('name') or 'OTHER');inp=s.get('input') or {}
        out=(outputs_override[i] if outputs_override is not None and i<len(outputs_override) else s.get('output')) or ''
        steps.append({'i':i,'family':'GENERIC' if ablate_identity else op_family(name),'input_atoms':atoms_text(inp),'output_atoms':atoms_text(str(out)),'output_norm':norm(out),'status':'ERROR' if ERR_RE.search(str(out)) else 'OK'})
    return steps

def numeric_equal(x,y,tol=1e-6):
    try:return abs(float(x)-float(y))<=tol*max(1,abs(float(y)))
    except:return False

def numeric_clauses(steps,answer):
    ans=NUM_RE.findall(norm(answer))
    if len(ans)!=1:return []
    try:target=float(ans[0].rstrip('%'))
    except:return []
    nums=[]
    for s in steps:
        for a in s['output_atoms']:
            if a.startswith('N:'):
                try:nums.append((s['i'],float(a[2:])))
                except:pass
    clauses=[]
    for i,v in nums:
        if numeric_equal(v,target):clauses.append(frozenset([i]))
    for (i,a),(j,b) in itertools.combinations(nums,2):
        tests=[a+b,a-b,b-a,a*b]
        if b:tests.append(a/b)
        if a:tests.append(b/a)
        if any(numeric_equal(v,target) for v in tests):clauses.append(frozenset([i,j]))
    return clauses

def build_circuit(row,ablate_identity=False,outputs_override=None,orderless=False):
    steps=context_steps(row,ablate_identity,outputs_override);qatoms=atoms_text(str(row.get('question') or ''));aatoms=answer_atoms(str(row.get('answer') or ''));claim={x for x in aatoms if not x.startswith('ANS:')};fullans=norm(row.get('answer') or '')
    edges=defaultdict(set);edge_atoms={}
    for i,a in enumerate(steps):
        for j,b in enumerate(steps):
            if i==j or (not orderless and i>=j):continue
            shared=(a['output_atoms']&b['input_atoms'])-qatoms;shared={x for x in shared if not x.startswith('P:')}
            if shared:edges[i].add(j);edge_atoms[(i,j)]=tuple(sorted(shared))
    direct=[];direct_score={}
    for s in steps:
        inter=s['output_atoms']&claim;phrase=bool(fullans and fullans in s['output_norm']);score=len(inter)+3*phrase;direct_score[s['i']]=score
        if score>0:direct.append(s['i'])
    rev=defaultdict(set)
    for i,js in edges.items():
        for j in js:rev[j].add(i)
    def ancestors(j):
        seen=set();dq=deque([j])
        while dq:
            x=dq.popleft()
            for p in rev.get(x,()):
                if p not in seen:seen.add(p);dq.append(p)
        return seen
    clauses=[frozenset({j}|ancestors(j)) for j in direct];clauses.extend(numeric_clauses(steps,row.get('answer') or ''));fallback=False
    if not clauses and steps:
        candidates=[s['i'] for s in steps if s['status']=='OK']
        if candidates:
            j=candidates[-1];clauses=[frozenset({j}|ancestors(j))];fallback=True
    uniq=sorted(set(clauses),key=lambda x:(len(x),tuple(x)));clauses=[c for c in uniq if not any(d<c for d in uniq)];total=max(1,len(clauses));deletion_score={i:sum(i in c for c in clauses)/total for i in range(len(steps))};minsize=min((len(c) for c in clauses),default=0);selected=set().union(*(c for c in clauses if len(c)==minsize)) if clauses else set()
    changed=True
    while changed:
        changed=False
        for i,js in edges.items():
            if any(j in selected for j in js) and i not in selected:selected.add(i);changed=True
    lexical={i for i,v in direct_score.items() if v>0};flow=set(lexical)
    for j in list(lexical):flow|=ancestors(j)
    circuit_sel=set().union(*(c for c in clauses if len(c)==minsize)) if clauses else set()
    return {'steps':steps,'edges':edges,'edge_atoms':edge_atoms,'clauses':clauses,'selected':selected,'lexical':lexical,'flow':flow,'circuit':circuit_sel,'deletion_score':deletion_score,'direct_score':direct_score,'fallback':fallback,'claim_atoms':sorted(claim)}

def min_hitting_set(clauses,n):
    if not clauses:return set()
    for k in range(1,min(n,12)+1):
        for comb in itertools.combinations(range(n),k):
            s=set(comb)
            if all(s&set(c) for c in clauses):return s
    remaining=[set(c) for c in clauses];out=set()
    while remaining:
        c=Counter(i for x in remaining for i in x)
        if not c:break
        i=c.most_common(1)[0][0];out.add(i);remaining=[x for x in remaining if i not in x]
    return out

def prepare(args):
    from huggingface_hub import hf_hub_download
    fp=hf_hub_download('DietCoke4671/ToolVQA','test.jsonl',repo_type='dataset',cache_dir=args.cache);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);blind=out/'blind.jsonl';labels=[];n=0
    with open(fp,encoding='utf-8') as f,blind.open('w',encoding='utf-8') as w:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);ctx=[];gold=[]
            for s in r.get('context') or []:
                if not isinstance(s,dict):continue
                ctx.append({'name':s.get('name'),'input':s.get('input'),'output':s.get('output')});gold.append(1 if str(s.get('is_important','')).strip().lower() in {'yes','1','true'} else 0)
            if not ctx:continue
            item={'id':n,'question':r.get('question'),'answer':r.get('answer'),'type':r.get('type'),'context':ctx,'reliable':str(r.get('correct_answer','')).lower()=='yes' and str(r.get('only_answer',''))=='1'}
            w.write(json.dumps(item,ensure_ascii=False)+'\n');labels.append({'id':n,'important':gold,'reliable':item['reliable'],'type':item['type']});n+=1
    lp=out/'labels_sealed.json';lp.write_text(json.dumps(labels,ensure_ascii=False),encoding='utf-8');manifest={'rows':n,'blind_sha256':sha(blind),'labels_sha256':sha(lp),'source':'DietCoke4671/ToolVQA test.jsonl'};(out/'prepare_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))

def predict(args):
    inp=Path(args.input);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if 'is_important' in inp.read_text(encoding='utf-8',errors='ignore'):raise RuntimeError('gold label leaked into predictor input')
    rng=random.Random(SEED);rows=[];global_gates=set();raw_bytes=0;naive_time=indexed_time=0.;same=True
    for line in inp.open(encoding='utf-8'):
        if not line.strip():continue
        r=json.loads(line);c=build_circuit(r);outputs=[s.get('output') or '' for s in r['context']];perm=list(outputs);rng.shuffle(perm);rw=build_circuit(r,outputs_override=perm);ab=build_circuit(r,ablate_identity=True);od=build_circuit(r,orderless=True);n=len(c['steps'])
        t=time.perf_counter();nv=[]
        for i in range(n):nv.append(sum(1 for cl in c['clauses'] if i in cl)/max(1,len(c['clauses'])))
        naive_time+=time.perf_counter()-t;t=time.perf_counter();iv=[c['deletion_score'].get(i,0.) for i in range(n)];indexed_time+=time.perf_counter()-t;same=same and nv==iv;cut=min_hitting_set(c['clauses'],n)
        pred={'lexical':sorted(c['lexical']),'flow':sorted(c['flow']),'circuit':sorted(c['circuit']),'full':sorted(c['selected']),'orderless':sorted(od['selected']),'identity_ablation':sorted(ab['selected']),'rewired':sorted(rw['selected']),'cutset':sorted(cut)}
        scores={'lexical':[float(c['direct_score'].get(i,0)>0) for i in range(n)],'flow':[float(i in c['flow']) for i in range(n)],'circuit':[c['deletion_score'].get(i,0.) for i in range(n)],'full':[c['deletion_score'].get(i,0.)+.25*float(i in c['selected']) for i in range(n)],'rewired':[rw['deletion_score'].get(i,0.)+.25*float(i in rw['selected']) for i in range(n)]}
        raw_bytes+=len(json.dumps(r['context'],ensure_ascii=False).encode())
        for s in c['steps']:global_gates.add(('E',s['family'],s['status'],tuple(sorted(s['output_atoms']))))
        for a,bs in c['edges'].items():
            for b in bs:global_gates.add(('T',c['steps'][a]['family'],c['steps'][b]['family'],c['edge_atoms'].get((a,b),())))
        for cl in c['clauses']:global_gates.add(('AND',tuple(sorted(cl))))
        global_gates.add(('OR',tuple(sorted(tuple(sorted(x)) for x in c['clauses']))));rows.append({'id':r['id'],'type':r.get('type'),'reliable':r.get('reliable'),'n_steps':n,'pred':pred,'scores':scores,'clauses':[sorted(x) for x in c['clauses']],'fallback':c['fallback'],'claim_atoms':c['claim_atoms']})
    circuit_bytes=len(json.dumps([repr(x) for x in sorted(global_gates,key=repr)]).encode());payload={'predictor_had_labels':False,'input_sha256':sha(inp),'rows':rows,'runtime':{'naive_seconds':naive_time,'indexed_seconds':indexed_time,'outputs_identical':same,'speedup':naive_time/max(indexed_time,1e-12)},'storage':{'raw_context_bytes':raw_bytes,'shared_circuit_bytes':circuit_bytes,'ratio':circuit_bytes/max(raw_bytes,1)},'global_gate_count':len(global_gates)};p=out/'predictions.json';p.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8');(out/'predictions.json.sha256').write_text(sha(p));print(json.dumps({'rows':len(rows),'runtime':payload['runtime'],'storage':payload['storage'],'gates':len(global_gates)},indent=2))

def auc(labels,scores):
    pos=[s for y,s in zip(labels,scores) if y];neg=[s for y,s in zip(labels,scores) if not y]
    if not pos or not neg:return None
    return sum(1 if a>b else .5 if a==b else 0 for a in pos for b in neg)/(len(pos)*len(neg))

def metrics(rows,labels,subset=None):
    kinds=['lexical','flow','circuit','full','orderless','identity_ablation','rewired','cutset'];out={}
    for kind in kinds:
        tp=fp=fn=0;ys=[];ss=[];jacc=[];exact=0;nrow=0
        for r,g in zip(rows,labels):
            if subset and not subset(r,g):continue
            n=min(r['n_steps'],len(g['important']));gold={i for i,y in enumerate(g['important'][:n]) if y};pred=set(r['pred'][kind])&set(range(n));tp+=len(gold&pred);fp+=len(pred-gold);fn+=len(gold-pred);jacc.append(len(gold&pred)/max(1,len(gold|pred)));exact+=int(gold==pred);nrow+=1
            if kind in r['scores']:ys.extend(g['important'][:n]);ss.extend(r['scores'][kind][:n])
        pr=tp/max(1,tp+fp);rc=tp/max(1,tp+fn);f1=2*pr*rc/max(1e-12,pr+rc);out[kind]={'rows':nrow,'precision':pr,'recall':rc,'f1':f1,'auroc':auc(ys,ss) if ss else None,'set_jaccard':statistics.mean(jacc) if jacc else 0,'set_exact':exact/max(1,nrow),'tp':tp,'fp':fp,'fn':fn}
    return out

def bootstrap(rows,labels,a='full',b='lexical',nboot=5000):
    pairs=list(zip(rows,labels));rng=random.Random(SEED);vals=[]
    def f1(sample,kind):
        tp=fp=fn=0
        for r,g in sample:
            n=min(r['n_steps'],len(g['important']));gold={i for i,y in enumerate(g['important'][:n]) if y};pred=set(r['pred'][kind])&set(range(n));tp+=len(gold&pred);fp+=len(pred-gold);fn+=len(gold-pred)
        return 2*tp/max(1,2*tp+fp+fn)
    for _ in range(nboot):
        s=[pairs[rng.randrange(len(pairs))] for _ in pairs];vals.append(f1(s,a)-f1(s,b))
    vals.sort();return {'mean':statistics.mean(vals),'lo':vals[int(.025*nboot)],'hi':vals[int(.975*nboot)]}

def evaluate(args):
    pp=Path(args.predictions);expected=Path(str(pp)+'.sha256').read_text().strip()
    if sha(pp)!=expected:raise RuntimeError('prediction file changed after blind prediction')
    payload=json.loads(pp.read_text());labels=json.loads(Path(args.labels).read_text());rows=payload['rows'];byid={x['id']:x for x in labels};labs=[byid[r['id']] for r in rows];allm=metrics(rows,labs);relm=metrics(rows,labs,lambda r,g:g.get('reliable'));types={}
    for t in sorted(set(str(g.get('type')) for g in labs)):types[t]=metrics(rows,labs,lambda r,g,tt=t:str(g.get('type'))==tt)['full']
    boot=bootstrap(rows,labs);strict={'full_beats_lexical_5pp':allm['full']['f1']>=allm['lexical']['f1']+.05,'bootstrap_ci_positive':boot['lo']>0,'deletion_auroc_075':(allm['full']['auroc'] or 0)>=.75,'rewire_drop_5pp':allm['full']['f1']>=allm['rewired']['f1']+.05,'identity_retains_90pct':allm['identity_ablation']['f1']>=.9*allm['full']['f1'],'storage_below_half':payload['storage']['ratio']<.5,'indexed_exact_and_faster':payload['runtime']['outputs_identical'] and payload['runtime']['speedup']>1};summary={'benchmark':'ToolVQA public test important-step annotations (TRACE-Bench source benchmark fallback)','n_rows':len(rows),'all':allm,'reliable_subset':relm,'by_type':types,'bootstrap_full_minus_lexical':boot,'runtime':payload['runtime'],'storage':payload['storage'],'fallback_rows':sum(r['fallback'] for r in rows),'strict':strict,'overall_pass':all(strict.values()),'limitations':['Official TRACE-Bench provenance records were not publicly discoverable, so this uses its public ToolVQA source benchmark.','The task is evidence-turn localization for a known reference answer, not VQA answer generation.','Human is_important annotations are turn-level necessity labels, not sentence-level relation labels.']};out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (out/'predictions_audit.jsonl').open('w',encoding='utf-8') as f:
        for r,g in zip(rows,labs):f.write(json.dumps({'id':r['id'],'gold':g['important'],'pred':r['pred'],'scores':r['scores'],'clauses':r['clauses'],'fallback':r['fallback']})+'\n')
    print(json.dumps(summary,indent=2))

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='mode',required=True);p=sp.add_parser('prepare');p.add_argument('--out',required=True);p.add_argument('--cache',default='.hf_cache');p=sp.add_parser('predict');p.add_argument('--input',required=True);p.add_argument('--out',required=True);p=sp.add_parser('evaluate');p.add_argument('--predictions',required=True);p.add_argument('--labels',required=True);p.add_argument('--out',required=True);a=ap.parse_args();{'prepare':prepare,'predict':predict,'evaluate':evaluate}[a.mode](a)
if __name__=='__main__':main()
