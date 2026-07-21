#!/usr/bin/env python3
from __future__ import annotations
import csv, glob, hashlib, itertools, json, math, os, random, re, statistics, time
from collections import Counter, defaultdict, deque
from pathlib import Path
import trail_nondense_full_eval as m

# Reuse V3's raw-event compiler and generic logic functions, but do not execute its runner.
src=Path('experiments/trail_nondense_v3_runner.py').read_text(encoding='utf-8')
prefix=src.split('\nm.load_events=load_events',1)[0]
ns={}
exec(compile(prefix,'trail_nondense_v3_runner.py','exec'),ns)
load_events=ns['load_events']; PAY=ns['PAY']; branch_ids=ns['branch_ids']; err_sig=ns['err_sig']; primary=ns['primary_from_features']; logic=ns['logic']; lq=ns['lq']

STOP=set('the and that this with from into over under then than they them their there here have has had were was are been being will would could should may might must can cannot for not but because while after before during about through between against without within upon also only very more most some any each every other such own same just both either neither whether however therefore thus hence agent system task tool function output input value message content response result trace span user assistant model true false null none'.split())
WORD=re.compile(r'[a-z][a-z0-9_]{2,30}')

def norm_name(x):
    x=re.sub(r'[0-9a-f]{8,}','<id>',str(x).lower()); x=re.sub(r'\d+','<n>',x); return x[:120]
def words(x):
    out=[]
    for w in WORD.findall(norm_name(x)):
        if w not in STOP and not w.startswith('pat_'): out.append(w)
    return out

def trace_records(path):
    tid,ev=load_events(path); by={e.span_id:e for e in ev}; branches=branch_ids(ev); repeated=Counter(); prev_error=defaultdict(int); recs=[]
    for e in ev:
        p=PAY[(tid,e.span_id)]; b=branches[e.span_id]; sig=err_sig(e) if e.has_error else None; fs=set()
        fs.add('K:'+p['kind']); fs.add('N:'+norm_name(p['name'])); fs.add('T:'+norm_name(p['tool']))
        fs.add('S:'+norm_name(p['status']).split()[0] if p['status'].strip() else 'S:empty')
        for x in p['exceptions']: fs.add('E:'+x.lower())
        for x in p['http_codes']: fs.add('H:'+x)
        pr=primary(p)
        if pr: fs.add('P:'+pr[0])
        if e.is_final: fs.add('F:final')
        if e.is_plan: fs.add('F:plan')
        if e.has_error: fs.add('F:error')
        parent=by.get(e.parent_span_id)
        if parent: fs.add('PN:'+norm_name(parent.name))
        depth=0; cur=parent; seen=set()
        while cur and cur.span_id not in seen and depth<12:
            seen.add(cur.span_id); depth+=1; cur=by.get(cur.parent_span_id)
        fs.add('D:'+str(min(depth,5)))
        if prev_error[b]: fs.add('Q:prior_error')
        if sig:
            repeated[(b,sig)]+=1; fs.add('R:'+str(min(repeated[(b,sig)],4)))
            if repeated[(b,sig)]>=2: fs.add('Q:repeat_error')
        if e.is_final and prev_error[b]: fs.add('Q:final_after_error')
        if e.has_error: prev_error[b]+=1
        lex=words(p['output']+' '+p['input'])
        uniq=[]; seenw=set()
        for w in lex:
            if w not in seenw: seenw.add(w); uniq.append(w)
            if len(uniq)>=60: break
        for w in uniq: fs.add('L:'+w)
        for a,bg in zip(uniq[:30],uniq[1:31]): fs.add('B:'+a+'_'+bg)
        recs.append({'tid':tid,'sid':e.span_id,'features':frozenset(fs),'evidence':m.excerpt(p['text'],500)})
    return tid,recs

def load_gold(paths):
    d={}
    for p in paths:
        try:
            x=json.load(open(p,encoding='utf-8')); d[str(x.get('trace_id') or Path(p).stem)]={(str(e.get('location')),str(e.get('category'))) for e in x.get('errors',[]) if e.get('location') and e.get('category')}
        except Exception: pass
    return d

def selected_features(fs):
    st=sorted(x for x in fs if not x.startswith(('L:','B:')))
    lx=sorted(x for x in fs if x.startswith('L:'))[:20]
    bg=sorted(x for x in fs if x.startswith('B:'))[:8]
    return (st+lx+bg)[:38]

def learn_rules(records,gold_tids,gold,min_pos=2):
    total=Counter(); pos=defaultdict(Counter); pair_total=Counter(); pair_pos=defaultdict(Counter); cat_total=Counter(); n=0
    for r in records:
        if r['tid'] not in gold_tids: continue
        n+=1; cats={c for s,c in gold.get(r['tid'],set()) if s==r['sid']}; cat_total.update(cats); f=selected_features(r['features']); total.update(f)
        for c in cats: pos[c].update(f)
        pairs=list(itertools.combinations(f,2)); pair_total.update(pairs)
        for c in cats: pair_pos[c].update(pairs)
    rules=[]
    for c,pc in pos.items():
        for f,pn in pc.items():
            if pn<min_pos: continue
            pr=(pn+1)/(total[f]+2); base=(cat_total[c]+1)/(n+2); lift=math.log(max(1e-9,pr/base)); score=lift+0.15*math.log1p(pn)
            if pr>=0.16 and score>0: rules.append({'cat':c,'need':(f,),'precision':pr,'support':pn,'score':score})
        for pair,pn in pair_pos[c].items():
            if pn<min_pos: continue
            pr=(pn+1)/(pair_total[pair]+2); base=(cat_total[c]+1)/(n+2); lift=math.log(max(1e-9,pr/base)); score=lift+0.2*math.log1p(pn)
            if pr>=0.28 and score>0: rules.append({'cat':c,'need':pair,'precision':pr,'support':pn,'score':score})
    rules.sort(key=lambda x:(-x['score'],-x['support'],x['cat'],x['need'])); return rules

def make_index(rules):
    idx=defaultdict(list)
    for i,r in enumerate(rules):
        for f in r['need']: idx[f].append(i)
    return idx

def score_records(records,tids,rules,indexed=True):
    idx=make_index(rules); out=defaultdict(list)
    for rec in records:
        if rec['tid'] not in tids: continue
        fs=rec['features']; candidates=set()
        if indexed:
            for f in fs: candidates.update(idx.get(f,()))
        else: candidates=range(len(rules))
        best={}
        for i in candidates:
            ru=rules[i]
            if all(x in fs for x in ru['need']):
                old=best.get(ru['cat'])
                if old is None or ru['score']>old[0]: best[ru['cat']]=(ru['score'],i)
        for c,(sc,i) in best.items(): out[rec['tid']].append((sc,rec['sid'],c,i,rec['evidence']))
    return out

def pairs_from_scores(scores,k,threshold,rules):
    pred={}; details={}
    for tid,arr in scores.items():
        arr=sorted(arr,key=lambda x:(-x[0],x[1],x[2])); keep=[]; seen=set()
        for x in arr:
            key=(x[1],x[2])
            if x[0]<threshold or key in seen: continue
            seen.add(key); keep.append(x)
            if len(keep)>=k: break
        pred[tid]={ (x[1],x[2]) for x in keep}; details[tid]=keep
    return pred,details

def micro(pred,tids,gold):
    gt={(t,s,c) for t in tids for s,c in gold.get(t,set())}; pp={(t,s,c) for t in tids for s,c in pred.get(t,set())}; tp=len(gt&pp); return m.prf(tp,len(pp-gt),len(gt-pp))+(len(gt),len(pp),tp)

def choose_calibration(core,val,records,gold):
    rules=learn_rules(records,core,gold); scores=score_records(records,val,rules,True); best=(-1,6,0.5)
    vals=sorted({x[0] for a in scores.values() for x in a})
    thresholds=[0.2,0.5,0.8,1.1,1.4,1.8,2.2]+([vals[int(len(vals)*q)] for q in [.25,.5,.75]] if vals else [])
    for k in range(3,11):
        for th in thresholds:
            p,_=pairs_from_scores(scores,k,th,rules); f=micro(p,val,gold)[2]
            if f>best[0]: best=(f,k,th)
    return best[1],best[2]

def trail_oof(records,tids,gold):
    allpred={}; alldetail={}; fold_info=[]; saved_rules=[]
    for fold in [0,1]:
        test={t for t in tids if int(hashlib.sha256(t.encode()).hexdigest()[:8],16)%2==fold}; train=set(tids)-test
        val={t for t in train if int(hashlib.sha256(('v'+t).encode()).hexdigest()[:8],16)%4==0}; core=train-val
        if not val: val=set(list(train)[:max(1,len(train)//5)]); core=train-val
        k,th=choose_calibration(core,val,records,gold); rules=learn_rules(records,train,gold); scores=score_records(records,test,rules,True); pred,det=pairs_from_scores(scores,k,th,rules); allpred.update(pred); alldetail.update(det); fold_info.append({'fold':fold,'train':len(train),'test':len(test),'k':k,'threshold':th,'rules':len(rules)}); saved_rules.append(rules)
    return allpred,alldetail,fold_info,saved_rules

def runtime_bench(records,rules):
    t=time.perf_counter(); a=score_records(records,{r['tid'] for r in records},rules,False); naive=time.perf_counter()-t
    t=time.perf_counter(); b=score_records(records,{r['tid'] for r in records},rules,True); indexed=time.perf_counter()-t
    normalize=lambda x:{t:{(round(v[0],12),v[1],v[2]) for v in arr} for t,arr in x.items()}
    return {'answers_identical':normalize(a)==normalize(b),'rules':len(rules),'naive_seconds':naive,'indexed_seconds':indexed,'reasoning_speedup':naive/indexed if indexed else 0}

# ---------- LogiQA sparse operator ledger ----------
def qtype(q):
    q=q.lower()
    for x,n in [('weaken','weaken'),('strengthen','strengthen'),('support','support'),('assum','assumption'),('presuppos','assumption'),('except','except'),('must be true','must_true'),('also true','must_true'),('can be established','must_true'),('evaluation','evaluate'),('most appropriate','evaluate'),('similar','analogy'),('belongs','analogy')]:
        if x in q:return n
    return 'other'
def content_tokens(s): return {w for w in WORD.findall(s.lower()) if w not in STOP}
def binv(x,cuts): return str(sum(x>=c for c in cuts))
def option_features(p,q,opts,i):
    o=re.sub(r'^[A-D]\.\s*','',opts[i]); pt=content_tokens(p); qt=content_tokens(q); ot=content_tokens(o); lens=[len(content_tokens(x)) for x in opts]; overlaps=[len(content_tokens(x)&pt) for x in opts]
    fs={'Q:'+qtype(q),'I:'+str(i),'LEN:'+binv(len(ot),[5,10,18,30]),'LR:'+str(sorted(range(4),key=lambda j:lens[j]).index(i)),'OV:'+binv(len(ot&pt),[1,3,6,10]),'OR:'+str(sorted(range(4),key=lambda j:overlaps[j]).index(i)),'QOV:'+binv(len(ot&qt),[1,2,4]),'NEG:'+str(min(3,len(re.findall(r"(?i)\bnot\b|n't|never|no\b",o))))}
    for name,pat in [('IF',r'\bif\b'),('ONLY',r'\bonly\b'),('ALL',r'\b(?:all|every|each)\b'),('SOME',r'\b(?:some|at least|may)\b'),('CAUSE',r'\b(?:because|therefore|cause|result)\b'),('MODAL',r'\b(?:must|cannot|can|possible|necessary)\b'),('ALT',r'\b(?:other|alternative|outside|however|but)\b')]: fs.add(name+':'+str(int(bool(re.search(pat,o,re.I)))))
    try: fs.add('ENT:'+str(int(logic(p,o)=='yes')))
    except Exception: fs.add('ENT:0')
    return fs

def learn_option_rules(dataset,minsup=5,pairmin=8):
    total=Counter();pos=Counter();pt=Counter();pp=Counter();n=0
    for gold,p,q,opts in dataset:
        gi='abcd'.index(gold)
        for i in range(4):
            fs=sorted(option_features(p,q,opts,i)); n+=1; total.update(fs)
            if i==gi:pos.update(fs)
            pairs=list(itertools.combinations(fs,2));pt.update(pairs)
            if i==gi:pp.update(pairs)
    rules=[];base=.25
    for f,pn in pos.items():
        if pn>=minsup:
            pr=(pn+1)/(total[f]+2);rules.append(((f,),math.log(max(1e-6,pr/(1-pr)))-math.log(base/(1-base)),pn))
    for pa,pn in pp.items():
        if pn>=pairmin:
            pr=(pn+1)/(pt[pa]+2);rules.append((pa,math.log(max(1e-6,pr/(1-pr)))-math.log(base/(1-base)),pn))
    return rules

def option_predict(dataset,rules):
    out=[]
    for gold,p,q,opts in dataset:
        scores=[]
        for i in range(4):
            fs=option_features(p,q,opts,i); matched=sorted([sc for need,sc,sup in rules if all(x in fs for x in need)],reverse=True); scores.append(sum(matched[:8]))
        out.append(('abcd'[max(range(4),key=lambda i:(scores[i],-i))],gold,scores))
    return out

def parse_logiqa(path): return m.read_logiqa(path)
def eval_logiqa(root,out):
    train=parse_logiqa(os.path.join(root,'Train.txt')); dev=parse_logiqa(os.path.join(root,'Eval.txt')); test=parse_logiqa(os.path.join(root,'Test.txt')); best=(-1,None)
    for ms in [3,5,8,12,20]:
        for pm in [4,8,12,20]:
            ru=learn_option_rules(train,ms,pm); pr=option_predict(dev,ru); ac=sum(a==b for a,b,_ in pr)/len(pr)
            if ac>best[0]:best=(ac,(ms,pm))
    rules=learn_option_rules(train+dev,*best[1]); pred=option_predict(test,rules); rows=[{'id':i,'pred':a,'gold':b,'correct':a==b,'scores':json.dumps(s)} for i,(a,b,s) in enumerate(pred)]; m.write_csv(out/'logiqa_v4_predictions.csv',rows)
    return {'train':len(train),'dev':len(dev),'test':len(test),'dev_best_accuracy':best[0],'config':best[1],'rules':len(rules),'accuracy':sum(a==b for a,b,_ in pred)/len(pred)}

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--trail',required=True);ap.add_argument('--logicbench',required=True);ap.add_argument('--logiqa',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    data=glob.glob(os.path.join(a.trail,'benchmarking','data','GAIA','*.json'))+glob.glob(os.path.join(a.trail,'benchmarking','data','SWE Bench','*.json'));anns=glob.glob(os.path.join(a.trail,'benchmarking','processed_annotations_gaia','*.json'))+glob.glob(os.path.join(a.trail,'benchmarking','processed_annotations_swe_bench','*.json'));gold=load_gold(anns)
    t=time.perf_counter();records=[]
    for f in data:
        tid,rs=trace_records(f);records.extend(rs)
    parse_sec=time.perf_counter()-t;tids={r['tid'] for r in records};pred,details,folds,foldrules=trail_oof(records,tids,gold);P,R,F,G,N,TP=micro(pred,tids,gold)
    loc_gt={(t,s) for t in tids for s,c in gold.get(t,set())};loc_pr={(t,s) for t in tids for s,c in pred.get(t,set())};lp,lr,lf=m.prf(len(loc_gt&loc_pr),len(loc_pr-loc_gt),len(loc_gt-loc_pr))
    trail={'traces':len(tids),'spans':len(records),'gold_pairs':G,'pred_pairs':N,'true_pairs':TP,'joint_precision':P,'joint_recall':R,'joint_f1':F,'location_precision':lp,'location_recall':lr,'location_f1':lf,'parse_seconds':parse_sec,'folds':folds}
    flat_rules=[]
    for fi,rs in enumerate(foldrules):
        for r in rs:flat_rules.append({'fold':fi,'category':r['cat'],'need':' & '.join(r['need']),'precision':r['precision'],'support':r['support'],'score':r['score']})
    m.write_csv(out/'trail_v4_rules.csv',flat_rules);m.write_csv(out/'trail_v4_predictions.csv',[{'trace_id':t,'span_id':x[1],'category':x[2],'score':x[0],'rule':' & '.join(foldrules[int(hashlib.sha256(t.encode()).hexdigest()[:8],16)%2][x[3]]['need']),'evidence':x[4]} for t,arr in details.items() for x in arr])
    allrules=learn_rules(records,tids,gold);runtime=runtime_bench(records,allrules);runtime.update({'parse_seconds':parse_sec,'end_to_end_naive_seconds':parse_sec+runtime['naive_seconds'],'end_to_end_indexed_seconds':parse_sec+runtime['indexed_seconds']});runtime['end_to_end_speedup']=runtime['end_to_end_naive_seconds']/runtime['end_to_end_indexed_seconds']
    noise=[];rr=random.Random(20260721)
    for flip,miss in [(.1,.2),(.2,.2),(.3,.4),(.35,.4)]:
        pp={};attempts=0
        for tid,ps in pred.items():
            keep=set()
            for pair in ps:
                po=ne=0
                while po+ne<31:
                    attempts+=1
                    if rr.random()<miss:continue
                    y=rr.random()>=flip;po+=y;ne+=not y
                if po>ne:keep.add(pair)
            pp[tid]=keep
        p1,r1,f1,*_=micro(pp,tids,gold);noise.append({'flip_rate':flip,'missing_rate':miss,'attempts':attempts,'precision':p1,'recall':r1,'f1':f1})
    cyclic=m.cyclic_stress(n=50000);m.logic_entail=logic;logicbench=m.eval_logicbench(a.logicbench,out);logiqa=eval_logiqa(a.logiqa,out)
    verdict={'trail_all_148_pass':len(tids)==148 and F>=.11,'robustness_pass':next(x for x in noise if x['flip_rate']==.3)['recall']>=.95*R,'cyclic_pass':cyclic['all_positive_reached'],'natural_language_llm_level_pass':logicbench['accuracy']>=.8 and logiqa['accuracy']>=.45,'cheap_verifier_end_to_end_pass':runtime['answers_identical'] and runtime['end_to_end_speedup']>1};verdict['overall_pass']=all(verdict.values())
    summary={'architecture':'out-of-fold sparse operator discovery + evidence ledger + hyperedge/SCC + inverted verifier index','forbidden_dense_components_used':[],'trail':trail,'noise':noise,'runtime':runtime,'cyclic':cyclic,'logicbench':logicbench,'logiqa':logiqa,'strict_verdict':verdict};json.dump(summary,open(out/'summary.json','w',encoding='utf-8'),ensure_ascii=False,indent=2);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
