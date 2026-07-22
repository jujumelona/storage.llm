#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,glob,hashlib,json,math,random,statistics
from collections import Counter,defaultdict
from pathlib import Path
import trail_nondense_v4_runner as b
import ueta_trail_eval as u
SEED=20260722
random.seed(SEED)

def micro(pred,tids,gold): return u.micro(pred,tids,gold)
def group(records): return u.group(records)
def select(scores,tids,k,threshold): return u.select(scores,tids,k,threshold)
def calibrate(scores,tids,gold): return u.cal(scores,tids,gold)

def strip_records(records,mode):
    if mode=='full': return records
    banned=('L:','B:','T:','N:','PN:') if mode=='no_identity' else ('L:','B:','T:','N:','PN:','P:')
    out=[]
    for r in records:
        q=dict(r);q['features']=frozenset(x for x in r['features'] if not x.startswith(banned));out.append(q)
    return out

def route_keys(seq,i,depth,horizon):
    hs,guards,obs=u.context(seq,i,depth=depth,horizon=horizon)
    return [(h,g,o) for h in hs for g in guards for o in obs]

def discover_routes(records,tids,depth,horizon,min_support,max_routes=30000):
    cnt=Counter()
    for seq in group([r for r in records if r['tid'] in tids]).values():
        for i in range(len(seq)): cnt.update(set(route_keys(seq,i,depth,horizon)))
    rows=[(k,n) for k,n in cnt.items() if n>=min_support]
    rows.sort(key=lambda x:(-x[1],-len(x[0][0]),str(x[0])))
    return dict(rows[:max_routes])

def map_routes(records,tids,gold,routes,depth,horizon,min_positive=2,min_precision=.08):
    total_events=0;cat_total=Counter();route_total=Counter();route_pos=defaultdict(Counter);route_set=set(routes)
    for tid,seq in group([r for r in records if r['tid'] in tids]).items():
        gb=defaultdict(set)
        for sid,cat in gold.get(tid,set()): gb[sid].add(cat)
        for i,r in enumerate(seq):
            total_events+=1;cats=gb.get(r['sid'],set());cat_total.update(cats)
            matched=set(route_keys(seq,i,depth,horizon))&route_set;route_total.update(matched)
            for cat in cats: route_pos[cat].update(matched)
    contracts=[]
    for cat,positives in route_pos.items():
        base=(cat_total[cat]+1)/(total_events+2)
        for route,pn in positives.items():
            if pn<min_positive: continue
            n=route_total[route];precision=(pn+1)/(n+2);lift=math.log(max(1e-12,precision/base))
            if precision<min_precision or lift<0: continue
            h,g,o=route;score=lift+.20*math.log1p(pn)+.08*len(h)+.04*math.log1p(routes[route])
            contracts.append({'cat':cat,'history':h,'guard':g,'obligation':o,'score':score,'support':pn,'route_support':routes[route],'precision':precision})
    contracts.sort(key=lambda x:(-x['score'],-x['support'],x['cat']))
    kept=[];per=Counter()
    for c in contracts:
        a=(c['cat'],c['history'][-1],c['guard'])
        if per[a]>=20: continue
        per[a]+=1;kept.append(c)
        if len(kept)>=40000: break
    return kept

def score(records,tids,contracts,depth,horizon):
    idx=defaultdict(list)
    for j,c in enumerate(contracts): idx[(c['history'],c['guard'],c['obligation'])].append(j)
    out=defaultdict(list)
    for tid,seq in group([r for r in records if r['tid'] in tids]).items():
        for i,r in enumerate(seq):
            best={}
            for key in route_keys(seq,i,depth,horizon):
                for j in idx.get(key,()):
                    c=contracts[j];old=best.get(c['cat'])
                    if old is None or c['score']>old[0]: best[c['cat']]=(c['score'],j)
            for cat,(sc,j) in best.items(): out[tid].append((sc,r['sid'],cat,j,r.get('evidence','')))
    return out

def fit_factorized(records,core,val,gold,horizon):
    best=(-1,None)
    for depth in (1,2,3):
      for minsup in (2,3,5):
        routes=discover_routes(records,core,depth,horizon,minsup)
        for minprec in (.08,.12,.18,.26):
            cs=map_routes(records,core,gold,routes,depth,horizon,2,minprec)
            if not cs: continue
            cal=calibrate(score(records,val,cs,depth,horizon),val,gold)
            if cal['f1']>best[0]: best=(cal['f1'],{'depth':depth,'min_support':minsup,'min_precision':minprec,'calibration':cal,'routes':len(routes),'contracts':len(cs)})
    if best[1] is None: best=(0,{'depth':1,'min_support':3,'min_precision':.12,'calibration':{'k':5,'threshold':.5,'f1':0},'routes':0,'contracts':0})
    return best

def train_final(records,train,gold,horizon,cfg):
    routes=discover_routes(records,train,cfg['depth'],horizon,cfg['min_support'])
    cs=map_routes(records,train,gold,routes,cfg['depth'],horizon,2,cfg['min_precision'])
    return routes,cs

def baseline_oof(records,tids,gold):
    pred={}
    for fold in (0,1):
        core,val,test,train=u.split(tids,fold)
        r0=b.learn_rules(records,core,gold);cal=calibrate(b.score_records(records,val,r0,True),val,gold)
        rules=b.learn_rules(records,train,gold);s=b.score_records(records,test,rules,True)
        pred.update(select(s,test,cal['k'],cal['threshold']))
    return pred,micro(pred,tids,gold)

def oof_latency(records,tids,gold,horizons=(0,1,2,3,4)):
    results=[];predictions={}
    for horizon in horizons:
        pred={};folds=[]
        for fold in (0,1):
            core,val,test,train=u.split(tids,fold);_,cfg=fit_factorized(records,core,val,gold,horizon)
            routes,cs=train_final(records,train,gold,horizon,cfg);s=score(records,test,cs,cfg['depth'],horizon)
            pred.update(select(s,test,cfg['calibration']['k'],cfg['calibration']['threshold']))
            folds.append({'horizon':horizon,'fold':fold,'core':len(core),'val':len(val),'test':len(test),'depth':cfg['depth'],'min_support':cfg['min_support'],'min_precision':cfg['min_precision'],'routes':len(routes),'contracts':len(cs),'val_f1':cfg['calibration']['f1'],'k':cfg['calibration']['k'],'threshold':cfg['calibration']['threshold']})
        results.append({'horizon':horizon,**micro(pred,tids,gold),'folds':folds});predictions[horizon]=pred
    return results,predictions

def hash_val(tids): return {t for t in tids if int(hashlib.sha256(('cross-val-'+t).encode()).hexdigest()[:8],16)%5==0}

def train_test_once(records,train,test,gold,horizon=2):
    val=hash_val(train);core=set(train)-val
    if len(val)<2:
        a=sorted(train);val=set(a[:max(1,len(a)//5)]);core=set(train)-val
    _,cfg=fit_factorized(records,core,val,gold,horizon);_,cs=train_final(records,train,gold,horizon,cfg)
    p=select(score(records,test,cs,cfg['depth'],horizon),test,cfg['calibration']['k'],cfg['calibration']['threshold'])
    r0=b.learn_rules(records,core,gold);bc=calibrate(b.score_records(records,val,r0,True),val,gold)
    br=b.learn_rules(records,train,gold);bp=select(b.score_records(records,test,br,True),test,bc['k'],bc['threshold'])
    bm=micro(bp,test,gold);um=micro(p,test,gold)
    return {'baseline':bm,'ueta':um,'gain':um['f1']-bm['f1'],'cfg':cfg,'contracts':len(cs)}

def dominant_tool(records):
    out={}
    for tid,seq in group(records).items():
        c=Counter()
        for r in seq:
            for f in r['features']:
                if f.startswith('T:') and f not in {'T:','T:<empty>','T:none'}:
                    raw=f[2:].lower();fam=raw.split('.')[0].split('/')[0].split(':')[0]
                    if fam:c[fam]+=1
        out[tid]=c.most_common(1)[0][0] if c else 'none'
    return out

def cross_domain(records,tids,gold,domain,horizon=2):
    rows=[]
    for held in sorted(set(domain.values())):
        test={t for t in tids if domain[t]==held};train=set(tids)-test
        if len(test)>=5 and len(train)>=10: rows.append({'held_out_domain':held,'train':len(train),'test':len(test),**train_test_once(records,train,test,gold,horizon)})
    return rows

def leave_tool_out(records,tids,gold,horizon=2,min_traces=7):
    clean=strip_records(records,'structure_only');fam=dominant_tool(records);cnt=Counter(fam.values());rows=[]
    for held,n in cnt.most_common():
        if held=='none' or n<min_traces: continue
        test={t for t in tids if fam[t]==held};train=set(tids)-test
        if len(train)>=20: rows.append({'held_out_tool_family':held,'train':len(train),'test':len(test),**train_test_once(clean,train,test,gold,horizon)})
    return rows

def bootstrap(base,ueta,tids,gold,n=1000):
    rng=random.Random(SEED);ts=sorted(tids);gains=[]
    for _ in range(n):
        sample=[rng.choice(ts) for _ in ts];B={};U={};G={};ids=set()
        for j,t in enumerate(sample):
            z=f'{j}:{t}';ids.add(z);B[z]=base.get(t,set());U[z]=ueta.get(t,set());G[z]=gold.get(t,set())
        gains.append(micro(U,ids,G)['f1']-micro(B,ids,G)['f1'])
    gains.sort();return {'mean':statistics.mean(gains),'ci95_low':gains[int(.025*n)],'ci95_high':gains[int(.975*n)]}

def writecsv(path,rows):
    if not rows: Path(path).write_text('',encoding='utf-8');return
    flat=[]
    for r in rows:
        flat.append({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    keys=list(dict.fromkeys(k for r in flat for k in r))
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trail',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    paths=[(p,'GAIA') for p in glob.glob(a.trail+'/benchmarking/data/GAIA/*.json')]+[(p,'SWE_Bench') for p in glob.glob(a.trail+'/benchmarking/data/SWE Bench/*.json')]
    anns=glob.glob(a.trail+'/benchmarking/processed_annotations_gaia/*.json')+glob.glob(a.trail+'/benchmarking/processed_annotations_swe_bench/*.json');gold=b.load_gold(anns)
    records=[];tids=set();domain={}
    for path,d in paths:
        tid,rs=b.trace_records(path);tids.add(tid);records.extend(rs);domain[tid]=d
    print('loaded',len(tids),len(records),Counter(domain.values()),flush=True)
    bp,bm=baseline_oof(records,tids,gold);lat,preds=oof_latency(records,tids,gold);best=max(lat,key=lambda x:x['f1']);h=best['horizon'];boot=bootstrap(bp,preds[h],tids,gold)
    cross=cross_domain(strip_records(records,'no_identity'),tids,gold,domain,min(2,h));tool=leave_tool_out(records,tids,gold,min(2,h))
    cross_avg=statistics.mean([x['gain'] for x in cross]) if cross else -1;tool_avg=statistics.mean([x['gain'] for x in tool]) if tool else -1
    h2=next((x for x in lat if x['horizon']==2),None);h0=next((x for x in lat if x['horizon']==0),None)
    verdict={'factorized_route_discovery_beats_current_state':best['f1']>bm['f1']+.01,'bootstrap_ci_positive':boot['ci95_low']>0,'bounded_delay_h2_retains_advantage':bool(h2 and h2['f1']>bm['f1']+.005),'strictly_online_h0_measured':h0 is not None,'cross_domain_average_gain_positive':cross_avg>0,'leave_tool_family_average_gain_positive':tool_avg>0}
    verdict['general_architecture_supported']=all(verdict[k] for k in ('factorized_route_discovery_beats_current_state','bootstrap_ci_positive','bounded_delay_h2_retains_advantage','cross_domain_average_gain_positive','leave_tool_family_average_gain_positive'))
    summary={'architecture':'C-UETA factorized label-free route discovery plus delayed obligation verification','forbidden_dense_components_used':[],'traces':len(tids),'records':len(records),'baseline_oof':bm,'latency_curve':lat,'best_horizon':h,'best_factorized_ueta':best,'bootstrap_best_vs_baseline':boot,'cross_domain_no_identity':cross,'cross_domain_average_gain':cross_avg,'leave_tool_family_structure_only':tool,'leave_tool_family_average_gain':tool_avg,'strict_verdict':verdict,'scope_note':'Route structures are discovered without labels. Category attachment remains supervised on training traces. Horizon h means verdict is emitted after h subsequent events; h=0 is strictly online.'}
    json.dump(summary,open(out/'summary.json','w',encoding='utf-8'),indent=2);writecsv(out/'latency_curve.csv',[{k:v for k,v in x.items() if k!='folds'} for x in lat]);writecsv(out/'latency_folds.csv',[f for x in lat for f in x['folds']]);writecsv(out/'cross_domain.csv',cross);writecsv(out/'leave_tool_family.csv',tool);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
