#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import product_lifted_agentprocess as core
import product_lifted_agentprocess_v2 as v2

BITS=(core.UNRESOLVED_ERROR,core.NEED_VERIFY,core.PARTIAL_EVIDENCE,core.NEED_AUTH,core.NEED_CONFIRM,core.UNSUPPORTED,core.REPEAT_NO_PROGRESS)
BIT_NAME={core.UNRESOLVED_ERROR:'UNRESOLVED_ERROR',core.NEED_VERIFY:'NEED_VERIFY',core.PARTIAL_EVIDENCE:'PARTIAL_EVIDENCE',core.NEED_AUTH:'NEED_AUTH',core.NEED_CONFIRM:'NEED_CONFIRM',core.UNSUPPORTED:'UNSUPPORTED',core.REPEAT_NO_PROGRESS:'REPEAT_NO_PROGRESS'}
HORIZON=4

def replay(events):
    auth=any('POLICY:AUTH' in e.atoms for e in events);mask=core.NEED_AUTH if auth else 0;out=[];prev=None;repeat=0
    for e in events:
        if e.status=='SUCCESS':mask&=~core.UNRESOLVED_ERROR
        if e.status=='ERROR':mask|=core.UNRESOLVED_ERROR
        if e.status=='PARTIAL':mask|=core.PARTIAL_EVIDENCE
        if e.op in {'READ','VERIFY','AUTH'} and e.status=='SUCCESS':
            mask&=~core.PARTIAL_EVIDENCE
            if e.op=='VERIFY':mask&=~core.NEED_VERIFY
            if e.op=='AUTH':mask&=~core.NEED_AUTH
        if e.op=='MUTATE':
            if 'POLICY:CONFIRM' in e.atoms and 'USER:CONFIRM' not in e.atoms:mask|=core.NEED_CONFIRM
            else:mask&=~core.NEED_CONFIRM
            mask|=core.NEED_VERIFY
        if prev and e.op==prev.op and e.resources==prev.resources and e.status in {'ERROR','NO_RESULT'} and prev.status in {'ERROR','NO_RESULT'}:
            repeat+=1;mask|=core.REPEAT_NO_PROGRESS
        else:repeat=0;mask&=~core.REPEAT_NO_PROGRESS
        if 'POLICY:LISTED' in e.atoms and e.hard>=4:mask|=core.UNSUPPORTED
        out.append((e,mask));prev=e
    return out

def ckey(e,bit):return (e.op,e.status,e.resources[0] if e.resources else 'none',bit)
def learn_contracts(traces,horizon=HORIZON):
    total=Counter();cleared=Counter();offsets=defaultdict(Counter);nextops=defaultdict(Counter);raw_windows=[]
    for tr in traces:
        rs=replay(tr['events'])
        for i,(e,m) in enumerate(rs):
            for bit in BITS:
                if not (m&bit):continue
                k=ckey(e,bit);total[k]+=1;clear=None
                for j in range(i+1,min(len(rs),i+1+horizon)):
                    if not (rs[j][1]&bit):clear=j;break
                raw_windows.append((k,tuple((rs[j][0].op,rs[j][0].status,rs[j][1]&bit) for j in range(i+1,min(len(rs),i+1+horizon)))))
                if clear is not None:
                    cleared[k]+=1;offsets[k][clear-i]+=1;nextops[k][(rs[clear][0].op,rs[clear][0].status)]+=1
    cs={}
    for k,n in total.items():
        cov=cleared[k]/n
        if n>=3 and cov>=.60:
            cs[k]={'support':n,'coverage':cov,'offset':offsets[k].most_common(1)[0][0] if offsets[k] else horizon,'clear_ops':[list(x) for x,_ in nextops[k].most_common(4)],'weight':math.log1p(n)*cov}
    return cs,raw_windows

def contract_instances(events,contracts,horizon=HORIZON):
    rs=replay(events);inst=[]
    for i,(e,m) in enumerate(rs):
        for bit in BITS:
            if not (m&bit):continue
            k=ckey(e,bit);c=contracts.get(k)
            if not c:continue
            clear=None
            for j in range(i+1,min(len(rs),i+1+horizon)):
                if not (rs[j][1]&bit):clear=j;break
            inst.append((c['weight'],i,bit,clear,c))
    return rs,inst

def anomaly(events,contracts):
    _,inst=contract_instances(events,contracts);viol=0.
    for w,i,bit,clear,c in inst:
        if clear is None:viol+=w
    return viol

def targeted_contract_broken(events,source,bit):
    rs=replay(events)
    if source>=len(rs) or not (rs[source][1]&bit):return False
    return all(rs[j][1]&bit for j in range(source+1,min(len(rs),source+1+HORIZON)))

def mutate(events,contracts):
    _,inst=contract_instances(events,contracts)
    sat=sorted((x for x in inst if x[3] is not None),key=lambda x:(-x[0],x[1],x[2]))
    for _,i,bit,j,c in sat:
        arr=list(events);x=arr.pop(j);target=min(len(arr),i+HORIZON+1);arr.insert(target,x)
        if targeted_contract_broken(arr,i,bit):
            return arr,{'source':i,'clearing':j,'moved_to':target,'bit':BIT_NAME[bit],'contract':c}
    return None

def learn_bigram(traces):
    edge=Counter();src=Counter()
    for tr in traces:
        for a,z in zip(tr['events'],tr['events'][1:]):edge[(core.pstate(a),core.pstate(z))]+=1;src[core.pstate(a)]+=1
    return edge,src

def bigram_anomaly(events,model):
    edge,src=model
    if len(events)<2:return 0.
    vals=[]
    for a,z in zip(events,events[1:]):
        s=core.pstate(a);vals.append(-math.log((edge[(s,core.pstate(z))]+1)/(src[s]+30)))
    return sum(vals)/len(vals)

def pair_auc(orig,mut):return sum(1. if b>a else .5 if b==a else 0. for a,b in zip(orig,mut))/len(orig) if orig else .5

def oof_counterfactual(traces):
    ao=[];am=[];bo=[];bm=[];witness=[];fold_rows=[]
    for fold in range(5):
        test=[t for t in traces if core.hashfold(t['tid'])==fold];train=[t for t in traces if core.hashfold(t['tid'])!=fold]
        contracts,_=learn_contracts(train);bg=learn_bigram(train);used=0
        for tr in test:
            z=mutate(tr['events'],contracts)
            if z is None:continue
            broken,why=z;ao.append(anomaly(tr['events'],contracts));am.append(anomaly(broken,contracts));bo.append(bigram_anomaly(tr['events'],bg));bm.append(bigram_anomaly(broken,bg));used+=1
            witness.append({'tid':tr['tid'],'dataset':tr['dataset'],'fold':fold,**why,'original_anomaly':ao[-1],'mutated_anomaly':am[-1],'original_bigram':bo[-1],'mutated_bigram':bm[-1]})
        fold_rows.append({'fold':fold,'train':len(train),'test':len(test),'contracts':len(contracts),'mutations':used})
    return {'n_pairs':len(ao),'auc':pair_auc(ao,am),'bigram_auc':pair_auc(bo,bm),'mean_margin':statistics.mean(b-a for a,b in zip(ao,am)) if ao else 0.,'folds':fold_rows},witness

def storage_runtime(traces):
    contracts,windows=learn_contracts(traces);records=[{'key':list(k),'value':v} for k,v in contracts.items()]
    contract_bytes=len(json.dumps(records,separators=(',',':')).encode());window_bytes=len(json.dumps(windows,separators=(',',':')).encode())
    items=list(contracts.items());all_events=[e for tr in traces for e in tr['events']];events=all_events[:3000];index=contracts
    def indexed():
        s=0.
        for e in events:
            for bit in BITS:
                c=index.get(ckey(e,bit))
                if c:s+=c['weight']
        return s
    def naive():
        s=0.
        for e in events:
            for bit in BITS:
                k=ckey(e,bit)
                for ck,c in items:
                    if ck==k:s+=c['weight'];break
        return s
    a=indexed();b=naive();assert abs(a-b)<1e-9
    t0=time.perf_counter();indexed();ti=time.perf_counter()-t0;t0=time.perf_counter();naive();tn=time.perf_counter()-t0
    return {'contracts':len(contracts),'raw_windows':len(windows),'contract_bytes':contract_bytes,'raw_window_bytes':window_bytes,'storage_ratio':contract_bytes/max(1,window_bytes),'runtime_sample_events':len(events),'indexed_value':a,'naive_value':b,'outputs_identical':True,'indexed_seconds':ti,'naive_seconds':tn,'speedup':tn/max(ti,1e-12),'indexed_lookups':len(all_events)*len(BITS),'naive_worst_comparisons':len(all_events)*len(BITS)*max(1,len(items))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    old=sys.argv;sys.argv=[old[0],'--data',args.data,'--out',args.out];core.counterfactual_auc=lambda traces,m:{'n_pairs':0,'auc':0.,'bigram_auc':0.,'mean_margin':0.};core.main();sys.argv=old
    traces=core.load_all(args.data);cf,witness=oof_counterfactual(traces);sr=storage_runtime(traces)
    summary=json.load(open(out/'summary.json'));summary['counterfactual']=cf;summary['storage_runtime']=sr
    summary['strict']['counterfactual_auc_075']=cf['auc']>=.75;summary['strict']['counterfactual_beats_bigram']=cf['auc']>cf['bigram_auc'];summary['strict']['storage_reduction']=sr['storage_ratio']<1.;summary['strict']['indexed_exact_output']=sr['outputs_identical'];summary['strict']['indexed_operation_reduction']=sr['indexed_lookups']<sr['naive_worst_comparisons'];summary['overall_pass']=all(summary['strict'].values())
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8');(out/'counterfactual_witnesses.json').write_text(json.dumps(witness,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
