#!/usr/bin/env python3
from __future__ import annotations
import math, statistics
from collections import Counter
import product_lifted_agentprocess as b

b.CONFIGS=[]
for lw in (.15,.30,.45):
    for rw in (0.,.15,.30):
        for boundary in ('score','early'):
            b.CONFIGS.append({'hist':2,'future':0,'bad_precision':.35,'support':2,'hard_bad':4,'topk':4,'hard_weight':.55,'transition_weight':.45,'unavoidable_bonus':1.3,'distance_bonus':.7,'local_weight':lw,'route_weight':rw,'boundary_mode':boundary})

def scores_v2(tr,m,kind):
    out=[];base=m['base'];cfg=m['cfg']
    for i,e in enumerate(tr['events']):
        local=sorted((b.loglift(k,m['cn'],m['cp'],base) for k in b.current_keys(tr,e)),reverse=True);local=sum(local[:cfg['topk']])
        route=sum(b.loglift(k,m['rn'],m['rp'],base) for k in b.route_keys(tr,e,cfg['hist'],cfg['future']))
        if kind=='current':sc=local
        elif kind=='ueta':sc=route+.15*e.hard
        elif kind=='graph':
            s=b.node(e);sc=b.loglift(('N',s),m['nn'],m['np'],base)
            if s in m['un']:sc+=cfg['unavoidable_bonus']
            if s in m['dn']:sc+=cfg['distance_bonus']/(1+m['dn'][s])
        elif kind in {'product','product_no_topology'}:
            s=b.pstate(e);sc=sum(b.loglift(k,m['pn'],m['pp'],base) for k in [('P',s),('PM',(e.op,e.mask,min(e.hard,6)))])
            sc+=cfg['hard_weight']*e.hard+cfg['local_weight']*local+cfg['route_weight']*route
            if i:sc+=cfg['transition_weight']*b.loglift(('T',b.pstate(tr['events'][i-1]),s),m['tn'],m['tp'],base)
            if kind=='product':
                if s in m['up']:sc+=cfg['unavoidable_bonus']
                if s in m['dp']:sc+=cfg['distance_bonus']/(1+m['dp'][s])
        else:raise ValueError(kind)
        out.append(sc)
    return out

def predict_v2(tr,m,kind):
    ss=scores_v2(tr,m,kind)
    if not ss:return -1
    if kind=='product' and m['cfg'].get('boundary_mode')=='early':
        ps=[b.pstate(e) for e in tr['events']]
        for i,s in enumerate(ps):
            if s in m['up'] and (i==0 or ps[i-1] not in m['up']):return i
    return max(range(len(ss)),key=lambda i:(ss[i],-i))
b.scores=scores_v2;b.predict=predict_v2

def replay_validity(events,m):
    unresolved=False;need_verify=False;partial=False;repeat=0;penalty=0.;prev=None
    for e in events:
        if e.status=='ERROR':unresolved=True
        if e.status=='PARTIAL':partial=True
        if e.op in {'READ','VERIFY','AUTH'} and e.status=='SUCCESS':
            unresolved=False;partial=False
            if e.op in {'VERIFY','READ'}:need_verify=False
        if e.op=='MUTATE':need_verify=True
        if prev and e.op==prev.op and e.resources==prev.resources and e.status in {'ERROR','NO_RESULT'} and prev.status in {'ERROR','NO_RESULT'}:repeat+=1;penalty+=2+repeat
        else:repeat=0
        if e.op=='FINAL':penalty+=5*unresolved+3*partial+2*need_verify
        if prev:
            a=b.pstate(prev);z=b.pstate(e);out=m['pg'].get(a,{});count=out.get(z,0);tot=sum(out.values());penalty+=.25*(-math.log((count+1)/(tot+max(1,len(out))+1)))
        prev=e
    penalty+=2*unresolved+1.5*partial+1.5*need_verify
    return -penalty

def mutate_break_obligation(events):
    unresolved=False;need_verify=False;partial=False
    for i,e in enumerate(events):
        before=(unresolved,need_verify,partial)
        if e.status=='ERROR':unresolved=True
        if e.status=='PARTIAL':partial=True
        if e.op=='MUTATE':need_verify=True
        if e.op in {'READ','VERIFY','AUTH'} and e.status=='SUCCESS':
            unresolved=False;partial=False
            if e.op in {'READ','VERIFY'}:need_verify=False
        after=(unresolved,need_verify,partial)
        if i>0 and any(before) and sum(after)<sum(before):return events[:i]+events[i+1:]
    for i,e in enumerate(events[:-1]):
        if e.op in {'READ','VERIFY','AUTH'} and e.status=='SUCCESS':return events[:i]+events[i+1:]
    for i in range(len(events)-1):
        if events[i].op!=events[i+1].op:
            x=list(events);x[i],x[i+1]=x[i+1],x[i];return x
    return events[:-1]

def auc_pairs(orig,mut):
    if not orig:return .5
    return sum(1. if a>b else .5 if a==b else 0. for a,b in zip(orig,mut))/len(orig)

def counterfactual_v2(traces,m):
    orig=[];mut=[];big_o=[];big_m=[];edge=Counter();source=Counter()
    for tr in traces:
        for a,z in zip(tr['events'],tr['events'][1:]):edge[(b.node(a),b.node(z))]+=1;source[b.node(a)]+=1
    def bigram_valid(es):
        if len(es)<2:return 0.
        return sum(math.log((edge[(b.node(a),b.node(z))]+1)/(source[b.node(a)]+20)) for a,z in zip(es,es[1:]))/(len(es)-1)
    for tr in traces:
        es=tr['events']
        if len(es)<3:continue
        broken=mutate_break_obligation(es)
        if len(broken)==len(es):continue
        orig.append(replay_validity(es,m));mut.append(replay_validity(broken,m));big_o.append(bigram_valid(es));big_m.append(bigram_valid(broken))
    return {'n_pairs':len(orig),'auc':auc_pairs(orig,mut),'bigram_auc':auc_pairs(big_o,big_m),'mean_margin':statistics.mean(a-b for a,b in zip(orig,mut)) if orig else 0.}
b.counterfactual_auc=counterfactual_v2
if __name__=='__main__':b.main()
