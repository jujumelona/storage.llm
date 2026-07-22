#!/usr/bin/env python3
from __future__ import annotations
import argparse,glob,hashlib,json,math,os
from collections import Counter,defaultdict
from pathlib import Path
import trail_nondense_v4_runner as b
import ueta_trail_eval as u

def transform(records,mode):
    drops={
      'no_primary':('P:',),
      'no_identity':('L:','N:','PN:','T:'),
      'structure_only':('P:','L:','N:','PN:','T:'),
    }[mode]
    out=[]
    for r in records:
      q=dict(r);q['features']=frozenset(x for x in r['features'] if not x.startswith(drops));out.append(q)
    return out

def evaluate_full(records,tids,gold):
    bp,up,cp,folds,saved=u.oof(records,tids,gold)
    return {'baseline':u.micro(bp,tids,gold),'ueta':u.micro(up,tids,gold),'union':u.micro(cp,tids,gold),'folds':folds,
            'ueta_minus_baseline':u.micro(up,tids,gold)['f1']-u.micro(bp,tids,gold)['f1']}

def learn_history(records,tids,gold):
    gs=u.group([r for r in records if r['tid'] in tids]);tot=Counter();pos=defaultdict(Counter);cats=Counter();n=0
    for tid,seq in gs.items():
      gb=defaultdict(set)
      for s,c in gold.get(tid,set()):gb[s].add(c)
      for i,r in enumerate(seq):
        n+=1;cs=gb.get(r['sid'],set());cats.update(cs);hs,guards,_=u.context(seq,i)
        for h in hs:
          for gd in guards:
            k=(h,gd);tot[k]+=1
            for c in cs:pos[c][k]+=1
    out=[]
    for c,pc in pos.items():
      base=(cats[c]+1)/(n+2)
      for (h,gd),pn in pc.items():
        if pn<2:continue
        prec=(pn+1)/(tot[(h,gd)]+2);sc=math.log(max(1e-9,prec/base))+.22*math.log1p(pn)+.08*len(h)
        if prec>=.08 and sc>0:out.append({'cat':c,'history':h,'guard':gd,'score':sc,'support':pn,'precision':prec})
    return sorted(out,key=lambda x:(-x['score'],-x['support']))[:40000]

def hscore(records,tids,rules):
    idx=defaultdict(list)
    for j,c in enumerate(rules):idx[(c['history'],c['guard'])].append(j)
    out=defaultdict(list)
    for tid,seq in u.group([r for r in records if r['tid'] in tids]).items():
      for i,r in enumerate(seq):
        hs,guards,_=u.context(seq,i);best={}
        for h in hs:
          for gd in guards:
            for j in idx.get((h,gd),()):
              c=rules[j]
              if c['cat'] not in best or c['score']>best[c['cat']][0]:best[c['cat']]=(c['score'],j)
        for cat,(sc,j) in best.items():out[tid].append((sc,r['sid'],cat,j,''))
    return out

def component_oof(records,tids,gold,mode):
    pred={};folds=[]
    for fold in (0,1):
      core,val,test,train=u.split(tids,fold)
      if mode=='history_only':
        r0=learn_history(records,core,gold);sv=hscore(records,val,r0);cal=u.cal(sv,val,gold);rules=learn_history(records,train,gold);st=hscore(records,test,rules)
      else:
        c0=u.learn(records,core,gold)
        if mode=='future_only':c0=[x for x in c0 if len(x['history'])==1]
        elif mode=='orderless_future':c0=[x for x in c0 if x['obligation'][0] in {'EV','END'}]
        elif mode=='ordered_future':c0=[x for x in c0 if x['obligation'][0].startswith('AT') or x['obligation'][0]=='SEQ']
        sv=u.score(records,val,c0);cal=u.cal(sv,val,gold);rules=u.learn(records,train,gold)
        if mode=='future_only':rules=[x for x in rules if len(x['history'])==1]
        elif mode=='orderless_future':rules=[x for x in rules if x['obligation'][0] in {'EV','END'}]
        elif mode=='ordered_future':rules=[x for x in rules if x['obligation'][0].startswith('AT') or x['obligation'][0]=='SEQ']
        st=u.score(records,test,rules)
      pred.update(u.select(st,test,cal['k'],cal['threshold']));folds.append({'fold':fold,'rules':len(rules),'val_f1':cal['f1'],'k':cal['k'],'threshold':cal['threshold']})
    return {'metrics':u.micro(pred,tids,gold),'folds':folds}

def build_trie(paths):
    trans=[];edge={};terminal={};next_state=1
    for path in paths:
      state=0
      for tok in path:
        k=(state,tok)
        if k not in edge:
          edge[k]=next_state;trans.append((state,tok,next_state));next_state+=1
        state=edge[k]
      terminal[tuple(path)]=state
    return trans,terminal,next_state

def functional_storage(records,gold,saved):
    explicit=[];contracts=[]
    for x in saved:
      for tid,seq in u.group([r for r in records if r['tid'] in x['train']]).items():
        gb=defaultdict(set)
        for sid,c in gold.get(tid,set()):gb[sid].add(c)
        for i,r in enumerate(seq):
          for c in gb.get(r['sid'],()):
            hs,gs,obs=u.context(seq,i);explicit.append({'cat':c,'history':hs,'guards':gs,'future':obs})
      contracts.extend(x['cs'])
    ht,hm,hn=build_trie([c['history'] for c in contracts]);ft,fm,fn=build_trie([c['obligation'] for c in contracts])
    endpoints=[(hm[tuple(c['history'])],fm[tuple(c['obligation'])],c['cat'],c['guard'],round(c['score'],5),c['support']) for c in contracts]
    machine={'history_transitions':ht,'future_transitions':ft,'contract_endpoints':endpoints}
    eb=len(json.dumps(explicit,separators=(',',':')).encode());mb=len(json.dumps(machine,separators=(',',':')).encode())
    return {'explicit_bytes':eb,'functional_automaton_bytes':mb,'ratio':mb/eb if eb else 0,'contracts':len(contracts),'history_states':hn,'future_states':fn,'transitions':len(ht)+len(ft),'endpoints':len(endpoints)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--trail',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    data=glob.glob(a.trail+'/benchmarking/data/GAIA/*.json')+glob.glob(a.trail+'/benchmarking/data/SWE Bench/*.json');anns=glob.glob(a.trail+'/benchmarking/processed_annotations_gaia/*.json')+glob.glob(a.trail+'/benchmarking/processed_annotations_swe_bench/*.json');gold=b.load_gold(anns);records=[];tids=set()
    for z in data:tid,rs=b.trace_records(z);tids.add(tid);records.extend(rs)
    base_bp,base_up,base_cp,base_folds,saved=u.oof(records,tids,gold)
    results={'full':{'baseline':u.micro(base_bp,tids,gold),'ueta':u.micro(base_up,tids,gold)},'feature_ablations':{},'component_ablations':{}}
    for mode in ('no_primary','no_identity','structure_only'):
      print('feature ablation',mode,flush=True);results['feature_ablations'][mode]=evaluate_full(transform(records,mode),tids,gold)
    for mode in ('history_only','future_only','orderless_future','ordered_future'):
      print('component ablation',mode,flush=True);results['component_ablations'][mode]=component_oof(records,tids,gold,mode)
    results['functional_storage']=functional_storage(records,gold,saved)
    f=results['full']['ueta']['f1'];results['claims']={
      'future_obligations_add_value':f>results['component_ablations']['history_only']['metrics']['f1']+.005,
      'ordered_future_adds_value':f>results['component_ablations']['orderless_future']['metrics']['f1']+.005,
      'not_dependent_on_primary_category_feature':results['feature_ablations']['no_primary']['ueta']['f1']>results['feature_ablations']['no_primary']['baseline']['f1']+.005,
      'structure_only_still_improves':results['feature_ablations']['structure_only']['ueta']['f1']>results['feature_ablations']['structure_only']['baseline']['f1']+.005,
      'functional_storage_compresses':results['functional_storage']['ratio']<1,
    }
    json.dump(results,open(out/'ablation_summary.json','w'),indent=2);print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':main()
