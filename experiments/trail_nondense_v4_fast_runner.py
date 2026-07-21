#!/usr/bin/env python3
import json, os
from collections import defaultdict
from pathlib import Path

src=Path('experiments/trail_nondense_v4_runner.py').read_text(encoding='utf-8')
prefix=src.rsplit("\nif __name__=='__main__':main()",1)[0]
ns={}
exec(compile(prefix,'trail_nondense_v4_runner.py','exec'),ns)

def option_predict_fast(dataset,rules):
    idx=defaultdict(list)
    for i,(need,sc,sup) in enumerate(rules):
        for f in need: idx[f].append(i)
    out=[]
    for gold,p,q,opts in dataset:
        scores=[]
        for oi in range(4):
            fs=ns['option_features'](p,q,opts,oi); cand=set()
            for f in fs: cand.update(idx.get(f,()))
            matched=sorted((rules[i][1] for i in cand if all(x in fs for x in rules[i][0])),reverse=True)
            scores.append(sum(matched[:8]))
        out.append(('abcd'[max(range(4),key=lambda i:(scores[i],-i))],gold,scores))
    return out

def eval_logiqa_fast(root,out):
    train=ns['parse_logiqa'](os.path.join(root,'Train.txt')); dev=ns['parse_logiqa'](os.path.join(root,'Eval.txt')); test=ns['parse_logiqa'](os.path.join(root,'Test.txt'))
    base=ns['learn_option_rules'](train,3,4); best=(-1,None)
    for ms in [3,5,8,12,20]:
        for pm in [4,8,12,20]:
            ru=[r for r in base if (len(r[0])==1 and r[2]>=ms) or (len(r[0])==2 and r[2]>=pm)]
            pr=option_predict_fast(dev,ru); ac=sum(a==b for a,b,_ in pr)/len(pr)
            if ac>best[0]: best=(ac,(ms,pm))
    full=ns['learn_option_rules'](train+dev,3,4); rules=[r for r in full if (len(r[0])==1 and r[2]>=best[1][0]) or (len(r[0])==2 and r[2]>=best[1][1])]
    pred=option_predict_fast(test,rules); rows=[{'id':i,'pred':a,'gold':b,'correct':a==b,'scores':json.dumps(s)} for i,(a,b,s) in enumerate(pred)]; ns['m'].write_csv(out/'logiqa_v4_predictions.csv',rows)
    return {'train':len(train),'dev':len(dev),'test':len(test),'dev_best_accuracy':best[0],'config':best[1],'rules':len(rules),'accuracy':sum(a==b for a,b,_ in pred)/len(pred)}

ns['option_predict']=option_predict_fast
ns['eval_logiqa']=eval_logiqa_fast
ns['main']()
