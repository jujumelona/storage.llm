#!/usr/bin/env python3
import argparse,gzip,json
from pathlib import Path
import cueta_traceelephant_eval as e
import ueta_causal_generalization as cu
def freeze(x):return tuple(freeze(v) for v in x) if isinstance(x,list) else x
def main():
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--trace-root',required=True);a.add_argument('--output',required=True);x=a.parse_args()
 with gzip.open(x.checkpoint,'rt') as f:c=json.load(f)
 cs=[]
 for z in c['contracts']:
  q=dict(z);q['history']=freeze(q['history']);q['obligation']=freeze(q['obligation']);cs.append(q)
 ts,skip=e.load(Path(x.trace_root));r=e.flatten_traces(ts);ids={t['tid'] for t in ts};scores=cu.score(r,ids,cs,c['config']['depth'],c['horizon']);pred=e.top1(scores,ts);m,rows=e.metrics(pred,ts);Path(x.output).write_text('\n'.join(json.dumps(z) for z in rows)+'\n');print(json.dumps({'metrics':m,'traces':len(ts),'skipped':skip},indent=2))
if __name__=='__main__':main()
