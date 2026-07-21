#!/usr/bin/env python3
import argparse, glob, json, os
from pathlib import Path
import trail_nondense_full_eval as m

ap=argparse.ArgumentParser()
ap.add_argument('--trail',required=True)
ap.add_argument('--logicbench',required=True)
ap.add_argument('--logiqa',required=True)
ap.add_argument('--out',required=True)
a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
data=glob.glob(os.path.join(a.trail,'benchmarking','data','GAIA','*.json'))+glob.glob(os.path.join(a.trail,'benchmarking','data','SWE Bench','*.json'))
anns=glob.glob(os.path.join(a.trail,'benchmarking','processed_annotations_gaia','*.json'))+glob.glob(os.path.join(a.trail,'benchmarking','processed_annotations_swe_bench','*.json'))
gold=m.load_gold(anns)
print(json.dumps({'stage':'start','data_files':len(data),'annotation_files':len(anns)}),flush=True)
trail=m.eval_trail(data,gold,out); print(json.dumps({'stage':'trail','result':trail}),flush=True)
noise=m.eval_noise(data,gold,out); print(json.dumps({'stage':'noise','result':noise}),flush=True)
runtime=m.benchmark_runtime(data,rounds=1); print(json.dumps({'stage':'runtime','result':runtime}),flush=True)
cyclic=m.cyclic_stress(n=50000); print(json.dumps({'stage':'cyclic','result':cyclic}),flush=True)
logic=m.eval_logicbench(a.logicbench,out); print(json.dumps({'stage':'logicbench','result':logic}),flush=True)
logiqa=m.eval_logiqa(a.logiqa,out); print(json.dumps({'stage':'logiqa','result':logiqa}),flush=True)
verdict={
 'trail_all_148_pass':len(data)==148 and trail['joint_f1']>=0.11,
 'robustness_pass':any(r['flip_rate']==0.3 and r['missing_rate']==0.4 and r['joint_recall']>=0.95*trail['joint_recall_official_style'] for r in noise),
 'cyclic_pass':cyclic['all_positive_reached'] and cyclic['atoms']>=50000,
 'natural_language_llm_level_pass':logic['accuracy']>=0.80 and logiqa['accuracy']>=0.45,
 'cheap_verifier_end_to_end_pass':runtime['answers_identical'] and runtime['end_to_end_speedup']>1.0,
}
verdict['overall_pass']=all(verdict.values())
summary={'architecture':'explicit discrete predicates + evidence ledger + sparse hyperedges + SCC agenda + route index','forbidden_dense_components_used':[],'trail':trail,'noise':noise,'runtime':runtime,'cyclic':cyclic,'logicbench':logic,'logiqa':logiqa,'strict_verdict':verdict}
json.dump(summary,open(out/'summary.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print(json.dumps({'stage':'complete','summary':summary},ensure_ascii=False),flush=True)
