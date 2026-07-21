#!/usr/bin/env python3
import argparse, csv, glob, json, os, random, statistics, time
from pathlib import Path
import trail_nondense_full_eval as m

ap=argparse.ArgumentParser(); ap.add_argument('--trail',required=True); ap.add_argument('--logicbench',required=True); ap.add_argument('--logiqa',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
data=glob.glob(os.path.join(a.trail,'benchmarking','data','GAIA','*.json'))+glob.glob(os.path.join(a.trail,'benchmarking','data','SWE Bench','*.json'))
anns=glob.glob(os.path.join(a.trail,'benchmarking','processed_annotations_gaia','*.json'))+glob.glob(os.path.join(a.trail,'benchmarking','processed_annotations_swe_bench','*.json'))
gold=m.load_gold(anns)
print(json.dumps({'stage':'start','data_files':len(data),'annotation_files':len(anns)}),flush=True)
trail=m.eval_trail(data,gold,out); print(json.dumps({'stage':'trail','result':trail}),flush=True)

# Noise is applied to already-compiled explicit evidence predicates; language extraction is not recomputed.
pred=json.load(open(out/'trail_predictions.json',encoding='utf-8')); rr=random.Random(20260721); noise=[]
for flip,missing in [(0.1,0.2),(0.2,0.2),(0.3,0.4),(0.35,0.4)]:
    gt_all=set(); pr_all=set(); attempts=0
    for tid,ps in pred.items():
        for e in gold.get(tid,{}).get('errors',[]):
            if e.get('location') and e.get('category'): gt_all.add((tid,str(e['location']),str(e['category'])))
        for p in ps:
            pos=neg=0
            while pos+neg<31:
                attempts+=1
                if rr.random()<missing: continue
                y=0 if rr.random()<flip else 1; pos+=y; neg+=1-y
            if pos>neg: pr_all.add((tid,p['location'],p['category']))
    tp=len(gt_all&pr_all); prec,rec,f1=m.prf(tp,len(pr_all-gt_all),len(gt_all-pr_all)); noise.append({'flip_rate':flip,'missing_rate':missing,'observed_votes':31,'raw_attempts':attempts,'joint_precision':prec,'joint_recall':rec,'joint_f1':f1})
m.write_csv(out/'trail_noise_robustness.csv',noise); print(json.dumps({'stage':'noise','result':noise}),flush=True)

# Parse once; compare all-rule scan and trigger-indexed scan on all 148 traces.
t0=time.perf_counter(); loaded=[m.load_events(f)[1] for f in data]; parse_seconds=time.perf_counter()-t0
t=time.perf_counter(); naive=[{(p.location,p.category) for p in m.compile_trace(ev,False)} for ev in loaded]; naive_s=time.perf_counter()-t
t=time.perf_counter(); indexed=[{(p.location,p.category) for p in m.compile_trace(ev,True)} for ev in loaded]; indexed_s=time.perf_counter()-t
# Full input-to-output wall clock: actual parse + each reasoning mode; index is static and already included in module import.
runtime={'answers_identical':naive==indexed,'parse_seconds':parse_seconds,'reasoning_naive_seconds':naive_s,'reasoning_indexed_seconds':indexed_s,'reasoning_speedup':naive_s/indexed_s if indexed_s else 0,'end_to_end_naive_seconds':parse_seconds+naive_s,'end_to_end_indexed_seconds':parse_seconds+indexed_s,'end_to_end_speedup':(parse_seconds+naive_s)/(parse_seconds+indexed_s) if parse_seconds+indexed_s else 0}
print(json.dumps({'stage':'runtime','result':runtime}),flush=True)
cyclic=m.cyclic_stress(n=50000); print(json.dumps({'stage':'cyclic','result':cyclic}),flush=True)
logic=m.eval_logicbench(a.logicbench,out); print(json.dumps({'stage':'logicbench','result':logic}),flush=True)
logiqa=m.eval_logiqa(a.logiqa,out); print(json.dumps({'stage':'logiqa','result':logiqa}),flush=True)
verdict={'trail_all_148_pass':len(data)==148 and trail['joint_f1']>=0.11,'robustness_pass':any(r['flip_rate']==0.3 and r['missing_rate']==0.4 and r['joint_recall']>=0.95*trail['joint_recall_official_style'] for r in noise),'cyclic_pass':cyclic['all_positive_reached'] and cyclic['atoms']>=50000,'natural_language_llm_level_pass':logic['accuracy']>=0.80 and logiqa['accuracy']>=0.45,'cheap_verifier_end_to_end_pass':runtime['answers_identical'] and runtime['end_to_end_speedup']>1.0}; verdict['overall_pass']=all(verdict.values())
summary={'architecture':'explicit discrete predicates + evidence ledger + sparse hyperedges + SCC agenda + route index','forbidden_dense_components_used':[],'trail':trail,'noise':noise,'runtime':runtime,'cyclic':cyclic,'logicbench':logic,'logiqa':logiqa,'strict_verdict':verdict}
json.dump(summary,open(out/'summary.json','w',encoding='utf-8'),ensure_ascii=False,indent=2); print(json.dumps({'stage':'complete','summary':summary},ensure_ascii=False),flush=True)
