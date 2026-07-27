from __future__ import annotations
import csv, gc, importlib.util, json, math, os
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('v13',HERE/'real_smollm2_v13_validated_task_arithmetic.py')
v13=importlib.util.module_from_spec(sp); sp.loader.exec_module(v13)
os.environ.setdefault('HF_HUB_DISABLE_XET','1'); os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
torch.set_num_threads(min(4,os.cpu_count() or 1)); torch.manual_seed(20260727); np.random.seed(20260727)
ROOT=Path('out/real_smollm2_v14_fisher_chebyshev'); ROOT.mkdir(parents=True,exist_ok=True)
NAMES=['parent_base','parent_instruct','parent_sql']; IDS=[v13.BASE_ID,v13.INSTRUCT_ID,v13.SQL_ID]

def group(name):
    if name in {'model.embed_tokens.weight','lm_head.weight'}: return 'embedding_head'
    if name.startswith('model.layers.'): return f"layer_{int(name.split('.')[2]):02d}"
    if name.startswith('model.norm.'): return 'final_norm'
    return 'other'

def fisher(model,bank):
    params=list(model.named_parameters()); acc={}; cnt={}
    for n,p in params:
        g=group(n); acc.setdefault(g,0.0); cnt[g]=cnt.get(g,0)+p.numel()
    for k,ids in enumerate(bank):
        model.zero_grad(set_to_none=True)
        loss=model(input_ids=ids[None],labels=ids[None],use_cache=False).loss
        if not torch.isfinite(loss): raise RuntimeError('nonfinite Fisher loss')
        loss.backward()
        for n,p in params:
            if p.grad is not None: acc[group(n)]+=float(p.grad.detach().float().square().sum())
        print('fisher',k,float(loss.detach()))
    out={g:acc[g]/(cnt[g]*len(bank)) for g in acc}
    if any((not math.isfinite(v) or v<=0) for v in out.values()): raise RuntimeError({'bad_fisher':out})
    model.zero_grad(set_to_none=True); return out

def pair_d2(models):
    states=[models[n].state_dict() for n in NAMES]; out={}
    for key in states[0]:
        g=group(key); out.setdefault(g,[0.0,0.0,0.0])
        x=[s[key].detach().cpu().double() for s in states]
        out[g][0]+=float((x[0]-x[1]).square().sum())
        out[g][1]+=float((x[0]-x[2]).square().sum())
        out[g][2]+=float((x[1]-x[2]).square().sum())
    return out

def dual_value(lam,F,D):
    total=0.0
    for g,d in D.items():
        a=lam*np.array([F[n][g] for n in NAMES]); A=float(a.sum())
        if A<=0: return -1e300
        total+=0.5*(a[0]*a[1]*d[0]+a[0]*a[2]*d[1]+a[1]*a[2]*d[2])/A
    return float(total)

def solve_dual(F,D):
    cons=({'type':'eq','fun':lambda x:float(x.sum()-1)},); bounds=[(0,1)]*3
    starts=[np.ones(3)/3,np.array([.8,.1,.1]),np.array([.1,.8,.1]),np.array([.1,.1,.8])]
    sols=[minimize(lambda x:-dual_value(x,F,D),s,method='SLSQP',bounds=bounds,constraints=cons,options={'ftol':1e-14,'maxiter':1000}) for s in starts]
    best=min(sols,key=lambda r:r.fun)
    if not best.success: raise RuntimeError(best.message)
    lam=np.maximum(best.x,0); lam/=lam.sum(); return lam,dual_value(lam,F,D)

def merge(candidate,models,F,lam):
    states=[models[n].state_dict() for n in NAMES]; dst=candidate.state_dict(); coeffs={}; err=0.0
    with torch.no_grad():
        for key,t in dst.items():
            g=group(key); raw=lam*np.array([F[n][g] for n in NAMES]); c=raw/raw.sum()
            coeffs.setdefault(g,{NAMES[i]:float(c[i]) for i in range(3)})
            y=sum(float(c[i])*states[i][key].detach().float() for i in range(3))
            t.copy_(y.to(t.dtype)); err=max(err,float((t.detach().float()-y).abs().max()))
    candidate.tie_weights(); return coeffs,err

def main():
    cfg=[AutoConfig.from_pretrained(x) for x in IDS]; sig=[v13.structural_signature(x) for x in cfg]
    if sig[1:]!=[sig[0],sig[0]]: raise RuntimeError('config mismatch')
    toks=[AutoTokenizer.from_pretrained(x) for x in IDS]; aud=v13.tokenizer_audit(toks)
    if not all(aud['vocab_equal']) or not all(all(x) for x in aud['probe_ids_equal']): raise RuntimeError('tokenizer mismatch')
    tok=toks[1]; tok.pad_token=tok.pad_token or tok.eos_token
    data={'wikitext':v13.build_wikitext(),'instruction':v13.build_instruction(tok),'text2sql':v13.build_text2sql(),'openbookqa':v13.build_openbookqa(tok),'piqa':v13.build_piqa(tok),'boolq':v13.build_boolq(tok)}
    models={NAMES[i]:AutoModelForCausalLM.from_pretrained(IDS[i],torch_dtype=torch.float32,low_cpu_mem_usage=True).eval() for i in range(3)}
    candidate=AutoModelForCausalLM.from_pretrained(IDS[0],torch_dtype=torch.float32,low_cpu_mem_usage=True).eval()
    gen=torch.Generator().manual_seed(42); bank=torch.randint(0,cfg[0].vocab_size,(8,64),generator=gen)
    F={n:fisher(models[n],bank) for n in NAMES}; D=pair_d2(models); lam,dual=solve_dual(F,D); coeffs,err=merge(candidate,models,F,lam)
    all_models=dict(models); all_models['fisher_chebyshev']=candidate
    records={}; summaries={}; timings={}
    for n,m in all_models.items():
        r,t=v13.evaluate_all(m,tok,data); records[n]=r; summaries[n]=v13.summarize(r); timings[n]=t; print(n,json.dumps(summaries[n]))
    pg1=v13.paired_delta(records['parent_instruct']['instruction'],records['parent_base']['instruction'],4101)
    pg2=v13.paired_delta(records['parent_sql']['text2sql'],records['parent_base']['text2sql'],4102)
    parent_pass=pg1['ci95'][1]<0 and pg2['ci95'][1]<0
    comps={n:v13.bootstrap_composite(records['fisher_chebyshev'],records[n],4200+i) for i,n in enumerate(NAMES)}
    domains=list(records['fisher_chebyshev']); best={d:min(NAMES,key=lambda n:summaries[n][d]['loss']) for d in domains}
    virtual={d:records[best[d]][d] for d in domains}; cv=v13.bootstrap_composite(records['fisher_chebyshev'],virtual,4301)
    ratios={d:summaries['fisher_chebyshev'][d]['loss']/summaries[best[d]][d]['loss'] for d in domains}
    acc={d:summaries['fisher_chebyshev'][d]['accuracy']-max(summaries[n][d]['accuracy'] for n in NAMES) for d in ['openbookqa','piqa','boolq']}
    gates={'parent_specialists_valid':parent_pass,'beats_each_parent':all(x['ci95'][1]<0 for x in comps.values()),'beats_virtual_best':cv['ci95'][1]<0,'no_loss_regression_over_2pct':all(x<=1.02 for x in ratios.values()),'no_accuracy_regression_over_2pp':all(x>=-.02 for x in acc.values()),'structure_pass':err==0 and v13.finite_audit(candidate)['all_finite']}
    gates['promoted']=all(gates.values())
    result={'status':'FISHER_CHEBYSHEV_PASS' if gates['promoted'] else 'FISHER_CHEBYSHEV_NOT_PROMOTED','objective':'min_theta max_i 0.5 sum_g F_i,g ||theta_g-theta_i,g||^2','random_fisher':{'samples':8,'sequence_length':64,'seed':42,'external_data':False},'dual_lambdas':{NAMES[i]:float(lam[i]) for i in range(3)},'dual_value':dual,'group_fishers':F,'group_coefficients':coeffs,'formula_error':err,'summaries':summaries,'timings_seconds':timings,'parent_gates':{'instruction':pg1,'sql':pg2},'comparisons_vs_parents':comps,'best_parent_by_domain':best,'comparison_vs_virtual_best':cv,'loss_ratio_vs_best':ratios,'accuracy_delta_vs_best':acc,'gates':gates,'parameter_ratio':sum(p.numel() for p in candidate.parameters())/sum(p.numel() for p in models['parent_base'].parameters()),'checkpoint_created':False}
    (ROOT/'RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    rows=[]
    for n,s in summaries.items():
        for d,v in s.items():
            if isinstance(v,dict): rows.append({'model':n,'domain':d,'n':v['n'],'loss':v['loss'],'accuracy':v.get('accuracy','')})
    with (ROOT/'METRICS.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['model','domain','n','loss','accuracy']); w.writeheader(); w.writerows(rows)
    (ROOT/'REPORT.md').write_text(f"# Fisher Chebyshev Merge\n\nStatus: **{result['status']}**\n\nDual lambdas: `{result['dual_lambdas']}`\n\nGates: `{gates}`\n",encoding='utf-8')
    del candidate; models.clear(); gc.collect(); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
