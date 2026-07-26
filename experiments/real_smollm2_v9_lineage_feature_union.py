from __future__ import annotations
import copy,gc,importlib.util,json,math,os,shutil,time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer

HERE=Path(__file__).resolve().parent
def mod(name,file):
 s=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v8=mod('v8','real_smollm2_v8_proper_choice_loss.py');v1=v8.v1;v5=v8.v5;v7=v8.v7
os.environ.setdefault('HF_HUB_DISABLE_XET','1');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
torch.set_num_threads(min(4,os.cpu_count() or 1));torch.manual_seed(20260726);np.random.seed(20260726)
ROOT=Path('out/real_smollm2_v9_lineage_feature_union');ROOT.mkdir(parents=True,exist_ok=True)
BASE=v1.BASE_ID;SPEC=v1.INSTRUCT_ID;N=96;v8.BOOTSTRAPS=5000;v8.BATCH_SEQUENCES=16;v1.WIKI_BLOCKS=24;v1.BLOCK_SIZE=256

def beta_rule(a,b):
 par=orth=de=be=0.
 sa,sb=a.state_dict(),b.state_dict()
 for k,x0 in sa.items():
  if not torch.is_floating_point(x0):continue
  x=x0.detach().cpu().double().reshape(-1);d=(sb[k].detach().cpu().double()-x0.detach().cpu().double()).reshape(-1)
  xx=float(x@x);dd=float(d@d);be+=xx;de+=dd
  if xx>1e-30:
   ps=(float(x@d)**2)/xx;par+=ps;orth+=max(0.,dd-ps)
  else:orth+=dd
 pn,on=math.sqrt(par),math.sqrt(orth)
 return on/max(pn+on,1e-30),{'parallel_delta_norm':pn,'orthogonal_delta_norm':on,'relative_delta_norm':math.sqrt(de/max(be,1e-30)),'orthogonal_delta_energy_fraction':orth/max(par+orth,1e-30),'formula':'||Delta_perp||/(||Delta_parallel||+||Delta_perp||)'}

def interpolation(cfg,a,b,beta):
 m=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32);sa,sb=a.state_dict(),b.state_dict();st={}
 for k,x in m.state_dict().items():
  if k not in sa or sa[k].shape!=x.shape:raise RuntimeError({'bad_key':k})
  st[k]=((1-beta)*sa[k].float()+beta*sb[k].float()).to(x.dtype) if torch.is_floating_point(x) else sa[k]
 m.load_state_dict(st,strict=True);return m.eval()

def union(cfg,a,b,beta):
 c=copy.deepcopy(cfg);hd=int(getattr(cfg,'head_dim',cfg.hidden_size//cfg.num_attention_heads));c.num_attention_heads*=2;c.num_key_value_heads*=2;c.head_dim=hd;c.intermediate_size*=2;c.tie_word_embeddings=True;c._attn_implementation='sdpa'
 m=AutoModelForCausalLM.from_config(c,torch_dtype=torch.float32)
 with torch.no_grad():
  m.model.embed_tokens.weight.copy_((1-beta)*a.model.embed_tokens.weight+beta*b.model.embed_tokens.weight)
  m.model.norm.weight.copy_((1-beta)*a.model.norm.weight+beta*b.model.norm.weight)
  for d,x,y in zip(m.model.layers,a.model.layers,b.model.layers):
   for z in [x.self_attn.q_proj,x.self_attn.k_proj,x.self_attn.v_proj,x.self_attn.o_proj,x.mlp.gate_proj,x.mlp.up_proj,x.mlp.down_proj,y.self_attn.q_proj,y.self_attn.k_proj,y.self_attn.v_proj,y.self_attn.o_proj,y.mlp.gate_proj,y.mlp.up_proj,y.mlp.down_proj]:
    if z.bias is not None:raise RuntimeError('V9 expects bias-free SmolLM2 projections')
   d.input_layernorm.weight.fill_(1);d.post_attention_layernorm.weight.fill_(1)
   ga,gb=x.input_layernorm.weight,y.input_layernorm.weight
   d.self_attn.q_proj.weight.copy_(torch.cat([x.self_attn.q_proj.weight*ga[None,:],y.self_attn.q_proj.weight*gb[None,:]],0))
   d.self_attn.k_proj.weight.copy_(torch.cat([x.self_attn.k_proj.weight*ga[None,:],y.self_attn.k_proj.weight*gb[None,:]],0))
   d.self_attn.v_proj.weight.copy_(torch.cat([x.self_attn.v_proj.weight*ga[None,:],y.self_attn.v_proj.weight*gb[None,:]],0))
   d.self_attn.o_proj.weight.copy_(torch.cat([(1-beta)*x.self_attn.o_proj.weight,beta*y.self_attn.o_proj.weight],1))
   fa,fb=x.post_attention_layernorm.weight,y.post_attention_layernorm.weight
   d.mlp.gate_proj.weight.copy_(torch.cat([x.mlp.gate_proj.weight*fa[None,:],y.mlp.gate_proj.weight*fb[None,:]],0))
   d.mlp.up_proj.weight.copy_(torch.cat([x.mlp.up_proj.weight*fa[None,:],y.mlp.up_proj.weight*fb[None,:]],0))
   d.mlp.down_proj.weight.copy_(torch.cat([(1-beta)*x.mlp.down_proj.weight,beta*y.mlp.down_proj.weight],1))
  m.tie_weights()
 return m.eval(),c

def branch(a,b,beta,ids):
 with torch.inference_mode():
  e=(1-beta)*a.model.embed_tokens.weight+beta*b.model.embed_tokens.weight;h=F.embedding(ids,e);bs,t=ids.shape;pos=torch.arange(t).unsqueeze(0).expand(bs,-1);pe=a.model.rotary_emb(h,pos);mask=torch.triu(torch.full((bs,1,t,t),float('-inf')),1)
  for x,y in zip(a.model.layers,b.model.layers):
   ax=x.self_attn(x.input_layernorm(h),position_embeddings=pe,attention_mask=mask,use_cache=False)[0];ay=y.self_attn(y.input_layernorm(h),position_embeddings=pe,attention_mask=mask,use_cache=False)[0];h=h+(1-beta)*ax+beta*ay
   h=h+(1-beta)*x.mlp(x.post_attention_layernorm(h))+beta*y.mlp(y.post_attention_layernorm(h))
  g=(1-beta)*a.model.norm.weight+beta*b.model.norm.weight;eps=a.model.norm.variance_epsilon;h=h*torch.rsqrt(h.float().square().mean(-1,keepdim=True)+eps).to(h.dtype)*g
  return F.linear(h,e).float()

def err(a,b):
 d=(a.float()-b.float()).reshape(-1);r=a.float().reshape(-1);return {'max_abs':float(d.abs().max()),'rms':float(d.square().mean().sqrt()),'relative_rms':float(d.square().mean().sqrt()/(r.square().mean().sqrt()+1e-30))}

def evaluate(name,m,tok,data,wiki):
 t=time.time();rec={'wikitext':v5.evaluate_wiki_blocks(m,tok,wiki)}
 for dom,rows in data.items():rec[dom]=v8.evaluate_mcq_proper(m,tok,rows)
 s=v8.summarize(rec);s['seconds']=time.time()-t;print(name,json.dumps(s,indent=2),flush=True);return rec,s

def compare(name,records,rng):
 return {p:{'composite_relative_proper_loss':v8.bootstrap_composite(records[name],records[p],rng),'domains':{d:v8.bootstrap_domain(records[name][d],records[p][d],d,rng) for d in records[name]}} for p in ['parent_base','parent_specialist']}

def main():
 cfg=AutoConfig.from_pretrained(BASE);cfgs=AutoConfig.from_pretrained(SPEC);fields=['hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size','hidden_act','rope_theta','tie_word_embeddings'];audit={f:[getattr(cfg,f),getattr(cfgs,f)] for f in fields}
 if any(x!=y for x,y in audit.values()):raise RuntimeError({'config_mismatch':audit})
 tok=AutoTokenizer.from_pretrained(BASE);chat=AutoTokenizer.from_pretrained(SPEC)
 if tok.get_vocab()!=chat.get_vocab():raise RuntimeError('tokenizer mismatch')
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 print('Loading public parents',flush=True);a=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32,low_cpu_mem_usage=True).eval();b=AutoModelForCausalLM.from_pretrained(SPEC,torch_dtype=torch.float32,low_cpu_mem_usage=True).eval();beta,bst=beta_rule(a,b);print(json.dumps({'beta':beta,'stats':bst},indent=2),flush=True)
 data={'openbookqa_chat':v7.deterministic_sample(v7.openbookqa_rows(chat),N,201),'commonsenseqa_chat':v7.deterministic_sample(v7.commonsenseqa_rows(chat),N,202),'winogrande_chat':v7.deterministic_sample(v7.winogrande_rows(chat),N,203),'piqa_chat':v7.deterministic_sample(v7.piqa_rows(chat),N,204),'boolq_chat':v7.deterministic_sample(v7.boolq_rows(chat),N,205)}
 wiki=[r['text'] for r in load_dataset('wikitext','wikitext-2-raw-v1',split='test') if r['text'].strip()][2200:2450]
 records={};summaries={}
 for n,m in [('parent_base',a),('parent_specialist',b)]:records[n],summaries[n]=evaluate(n,m,tok,data,wiki);(ROOT/'PARTIAL.json').write_text(json.dumps({'beta':beta,'summaries':summaries},indent=2))
 lin=interpolation(cfg,a,b,beta);records['tensor_interpolation'],summaries['tensor_interpolation']=evaluate('tensor_interpolation',lin,tok,data,wiki);del lin;gc.collect()
 child,cc=union(cfg,a,b,beta);ids=tok('The copper wire conducts',return_tensors='pt',add_special_tokens=False).input_ids[:,:8];fold=err(branch(a,b,beta,ids),child(input_ids=ids,use_cache=False).logits);records['lineage_feature_union'],summaries['lineage_feature_union']=evaluate('lineage_feature_union',child,tok,data,wiki)
 pp=sum(p.numel() for p in a.parameters());cp=sum(p.numel() for p in child.parameters());ratio=cp/pp
 ck=ROOT/'TEMP_CHECKPOINT';shutil.rmtree(ck,ignore_errors=True);child.save_pretrained(ck,safe_serialization=True);tok.save_pretrained(ck);ref=child(input_ids=ids,use_cache=False).logits.detach().cpu();del child,a,b;gc.collect();reload=AutoModelForCausalLM.from_pretrained(ck,torch_dtype=torch.float32,low_cpu_mem_usage=True).eval();ra=err(ref,reload(input_ids=ids,use_cache=False).logits.detach().cpu());finite=bool(torch.isfinite(reload(input_ids=ids,use_cache=False).logits).all());del reload;gc.collect();ckbytes=sum(p.stat().st_size for p in ck.rglob('*') if p.is_file());shutil.rmtree(ck)
 rng=np.random.default_rng(20260726);fc=compare('lineage_feature_union',records,rng);ic=compare('tensor_interpolation',records,rng);fi={'composite_relative_proper_loss':v8.bootstrap_composite(records['lineage_feature_union'],records['tensor_interpolation'],rng),'domains':{d:v8.bootstrap_domain(records['lineage_feature_union'][d],records['tensor_interpolation'][d],d,rng) for d in records['lineage_feature_union']}}
 dom=list(records['lineage_feature_union']);mcq=[d for d in dom if d!='wikitext'];bestloss={d:min(summaries['parent_base'][d]['loss'],summaries['parent_specialist'][d]['loss']) for d in dom};bestacc={d:max(summaries['parent_base'][d]['accuracy'],summaries['parent_specialist'][d]['accuracy']) for d in mcq}
 noloss=all(summaries['lineage_feature_union'][d]['loss']<=1.03*bestloss[d] for d in dom);noacc=all(summaries['lineage_feature_union'][d]['accuracy']>=bestacc[d]-.05 for d in mcq);beats=all(fc[p]['composite_relative_proper_loss']['ci95'][1]<0 for p in fc);beatslin=fi['composite_relative_proper_loss']['ci95'][1]<0;struct=ratio<=1.8 and fold['relative_rms']<1e-5 and ra['max_abs']==0 and finite;promoted=beats and noloss and noacc and struct
 result={'status':'REAL_PUBLIC_LINEAGE_FEATURE_UNION_PASS' if promoted else 'REAL_PUBLIC_LINEAGE_FEATURE_UNION_NOT_PROMOTED','candidate_frozen':True,'parents':{'base':BASE,'specialist':SPEC},'config_audit':audit,'beta':beta,'beta_stats':bst,'beta_uses_evaluation_data':False,'compiler_uses_parent_forward_or_logits':False,'compiler_uses_gradients':False,'runtime_logit_or_probability_mixture':False,'runtime_router':False,'single_residual_stream':True,'single_tied_lm_head':True,'widened_config':{'hidden_size':cc.hidden_size,'head_dim':cc.head_dim,'num_attention_heads':cc.num_attention_heads,'num_key_value_heads':cc.num_key_value_heads,'intermediate_size':cc.intermediate_size,'num_hidden_layers':cc.num_hidden_layers},'parent_parameter_count':pp,'child_parameter_count':cp,'parameter_ratio':ratio,'checkpoint_bytes_before_deletion':ckbytes,'fold_equivalence':fold,'parent_free_reload':{'finite':finite,**ra},'evaluation':{'n_per_mcq_domain':N,'wikitext_blocks':v1.WIKI_BLOCKS,'bootstrap_resamples':v8.BOOTSTRAPS},'summaries':summaries,'feature_union_vs_parents':fc,'interpolation_vs_parents':ic,'feature_union_vs_same_beta_interpolation':fi,'no_domain_proper_loss_regression_over_3pct':noloss,'no_mcq_accuracy_regression_over_5pp':noacc,'composite_proper_loss_significantly_beats_both_parents':beats,'composite_proper_loss_significantly_beats_interpolation':beatslin,'structural_pass':struct,'promoted':promoted}
 (ROOT/'RESULTS.json').write_text(json.dumps(result,indent=2));(ROOT/'REPORT.md').write_text('\n'.join(['# Real SmolLM2 Base-to-Specialist Lineage Feature Union','',f"Status: **{result['status']}**",f'Frozen tensor-only beta: **{beta:.8f}**',f'Parameter ratio: **{ratio:.6f}x**',f"Fold relative RMS: **{fold['relative_rms']:.8g}**",f"Parent-free reload max error: **{ra['max_abs']:.8g}**",f'Significantly beats both parents: **{beats}**',f'Significantly beats same-beta interpolation: **{beatslin}**',f'All proper-loss domains within 3%: **{noloss}**',f'All MCQ accuracies within 5pp: **{noacc}**','','One standard widened Llama checkpoint; no parent modules, router, or output mixture remain at runtime.']))
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
