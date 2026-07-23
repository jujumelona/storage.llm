#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import product_lifted_codetrace_full_blind as full
import codetrace_full_parser_v2 as parser


def process(item,cache):
    idx,row=item;tid=str(row['traj_id']);agent=str(row['agent']);bad=full.incorrect_ids(row);ap=row.get('artifact_path');pairs=[];err='';pname='none';cands=[];expected=int(row['step_count'])
    try:
        fp=hf_hub_download('NJU-LINK/CodeTraceBench',ap,repo_type='dataset',cache_dir=str(cache))
        with tempfile.TemporaryDirectory(prefix='ctbp_') as td:
            full.extract_tar_zst(fp,td);pairs,pname,cands=parser.parse_artifact(agent,Path(td),expected)
    except Exception as e:err=type(e).__name__+':'+str(e)[:300]
    ev=parser.compile_pairs(tid,pairs,False);eva=parser.compile_pairs(tid,pairs,True)
    blind={'instance_id':tid,'agent':agent,'expected_step_count':expected,'events':ev,'events_ablation':eva} if ev else None
    label={'instance_id':tid,'agent':agent,'gold_step':min(bad),'incorrect_steps':bad,'expected_step_count':expected} if bad else None
    stat={'index':idx,'instance_id':tid,'agent':agent,'expected':expected,'parsed':len(ev),'selected_parser':pname,'parser_candidates':cands,'has_gold':bool(bad),'max_gold':max(bad) if bad else None,'parse_error':err}
    return idx,blind,label,stat

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--cache',required=True);ap.add_argument('--workers',type=int,default=12);ap.add_argument('--split',default='verified');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache);cache.mkdir(parents=True,exist_ok=True)
    ds=load_dataset('NJU-LINK/CodeTraceBench',split=a.split);items=list(enumerate(ds));results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n,r in enumerate(ex.map(lambda x:process(x,cache),items),1):
            results.append(r)
            if n%25==0:print(json.dumps({'processed':n}),flush=True)
    results.sort(key=lambda x:x[0]);blinds=[x[1] for x in results if x[1]];labels=[x[2] for x in results if x[2]];stats=[x[3] for x in results]
    with (out/'blind_events.jsonl').open('w',encoding='utf-8') as f:
        for x in blinds:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    (out/'labels_sealed.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf-8');(out/'parse_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    parsed=sum(x['parsed']>0 for x in stats);exact=sum(x['parsed']==x['expected'] for x in stats);cover=sum(x['parsed']>=x['max_gold'] for x in stats if x['has_gold'] and x['parsed']>0);goldn=sum(x['has_gold'] for x in stats)
    manifest={'split':a.split,'rows':len(ds),'parsed':parsed,'exact_step_count':exact,'gold_covered':cover,'gold_rows':goldn,'workers':a.workers,'blind_sha256':hashlib.sha256((out/'blind_events.jsonl').read_bytes()).hexdigest(),'labels_sha256':hashlib.sha256((out/'labels_sealed.json').read_bytes()).hexdigest()}
    (out/'prepare_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
