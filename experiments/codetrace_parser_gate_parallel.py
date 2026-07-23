#!/usr/bin/env python3
from __future__ import annotations
import gzip, hashlib, json, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datasets import load_dataset
import codetrace_parser_gate as g

WORKERS=3
OUT=Path('codetrace_parser_gate_parallel');OUT.mkdir(exist_ok=True)

def process(item):
    i,row=item;tid=str(row['traj_id']);agent=str(row['agent']);expected=int(row['step_count']);ap=str(row['artifact_path']);cands=[];chosen=None;err='';t=time.perf_counter()
    try:
        fp=g.download_retry(ap)
        with tempfile.TemporaryDirectory(prefix='ctgatep_') as td:
            g.base.extract_tar_zst(fp,td);cands=g.candidate_set(agent,Path(td),expected);chosen=g.select(cands,expected)
    except Exception as e:err=type(e).__name__+':'+str(e)[:500]
    pairs=chosen[1] if chosen else [];events=g.old.compile_pairs(tid,pairs,False)
    stat={'index':i,'instance_id':tid,'agent':agent,'expected':expected,'parsed':len(events),'selected_parser':chosen[0] if chosen else 'none','candidate_counts':{n:len(p) for n,p,_ in cands},'error':err,'elapsed_sec':time.perf_counter()-t}
    blind={'instance_id':tid,'agent':agent,'expected_step_count':expected,'selected_parser':chosen[0] if chosen else 'none','events':events} if events else None
    return i,stat,blind

def main():
    ds=load_dataset(g.DATASET,split='verified');items=list(enumerate(ds));rows=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for n,result in enumerate(ex.map(process,items),1):
            rows.append(result)
            if n%10==0:
                _,st,_=result;print(json.dumps({'processed':n,'total':len(items),'agent':st['agent'],'expected':st['expected'],'parsed':st['parsed'],'parser':st['selected_parser'],'error':st['error'][:120]},ensure_ascii=False),flush=True)
    rows.sort(key=lambda x:x[0]);stats=[x[1] for x in rows];blind_path=OUT/'blind_events.jsonl.gz'
    with gzip.open(blind_path,'wt',encoding='utf-8') as f:
        for _,_,b in rows:
            if b:f.write(json.dumps(b,ensure_ascii=False,separators=(',',':'))+'\n')
    s=g.summary(stats);(OUT/'parse_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8');contract={'version':'codetrace-parser-gate-parallel-v1','workers':WORKERS,'selection_uses':['agent','artifact schema','official step_count'],'selection_does_not_use':['incorrect_stages','incorrect_step_ids','failure labels','model predictions'],'blind_events_sha256':hashlib.sha256(blind_path.read_bytes()).hexdigest(),'stats_sha256':hashlib.sha256((OUT/'parse_stats.json').read_bytes()).hexdigest(),'summary_sha256':hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest()};(OUT/'parser_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(s,ensure_ascii=False,indent=2),flush=True);print('PARSER_GATE_PASSED' if s['strict_gate']['overall_pass'] else 'PARSER_GATE_FAILED',flush=True)
if __name__=='__main__':main()
