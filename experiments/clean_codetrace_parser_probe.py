#!/usr/bin/env python3
from __future__ import annotations
import importlib, json, tarfile, tempfile, traceback
from pathlib import Path
import zstandard as zstd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

OUT=Path("probe_out")
OUT.mkdir(exist_ok=True)
print("clean-parser-probe-start", flush=True)

MODS={
    "miniswe":"codetracer.skills.seed.miniswe.parser",
    "openhands":"codetracer.skills.seed.openhands.parser",
    "terminus2":"codetracer.skills.seed.terminus2.parser",
}
parsers={}
import_errors={}
for name,mod in MODS.items():
    try:
        parsers[name]=importlib.import_module(mod).parser
    except Exception as e:
        import_errors[name]={
            "type":type(e).__name__,
            "message":str(e),
            "traceback":traceback.format_exc(limit=12),
        }

def extract_tar_zst(src: str|Path, dst: str|Path):
    with open(src,"rb") as f:
        with zstd.ZstdDecompressor().stream_reader(f) as r:
            with tarfile.open(fileobj=r,mode="r|") as tf:
                tf.extractall(dst,filter="data")

def candidate_roots(root:Path):
    out={root}
    for p in root.rglob("*"):
        if not p.exists():
            continue
        name=p.name
        if p.is_file() and name in {"mini.traj.json","commands.txt","agent.log","results.json"}:
            for a in [p.parent,*list(p.parents)[:5]]:
                if root==a or root in a.parents: out.add(a)
        if p.is_dir() and (name=="agent-logs" or name=="sessions"):
            for a in [p.parent,*list(p.parents)[:4]]:
                if root==a or root in a.parents: out.add(a)
    return sorted(out,key=lambda p:(len(p.parts),str(p)))

def run_parser(parser, roots, expected):
    vals=[]
    for r in roots:
        try:
            if parser.can_parse(r):
                n=len(parser.parse(r).steps)
                vals.append({"root":str(r),"count":n})
        except Exception as e:
            vals.append({"root":str(r),"error":type(e).__name__+":"+str(e)[:300]})
    good=[x for x in vals if isinstance(x.get("count"),int)]
    best=min(good,key=lambda x:(abs(x["count"]-expected),-x["count"],x["root"])) if good else None
    return vals,best

ds=load_dataset("NJU-LINK/CodeTraceBench",split="verified")
chosen=[]
for agent in ("mini-SWE-agent","OpenHands","SWE-agent","Terminus2"):
    chosen += [(i,r) for i,r in enumerate(ds) if str(r["agent"])==agent][:8]

rows=[]
for num,(idx,row) in enumerate(chosen,1):
    rec={"index":idx,"traj_id":str(row["traj_id"]),"agent":str(row["agent"]),"expected":int(row["step_count"])}
    try:
        fp=hf_hub_download("NJU-LINK/CodeTraceBench",row["artifact_path"],repo_type="dataset",cache_dir="/tmp/ct_clean")
        with tempfile.TemporaryDirectory(prefix="ctclean_") as td:
            extract_tar_zst(fp,td)
            roots=candidate_roots(Path(td))
            rec["candidate_root_count"]=len(roots)
            for name,p in parsers.items():
                vals,best=run_parser(p,roots,rec["expected"])
                rec[name+"_runs"]=vals
                if best:
                    rec[name+"_count"]=best["count"]
                    rec[name+"_root"]=best["root"]
    except Exception as e:
        rec["artifact_error"]=type(e).__name__+":"+str(e)[:500]
    rows.append(rec)
    print(json.dumps({"processed":num,"agent":rec["agent"],"expected":rec["expected"],
                      "counts":{k:v for k,v in rec.items() if k.endswith("_count")}},ensure_ascii=False),flush=True)

summary={"imports":{"available":sorted(parsers),"errors":import_errors},"agents":{}}
for agent in sorted(set(r["agent"] for r in rows)):
    part=[r for r in rows if r["agent"]==agent]
    summary["agents"][agent]={}
    for name in parsers:
        key=name+"_count"
        vals=[r[key] for r in part if isinstance(r.get(key),int)]
        summary["agents"][agent][name]={
            "available":len(vals),
            "exact":sum(r.get(key)==r["expected"] for r in part if isinstance(r.get(key),int)),
            "mean_abs_error":sum(abs(r[key]-r["expected"]) for r in part if isinstance(r.get(key),int))/max(1,len(vals)),
        }
(OUT/"rows.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
