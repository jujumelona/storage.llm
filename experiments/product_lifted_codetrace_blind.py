#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, random, re
from pathlib import Path

import product_lifted_agentprocess as core
import product_lifted_agentprocess_v2 as v2

SEED = 20260723
random.seed(SEED)
KINDS = ("current", "ueta", "graph", "product_no_topology", "product")
CFG = {
    "hist": 2, "future": 0, "bad_precision": 0.35, "support": 2,
    "hard_bad": 4, "topk": 4, "hard_weight": 0.55,
    "transition_weight": 0.45, "unavoidable_bonus": 1.3,
    "distance_bonus": 0.7, "local_weight": 0.15,
    "route_weight": 0.15, "boundary_mode": "early",
}
PATH_RE = re.compile(r"(?:/[A-Za-z0-9_.-]+)+|https?://\S+|[A-Za-z0-9_.-]+\.(?:py|js|ts|json|yaml|yml|txt|md|csv|parquet)", re.I)
TEST_RE = re.compile(r"\b(test|pytest|unittest|check|verify|validate|assert|compare|diff|lint|mypy|compile)\b", re.I)
MUTATE_RE = re.compile(r"\b(edit|write|patch|replace|create|delete|remove|modify|update|apply_patch|str_replace|sed|cat\s*>)\b", re.I)
READ_RE = re.compile(r"\b(read|view|open|cat|grep|find|search|list|ls|inspect|head|tail|show)\b", re.I)
EXEC_RE = re.compile(r"\b(run|execute|bash|shell|python|node|npm|make|cmake|cargo|go test|java)\b", re.I)
FINAL_RE = re.compile(r"\b(final|submit|done|completed|solution|answer)\b", re.I)

def as_obj(x):
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return x
        try:
            return json.loads(s)
        except Exception:
            return x
    return x

def nested_text(x):
    x = as_obj(x)
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(nested_text(z) for z in x)
    if isinstance(x, dict):
        parts = []
        for k in ("content", "text", "message", "thought", "action", "observation", "output", "command"):
            if k in x:
                parts.append(nested_text(x[k]))
        if not parts:
            for k, v in x.items():
                if k not in {"labels", "incorrect_step_ids", "stage_id", "step_id"}:
                    parts.append(nested_text(v))
        return "\n".join(p for p in parts if p)
    return str(x)

def parse_list(x):
    x = as_obj(x)
    return x if isinstance(x, list) else []

def operation(action_text, observation_text):
    t = action_text.lower()
    if FINAL_RE.search(t):
        return "FINAL"
    if TEST_RE.search(t):
        return "VERIFY"
    if MUTATE_RE.search(t):
        return "MUTATE"
    if EXEC_RE.search(t):
        return "EXECUTE"
    if READ_RE.search(t):
        return "READ"
    if "?" in action_text or "need" in t or "plan" in t or "thought" in t:
        return "PLAN"
    return "TOOL_OTHER"

def resources(text, ablate=False):
    if ablate:
        return ("none",)
    out = set()
    for s in PATH_RE.findall(text):
        if s.startswith("http"):
            out.add("url")
        elif "/" in s:
            out.add("path")
        else:
            out.add("file")
    low = text.lower()
    for k in ("repo", "file", "path", "url", "query", "package", "test", "module", "function", "class"):
        if k in low:
            out.add(k)
    return tuple(sorted(out)[:5]) or ("none",)

def status(action, observation):
    z = f"{action}\n{observation}"
    if core.ERROR_RE.search(z):
        return "ERROR"
    if core.PARTIAL_RE.search(z):
        return "PARTIAL"
    if observation.strip():
        return "SUCCESS"
    return "NO_RESULT"

def compile_instance(inst, ablate=False):
    tid = inst["instance_id"]
    events = []
    last_op, last_res, last_status = "START", ("none",), "NONE"
    mask = 0
    repeats = 0
    steps = sorted(inst["steps"], key=lambda x: int(x["step_id"]))
    for pos, st in enumerate(steps):
        action = nested_text(st.get("action_ref"))
        obs = nested_text(st.get("observation_ref"))
        op = operation(action, obs)
        res = resources(action + "\n" + obs, ablate)
        st_status = status(action, obs)
        hard = 0
        if st_status == "SUCCESS":
            mask &= ~core.UNRESOLVED_ERROR
        if st_status == "ERROR":
            mask |= core.UNRESOLVED_ERROR
            hard += 1
        if st_status == "PARTIAL":
            mask |= core.PARTIAL_EVIDENCE
            hard += 1
        if op in {"READ", "VERIFY"} and st_status == "SUCCESS":
            mask &= ~core.PARTIAL_EVIDENCE
            if op == "VERIFY":
                mask &= ~core.NEED_VERIFY
        if op == "MUTATE":
            mask |= core.NEED_VERIFY
        if op == last_op and res == last_res and st_status in {"ERROR", "NO_RESULT"} and last_status in {"ERROR", "NO_RESULT"}:
            repeats += 1
            mask |= core.REPEAT_NO_PROGRESS
            hard += 2 + min(2, repeats)
        else:
            repeats = 0
            mask &= ~core.REPEAT_NO_PROGRESS
        if op == "FINAL":
            if mask & core.UNRESOLVED_ERROR:
                hard += 5
            if mask & core.PARTIAL_EVIDENCE:
                hard += 3
            if mask & core.NEED_VERIFY:
                hard += 2
        atoms = (
            f"OP:{op}", f"ST:{st_status}", f"MASK:{mask}", f"RES:{'+'.join(res)}",
            f"TLEN:{core.bucket(len(action))}", f"OLEN:{core.bucket(len(obs))}",
            f"PREV:{last_op}", f"PREVST:{last_status}", f"HARD:{min(hard,6)}",
        )
        events.append(core.Event(tid, "codetrace", pos, int(st["step_id"]), "0", op, st_status,
                                 res, atoms, hard, mask, len(action), 0, False))
        last_op, last_res, last_status = op, res, st_status
    return {"tid": tid, "dataset": "codetrace", "events": events, "has_error": False,
            "first_error_msg": None, "step_ids": [int(x["step_id"]) for x in steps]}

def prepare(args):
    from datasets import load_dataset
    ds = load_dataset("NJU-LINK/CodeTraceBench", split="verified")
    blind, labels = [], []
    for row in ds:
        traj_id = str(row.get("traj_id"))
        agent = str(row.get("agent") or "")
        model = str(row.get("model") or "")
        category = str(row.get("category") or "")
        stages = parse_list(row.get("incorrect_stages"))
        for s in stages:
            if not isinstance(s, dict):
                continue
            steps = parse_list(s.get("steps"))
            if len(steps) < 2:
                continue
            bad = set(int(x) for x in parse_list(s.get("incorrect_step_ids")) if str(x).lstrip("-").isdigit())
            clean_steps = []
            for st in steps:
                if not isinstance(st, dict) or "step_id" not in st:
                    continue
                sid = int(st["step_id"])
                clean_steps.append({
                    "step_id": sid,
                    "action_ref": st.get("action_ref"),
                    "observation_ref": st.get("observation_ref"),
                })
                if not bad:
                    labs = parse_list(st.get("labels"))
                    if any(str(x).lower() == "incorrect" for x in labs):
                        bad.add(sid)
            if len(clean_steps) < 2 or not bad:
                continue
            instance_id = f"{traj_id}:stage:{s.get('stage_id','?')}"
            blind.append({"instance_id": instance_id, "steps": clean_steps})
            labels.append({
                "instance_id": instance_id,
                "traj_id": traj_id,
                "stage_id": s.get("stage_id"),
                "gold_step": min(bad),
                "incorrect_steps": sorted(bad),
                "agent": agent,
                "model": model,
                "category": category,
            })
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out/"blind_instances.jsonl").open("w", encoding="utf-8") as f:
        for x in blind:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (out/"labels_sealed.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "dataset": "NJU-LINK/CodeTraceBench verified",
        "rows": len(ds),
        "blind_instances": len(blind),
        "label_sha256": hashlib.sha256((out/"labels_sealed.json").read_bytes()).hexdigest(),
        "blind_sha256": hashlib.sha256((out/"blind_instances.jsonl").read_bytes()).hexdigest(),
    }
    (out/"prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

def train_source(root):
    traces = core.load_all(root)
    return traces, core.train_model(traces, CFG)

def predict(args):
    blind_path = Path(args.blind)
    if "label" in blind_path.name.lower():
        raise RuntimeError("Predictor received a label-like file")
    source, model = train_source(Path(args.source))
    instances = [json.loads(x) for x in blind_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = []
    for inst in instances:
        tr = compile_instance(inst, False)
        ta = compile_instance(inst, True)
        if not tr["events"]:
            continue
        item = {"instance_id": inst["instance_id"], "step_ids": tr["step_ids"],
                "predictions": {}, "identity_ablation_predictions": {}}
        for kind in KINDS:
            p = core.predict(tr, model, kind)
            pa = core.predict(ta, model, kind)
            item["predictions"][kind] = int(tr["step_ids"][p]) if 0 <= p < len(tr["step_ids"]) else -1
            item["identity_ablation_predictions"][kind] = int(ta["step_ids"][pa]) if 0 <= pa < len(ta["step_ids"]) else -1
        rows.append(item)
    payload = {
        "protocol": "Frozen AgentProcessBench Product-Lifted model; CodeTrace labels unavailable to predictor",
        "source_config": CFG,
        "source_training_trajectories": len(source),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "blind_sha256": hashlib.sha256(blind_path.read_bytes()).hexdigest(),
        "predictor_had_labels": False,
        "predictions": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(str(out)+".sha256").write_text(hashlib.sha256(out.read_bytes()).hexdigest()+"\n", encoding="utf-8")
    print(json.dumps({"instances": len(rows), "code_sha256": payload["code_sha256"],
                      "prediction_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}, indent=2))

def metric(rows, kind, field="predictions"):
    if not rows:
        return {"n":0,"exact":0.0,"near1":0.0,"mrr":0.0}
    exact = near = 0
    rr = 0.0
    for r in rows:
        pred = int(r[field][kind])
        gold = int(r["gold_step"])
        exact += pred == gold
        near += abs(pred-gold) <= 1
        ordered = r["step_ids"]
        if pred in ordered and gold in ordered:
            rr += 1.0/(1+abs(ordered.index(pred)-ordered.index(gold)))
    n=len(rows)
    return {"n":n,"exact":exact/n,"near1":near/n,"mrr":rr/n}

def paired(rows, a="product", b="current", nboot=10000):
    vals=[];im=de=0
    for r in rows:
        ca=int(r["predictions"][a]==r["gold_step"])
        cb=int(r["predictions"][b]==r["gold_step"])
        vals.append(ca-cb);im+=int(ca and not cb);de+=int(cb and not ca)
    rng=random.Random(SEED)
    boots=[]
    for _ in range(nboot):
        boots.append(sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals))
    boots.sort()
    return {"n":len(vals),"gain":sum(vals)/len(vals),
            "bootstrap_95_ci":[boots[int(.025*nboot)],boots[int(.975*nboot)]],
            "improved":im,"degraded":de}

def evaluate(args):
    predp=Path(args.predictions)
    pred=json.loads(predp.read_text(encoding="utf-8"))
    expected=Path(str(predp)+".sha256").read_text().strip()
    actual=hashlib.sha256(predp.read_bytes()).hexdigest()
    if expected!=actual:
        raise RuntimeError("Prediction file changed after blind generation")
    labels=json.loads(Path(args.labels).read_text(encoding="utf-8"))
    by={x["instance_id"]:x for x in labels}
    rows=[]
    for p in pred["predictions"]:
        if p["instance_id"] not in by:
            continue
        z=dict(p);z.update(by[p["instance_id"]]);rows.append(z)
    metrics={k:metric(rows,k) for k in KINDS}
    ablation={k:metric(rows,k,"identity_ablation_predictions") for k in KINDS}
    agents={}
    for ag in sorted(set(r["agent"] for r in rows)):
        part=[r for r in rows if r["agent"]==ag]
        agents[ag]={k:metric(part,k) for k in KINDS}
    paired_pc=paired(rows,"product","current")
    paired_pu=paired(rows,"product","ueta")
    paired_pt=paired(rows,"product","product_no_topology")
    deltas=[agents[a]["product"]["exact"]-agents[a]["current"]["exact"] for a in agents if agents[a]["product"]["n"]>=10]
    strict={
        "hash_verified":expected==actual and pred.get("predictor_had_labels") is False,
        "at_least_300_instances":len(rows)>=300,
        "product_plus_5pp_over_current":metrics["product"]["exact"]>=metrics["current"]["exact"]+.05,
        "bootstrap_lower_positive":paired_pc["bootstrap_95_ci"][0]>0,
        "product_beats_ueta":metrics["product"]["exact"]>metrics["ueta"]["exact"],
        "topology_ablation_positive":metrics["product"]["exact"]>metrics["product_no_topology"]["exact"],
        "all_large_agent_groups_nonnegative":all(x>=-1e-12 for x in deltas) if deltas else False,
        "identity_ablation_product_not_below_current":ablation["product"]["exact"]>=ablation["current"]["exact"],
    }
    summary={
        "benchmark":"CodeTraceBench verified incorrect-stage first-error localization",
        "instances":len(rows),
        "metrics":metrics,
        "identity_ablation":ablation,
        "agent_groups":agents,
        "paired_product_minus_current":paired_pc,
        "paired_product_minus_ueta":paired_pu,
        "paired_product_minus_no_topology":paired_pt,
        "prediction_sha256":actual,
        "code_sha256":pred["code_sha256"],
        "strict":strict,
        "overall_pass":all(strict.values()),
        "scope":"Candidate steps are the human-verified incorrect stages, with step labels removed before prediction.",
    }
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["instance_id","agent","gold_step"]+[f"{k}_step" for k in KINDS]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            z={"instance_id":r["instance_id"],"agent":r["agent"],"gold_step":r["gold_step"]}
            z.update({f"{k}_step":r["predictions"][k] for k in KINDS});w.writerow(z)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="mode",required=True)
    p=sp.add_parser("prepare");p.add_argument("--out",required=True)
    p=sp.add_parser("predict");p.add_argument("--source",required=True);p.add_argument("--blind",required=True);p.add_argument("--out",required=True)
    p=sp.add_parser("evaluate");p.add_argument("--predictions",required=True);p.add_argument("--labels",required=True);p.add_argument("--out",required=True)
    a=ap.parse_args()
    {"prepare":prepare,"predict":predict,"evaluate":evaluate}[a.mode](a)

if __name__=="__main__":
    main()
