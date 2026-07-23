#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, re, statistics
from collections import Counter
from pathlib import Path

import product_lifted_agentprocess as core
import product_lifted_agentprocess_v2 as v2  # patches core.scores/core.predict to frozen V2/V3 scorer

SEED = 20260723
random.seed(SEED)
KINDS = ("current", "ueta", "graph", "product_no_topology", "product")
CFG = {
    "hist": 2,
    "future": 0,
    "bad_precision": 0.35,
    "support": 2,
    "hard_bad": 4,
    "topk": 4,
    "hard_weight": 0.55,
    "transition_weight": 0.45,
    "unavoidable_bonus": 1.3,
    "distance_bonus": 0.7,
    "local_weight": 0.15,
    "route_weight": 0.15,
    "boundary_mode": "early",
}
PATH_RE = re.compile(r"(?:/[A-Za-z0-9_.-]+)+|https?://\S+|#[A-Z0-9]{4,}", re.I)

def safe_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def domain_from_path(path: Path) -> str:
    s = str(path).lower()
    if "magentic" in s:
        return "magentic"
    if "tau" in s or "retail" in s:
        return "tau"
    if "flash" in s or "incident" in s:
        return "flash"
    return "unknown"

def message_list(obj):
    if isinstance(obj, dict):
        for k in ("traj", "messages", "trajectory", "steps", "history"):
            x = obj.get(k)
            if isinstance(x, list) and x and all(isinstance(z, dict) for z in x):
                return x
    if isinstance(obj, list) and obj and all(isinstance(z, dict) for z in obj):
        if sum(("role" in z or "speaker" in z or "agent" in z or "name" in z) for z in obj) >= max(1, len(obj)//3):
            return obj
    return None

def text_of(m):
    x = m.get("content", m.get("text", m.get("message", m.get("output", ""))))
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    return "" if x is None else str(x)

def role_of(m):
    return str(m.get("role", m.get("speaker", m.get("agent", m.get("name", "unknown")))))

def original_step(m, idx):
    for k in ("index", "step", "step_number", "turn", "id"):
        x = m.get(k)
        if isinstance(x, int):
            return x
        if isinstance(x, str) and x.isdigit():
            return int(x)
    return idx + 1

def calls_of(m):
    out = []
    for tc in m.get("tool_calls", []) or []:
        if not isinstance(tc, dict):
            continue
        f = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = str(f.get("name", tc.get("name", "")))
        args = f.get("arguments", tc.get("arguments", tc.get("args", "{}")))
        out.append((name, args))
    fc = m.get("function_call")
    if isinstance(fc, dict):
        out.append((str(fc.get("name", "")), fc.get("arguments", "{}")))
    if m.get("tool_name"):
        out.append((str(m.get("tool_name")), m.get("tool_args", m.get("arguments", "{}"))))
    return out

def op_from_role_text(role, text, calls):
    if calls:
        fams = [core.family(n) for n, _ in calls]
        return fams[0] if len(set(fams)) == 1 else "MULTI_TOOL"
    r = role.lower()
    t = text.lower()
    if r in {"tool", "function"}:
        return "TOOL_RESULT"
    if "user" in r or "human" in r:
        return "USER"
    if "terminal" in r or "computer" in r or "shell" in r:
        return "EXECUTE"
    if "file" in r:
        return "MUTATE" if any(w in t for w in core.MUTATE_WORDS) else "READ"
    if "web" in r or "browser" in r or "surfer" in r:
        return "READ"
    if "orchestrator" in r and ("thought" in r or "ledger" in t or "next_speaker" in t):
        return "PLAN"
    if core.ASK_RE.search(text):
        return "ASK"
    if core.FINAL_RE.search(text):
        return "FINAL"
    if core.PLAN_RE.search(text):
        return "PLAN"
    if any(w in t for w in core.VERIFY_WORDS):
        return "VERIFY"
    if any(w in t for w in core.MUTATE_WORDS):
        return "MUTATE"
    if any(w in t for w in core.READ_WORDS):
        return "READ"
    return "FINAL"

def text_resources(text):
    out = set()
    for x in PATH_RE.findall(text or ""):
        if x.startswith("http"):
            out.add("url")
        elif x.startswith("#"):
            out.add("entity_id")
        else:
            out.add("path")
    low = (text or "").lower()
    for k in ("order_id", "user_id", "payment", "address", "email", "file", "query", "url"):
        if k in low:
            out.add(k)
    return tuple(sorted(out)[:4]) or ("none",)

def compile_trajectory(path: Path, obj, identity_ablation=False):
    msgs = message_list(obj)
    if not msgs:
        return None
    domain = domain_from_path(path)
    top = obj if isinstance(obj, dict) else {}
    raw_id = top.get("trajectory_id", top.get("task_id", top.get("id", path.stem)))
    tid = f"{domain}:{raw_id}:{path.name}"
    policy = core.policy_flags(msgs)
    declared = core.declared_tool_names(top)
    mask = core.NEED_AUTH if policy["auth"] else 0
    events, steps, roles = [], [], []
    last_op, last_res, last_status = "START", ("none",), "NONE"
    repeats = 0
    for idx, m in enumerate(msgs):
        role = role_of(m)
        if role.lower() == "system":
            continue
        calls = calls_of(m)
        content = text_of(m)
        if role.lower() in {"tool", "function"}:
            continue
        result_text = ""
        for z in msgs[idx+1: min(len(msgs), idx+4)]:
            zr = role_of(z).lower()
            if zr in {"tool", "function"}:
                result_text += " " + text_of(z)
            else:
                break
        op = op_from_role_text(role, content, calls)
        if calls:
            res = tuple(sorted(set(x for _, a in calls for x in core.resource_types(a))))[:5] or ("none",)
        else:
            res = text_resources(content)
        if identity_ablation:
            res = ("none",)
        status = (
            "ERROR" if (result_text and core.ERROR_RE.search(result_text)) or core.ERROR_RE.search(content)
            else "PARTIAL" if (result_text and core.PARTIAL_RE.search(result_text)) or core.PARTIAL_RE.search(content)
            else "SUCCESS" if result_text
            else "NO_RESULT" if calls
            else "TEXT"
        )
        hard = 0
        if status == "SUCCESS":
            mask &= ~core.UNRESOLVED_ERROR
        if status == "ERROR":
            mask |= core.UNRESOLVED_ERROR
            hard += 1
        if status == "PARTIAL":
            mask |= core.PARTIAL_EVIDENCE
            hard += 1
        if op in {"READ", "VERIFY", "AUTH"} and status == "SUCCESS":
            mask &= ~core.PARTIAL_EVIDENCE
            if op == "VERIFY":
                mask &= ~core.NEED_VERIFY
            if op == "AUTH":
                mask &= ~core.NEED_AUTH
        prev_user = ""
        for z in reversed(msgs[:idx]):
            if "user" in role_of(z).lower() or "human" in role_of(z).lower():
                prev_user = text_of(z)
                break
        if calls and policy["listed_only"] and declared and any(n not in declared for n, _ in calls):
            mask |= core.UNSUPPORTED
            hard += 4
        if policy["one_tool"] and len(calls) > 1:
            hard += 3
        if op == "MUTATE":
            if policy["auth"] and mask & core.NEED_AUTH:
                hard += 4
            if policy["confirm"] and not core.YES_RE.search(prev_user):
                mask |= core.NEED_CONFIRM
                hard += 4
            else:
                mask &= ~core.NEED_CONFIRM
            mask |= core.NEED_VERIFY
        if op == last_op and res == last_res and status in {"ERROR", "NO_RESULT"} and last_status in {"ERROR", "NO_RESULT"}:
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
            if mask & core.NEED_CONFIRM:
                hard += 4
            if mask & core.UNSUPPORTED:
                hard += 3
            if mask & core.NEED_VERIFY:
                hard += 1
        atoms = [
            f"OP:{op}", f"ST:{status}", f"MASK:{mask}", f"RES:{'+'.join(res)}",
            f"CALLS:{core.bucket(len(calls))}", f"TLEN:{core.bucket(len(content))}",
            f"PREV:{last_op}", f"PREVST:{last_status}", f"HARD:{min(hard,6)}",
        ]
        if policy["confirm"]:
            atoms.append("POLICY:CONFIRM")
        if policy["auth"]:
            atoms.append("POLICY:AUTH")
        if policy["listed_only"]:
            atoms.append("POLICY:LISTED")
        if core.YES_RE.search(prev_user):
            atoms.append("USER:CONFIRM")
        if core.ERROR_RE.search(content):
            atoms.append("TEXT:ERROR")
        if core.PARTIAL_RE.search(content):
            atoms.append("TEXT:PARTIAL")
        pos = len(events)
        events.append(core.Event(tid, domain, pos, original_step(m, idx), "0", op, status, res, tuple(atoms), hard, mask, len(content), len(calls), False))
        steps.append(original_step(m, idx))
        roles.append(role)
        last_op, last_res, last_status = op, res, status
    if not events:
        return None
    ids = {str(raw_id), path.stem, path.name}
    for k in ("trajectory_id", "task_id", "id"):
        if isinstance(top, dict) and top.get(k) is not None:
            ids.add(str(top[k]))
    return {"tid": tid, "dataset": domain, "events": events, "has_error": False, "first_error_msg": None,
            "steps": steps, "roles": roles, "ids": sorted(ids), "file": path.name, "path": str(path)}

def discover_trajectories(root: Path, identity_ablation=False):
    out = []
    for path in sorted(root.rglob("*.json")):
        low = str(path).lower()
        if any(x in low for x in ("ground_truth", "groundtruth", "annotation", "labels", "failure_metadata")):
            raise RuntimeError(f"Blind predictor was given a forbidden label-like path: {path}")
        obj = safe_json(path)
        tr = compile_trajectory(path, obj, identity_ablation) if obj is not None else None
        if tr:
            out.append(tr)
    dedup = {}
    for tr in out:
        key = (tr["dataset"], tuple(tr["ids"]), tuple(tr["steps"]))
        dedup.setdefault(key, tr)
    return list(dedup.values())

def train_source(apb_root: Path):
    traces = core.load_all(apb_root)
    model = core.train_model(traces, CFG)
    return traces, model

def predict_mode(args):
    blind = Path(args.trajectories)
    forbidden = [p for p in blind.rglob("*") if p.is_file() and any(x in str(p).lower() for x in ("ground_truth", "groundtruth", "annotation", "labels"))]
    if forbidden:
        raise RuntimeError(f"Forbidden label files present in predictor input: {forbidden[:3]}")
    source, model = train_source(Path(args.source))
    traces = discover_trajectories(blind, identity_ablation=False)
    traces_abl = discover_trajectories(blind, identity_ablation=True)
    abl_by_tid = {x["tid"]: x for x in traces_abl}
    rows = []
    for tr in traces:
        item = {
            "tid": tr["tid"], "domain": tr["dataset"], "ids": tr["ids"], "file": tr["file"],
            "n_events": len(tr["events"]), "steps": tr["steps"], "roles": tr["roles"],
            "predictions": {}, "identity_ablation_predictions": {},
        }
        for kind in KINDS:
            p = core.predict(tr, model, kind)
            item["predictions"][kind] = {
                "event_position": int(p),
                "step": int(tr["steps"][p]) if 0 <= p < len(tr["steps"]) else -1,
                "agent": tr["roles"][p] if 0 <= p < len(tr["roles"]) else "",
            }
            ta = abl_by_tid.get(tr["tid"])
            pa = core.predict(ta, model, kind) if ta else -1
            item["identity_ablation_predictions"][kind] = {
                "event_position": int(pa),
                "step": int(ta["steps"][pa]) if ta and 0 <= pa < len(ta["steps"]) else -1,
            }
        rows.append(item)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = {
        "protocol": "AgentRx label-blind transfer from frozen AgentProcessBench Product-Lifted model",
        "source_training_trajectories": len(source),
        "source_config": CFG,
        "code_sha256": code_hash,
        "predictor_had_ground_truth": False,
        "trajectory_count": len(rows),
        "predictions": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(str(out) + ".sha256").write_text(hashlib.sha256(out.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("protocol", "source_training_trajectories", "code_sha256", "trajectory_count")}, indent=2))

def infer_gt_domain(path):
    return domain_from_path(path)

def add_gt_record(records, domain, ids, step, agent="", category="", source=""):
    try:
        step = int(step)
    except Exception:
        return
    ids = {str(x) for x in ids if x is not None and str(x)}
    if not ids:
        return
    records.append({"domain": domain, "ids": sorted(ids), "step": step, "agent": str(agent or ""),
                    "category": str(category or ""), "source": source})

def scan_ground_truth(root: Path):
    records = []
    for path in sorted(root.rglob("*.json")):
        obj = safe_json(path)
        if obj is None:
            continue
        domain = infer_gt_domain(path)
        seq = obj if isinstance(obj, list) else [obj] if isinstance(obj, dict) and "failures" in obj else []
        for rec in seq:
            if not isinstance(rec, dict) or not isinstance(rec.get("failures"), list):
                continue
            rc = rec.get("root_cause") or {}
            rid = rc.get("failure_id") if isinstance(rc, dict) else None
            failure = next((f for f in rec["failures"] if isinstance(f, dict) and f.get("failure_id") == rid), None)
            if failure is None and len(rec["failures"]) == 1:
                failure = rec["failures"][0]
            if not isinstance(failure, dict):
                continue
            ids = [rec.get("trajectory_id"), rec.get("task_id"), rec.get("id")]
            add_gt_record(records, domain, ids, failure.get("step_number", failure.get("step")),
                          failure.get("failed_agent", ""), failure.get("failure_category", ""), str(path))
        if isinstance(obj, dict):
            for category, group in obj.items():
                if not isinstance(group, dict):
                    continue
                for key, val in group.items():
                    if not isinstance(val, dict) or ("step" not in val and "step_number" not in val):
                        continue
                    name = val.get("name", "")
                    ids = [key, name, Path(name).stem if name else ""]
                    add_gt_record(records, domain, ids, val.get("step", val.get("step_number")),
                                  val.get("failed_agent", ""), category, str(path))
    uniq = {}
    for r in records:
        key = (r["domain"], tuple(r["ids"]), r["step"], r["agent"])
        uniq[key] = r
    return list(uniq.values())

def norm_agent(s):
    x = re.sub(r"[^a-z0-9]+", "", str(s).lower())
    for k in ("websurfer", "filesurfer", "orchestrator", "assistant", "user", "computerterminal", "terminal"):
        if k in x:
            return k
    return x

def match_gt(pred, gt_records):
    ids = set(pred["ids"]) | {pred["file"], Path(pred["file"]).stem}
    same_domain = [g for g in gt_records if g["domain"] == pred["domain"]]
    candidates = []
    for g in same_domain:
        overlap = ids & set(g["ids"])
        if overlap:
            candidates.append((len(overlap), g))
    if not candidates:
        for g in gt_records:
            overlap = ids & set(g["ids"])
            if overlap:
                candidates.append((len(overlap), g))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]["step"]))
    return candidates[0][1]

def metric(rows, kind, field="predictions"):
    rr = [r for r in rows if r.get("gold")]
    if not rr:
        return {"n": 0, "exact": 0.0, "near1": 0.0, "agent": 0.0}
    ex = [int(r[field][kind]["step"] == r["gold"]["step"]) for r in rr]
    nr = [int(abs(r[field][kind]["step"] - r["gold"]["step"]) <= 1) for r in rr]
    ag = []
    for r in rr:
        ga = norm_agent(r["gold"].get("agent", ""))
        pa = norm_agent(r[field][kind].get("agent", "")) if "agent" in r[field][kind] else ""
        if ga:
            ag.append(int(ga == pa or ga in pa or pa in ga))
    return {"n": len(rr), "exact": sum(ex)/len(ex), "near1": sum(nr)/len(nr),
            "agent": sum(ag)/len(ag) if ag else None}

def paired_stats(rows, a="product", b="current", nboot=10000):
    vals = []
    improve = degrade = 0
    for r in rows:
        if not r.get("gold"):
            continue
        ca = int(r["predictions"][a]["step"] == r["gold"]["step"])
        cb = int(r["predictions"][b]["step"] == r["gold"]["step"])
        vals.append(ca-cb)
        improve += int(ca and not cb)
        degrade += int(cb and not ca)
    if not vals:
        return {"n": 0, "gain": 0.0, "bootstrap_95_ci": [0.0, 0.0], "improved": 0, "degraded": 0}
    rng = random.Random(SEED)
    boots = []
    for _ in range(nboot):
        boots.append(sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals))
    boots.sort()
    return {"n": len(vals), "gain": sum(vals)/len(vals),
            "bootstrap_95_ci": [boots[int(.025*nboot)], boots[int(.975*nboot)]],
            "improved": improve, "degraded": degrade}

def evaluate_mode(args):
    pred_path = Path(args.predictions)
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    expected = Path(str(pred_path) + ".sha256").read_text().strip()
    actual = hashlib.sha256(pred_path.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError("Prediction file changed after blind generation")
    gt = scan_ground_truth(Path(args.repo))
    matched = []
    for p in payload["predictions"]:
        g = match_gt(p, gt)
        row = dict(p)
        row["gold"] = g
        matched.append(row)
    usable = [r for r in matched if r["gold"]]
    metrics = {k: metric(usable, k) for k in KINDS}
    ablation = {k: metric(usable, k, "identity_ablation_predictions") for k in KINDS}
    domains = {}
    for d in sorted(set(r["domain"] for r in usable)):
        part = [r for r in usable if r["domain"] == d]
        domains[d] = {k: metric(part, k) for k in KINDS}
    paired = paired_stats(usable)
    deltas = [domains[d]["product"]["exact"] - domains[d]["current"]["exact"] for d in domains]
    strict = {
        "blind_hash_verified": expected == actual and payload.get("predictor_had_ground_truth") is False,
        "matched_at_least_80": len(usable) >= 80,
        "product_beats_current": metrics["product"]["exact"] > metrics["current"]["exact"],
        "bootstrap_lower_positive": paired["bootstrap_95_ci"][0] > 0,
        "all_domains_nonnegative": all(x >= -1e-12 for x in deltas) if deltas else False,
        "identity_ablation_product_not_below_current": ablation["product"]["exact"] >= ablation["current"]["exact"],
    }
    summary = {
        "benchmark": "Microsoft AgentRx",
        "protocol": payload["protocol"],
        "code_sha256": payload["code_sha256"],
        "prediction_sha256": actual,
        "discovered_ground_truth_records": len(gt),
        "blind_trajectories": payload["trajectory_count"],
        "matched_trajectories": len(usable),
        "unmatched_files": [r["file"] for r in matched if not r["gold"]],
        "metrics": metrics,
        "identity_ablation": ablation,
        "domain_metrics": domains,
        "paired_product_minus_current": paired,
        "strict": strict,
        "overall_pass": all(strict.values()),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out/"blind_predictions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out/"matched_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["tid","domain","file","gold_step","gold_agent"] + [f"{k}_step" for k in KINDS]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in usable:
            z = {"tid":r["tid"],"domain":r["domain"],"file":r["file"],
                 "gold_step":r["gold"]["step"],"gold_agent":r["gold"].get("agent","")}
            z.update({f"{k}_step":r["predictions"][k]["step"] for k in KINDS})
            w.writerow(z)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    p = sp.add_parser("predict")
    p.add_argument("--source", required=True)
    p.add_argument("--trajectories", required=True)
    p.add_argument("--out", required=True)
    e = sp.add_parser("evaluate")
    e.add_argument("--repo", required=True)
    e.add_argument("--predictions", required=True)
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "predict":
        predict_mode(args)
    else:
        evaluate_mode(args)

if __name__ == "__main__":
    main()
