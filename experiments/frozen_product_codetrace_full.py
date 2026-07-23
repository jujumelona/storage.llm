#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, time
from pathlib import Path

SEED = 20260723
KINDS = ("current", "ueta", "graph", "product_no_topology", "product")
ALL_KINDS = ("first_step",) + KINDS
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


def official_map(n: int, expected: int) -> list[int]:
    """Monotone coordinate alignment using public step_count only; no error labels."""
    if n <= 0:
        return []
    if n == 1 or expected <= 1:
        return [1] * n
    out = []
    last = 1
    for i in range(n):
        x = 1 + int(math.floor(i * (expected - 1) / (n - 1) + 0.5))
        x = max(last, min(expected, x))
        out.append(x)
        last = x
    out[0] = 1
    out[-1] = expected
    return out


def retry_process(prep, item, cache: Path, attempts: int = 10):
    result = None
    for k in range(attempts):
        result = prep.process(item, cache)
        stat = result[3]
        err = str(stat.get("parse_error") or "")
        if not any(s in err for s in ("429", "Too Many Requests", "Network error", "ConnectionError", "Previous task error")):
            return result
        time.sleep(min(90, 3 * (2 ** min(k, 5))))
    return result


def prepare(args):
    from datasets import load_dataset
    import codetrace_full_parallel_prepare as prep

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("NJU-LINK/CodeTraceBench", split=args.split)

    blinds, labels, stats = [], [], []
    for idx, row in enumerate(ds):
        _, blind, label, stat = retry_process(prep, (idx, row), cache)
        if blind:
            mapping = official_map(len(blind["events"]), int(blind["expected_step_count"]))
            for event, step in zip(blind["events"], mapping):
                event["official_step"] = step
            for event, step in zip(blind["events_ablation"], mapping):
                event["official_step"] = step
            blind["coordinate_alignment"] = "monotone_rank_to_public_step_count"
            blinds.append(blind)
        if label:
            labels.append(label)
        stats.append(stat)
        if (idx + 1) % 25 == 0:
            print(json.dumps({
                "processed": idx + 1,
                "parsed": sum(int(x.get("parsed", 0)) > 0 for x in stats),
                "exact_count": sum(int(x.get("parsed", 0)) == int(x.get("expected", -1)) for x in stats),
                "rate_limit_failures": sum("429" in str(x.get("parse_error") or "") for x in stats),
            }), flush=True)

    blind_path = out / "blind_events.jsonl"
    with blind_path.open("w", encoding="utf-8") as f:
        for x in blinds:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    label_path = out / "labels_sealed.json"
    label_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "parse_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    parsed = sum(int(x.get("parsed", 0)) > 0 for x in stats)
    exact = sum(int(x.get("parsed", 0)) == int(x.get("expected", -1)) for x in stats)
    ratios = [int(x["parsed"]) / max(1, int(x["expected"])) for x in stats if int(x.get("parsed", 0)) > 0]
    parser_counts = {}
    agent_stats = {}
    for x in stats:
        parser_counts[x.get("selected_parser", "none")] = parser_counts.get(x.get("selected_parser", "none"), 0) + 1
        a = x.get("agent", "unknown")
        z = agent_stats.setdefault(a, {"rows": 0, "parsed": 0, "exact": 0, "ratios": []})
        z["rows"] += 1
        if int(x.get("parsed", 0)) > 0:
            z["parsed"] += 1
            z["ratios"].append(int(x["parsed"]) / max(1, int(x["expected"])))
        z["exact"] += int(int(x.get("parsed", 0)) == int(x.get("expected", -1)))
    for z in agent_stats.values():
        r = sorted(z.pop("ratios"))
        z["median_count_ratio"] = r[len(r)//2] if r else 0.0
        z["mean_count_ratio"] = sum(r)/len(r) if r else 0.0

    manifest = {
        "dataset": "NJU-LINK/CodeTraceBench",
        "split": args.split,
        "rows": len(ds),
        "parsed": parsed,
        "exact_step_count": exact,
        "mean_count_ratio": sum(ratios)/len(ratios) if ratios else 0.0,
        "blind_instances": len(blinds),
        "sealed_labels": len(labels),
        "parser_counts": parser_counts,
        "agent_parse_stats": agent_stats,
        "blind_sha256": hashlib.sha256(blind_path.read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
        "labels_not_used_for_parser_selection": True,
        "parser_selection_signal": "public step_count only",
    }
    (out / "prepare_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def train_source(root: Path):
    import product_lifted_agentprocess as core
    traces = core.load_all(root)
    return traces, core.train_model(traces, CFG)


def event_from(core, tid: str, pos: int, x: dict):
    return core.Event(
        tid, "codetrace_full", pos, int(x["official_step"]), "0",
        x["op"], x["status"], tuple(x["resources"]), tuple(x["atoms"]),
        int(x["hard"]), int(x["mask"]), int(x["text_len"]), 0, False,
    )


def predict(args):
    import product_lifted_agentprocess as core
    import product_lifted_agentprocess_v2  # patches frozen scorer

    blind_path = Path(args.blind)
    if "label" in blind_path.name.lower():
        raise RuntimeError("Predictor received a label-like input")
    source, model = train_source(Path(args.source))
    rows = []
    for line in blind_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        es = [event_from(core, x["instance_id"], i, z) for i, z in enumerate(x["events"])]
        ea = [event_from(core, x["instance_id"], i, z) for i, z in enumerate(x["events_ablation"])]
        tr = {"tid": x["instance_id"], "dataset": "codetrace_full", "events": es, "has_error": False, "first_error_msg": None}
        ta = {**tr, "events": ea}
        item = {
            "instance_id": x["instance_id"],
            "agent": x["agent"],
            "n_events": len(es),
            "expected_step_count": int(x["expected_step_count"]),
            "predictions": {"first_step": 1},
            "identity_ablation_predictions": {"first_step": 1},
        }
        for kind in KINDS:
            p = core.predict(tr, model, kind)
            q = core.predict(ta, model, kind)
            item["predictions"][kind] = int(es[p].msg_idx) if 0 <= p < len(es) else -1
            item["identity_ablation_predictions"][kind] = int(ea[q].msg_idx) if 0 <= q < len(ea) else -1
        rows.append(item)

    payload = {
        "protocol": "Frozen AgentProcessBench Product-Lifted model transferred to untouched CodeTraceBench full split",
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
    Path(str(out) + ".sha256").write_text(hashlib.sha256(out.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    print(json.dumps({"predictions": len(rows), "prediction_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}, indent=2))


def metric(rows, kind: str, field: str = "predictions"):
    if not rows:
        return {"n": 0, "exact": 0.0, "near1": 0.0, "near3": 0.0, "normalized_mae": 1.0}
    distances, norm = [], []
    for r in rows:
        d = abs(int(r[field][kind]) - int(r["gold_step"]))
        distances.append(d)
        norm.append(d / max(1, int(r["expected_step_count"]) - 1))
    n = len(rows)
    return {
        "n": n,
        "exact": sum(x == 0 for x in distances) / n,
        "near1": sum(x <= 1 for x in distances) / n,
        "near3": sum(x <= 3 for x in distances) / n,
        "normalized_mae": sum(norm) / n,
    }


def paired(rows, a="product", b="current", nboot=10000):
    vals, improved, degraded = [], 0, 0
    for r in rows:
        ca = int(r["predictions"][a] == r["gold_step"])
        cb = int(r["predictions"][b] == r["gold_step"])
        vals.append(ca - cb)
        improved += int(ca > cb)
        degraded += int(cb > ca)
    if not vals:
        return {"n": 0, "gain": 0.0, "bootstrap_95_ci": [0.0, 0.0], "improved": 0, "degraded": 0}
    rng = random.Random(SEED)
    boots = []
    for _ in range(nboot):
        boots.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    boots.sort()
    return {
        "n": len(vals),
        "gain": sum(vals) / len(vals),
        "bootstrap_95_ci": [boots[int(0.025*nboot)], boots[int(0.975*nboot)]],
        "improved": improved,
        "degraded": degraded,
    }


def evaluate(args):
    pred_path = Path(args.predictions)
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    expected_hash = Path(str(pred_path) + ".sha256").read_text().strip()
    actual_hash = hashlib.sha256(pred_path.read_bytes()).hexdigest()
    if expected_hash != actual_hash:
        raise RuntimeError("Predictions changed after blind sealing")

    labels = {x["instance_id"]: x for x in json.loads(Path(args.labels).read_text(encoding="utf-8"))}
    rows = []
    for p in payload["predictions"]:
        g = labels.get(p["instance_id"])
        if not g:
            continue
        z = dict(p)
        z.update(g)
        rows.append(z)

    metrics = {k: metric(rows, k) for k in ALL_KINDS}
    ablation = {k: metric(rows, k, "identity_ablation_predictions") for k in ALL_KINDS}
    groups = {}
    for agent in sorted(set(r["agent"] for r in rows)):
        part = [r for r in rows if r["agent"] == agent]
        groups[agent] = {k: metric(part, k) for k in ALL_KINDS}

    pc = paired(rows, "product", "current")
    pu = paired(rows, "product", "ueta")
    pt = paired(rows, "product", "product_no_topology")
    pf = paired(rows, "product", "first_step")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    large_group_deltas = [
        groups[a]["product"]["exact"] - groups[a]["current"]["exact"]
        for a in groups if groups[a]["product"]["n"] >= 50
    ]
    parse_ratio_ok = all(
        z["parsed"] / max(1, z["rows"]) >= 0.75
        for z in manifest["agent_parse_stats"].values()
    )
    strict = {
        "hash_verified": expected_hash == actual_hash and payload.get("predictor_had_labels") is False,
        "full_split_at_least_3000_rows": manifest["rows"] >= 3000,
        "parsed_at_least_80_percent": manifest["parsed"] / manifest["rows"] >= 0.80,
        "each_agent_parse_rate_at_least_75_percent": parse_ratio_ok,
        "evaluated_at_least_1000_labeled_trajectories": len(rows) >= 1000,
        "product_plus_5pp_over_current": metrics["product"]["exact"] >= metrics["current"]["exact"] + 0.05,
        "bootstrap_lower_positive": pc["bootstrap_95_ci"][0] > 0,
        "product_beats_ueta": metrics["product"]["exact"] > metrics["ueta"]["exact"],
        "topology_ablation_positive": metrics["product"]["exact"] > metrics["product_no_topology"]["exact"],
        "product_beats_first_step": metrics["product"]["exact"] > metrics["first_step"]["exact"],
        "all_large_agent_groups_nonnegative": all(x >= -1e-12 for x in large_group_deltas) if large_group_deltas else False,
        "identity_ablation_product_not_below_current": ablation["product"]["exact"] >= ablation["current"]["exact"],
    }
    summary = {
        "benchmark": "CodeTraceBench full split full-trajectory first incorrect step blind transfer",
        "scope": "Frozen AgentProcessBench model; parser chosen only by public step_count; labels opened after predictions.",
        "prepare_manifest": manifest,
        "evaluated": len(rows),
        "metrics": metrics,
        "identity_ablation": ablation,
        "agent_groups": groups,
        "paired_product_minus_current": pc,
        "paired_product_minus_ueta": pu,
        "paired_product_minus_no_topology": pt,
        "paired_product_minus_first_step": pf,
        "prediction_sha256": actual_hash,
        "code_sha256": payload["code_sha256"],
        "strict": strict,
        "overall_pass": all(strict.values()),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["instance_id", "agent", "expected_step_count", "gold_step"] + [f"{k}_step" for k in ALL_KINDS]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            z = {q: r[q] for q in ("instance_id", "agent", "expected_step_count", "gold_step")}
            z.update({f"{k}_step": r["predictions"][k] for k in ALL_KINDS})
            w.writerow(z)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    p = sp.add_parser("prepare")
    p.add_argument("--split", default="full")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", required=True)
    p = sp.add_parser("predict")
    p.add_argument("--source", required=True)
    p.add_argument("--blind", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("evaluate")
    p.add_argument("--predictions", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"prepare": prepare, "predict": predict, "evaluate": evaluate}[args.mode](args)

if __name__ == "__main__":
    main()
