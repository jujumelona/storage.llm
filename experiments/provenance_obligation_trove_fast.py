#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "provenance_obligation_trove.py"
spec = importlib.util.spec_from_file_location("trove_base", BASE_PATH)
b = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(b)


def rewired_features(row, fs):
    if len(fs) <= 1:
        return fs
    out = [dict(x) for x in fs]
    perm = list(range(len(fs)))
    b.random.Random(int(hashlib.sha256(row['rid'].encode()).hexdigest()[:12], 16)).shuffle(perm)
    ca_list = [fs[j]["ca"] for j in perm]
    for i, x in enumerate(out):
        x["ca"] = ca_list[i]
        x["inter"] = set(x["tw"]) & x["ca"]
        x["wrec"] = sum(x["tw"][a] for a in x["inter"]) / (sum(x["tw"].values()) or 1)
        x["rec"] = len(x["inter"]) / max(1, len(x["tw"]))
        x["prec"] = len(x["inter"]) / max(1, len(x["ca"]))
        x["score"] = .52*x["wrec"] + .18*x["rec"] + .12*x["prec"] + .10*x["lcs"] + .08*x["entity"]
    return out


def select(row, fs, indexed=True):
    target, cands = row["target"], row["candidates"]
    inv = defaultdict(list)
    for i, f in enumerate(fs):
        for a in f["ca"]:
            inv[a].append(i)
    target_atoms = set(fs[0]["tw"]) if fs else set()
    if indexed:
        eligible = set()
        for a in target_atoms:
            eligible.update(inv.get(a, ()))
    else:
        eligible = set(range(len(cands)))
    if not eligible:
        eligible = set(range(len(cands)))
    ranked = sorted(eligible, key=lambda i: (fs[i]["score"], -i), reverse=True)
    selected, covered = [], set()
    totalw = sum(fs[0]["tw"].values()) if fs else 1
    for i in ranked:
        f = fs[i]
        marginal = f["inter"] - covered
        mg = sum(f["tw"].get(a, 1) for a in marginal) / max(1, totalw)
        standalone = f["score"]
        if not selected or (standalone >= .19 and mg >= .055) or standalone >= .54:
            selected.append(i)
            covered.update(f["inter"])
        if len(selected) >= 6 or sum(fs[0]["tw"].get(a, 1) for a in covered) / max(1, totalw) >= .94:
            break
    if not selected and ranked:
        selected = [ranked[0]]
    joint = len(selected) > 1
    pred = {cands[i]["id"]: b.relation_of(fs[i], target, cands[i]["text"], joint) for i in selected}
    union = set().union(*(fs[i]["inter"] for i in selected)) if selected else set()
    cut = {}
    for i, f in enumerate(fs):
        others = set().union(*(fs[j]["inter"] for j in selected if j != i)) if i in selected else union
        unique = f["inter"] - others
        drop = sum(f["tw"].get(a, 1) for a in unique) / max(1, totalw)
        cut[cands[i]["id"]] = drop + .20*f["score"] + .10*f["entity"]
    meta = {
        "scores": {cands[i]["id"]: fs[i]["score"] for i in range(len(cands))},
        "cut_scores": cut,
        "selected": list(pred),
        "covered_weight": sum(fs[0]["tw"].get(a, 1) for a in union) / max(1, totalw),
        "atom_count": len(target_atoms) + sum(len(f["ca"]) for f in fs),
        "edge_count": sum(len(f["inter"]) for f in fs),
        "index_keys": len(inv),
    }
    return pred, meta


def baseline(row, fs, kind):
    c, t = row["candidates"], row["target"]
    if not c:
        return {}
    if kind == "top1":
        ids = [max(range(len(c)), key=lambda i: (fs[i]["score"], -i))]
    elif kind == "first":
        ids = [0]
    elif kind == "overlap":
        ids = [i for i, f in enumerate(fs) if f["wrec"] >= .32 or f["score"] >= .46]
        if not ids:
            ids = [max(range(len(c)), key=lambda i: fs[i]["score"])]
        ids = sorted(ids, key=lambda i: fs[i]["score"], reverse=True)[:6]
    elif kind == "all":
        ids = list(range(len(c)))
    else:
        raise ValueError(kind)
    return {c[i]["id"]: b.relation_of(fs[i], t, c[i]["text"], len(ids) > 1) for i in ids}


def predict(blind: Path, out: Path):
    rows = list(b.load_jsonl(blind))
    cached = []
    tfeat = time.perf_counter()
    for r in rows:
        fs = [b.feature(r["target"], c["text"]) for c in r["candidates"]]
        cached.append((r, fs, rewired_features(r, fs)))
    feature_seconds = time.perf_counter() - tfeat

    preds, circuit_bytes = [], 0
    t0 = time.perf_counter()
    for r, fs, rwfs in cached:
        p, m = select(r, fs, True)
        rw, _ = select(r, rwfs, True)
        preds.append({
            "rid": r["rid"], "group": r["group"], "circuit": p, "rewired": rw,
            "top1": baseline(r, fs, "top1"), "first": baseline(r, fs, "first"),
            "overlap": baseline(r, fs, "overlap"), "all": baseline(r, fs, "all"),
            "scores": m["scores"], "cut_scores": m["cut_scores"],
            "circuit_meta": {k: m[k] for k in ("selected", "covered_weight", "atom_count", "edge_count", "index_keys")},
        })
        circuit_bytes += len(json.dumps({"a": m["atom_count"], "e": m["edge_count"], "i": m["index_keys"], "p": p}, ensure_ascii=False).encode())
    production_seconds = time.perf_counter() - t0

    # Fair route/index audit: identical selection work, features already compiled once.
    t_index = time.perf_counter()
    indexed_audit = [select(r, fs, True)[0] for r, fs, _ in cached]
    indexed_seconds = time.perf_counter() - t_index
    t_naive = time.perf_counter()
    naive = [select(r, fs, False)[0] for r, fs, _ in cached]
    naive_seconds = time.perf_counter() - t_naive
    identical = all(x == y for x, y in zip(indexed_audit, naive)) and all(x["circuit"] == y for x, y in zip(preds, indexed_audit))
    payload = {
        "model": "Counterfactual_Provenance_Obligation_Circuit_V2",
        "predictions": preds,
        "storage_runtime": {
            "raw_bytes": blind.stat().st_size, "circuit_bytes": circuit_bytes,
            "storage_ratio": circuit_bytes / max(1, blind.stat().st_size),
            "feature_seconds": feature_seconds, "production_seconds": production_seconds,
            "indexed_seconds": indexed_seconds, "naive_seconds": naive_seconds,
            "speedup": naive_seconds / max(indexed_seconds, 1e-9),
            "outputs_identical": identical,
        },
        "forbidden": {"embedding": False, "tfidf": False, "pretrained_encoder": False, "llm": False, "nearest_vector": False, "external_solver": False},
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    Path(str(out) + ".sha256").write_text(hashlib.sha256(out.read_bytes()).hexdigest())
    print(json.dumps(payload["storage_runtime"], indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", required=True)
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    predict(Path(x.blind), Path(x.out))


if __name__ == "__main__":
    main()
