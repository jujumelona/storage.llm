#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, random, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

SEED = 20260725
METHODS = ("circuit", "rewired", "top1", "first", "overlap", "all")


def prf(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f = 2*p*r / max(1e-12, p+r)
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


class Acc:
    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.rtp = self.rfp = self.rfn = 0
        self.macro = []
    def add(self, pred, gold):
        ps, gs = set(pred), set(gold)
        a, b, c = len(ps & gs), len(ps - gs), len(gs - ps)
        self.tp += a; self.fp += b; self.fn += c
        self.macro.append(prf(a,b,c)["f1"])
        for k,v in pred.items():
            if k in gold and gold[k] == v: self.rtp += 1
            else: self.rfp += 1
        self.rfn += sum(1 for k,v in gold.items() if k not in pred or pred.get(k) != v)
    def result(self):
        x = prf(self.tp, self.fp, self.fn)
        x["macro_f1"] = statistics.mean(self.macro) if self.macro else 0.0
        x["relation"] = prf(self.rtp, self.rfp, self.rfn)
        return x


def auc(scores, labels):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.bool_)
    npos = int(y.sum()); nneg = int(len(y) - npos)
    if npos == 0 or nneg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ss = s[order]; yy = y[order]
    rank_sum = 0.0; i = 0; n = len(ss)
    while i < n:
        j = i + 1
        while j < n and ss[j] == ss[i]: j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * int(yy[i:j].sum())
        i = j
    return float((rank_sum - npos*(npos+1)/2.0) / (npos*nneg))


def claim_f1(pred, gold):
    p, g = set(pred), set(gold)
    return prf(len(p&g), len(p-g), len(g-p))["f1"]


def bootstrap(deltas, n=5000):
    rng = random.Random(SEED)
    vals = list(deltas)
    boots = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals) for _ in range(n))
    return {"mean": statistics.mean(vals), "ci95": [boots[int(.025*n)], boots[int(.975*n)]]}


def evaluate(predictions: Path, pred_manifest: Path, labels_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    expected = Path(str(predictions) + ".sha256").read_text().strip()
    actual = hashlib.sha256(predictions.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError("prediction hash mismatch")
    runtime = json.loads(pred_manifest.read_text())
    labels = {x["rid"]: x for x in json.loads(labels_path.read_text())}

    acc = {k: Acc() for k in METHODS}
    group_acc = defaultdict(lambda: {k: Acc() for k in METHODS})
    per_claim = []
    scores = []; cuts = []; ys = []
    slim_path = outdir / "predictions_with_gold.jsonl"
    claims = pairs = positives = 0

    with predictions.open(encoding="utf-8") as src, slim_path.open("w", encoding="utf-8") as slim:
        for line in src:
            if not line.strip(): continue
            row = json.loads(line)
            lab = labels.get(row["rid"])
            if not lab: continue
            gold = lab["ground_truth"]
            claims += 1
            for k in METHODS:
                acc[k].add(row[k], gold)
                group_acc[row["group"]][k].add(row[k], gold)
            best_row = {k: claim_f1(row[k], gold) for k in METHODS}
            per_claim.append(best_row)
            ids = row["candidate_ids"]
            pairs += len(ids)
            for cid, sc, cut in zip(ids, row["scores"], row["cut_scores"]):
                y = cid in gold
                scores.append(sc); cuts.append(cut); ys.append(y); positives += int(y)
            slim.write(json.dumps({
                "rid": row["rid"], "group": row["group"], "gold": gold,
                **{k: row[k] for k in METHODS}, "circuit_meta": row["circuit_meta"]
            }, ensure_ascii=False, separators=(",", ":")) + "\n")

    methods = {k: acc[k].result() for k in METHODS}
    baselines = ["top1", "first", "overlap", "all"]
    best = max(baselines, key=lambda k: methods[k]["f1"])
    deltas = [x["circuit"] - x[best] for x in per_claim]
    groups = {g: {k: a.result() for k,a in ks.items()} for g,ks in group_acc.items()}
    summary = {
        "claims": claims, "candidate_pairs": pairs, "positive_pairs": positives,
        "positive_rate": positives / max(1,pairs), "methods": methods,
        "best_baseline": best, "bootstrap_circuit_minus_best": bootstrap(deltas),
        "link_score_auc": auc(scores,ys), "deletion_cut_auc": auc(cuts,ys),
        "rewiring_drop": methods["circuit"]["f1"] - methods["rewired"]["f1"],
        "groups": groups, "storage_runtime": runtime["storage_runtime"],
    }
    summary["strict"] = {
        "hash_verified": expected == actual,
        "n_claims_ge_500": claims >= 500,
        "beats_best_by_3pp": methods["circuit"]["f1"] >= methods[best]["f1"] + .03,
        "bootstrap_lower_gt_0": summary["bootstrap_circuit_minus_best"]["ci95"][0] > 0,
        "link_auc_ge_070": summary["link_score_auc"] >= .70,
        "cut_auc_ge_070": summary["deletion_cut_auc"] >= .70,
        "rewiring_drop_ge_03": summary["rewiring_drop"] >= .03,
        "storage_ratio_lt_half": runtime["storage_runtime"]["storage_ratio"] < .5,
        "indexed_exact": runtime["storage_runtime"]["outputs_identical"],
        "indexed_speedup_gt_12": runtime["storage_runtime"]["speedup"] > 1.2,
    }
    summary["overall_pass"] = all(summary["strict"].values())
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--pred-manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    x = ap.parse_args()
    evaluate(Path(x.predictions), Path(x.pred_manifest), Path(x.labels), Path(x.outdir))


if __name__ == "__main__":
    main()
