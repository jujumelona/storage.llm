#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACCEL_PATH = HERE / "provenance_obligation_trove_bitset.py"
spec = importlib.util.spec_from_file_location("trove_accel", ACCEL_PATH)
a = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(a)
f = a.f


def run(blind: Path, output: Path, manifest: Path):
    raw_bytes = blind.stat().st_size
    circuit_bytes = 0
    feature_seconds = production_seconds = indexed_seconds = naive_seconds = 0.0
    outputs_identical = True
    rows = candidate_pairs = 0

    with output.open("w", encoding="utf-8") as dst:
        for row in f.b.load_jsonl(blind):
            rows += 1
            cands = row["candidates"]
            candidate_pairs += len(cands)

            t = time.perf_counter()
            fs = [f.b.feature(row["target"], c["text"]) for c in cands]
            rwfs = f.rewired_features(row, fs)
            feature_seconds += time.perf_counter() - t

            t = time.perf_counter()
            circuit, meta = f.select(row, fs, True)
            indexed_seconds += time.perf_counter() - t

            t = time.perf_counter()
            naive, _ = f.select(row, fs, False)
            naive_seconds += time.perf_counter() - t
            outputs_identical = outputs_identical and circuit == naive

            t = time.perf_counter()
            rewired, _ = f.select(row, rwfs, True)
            top1 = f.baseline(row, fs, "top1")
            first = f.baseline(row, fs, "first")
            overlap = f.baseline(row, fs, "overlap")
            all_pred = f.baseline(row, fs, "all")
            production_seconds += time.perf_counter() - t

            ids = [c["id"] for c in cands]
            record = {
                "rid": row["rid"], "group": row["group"],
                "circuit": circuit, "rewired": rewired,
                "top1": top1, "first": first, "overlap": overlap, "all": all_pred,
                "candidate_ids": ids,
                "scores": [meta["scores"][cid] for cid in ids],
                "cut_scores": [meta["cut_scores"][cid] for cid in ids],
                "circuit_meta": {k: meta[k] for k in ("selected", "covered_weight", "atom_count", "edge_count", "index_keys")},
            }
            dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            circuit_bytes += len(json.dumps({"a": meta["atom_count"], "e": meta["edge_count"], "i": meta["index_keys"], "p": circuit}, ensure_ascii=False).encode())
            if rows % 250 == 0:
                print(json.dumps({"rows": rows, "candidate_pairs": candidate_pairs}), flush=True)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    Path(str(output) + ".sha256").write_text(digest)
    info = {
        "model": "Counterfactual_Provenance_Obligation_Circuit_V2_stream_exact",
        "rows": rows, "candidate_pairs": candidate_pairs,
        "prediction_sha256": digest,
        "storage_runtime": {
            "raw_bytes": raw_bytes, "prediction_bytes": output.stat().st_size,
            "circuit_bytes": circuit_bytes, "storage_ratio": circuit_bytes / max(1, raw_bytes),
            "feature_seconds": feature_seconds, "production_seconds": production_seconds,
            "indexed_seconds": indexed_seconds, "naive_seconds": naive_seconds,
            "speedup": naive_seconds / max(indexed_seconds, 1e-9),
            "outputs_identical": outputs_identical,
        },
        "forbidden": {"embedding": False, "tfidf": False, "pretrained_encoder": False, "llm": False, "nearest_vector": False, "external_solver": False},
    }
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest", required=True)
    x = ap.parse_args()
    run(Path(x.blind), Path(x.output), Path(x.manifest))


if __name__ == "__main__":
    main()
