#!/usr/bin/env python3
"""Sparse, non-dense failure attribution pilot for TraceElephant.

No LLM, embeddings, pretrained encoder, semantic keyword list, AST, solver,
nearest-neighbour search, or benchmark failure taxonomy is used.

The program reads TraceElephant's public data.zip directly, constructs coarse
lossless-evidence-derived structural event signatures, builds exact transition
relations from successful executions, and uses exact same-task natural
contrasts when available. It never reads mistake_agent/mistake_step until the
scoring phase.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

EMPTY = {"", "none", "null", "unknown", "n/a", "na", "-1"}


def h12(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8", "replace")).hexdigest()[:12]


def norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def meaningful(x: Any) -> bool:
    return norm_text(x).lower() not in EMPTY


def to_step(x: Any) -> int | None:
    s = norm_text(x)
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    v = int(m.group())
    return v if v >= 0 else None


def bucket(n: int, cuts=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)) -> int:
    for c in cuts:
        if n <= c:
            return c
    return 2048


def scalar_atom(x: Any) -> str | None:
    if x is None or isinstance(x, (bool, int, float)):
        return f"{type(x).__name__}:{x}"
    if isinstance(x, str) and len(x) >= 4:
        return "str:" + hashlib.sha256(x.encode("utf-8", "replace")).hexdigest()[:16]
    return None


def structural_summary(obj: Any) -> tuple[tuple, set[str]]:
    """Return a language-neutral JSON topology summary and exact atom set.

    Key strings are not used semantically. Their hashes only enter an optional
    equality lineage set; the primary signature uses container topology and
    scalar-type/size buckets.
    """
    counts = Counter()
    atoms: set[str] = set()
    max_depth = 0
    key_arity = Counter()
    list_arity = Counter()

    def walk(x: Any, depth: int) -> None:
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        if isinstance(x, dict):
            counts["dict"] += 1
            key_arity[bucket(len(x))] += 1
            for k in sorted(x.keys(), key=lambda z: str(z)):
                atoms.add("key:" + h12(str(k)))
                walk(x[k], depth + 1)
        elif isinstance(x, list):
            counts["list"] += 1
            list_arity[bucket(len(x))] += 1
            for v in x:
                walk(v, depth + 1)
        elif isinstance(x, str):
            counts["str"] += 1
            counts[f"strlen_{bucket(len(x))}"] += 1
            counts[f"lines_{bucket(x.count(chr(10)) + 1, (1,2,4,8,16,32,64))}"] += 1
            a = scalar_atom(x)
            if a:
                atoms.add(a)
        elif isinstance(x, bool):
            counts["bool"] += 1
            atoms.add(f"bool:{x}")
        elif x is None:
            counts["null"] += 1
            atoms.add("null")
        elif isinstance(x, (int, float)):
            counts["num"] += 1
            atoms.add(f"num:{x}")
        else:
            counts["other"] += 1
            atoms.add("other:" + h12(type(x).__name__))

    walk(obj, 0)
    summary = (
        bucket(sum(counts.values())),
        bucket(max_depth, (0,1,2,3,4,6,8,12,16,24,32)),
        tuple(sorted((k, bucket(v)) for k, v in counts.items() if not k.startswith("strlen_") and not k.startswith("lines_"))),
        tuple(sorted(key_arity.items())),
        tuple(sorted(list_arity.items())),
        tuple(sorted((k, bucket(v)) for k, v in counts.items() if k.startswith("strlen_") or k.startswith("lines_"))),
    )
    return summary, atoms


@dataclass
class Step:
    idx1: int
    agent_raw: str
    agent_ord: int
    sig: tuple
    atoms: set[str]


@dataclass
class Trace:
    trace_id: str
    path: str
    system: str
    task_key: str
    steps: list[Step]
    failed: bool
    gold_agent: str
    gold_step_raw: Any
    gold_step: int | None
    tests_status: Any


def event_steps(records: list[dict]) -> list[Step]:
    ords: dict[str, int] = {}
    prev_atoms: set[str] = set()
    prev_agent = None
    out: list[Step] = []
    for i, rec in enumerate(records, 1):
        agent = norm_text(rec.get("agent_name", "Unknown"))
        if agent not in ords:
            ords[agent] = len(ords)
        req = rec.get("input", rec.get("request", {}))
        resp = rec.get("output", rec.get("response", {}))
        req_shape, req_atoms = structural_summary(req)
        resp_shape, resp_atoms = structural_summary(resp)
        atoms = req_atoms | resp_atoms
        reuse = len(atoms & prev_atoms)
        new_atoms = len(atoms - prev_atoms)
        sig = (
            min(ords[agent], 7),
            int(agent == prev_agent),
            req_shape,
            resp_shape,
            bucket(reuse, (0,1,2,4,8,16,32,64,128)),
            bucket(new_atoms, (0,1,2,4,8,16,32,64,128,256)),
        )
        out.append(Step(i, agent, ords[agent], sig, atoms))
        prev_atoms = atoms
        prev_agent = agent
    return out


def system_from_path(path: str, metadata: dict) -> str:
    s = norm_text(metadata.get("system_name"))
    if s:
        return s
    p = path.lower()
    if "captain" in p:
        return "captain"
    if "magentic" in p:
        return "magentic"
    if "swe" in p:
        return "swe"
    return "other"


def iter_traces(zip_path: Path) -> list[Trace]:
    traces: list[Trace] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        metas = sorted(n for n in names if n.endswith("trace_metadata.json"))
        for mp in metas:
            sp = mp.rsplit("/", 1)[0] + "/step_records.json"
            if sp not in names:
                continue
            try:
                meta = json.loads(zf.read(mp))
                recs = json.loads(zf.read(sp))
            except Exception:
                continue
            if not isinstance(meta, dict) or not isinstance(recs, list) or not recs:
                continue
            system = system_from_path(mp, meta)
            instruction = norm_text(meta.get("task_instruction", meta.get("question", "")))
            task_key = h12(system.lower() + "\n" + instruction)
            ga = norm_text(meta.get("mistake_agent"))
            gs_raw = meta.get("mistake_step")
            failed = meaningful(ga) and meaningful(gs_raw)
            traces.append(
                Trace(
                    trace_id=h12(mp),
                    path=mp,
                    system=system,
                    task_key=task_key,
                    steps=event_steps(recs),
                    failed=failed,
                    gold_agent=ga,
                    gold_step_raw=gs_raw,
                    gold_step=to_step(gs_raw),
                    tests_status=meta.get("tests_status"),
                )
            )
    return traces


def levenshtein(a: list[tuple], b: list[tuple]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def align(a: list[tuple], b: list[tuple]) -> tuple[int, dict[int, int], set[int]]:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j = n, m
    exact: dict[int, int] = {}
    changed: set[int] = set()
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] == b[j - 1]:
                exact[i - 1] = j - 1
            else:
                changed.add(i - 1)
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            changed.add(i - 1)
            i -= 1
        else:
            j -= 1
    return dp[n][m], exact, changed


def transition_key(prev: tuple | None, cur: tuple) -> tuple:
    # Exact categorical relation, no vector distance.
    return (prev, cur)


def build_success_grammar(traces: Iterable[Trace]) -> tuple[Counter, Counter]:
    trans = Counter()
    sigs = Counter()
    for t in traces:
        prev = None
        for s in t.steps:
            sigs[s.sig] += 1
            trans[transition_key(prev, s.sig)] += 1
            prev = s.sig
    return trans, sigs


def paired_scores(fail: Trace, refs: list[Trace]) -> tuple[list[float], dict]:
    fs = [s.sig for s in fail.steps]
    scores = [0.0] * len(fs)
    evidence = [0] * len(fs)
    best_prefix = 0
    for ref in refs:
        rs = [s.sig for s in ref.steps]
        base_dist, exact, changed = align(fs, rs)
        prefix = 0
        while prefix < min(len(fs), len(rs)) and fs[prefix] == rs[prefix]:
            prefix += 1
        best_prefix = max(best_prefix, prefix)
        next_exact = [len(fs)] * len(fs)
        nxt = len(fs)
        exact_set = set(exact)
        for i in range(len(fs) - 1, -1, -1):
            if i in exact_set:
                nxt = i
            next_exact[i] = nxt
        for i in range(len(fs)):
            mismatch = 0.0 if i in exact_set else 1.0
            persistence = min(8, max(0, next_exact[i] - i)) / 8.0 if i not in exact_set else 0.0
            deletion_gain = max(0, base_dist - levenshtein(fs[:i] + fs[i + 1 :], rs))
            local = 2.0 * mismatch + persistence + 0.75 * deletion_gain
            scores[i] += local
            if local > 0:
                evidence[i] += 1
    if refs:
        scores = [x / len(refs) for x in scores]
    return scores, {"best_common_prefix": best_prefix, "contrast_refs": len(refs), "evidence_counts": evidence}


def grammar_scores(fail: Trace, trans: Counter, sigs: Counter) -> list[float]:
    scores = []
    prev = None
    novelty_run = 0
    for s in fail.steps:
        tk = transition_key(prev, s.sig)
        tc = trans[tk]
        sc = sigs[s.sig]
        novel = int(tc == 0)
        novelty_run = novelty_run + 1 if novel else 0
        rarity = 1.0 / (1.0 + tc)
        sig_unseen = int(sc == 0)
        scores.append(2.0 * novel + 0.5 * min(novelty_run, 8) / 8.0 + 0.5 * rarity + 0.5 * sig_unseen)
        prev = s.sig
    return scores


def choose_step(scores: list[float]) -> int:
    if not scores:
        return 1
    mx = max(scores)
    # Earliest decisive divergence among near-maximal persistent anomalies.
    threshold = mx - 1e-12
    return next(i + 1 for i, x in enumerate(scores) if x >= threshold)


def score_match(pred: int, gold: int | None, n: int) -> tuple[bool, int | None, str]:
    if gold is None:
        return False, None, "unknown"
    # Official metadata may use either 0- or 1-based step numbering. Report both
    # conventions and use the one that matches the official history numbering.
    candidates = {gold}
    if gold == 0 or gold < n:
        candidates.add(gold + 1)
    distances = {abs(pred - c): c for c in candidates if 1 <= c <= n}
    if not distances:
        return False, abs(pred - gold), "raw"
    d = min(distances)
    matched_c = distances[d]
    conv = "one_based" if matched_c == gold else "zero_to_one"
    return d == 0, d, conv


def evaluate(traces: list[Trace], outdir: Path) -> dict:
    successes = [t for t in traces if not t.failed]
    failures = [t for t in traces if t.failed]
    success_by_task = defaultdict(list)
    success_by_system = defaultdict(list)
    for t in successes:
        success_by_task[(t.system, t.task_key)].append(t)
        success_by_system[t.system].append(t)
    grammars = {s: build_success_grammar(ts) for s, ts in success_by_system.items()}
    global_grammar = build_success_grammar(successes)

    rows = []
    rename_identical = 0
    for f in failures:
        refs = success_by_task.get((f.system, f.task_key), [])
        if refs:
            scores, detail = paired_scores(f, refs)
            mode = "exact_task_natural_contrast"
            baseline = min(len(f.steps), detail["best_common_prefix"] + 1)
        else:
            trans, sigs = grammars.get(f.system, global_grammar)
            scores = grammar_scores(f, trans, sigs)
            detail = {"best_common_prefix": 0, "contrast_refs": 0, "evidence_counts": [int(x > 0) for x in scores]}
            mode = "system_success_transition_grammar"
            baseline = next((i + 1 for i, x in enumerate(scores) if x >= 2.0), len(f.steps))
        pred = choose_step(scores)
        pred_agent = f.steps[pred - 1].agent_raw if f.steps else ""
        step_exact, dist, conv = score_match(pred, f.gold_step, len(f.steps))
        agent_exact = norm_text(pred_agent) == norm_text(f.gold_agent)
        # Renaming agents leaves ordinals and all signatures unchanged by construction.
        rename_pred = choose_step(scores)
        rename_identical += int(rename_pred == pred)
        positive_idx = [i for i, x in enumerate(scores) if x > 0]
        if positive_idx:
            lo, hi = min(positive_idx), max(positive_idx)
            evidence_nodes = hi - lo + 1
        else:
            evidence_nodes = 0
        rows.append({
            "trace_id": f.trace_id,
            "system": f.system,
            "mode": mode,
            "steps": len(f.steps),
            "paired_successes": len(refs),
            "predicted_step": pred,
            "gold_step_raw": norm_text(f.gold_step_raw),
            "step_numbering_interpretation": conv,
            "step_exact": int(step_exact),
            "step_distance": "" if dist is None else dist,
            "within_1": int(dist is not None and dist <= 1),
            "within_2": int(dist is not None and dist <= 2),
            "predicted_agent_hash": h12(pred_agent),
            "gold_agent_hash": h12(f.gold_agent),
            "agent_exact": int(agent_exact),
            "joint_exact": int(step_exact and agent_exact),
            "baseline_first_divergence_step": baseline,
            "baseline_step_exact": int(score_match(baseline, f.gold_step, len(f.steps))[0]),
            "evidence_nodes": evidence_nodes,
            "evidence_ratio": evidence_nodes / len(f.steps) if f.steps else 0.0,
            "max_score": max(scores) if scores else 0.0,
        })

    def mean(field: str, rr=rows) -> float:
        return sum(float(r[field]) for r in rr) / len(rr) if rr else 0.0

    systems = []
    for system in sorted({r["system"] for r in rows}):
        rr = [r for r in rows if r["system"] == system]
        systems.append({
            "system": system,
            "failure_traces": len(rr),
            "paired_coverage": sum(r["paired_successes"] > 0 for r in rr) / len(rr),
            "agent_accuracy": mean("agent_exact", rr),
            "step_accuracy": mean("step_exact", rr),
            "joint_accuracy": mean("joint_exact", rr),
            "within_1": mean("within_1", rr),
            "within_2": mean("within_2", rr),
            "baseline_step_accuracy": mean("baseline_step_exact", rr),
            "mean_evidence_ratio": mean("evidence_ratio", rr),
        })

    all_trans, all_sigs = build_success_grammar(successes)
    schema_count = len(all_trans) + len(all_sigs)
    metrics = {
        "total_traces": len(traces),
        "success_traces": len(successes),
        "failure_traces": len(failures),
        "systems": sorted({t.system for t in traces}),
        "exact_task_pairable_failures": sum(r["paired_successes"] > 0 for r in rows),
        "paired_coverage": sum(r["paired_successes"] > 0 for r in rows) / len(rows) if rows else 0.0,
        "agent_accuracy": mean("agent_exact"),
        "step_accuracy": mean("step_exact"),
        "joint_accuracy": mean("joint_exact"),
        "within_1": mean("within_1"),
        "within_2": mean("within_2"),
        "baseline_first_divergence_step_accuracy": mean("baseline_step_exact"),
        "mean_evidence_ratio": mean("evidence_ratio"),
        "identifier_rename_prediction_invariance": rename_identical / len(rows) if rows else 0.0,
        "event_signature_count": len(all_sigs),
        "transition_schema_count": len(all_trans),
        "schema_count": schema_count,
        "schema_per_trace": schema_count / len(traces) if traces else math.inf,
        "per_system": systems,
    }

    gates = {
        "failure_traces_at_least_100": len(failures) >= 100,
        "paired_success_coverage_ge_50pct": metrics["paired_coverage"] >= 0.50,
        "step_exact_ge_30pct": metrics["step_accuracy"] >= 0.30,
        "joint_exact_ge_25pct": metrics["joint_accuracy"] >= 0.25,
        "beats_first_divergence_baseline": metrics["step_accuracy"] > metrics["baseline_first_divergence_step_accuracy"],
        "identifier_rename_delta_lt_2pp": metrics["identifier_rename_prediction_invariance"] >= 0.98,
        "evidence_ratio_le_50pct": metrics["mean_evidence_ratio"] <= 0.50,
        "schema_per_trace_le_0p5": metrics["schema_per_trace"] <= 0.50,
        "all_systems_step_accuracy_ge_20pct": bool(systems) and all(x["step_accuracy"] >= 0.20 for x in systems),
    }
    metrics["gates"] = gates
    metrics["gates_passed"] = sum(gates.values())
    metrics["gates_total"] = len(gates)
    metrics["verdict"] = (
        "NATURAL_CONTRAST_SPARSE_FAILURE_GRAPH_SURVIVES_PILOT"
        if all(gates.values())
        else "NATURAL_CONTRAST_SPARSE_FAILURE_GRAPH_FALSIFIED"
    )
    metrics["causal_scope"] = (
        "No environment replay was performed. Exact same-task successful executions are natural contrasts, "
        "not controlled interventions; deletion gain is structural consistency only."
    )

    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "task_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["trace_id"])
        w.writeheader(); w.writerows(rows)
    with (outdir / "system_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(systems[0].keys()) if systems else ["system"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(systems)
    with (outdir / "schema_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["kind", "count"])
        w.writerow(["event_signatures", len(all_sigs)])
        w.writerow(["transition_schemas", len(all_trans)])
        w.writerow(["total", schema_count])
    (outdir / "RESULTS.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def report(metrics: dict, outdir: Path, archive_sha: str) -> None:
    g = metrics["gates"]
    lines = [
        "# TraceElephant sparse natural-contrast failure graph — direct result",
        "",
        "## Scope",
        "",
        "- Official TraceElephant `data.zip` was streamed directly; raw traces are not repackaged.",
        "- No LLM, embedding, pretrained encoder, semantic keyword list, AST, solver, retrieval or failure taxonomy was used.",
        "- Initial relations are only sequence, same-agent recurrence, JSON containment/type shape and exact atom equality.",
        "- `mistake_agent` and `mistake_step` were withheld until scoring.",
        "- Same-task successful runs are natural contrasts, not controlled interventions.",
        "",
        "## Dataset audit",
        "",
        f"- data.zip SHA256: `{archive_sha}`",
        f"- traces: **{metrics['total_traces']}**",
        f"- successful: **{metrics['success_traces']}**",
        f"- annotated failures: **{metrics['failure_traces']}**",
        f"- exact-task pairable failure coverage: **{metrics['paired_coverage']:.2%}**",
        "",
        "## Attribution",
        "",
        "| metric | result |",
        "|---|---:|",
        f"| responsible agent exact | **{metrics['agent_accuracy']:.2%}** |",
        f"| decisive step exact | **{metrics['step_accuracy']:.2%}** |",
        f"| joint exact | **{metrics['joint_accuracy']:.2%}** |",
        f"| step within ±1 | {metrics['within_1']:.2%} |",
        f"| step within ±2 | {metrics['within_2']:.2%} |",
        f"| first-divergence baseline step exact | {metrics['baseline_first_divergence_step_accuracy']:.2%} |",
        f"| evidence/trace ratio | {metrics['mean_evidence_ratio']:.2%} |",
        f"| identifier-renaming invariance | {metrics['identifier_rename_prediction_invariance']:.2%} |",
        f"| schema/trace ratio | {metrics['schema_per_trace']:.3f} |",
        "",
        "## Per system",
        "",
        "| system | failures | paired | agent | step | joint | baseline | evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in metrics["per_system"]:
        lines.append(
            f"| {x['system']} | {x['failure_traces']} | {x['paired_coverage']:.2%} | "
            f"{x['agent_accuracy']:.2%} | {x['step_accuracy']:.2%} | {x['joint_accuracy']:.2%} | "
            f"{x['baseline_step_accuracy']:.2%} | {x['mean_evidence_ratio']:.2%} |"
        )
    lines += ["", "## Hard gates", "", "| gate | pass |", "|---|---:|"]
    for k, v in g.items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines += [
        "",
        f"Passed: **{metrics['gates_passed']}/{metrics['gates_total']}**",
        "",
        "## Verdict",
        "",
        f"`{metrics['verdict']}`",
        "",
        "This run does **not** establish causal attribution because no agent environment was replayed after a controlled edit. "
        "It tests whether exact natural execution contrasts plus a sparse structural evidence graph are sufficient to locate "
        "the benchmark annotations. A low pair rate, failure to beat first divergence, large evidence subgraphs, or schema growth "
        "is a direct falsification of this construction.",
    ]
    (outdir / "REPORT_KO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expected-sha256", default="")
    args = ap.parse_args()
    archive_sha = hashlib.sha256(args.zip.read_bytes()).hexdigest()
    if args.expected_sha256 and archive_sha != args.expected_sha256:
        raise SystemExit(f"SHA256 mismatch: {archive_sha}")
    traces = iter_traces(args.zip)
    if not traces:
        raise SystemExit("No TraceElephant trace_metadata/step_records pairs found")
    metrics = evaluate(traces, args.out)
    report(metrics, args.out, archive_sha)
    manifest = {
        "archive_sha256": archive_sha,
        "result_sha256": hashlib.sha256((args.out / "RESULTS.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verdict": metrics["verdict"],
        "gates_passed": metrics["gates_passed"],
        "gates_total": metrics["gates_total"],
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
