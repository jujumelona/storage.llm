#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pddlpy

SIFT_COMMIT = "0f4518ffa609322a24521609f80d70728f07ac09"
TARGETS = {
    "gripper": ("gripper",),
    "ferry": ("ferry",),
    "delivery": ("delivery",),
    "hanoi": ("hanoi",),
    "n-puzzle": ("n-puzzle", "npuzzle", "puzzle"),
    "sokoban": ("sokoban",),
}
GROUND_CAP = 250_000


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def classify(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")[:4096].lower()
    if re.search(r"\(\s*define\s*\(\s*domain\b", text):
        return "domain"
    if re.search(r"\(\s*define\s*\(\s*problem\b", text):
        return "problem"
    return "unknown"


def pred_name(atom: Any) -> str:
    # pddlpy atoms are often tuples or objects with predicate attr.
    if isinstance(atom, tuple) and atom:
        return str(atom[0])
    for attr in ("predicate", "name"):
        if hasattr(atom, attr):
            value = getattr(atom, attr)
            if hasattr(value, "name"):
                return str(value.name)
            return str(value)
    text = str(atom).strip()
    text = text.lstrip("(")
    return text.split()[0] if text else ""


def atom_repr(atom: Any) -> str:
    if isinstance(atom, (tuple, list)):
        return "(" + " ".join(map(str, atom)) + ")"
    return str(atom)


def safe_list(value: Iterable[Any]) -> List[Any]:
    try:
        return list(value)
    except TypeError:
        return []


def inspect_pair(domain_file: Path, problem_file: Path) -> Dict[str, Any]:
    started = time.time()
    dp = pddlpy.DomainProblem(str(domain_file), str(problem_file))
    operators = sorted(list(dp.operators()))
    objects = dp.worldobjects()
    init = safe_list(dp.initialstate())
    goals = safe_list(dp.goals())

    effect_preds = set()
    precondition_preds = set()
    operator_stats = {}
    total_ground = 0
    truncated = False
    samples = []

    for op_name in operators:
        count = 0
        op_samples = []
        for grounded in dp.ground_operator(op_name):
            count += 1
            total_ground += 1
            for attr in ("effect_pos", "effect_neg"):
                for atom in safe_list(getattr(grounded, attr, [])):
                    effect_preds.add(pred_name(atom))
            for attr in ("precondition_pos", "precondition_neg"):
                for atom in safe_list(getattr(grounded, attr, [])):
                    precondition_preds.add(pred_name(atom))
            if len(op_samples) < 2:
                op_samples.append({
                    "operator": op_name,
                    "variable_list": {str(k): str(v) for k, v in dict(getattr(grounded, "variable_list", {})).items()},
                    "pre_pos": [atom_repr(x) for x in safe_list(getattr(grounded, "precondition_pos", []))],
                    "pre_neg": [atom_repr(x) for x in safe_list(getattr(grounded, "precondition_neg", []))],
                    "eff_pos": [atom_repr(x) for x in safe_list(getattr(grounded, "effect_pos", []))],
                    "eff_neg": [atom_repr(x) for x in safe_list(getattr(grounded, "effect_neg", []))],
                })
            if total_ground >= GROUND_CAP:
                truncated = True
                break
        operator_stats[op_name] = {"ground_count": count, "samples": op_samples}
        samples.extend(op_samples)
        if truncated:
            break

    init_preds = {pred_name(a) for a in init}
    static_preds = sorted(p for p in init_preds if p not in effect_preds)
    dynamic_preds = sorted(effect_preds)

    return {
        "domain_file": str(domain_file),
        "problem_file": str(problem_file),
        "objects": {str(k): str(v) for k, v in dict(objects).items()},
        "object_count": len(objects),
        "operators": operators,
        "operator_stats": operator_stats,
        "ground_action_count": total_ground,
        "grounding_truncated_at": GROUND_CAP if truncated else None,
        "initial_atom_count": len(init),
        "goal_atom_count": len(goals),
        "goals": [atom_repr(g) for g in goals],
        "dynamic_predicates": dynamic_preds,
        "static_predicates": static_preds,
        "precondition_predicates": sorted(precondition_preds),
        "elapsed_sec": round(time.time() - started, 6),
    }


def choose_target_dirs(root: Path) -> Dict[str, Path]:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    result: Dict[str, Path] = {}
    for target, aliases in TARGETS.items():
        ranked = []
        for d in dirs:
            nd = norm(d.name)
            score = max((len(norm(a)) if norm(a) in nd or nd in norm(a) else 0) for a in aliases)
            if score:
                ranked.append((score, -len(d.name), d))
        if ranked:
            result[target] = sorted(ranked, reverse=True)[0][2]
    return result


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: real_pddl_inventory.py SIFT_ROOT OUTPUT_DIR")
    sift_root = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pddl_root = sift_root / "pddl_files"
    if not pddl_root.is_dir():
        raise RuntimeError(f"missing pddl_files: {pddl_root}")

    selected_dirs = choose_target_dirs(pddl_root)
    result: Dict[str, Any] = {
        "source_repo": "https://github.com/JonasGoesgens/SIFT",
        "source_commit": SIFT_COMMIT,
        "parser": "pddlpy",
        "ground_cap": GROUND_CAP,
        "selected_dirs": {k: str(v.relative_to(sift_root)) for k, v in selected_dirs.items()},
        "domains": {},
    }

    for target in TARGETS:
        entry: Dict[str, Any] = {"status": "missing_directory", "pairs": [], "errors": []}
        result["domains"][target] = entry
        folder = selected_dirs.get(target)
        if folder is None:
            continue
        files = sorted(folder.rglob("*.pddl"))
        domains = [p for p in files if classify(p) == "domain"]
        problems = [p for p in files if classify(p) == "problem"]
        entry.update({
            "status": "no_domain_or_problem",
            "directory": str(folder.relative_to(sift_root)),
            "domain_candidates": [str(p.relative_to(sift_root)) for p in domains],
            "problem_candidates": [str(p.relative_to(sift_root)) for p in problems],
        })
        best_domain: Optional[Path] = None
        best_pairs: List[Dict[str, Any]] = []
        for domain_file in domains:
            parsed: List[Dict[str, Any]] = []
            for problem_file in problems:
                try:
                    parsed.append(inspect_pair(domain_file, problem_file))
                except Exception as exc:
                    entry["errors"].append({
                        "domain": str(domain_file.relative_to(sift_root)),
                        "problem": str(problem_file.relative_to(sift_root)),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            if len(parsed) > len(best_pairs):
                best_domain = domain_file
                best_pairs = parsed
        if best_domain is not None and best_pairs:
            entry["status"] = "parsed"
            entry["selected_domain"] = str(best_domain.relative_to(sift_root))
            entry["pairs"] = best_pairs
        else:
            entry["status"] = "parse_failed"

    parsed_domains = sum(1 for e in result["domains"].values() if e["status"] == "parsed")
    result["summary"] = {
        "target_domain_count": len(TARGETS),
        "parsed_domain_count": parsed_domains,
        "all_six_parsed": parsed_domains == len(TARGETS),
        "total_parseable_problems": sum(len(e.get("pairs", [])) for e in result["domains"].values()),
    }

    json_path = out_dir / "REAL_PDDL_INVENTORY.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Real SIFT PDDL parser/grounding inventory",
        "",
        f"- Commit: `{SIFT_COMMIT}`",
        f"- Parsed domains: **{parsed_domains}/{len(TARGETS)}**",
        f"- Parseable problems: **{result['summary']['total_parseable_problems']}**",
        "",
        "| Domain | Status | Parseable problems | Selected domain |",
        "|---|---:|---:|---|",
    ]
    for name, e in result["domains"].items():
        lines.append(f"| {name} | {e['status']} | {len(e.get('pairs', []))} | `{e.get('selected_domain', '')}` |")
    lines.extend(["", "## Problems", ""])
    for name, e in result["domains"].items():
        lines.append(f"### {name}")
        for pair in e.get("pairs", []):
            lines.append(
                f"- `{Path(pair['problem_file']).name}`: objects={pair['object_count']}, "
                f"ground_actions={pair['ground_action_count']}, init={pair['initial_atom_count']}, "
                f"goals={pair['goal_atom_count']}, truncated={pair['grounding_truncated_at'] is not None}"
            )
        lines.append("")
    (out_dir / "REAL_PDDL_INVENTORY.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(result["summary"], indent=2))
    return 0 if parsed_domains == len(TARGETS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
