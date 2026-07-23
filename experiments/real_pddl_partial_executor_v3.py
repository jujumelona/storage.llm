#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pddlpy
import real_pddl_partial_executor_v2 as base


def atom_parts(atom: Any) -> Tuple[str, Tuple[str, ...]]:
    if hasattr(atom, "predicate"):
        parts = list(atom.predicate)
    elif isinstance(atom, (tuple, list)):
        parts = list(atom)
    else:
        raise TypeError(f"unsupported pddlpy atom: type={type(atom)!r}, value={atom!r}")
    if not parts:
        raise TypeError(f"empty pddlpy atom: {atom!r}")
    return str(parts[0]), tuple(str(x) for x in parts[1:])


def atom_key(atom: Any) -> Tuple[str, ...]:
    predicate, args = atom_parts(atom)
    return (predicate, *args)


def static_atom_truth(atom: Any, initial_keys: Set[Tuple[str, ...]]) -> bool:
    predicate, args = atom_parts(atom)
    if predicate == "=":
        return len(args) == 2 and args[0] == args[1]
    return (predicate, *args) in initial_keys


def build_semantic_dynamic_universe(
    problem: Any,
    domain_file: Path,
    problem_file: Path,
    up_dynamic: Set[Any],
) -> Tuple[List[Any], Dict[str, Any]]:
    dp = pddlpy.DomainProblem(str(domain_file), str(problem_file))
    initial_atoms = list(dp.initialstate())
    initial_keys = {atom_key(atom) for atom in initial_atoms}
    grounded = []
    dynamic_predicates: Set[str] = set()

    for operator_name in sorted(dp.operators()):
        for op in dp.ground_operator(operator_name):
            grounded.append(op)
            for attr in ("effect_pos", "effect_neg"):
                for atom in list(getattr(op, attr, [])):
                    dynamic_predicates.add(atom_parts(atom)[0])

    support_keys: Set[Tuple[str, ...]] = set()
    admitted = 0
    rejected_static = 0
    for op in grounded:
        pre_pos = list(getattr(op, "precondition_pos", []))
        pre_neg = list(getattr(op, "precondition_neg", []))
        static_pos = [a for a in pre_pos if atom_parts(a)[0] not in dynamic_predicates]
        static_neg = [a for a in pre_neg if atom_parts(a)[0] not in dynamic_predicates]
        static_ok = all(static_atom_truth(a, initial_keys) for a in static_pos) and all(
            not static_atom_truth(a, initial_keys) for a in static_neg
        )
        if not static_ok:
            rejected_static += 1
            continue
        admitted += 1
        for attr in ("precondition_pos", "precondition_neg", "effect_pos", "effect_neg"):
            for atom in list(getattr(op, attr, [])):
                if atom_parts(atom)[0] in dynamic_predicates:
                    support_keys.add(atom_key(atom))

    for key in initial_keys:
        if key[0] in dynamic_predicates:
            support_keys.add(key)

    universe: Set[Any] = set()
    conversion_errors = []
    for key in sorted(support_keys):
        try:
            fexp = base.atom_to_up_fluent(problem, key)
            if fexp.fluent() in up_dynamic:
                universe.add(fexp)
        except Exception as exc:
            conversion_errors.append({"atom": repr(key), "error": f"{type(exc).__name__}: {exc}"})

    for goal in problem.goals:
        for fexp in base.fluent_leaves(goal.simplify()):
            if fexp.fluent() in up_dynamic:
                universe.add(fexp)

    if conversion_errors:
        raise RuntimeError(f"failed to convert dynamic support atoms: {conversion_errors[:5]}")

    stats = {
        "raw_ground_action_bindings": len(grounded),
        "statically_admitted_bindings": admitted,
        "statically_rejected_bindings": rejected_static,
        "dynamic_predicates": sorted(dynamic_predicates),
        "support_atom_count": len(support_keys),
        "semantic_dynamic_atom_count": len(universe),
        "initial_atom_count": len(initial_keys),
    }
    return sorted(universe, key=str), stats


base.atom_parts = atom_parts
base.static_atom_truth = static_atom_truth
base.build_semantic_dynamic_universe = build_semantic_dynamic_universe


if __name__ == "__main__":
    raise SystemExit(base.main())
