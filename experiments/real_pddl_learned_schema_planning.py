#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.io import PDDLReader
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import OneshotPlanner, SequentialSimulator, get_environment

import real_pddl_trace_schema_equivalence as base

SOLVE_TIMEOUT = 300
SEED = 911773

Atom = base.Atom


def pddl_symbol(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]", "_", text)
    if not value or not value[0].isalpha():
        value = "x_" + value
    return value.lower()


def object_name(node: Any) -> str:
    if hasattr(node, "is_object_exp") and node.is_object_exp():
        return node.object().name
    return str(node)


def fluent_atom(node: Any) -> Atom:
    if not node.is_fluent_exp():
        raise TypeError(f"not a fluent expression: {node}")
    return tuple([node.fluent().name] + [object_name(arg) for arg in node.args])


def collect_goal_literals(node: Any, positive: Set[Atom], negative: Set[Atom]) -> None:
    node = node.simplify()
    if node.is_true():
        return
    if node.is_and():
        for child in node.args:
            collect_goal_literals(child, positive, negative)
        return
    if node.is_not() and node.arg(0).is_fluent_exp():
        negative.add(fluent_atom(node.arg(0)))
        return
    if node.is_fluent_exp():
        positive.add(fluent_atom(node))
        return
    raise NotImplementedError(f"unsupported goal formula: {node}")


def original_goal_literals(domain_file: Path, problem_file: Path) -> Tuple[Set[Atom], Set[Atom]]:
    problem = PDDLReader().parse_problem(str(domain_file), str(problem_file))
    pos: Set[Atom] = set()
    neg: Set[Atom] = set()
    for goal in problem.goals:
        collect_goal_literals(goal, pos, neg)
    return pos, neg


def compiled_action_from_learned(
    model: Mapping[str, Any],
    action: base.GroundAction,
) -> Tuple[Set[Atom], Set[Atom], Set[Atom], Set[Atom]] | None:
    """Instantiate only learned rule/effect fields; never reads original action semantics."""
    key = base.key_string(action.key)
    rule = model["rules"].get(key)
    effect = model["effects"].get(key)
    if rule is None or rule.get("always_false") or effect is None:
        return None

    pre_pos: Set[Atom] = set()
    pre_neg: Set[Atom] = set()
    for raw_feature in rule.get("features", []):
        feature = tuple(raw_feature) if isinstance(raw_feature, tuple) else raw_feature
        sign, pred, roles = feature
        roles = tuple(roles)
        if pred == "__eq__":
            i, j = roles
            holds = action.args[i] == action.args[j]
            required = holds if sign == "+" else not holds
            if not required:
                return None
            continue
        atom = tuple([pred] + [action.args[i] for i in roles])
        (pre_pos if sign == "+" else pre_neg).add(atom)

    add, delete = base.instantiate_effect(effect, action.args)
    return pre_pos, pre_neg, add, delete


def emit_and(parts: Sequence[str]) -> str:
    return "(and)" if not parts else "(and " + " ".join(parts) + ")"


def compile_propositional_problem(
    case_name: str,
    model: Mapping[str, Any],
    test_initial: Set[Atom],
    test_actions: Sequence[base.GroundAction],
    goal_pos: Set[Atom],
    goal_neg: Set[Atom],
    out_dir: Path,
) -> Tuple[Path, Path, Dict[str, base.GroundAction], Dict[str, Any]]:
    # Intentionally restricted to name, args and learned schema. The compiler does
    # not access action.pre_pos/pre_neg/eff_add/eff_del.
    rows: List[Tuple[str, base.GroundAction, Set[Atom], Set[Atom], Set[Atom], Set[Atom]]] = []
    atoms: Set[Atom] = set(test_initial) | set(goal_pos) | set(goal_neg)

    for index, action in enumerate(test_actions):
        learned = compiled_action_from_learned(model, action)
        if learned is None:
            continue
        pre_pos, pre_neg, add, delete = learned
        action_id = f"ga_{index:06d}"
        rows.append((action_id, action, pre_pos, pre_neg, add, delete))
        atoms.update(pre_pos)
        atoms.update(pre_neg)
        atoms.update(add)
        atoms.update(delete)

    atom_list = sorted(atoms)
    atom_name = {atom: f"p_{i:06d}" for i, atom in enumerate(atom_list)}
    action_map: Dict[str, base.GroundAction] = {}

    domain_name = pddl_symbol(f"learned_{case_name}")
    domain_lines = [
        f"(define (domain {domain_name})",
        "  (:requirements :strips :negative-preconditions)",
        "  (:predicates",
    ]
    domain_lines.extend(f"    ({atom_name[atom]})" for atom in atom_list)
    domain_lines.append("  )")

    for action_id, action, pre_pos, pre_neg, add, delete in rows:
        action_map[action_id] = action
        pre_parts = [f"({atom_name[a]})" for a in sorted(pre_pos)]
        pre_parts += [f"(not ({atom_name[a]}))" for a in sorted(pre_neg)]
        eff_parts = [f"({atom_name[a]})" for a in sorted(add)]
        eff_parts += [f"(not ({atom_name[a]}))" for a in sorted(delete)]
        domain_lines.extend([
            f"  (:action {action_id}",
            "    :parameters ()",
            f"    :precondition {emit_and(pre_parts)}",
            f"    :effect {emit_and(eff_parts)}",
            "  )",
        ])
    domain_lines.append(")")

    problem_name = pddl_symbol(f"learned_{case_name}_problem")
    init_parts = [f"({atom_name[a]})" for a in sorted(test_initial) if a in atom_name]
    goal_parts = [f"({atom_name[a]})" for a in sorted(goal_pos)]
    goal_parts += [f"(not ({atom_name[a]}))" for a in sorted(goal_neg)]
    problem_lines = [
        f"(define (problem {problem_name})",
        f"  (:domain {domain_name})",
        "  (:init " + " ".join(init_parts) + ")",
        f"  (:goal {emit_and(goal_parts)})",
        ")",
    ]

    domain_path = out_dir / f"learned_{case_name}_domain.pddl"
    problem_path = out_dir / f"learned_{case_name}_problem.pddl"
    domain_path.write_text("\n".join(domain_lines) + "\n", encoding="utf-8")
    problem_path.write_text("\n".join(problem_lines) + "\n", encoding="utf-8")
    stats = {
        "compiled_actions": len(rows),
        "compiled_propositions": len(atom_list),
        "goal_positive": len(goal_pos),
        "goal_negative": len(goal_neg),
    }
    return domain_path, problem_path, action_map, stats


def solve_compiled(domain_path: Path, problem_path: Path, engine: str) -> Tuple[List[str], str, float]:
    problem = PDDLReader().parse_problem(str(domain_path), str(problem_path))
    start = time.time()
    with OneshotPlanner(name=engine) as planner:
        result = planner.solve(problem, timeout=SOLVE_TIMEOUT)
    elapsed = time.time() - start
    if result.status not in {
        PlanGenerationResultStatus.SOLVED_SATISFICING,
        PlanGenerationResultStatus.SOLVED_OPTIMALLY,
    } or result.plan is None:
        raise RuntimeError(f"learned compiled planner failed: {result.status}")
    names = [ai.action.name for ai in result.plan.actions]
    return names, str(result.status), elapsed


def validate_in_original(
    domain_file: Path,
    problem_file: Path,
    plan_names: Sequence[str],
    action_map: Mapping[str, base.GroundAction],
) -> Dict[str, Any]:
    problem = PDDLReader().parse_problem(str(domain_file), str(problem_file))
    mapped: List[ActionInstance] = []
    for name in plan_names:
        action = action_map[name]
        up_action = problem.action(action.name)
        params = tuple(problem.object(obj) for obj in action.args)
        mapped.append(ActionInstance(up_action, params))

    illegal_step = None
    with SequentialSimulator(problem) as simulator:
        state = simulator.get_initial_state()
        for i, action_instance in enumerate(mapped):
            if not simulator.is_applicable(state, action_instance):
                illegal_step = i
                break
            state = simulator.apply(state, action_instance)
            if state is None:
                illegal_step = i
                break
        goal = illegal_step is None and simulator.is_goal(state)
    return {
        "mapped_steps": len(mapped),
        "illegal_step": illegal_step,
        "goal": bool(goal),
        "pass": illegal_step is None and bool(goal),
        "mapped_plan": [f"{action_map[n].name}({','.join(action_map[n].args)})" for n in plan_names],
    }


def learned_rollout_check(
    model: Mapping[str, Any],
    initial: Set[Atom],
    plan_names: Sequence[str],
    action_map: Mapping[str, base.GroundAction],
) -> Dict[str, Any]:
    true_state = set(initial)
    learned_state = set(initial)
    mismatch = None
    for i, name in enumerate(plan_names):
        action = action_map[name]
        true_app = action.applicable(true_state)
        learned_app = base.predict_applicable(model, learned_state, action)
        if not true_app or learned_app is not True:
            mismatch = {"step": i, "reason": "applicability", "true": true_app, "learned": learned_app}
            break
        true_state = action.apply(true_state)
        next_state = base.predict_successor(model, learned_state, action)
        if next_state is None:
            mismatch = {"step": i, "reason": "missing_effect"}
            break
        learned_state = next_state
        if learned_state != true_state:
            mismatch = {"step": i, "reason": "successor_mismatch"}
            break
    return {"pass": mismatch is None, "exact_steps": len(plan_names) if mismatch is None else mismatch["step"], "mismatch": mismatch}


def compiler_leak_audit() -> Dict[str, Any]:
    source = inspect.getsource(compile_propositional_problem) + inspect.getsource(compiled_action_from_learned)
    forbidden = [".pre_pos", ".pre_neg", ".eff_add", ".eff_del"]
    hits = [token for token in forbidden if token in source]
    # Comments explicitly mention the forbidden field names; remove comment strings
    # from the substantive verdict by checking attribute AST-like patterns below.
    executable_hits = [token for token in hits if f"action{token}" in source.replace("does not access action.pre_pos/pre_neg/eff_add/eff_del", "")]
    return {"forbidden_attribute_hits": executable_hits, "pass": not executable_hits}


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: real_pddl_learned_schema_planning.py SIFT_ROOT OUTPUT_DIR")
    root = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = base.choose_engine()
    report: Dict[str, Any] = {
        "source_commit": base.SIFT_COMMIT,
        "engine": engine,
        "protocol": {
            "training": "small public problem transition outcomes only",
            "planner_domain": "new propositional PDDL compiled only from learned applicability/effects",
            "allowed_test_inputs": "large problem objects, grounded action name/arguments, initial state and goal",
            "forbidden_planner_inputs": "original large-problem action precondition/effect literals",
            "validation": "mapped learned plan replayed in original PDDL SequentialSimulator",
        },
        "compiler_leak_audit": compiler_leak_audit(),
        "cases": {},
    }
    master = random.Random(SEED)

    for name, (domain_rel, train_rel, test_rel) in base.CASES.items():
        case: Dict[str, Any] = {"domain": domain_rel, "train_problem": train_rel, "test_problem": test_rel}
        report["cases"][name] = case
        try:
            domain_file = root / domain_rel
            train_file = root / train_rel
            test_file = root / test_rel
            train_initial, train_actions, train_arities = base.load_grounded(domain_file, train_file)
            test_initial, test_actions, test_arities = base.load_grounded(domain_file, test_file)
            arities = base.predicate_arities(train_arities, test_arities)
            train_plan = base.solve_plan(domain_file, train_file, engine)
            original_test_plan = base.solve_plan(domain_file, test_file, engine)
            rng = random.Random(master.randrange(1 << 63))
            train_states = base.collect_states(train_initial, train_actions, train_plan, base.TRAIN_RANDOM_STATES, rng)
            model = base.learn_model(train_states, train_actions, arities, rng)
            goal_pos, goal_neg = original_goal_literals(domain_file, test_file)
            learned_domain, learned_problem, action_map, compile_stats = compile_propositional_problem(
                name, model, test_initial, test_actions, goal_pos, goal_neg, out_dir
            )
            learned_plan_names, solve_status, solve_sec = solve_compiled(learned_domain, learned_problem, engine)
            original_validation = validate_in_original(domain_file, test_file, learned_plan_names, action_map)
            learned_rollout = learned_rollout_check(model, test_initial, learned_plan_names, action_map)
            case.update({
                "train_states": len(train_states),
                "train_ground_actions": len(train_actions),
                "test_ground_actions": len(test_actions),
                "original_baseline_plan_length": len(original_test_plan),
                "learned_plan_length": len(learned_plan_names),
                "plan_length_ratio": len(learned_plan_names) / len(original_test_plan) if original_test_plan else None,
                "solve_status": solve_status,
                "solve_sec": round(solve_sec, 6),
                "compile": compile_stats,
                "original_validation": original_validation,
                "learned_rollout": learned_rollout,
                "model_conflicts": list(model.get("conflicts", [])),
            })
            case["pass"] = bool(
                report["compiler_leak_audit"]["pass"]
                and not model.get("conflicts")
                and original_validation["pass"]
                and learned_rollout["pass"]
            )
            print("CEDC_LEARNED_PLAN_CASE=" + json.dumps({
                "domain": name,
                "pass": case["pass"],
                "baseline_len": case["original_baseline_plan_length"],
                "learned_len": case["learned_plan_length"],
                "ratio": case["plan_length_ratio"],
                "original_valid": original_validation["pass"],
                "illegal_step": original_validation["illegal_step"],
                "learned_rollout": learned_rollout["pass"],
                "compiled_actions": compile_stats["compiled_actions"],
                "solve_sec": case["solve_sec"],
            }, sort_keys=True), flush=True)
        except Exception as exc:
            case["pass"] = False
            case["error"] = f"{type(exc).__name__}: {exc}"
            case["traceback"] = traceback.format_exc()
            print("CEDC_LEARNED_PLAN_CASE=" + json.dumps({"domain": name, "pass": False, "error": case["error"]}, sort_keys=True), flush=True)
        (out_dir / f"learned_plan_{name}.json").write_text(json.dumps(case, indent=2, sort_keys=True), encoding="utf-8")

    cases = list(report["cases"].values())
    summary = {
        "passed_cases": sum(bool(c.get("pass")) for c in cases),
        "total_cases": len(cases),
        "original_valid_cases": sum(bool(c.get("original_validation", {}).get("pass")) for c in cases),
        "learned_rollout_cases": sum(bool(c.get("learned_rollout", {}).get("pass")) for c in cases),
        "baseline_steps": sum(c.get("original_baseline_plan_length", 0) for c in cases),
        "learned_steps": sum(c.get("learned_plan_length", 0) for c in cases),
        "illegal_action_cases": sum(c.get("original_validation", {}).get("illegal_step") is not None for c in cases),
        "compiler_leak_audit_pass": report["compiler_leak_audit"]["pass"],
    }
    summary["aggregate_length_ratio"] = summary["learned_steps"] / summary["baseline_steps"] if summary["baseline_steps"] else None
    summary["all_passed"] = summary["passed_cases"] == summary["total_cases"] and summary["compiler_leak_audit_pass"]
    report["summary"] = summary
    (out_dir / "REAL_PDDL_LEARNED_SCHEMA_PLANNING.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# Autonomous real-PDDL planning from trace-learned schemas",
        "",
        f"- Passed: **{summary['passed_cases']}/{summary['total_cases']}**",
        f"- Original validator/replay: **{summary['original_valid_cases']}/{summary['total_cases']}**",
        f"- Illegal action cases: **{summary['illegal_action_cases']}**",
        f"- Baseline/learned steps: **{summary['baseline_steps']}/{summary['learned_steps']}**",
        f"- Aggregate length ratio: **{summary['aggregate_length_ratio']}**",
        "",
        "| Domain | Pass | Baseline | Learned | Ratio | Original valid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, case in report["cases"].items():
        md.append(f"| {name} | {case.get('pass', False)} | {case.get('original_baseline_plan_length', 0)} | {case.get('learned_plan_length', 0)} | {case.get('plan_length_ratio')} | {case.get('original_validation', {}).get('pass', False)} |")
    (out_dir / "REAL_PDDL_LEARNED_SCHEMA_PLANNING.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("CEDC_LEARNED_PLAN_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
