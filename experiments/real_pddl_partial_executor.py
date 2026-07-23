#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import pddlpy
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, SequentialSimulator, get_environment

SIFT_COMMIT = "0f4518ffa609322a24521609f80d70728f07ac09"
CASES = {
    "gripper": ("pddl_files/gripper/gripper.pddl", "pddl_files/gripper/gripper_8_3.pddl"),
    "ferry": ("pddl_files/ferry/ferry.pddl", "pddl_files/ferry/ferry-6-5.pddl"),
    "delivery": ("pddl_files/delivery/delivery.pddl", "pddl_files/delivery/delivery-3-3-3-2.pddl"),
    "hanoi": ("pddl_files/hanoi/hanoi.pddl", "pddl_files/hanoi/hanoi-3-10.pddl"),
    "n-puzzle": ("pddl_files/npuzzle/npuzzle.pddl", "pddl_files/npuzzle/npuzzle-4-4.pddl"),
    "sokoban": ("pddl_files/sokoban/sokoban.pddl", "pddl_files/sokoban/sokoban-5-5_3.pddl"),
}
MASK_RATES = (0.5, 0.7)
MASK_SEEDS = (41001, 41002, 41003, 41004, 41005)
SOLVE_TIMEOUT = 240


def choose_engine() -> str:
    engines = sorted(get_environment().factory.engines)
    for name in ("fast-downward", "fast-downward-opt"):
        if name in engines:
            return name
    raise RuntimeError(f"Fast Downward unavailable: {engines}")


def fluent_leaves(expr: Any) -> Set[Any]:
    out: Set[Any] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if node.is_fluent_exp():
            out.add(node)
        else:
            stack.extend(node.args)
    return out


def dynamic_fluents(problem: Any) -> Set[Any]:
    result: Set[Any] = set()
    for action in problem.actions:
        for effect in action.effects:
            result.add(effect.fluent.fluent())
    return result


def atom_parts(atom: Any) -> Tuple[str, Tuple[str, ...]]:
    if isinstance(atom, (tuple, list)) and atom:
        return str(atom[0]), tuple(str(x) for x in atom[1:])
    raise TypeError(f"unsupported pddlpy atom representation: {atom!r}")


def static_atom_truth(atom: Any, initial_state: Set[Any]) -> bool:
    predicate, args = atom_parts(atom)
    if predicate == "=":
        return len(args) == 2 and args[0] == args[1]
    return atom in initial_state


def atom_to_up_fluent(problem: Any, atom: Any) -> Any:
    predicate, args = atom_parts(atom)
    fluent = problem.fluent(predicate)
    objects = [problem.object(name) for name in args]
    return fluent(*objects)


def build_semantic_dynamic_universe(
    problem: Any,
    domain_file: Path,
    problem_file: Path,
    up_dynamic: Set[Any],
) -> Tuple[List[Any], Dict[str, Any]]:
    """Build only action-relevant state variables admitted by static PDDL facts.

    Several SIFT domains encode types through static unary predicates instead of
    PDDL :types. Enumerating every object Cartesian product therefore creates
    impossible pseudo-atoms. We enumerate pddlpy groundings, identify predicates
    changed by effects, reject bindings whose static positive/negative conditions
    are false in the public initial state, and collect only dynamic atoms appearing
    in the remaining action preconditions/effects plus initial/goal atoms.
    """
    dp = pddlpy.DomainProblem(str(domain_file), str(problem_file))
    initial = set(dp.initialstate())
    grounded = []
    dynamic_predicates: Set[str] = set()

    for operator_name in sorted(dp.operators()):
        for op in dp.ground_operator(operator_name):
            grounded.append(op)
            for attr in ("effect_pos", "effect_neg"):
                for atom in list(getattr(op, attr, [])):
                    dynamic_predicates.add(atom_parts(atom)[0])

    support_atoms: Set[Any] = set()
    admitted = 0
    rejected_static = 0
    for op in grounded:
        pre_pos = list(getattr(op, "precondition_pos", []))
        pre_neg = list(getattr(op, "precondition_neg", []))
        static_pos = [a for a in pre_pos if atom_parts(a)[0] not in dynamic_predicates]
        static_neg = [a for a in pre_neg if atom_parts(a)[0] not in dynamic_predicates]
        static_ok = all(static_atom_truth(a, initial) for a in static_pos) and all(
            not static_atom_truth(a, initial) for a in static_neg
        )
        if not static_ok:
            rejected_static += 1
            continue
        admitted += 1
        for attr in ("precondition_pos", "precondition_neg", "effect_pos", "effect_neg"):
            for atom in list(getattr(op, attr, [])):
                if atom_parts(atom)[0] in dynamic_predicates:
                    support_atoms.add(atom)

    for atom in initial:
        if atom_parts(atom)[0] in dynamic_predicates:
            support_atoms.add(atom)

    universe: Set[Any] = set()
    conversion_errors = []
    for atom in sorted(support_atoms, key=str):
        try:
            fexp = atom_to_up_fluent(problem, atom)
            if fexp.fluent() in up_dynamic:
                universe.add(fexp)
        except Exception as exc:
            conversion_errors.append({"atom": str(atom), "error": f"{type(exc).__name__}: {exc}"})

    for goal in problem.goals:
        for fexp in fluent_leaves(goal.simplify()):
            if fexp.fluent() in up_dynamic:
                universe.add(fexp)

    if conversion_errors:
        raise RuntimeError(f"failed to convert dynamic support atoms: {conversion_errors[:5]}")

    stats = {
        "raw_ground_action_bindings": len(grounded),
        "statically_admitted_bindings": admitted,
        "statically_rejected_bindings": rejected_static,
        "dynamic_predicates": sorted(dynamic_predicates),
        "support_atom_count": len(support_atoms),
        "semantic_dynamic_atom_count": len(universe),
    }
    return sorted(universe, key=str), stats


def bool_value(state: Any, fluent_exp: Any) -> bool:
    value = state.get_value(fluent_exp)
    if not value.is_bool_constant():
        raise TypeError(f"non-boolean value for {fluent_exp}: {value}")
    return value.is_true()


def ground_substitution(action_instance: Any) -> Dict[Any, Any]:
    return dict(zip(action_instance.action.parameters, action_instance.actual_parameters))


def grounded_precondition_dynamic_leaves(action_instance: Any, dynamic: Set[Any]) -> Set[Any]:
    subs = ground_substitution(action_instance)
    leaves: Set[Any] = set()
    for condition in action_instance.action.preconditions:
        grounded_condition = condition.substitute(subs).simplify()
        for fexp in fluent_leaves(grounded_condition):
            if fexp.fluent() in dynamic:
                leaves.add(fexp)
    return leaves


def apply_effects_to_belief(action_instance: Any, belief: Dict[Any, bool | None]) -> int:
    subs = ground_substitution(action_instance)
    updated = 0
    for effect in action_instance.action.effects:
        condition = effect.condition.substitute(subs).simplify()
        if not condition.is_true():
            raise NotImplementedError(f"conditional/nontrivial effect unsupported: {condition}")
        fluent_exp = effect.fluent.substitute(subs).simplify()
        value = effect.value.substitute(subs).simplify()
        if fluent_exp not in belief:
            raise RuntimeError(f"effect atom absent from semantic universe: {fluent_exp}")
        if not value.is_bool_constant():
            raise NotImplementedError(f"non-boolean effect unsupported: {value}")
        belief[fluent_exp] = value.is_true()
        updated += 1
    return updated


def goal_dynamic_leaves(problem: Any, dynamic: Set[Any]) -> Set[Any]:
    leaves: Set[Any] = set()
    for goal in problem.goals:
        for fexp in fluent_leaves(goal.simplify()):
            if fexp.fluent() in dynamic:
                leaves.add(fexp)
    return leaves


def solve_once(problem: Any, engine: str) -> Any:
    with OneshotPlanner(name=engine) as planner:
        result = planner.solve(problem, timeout=SOLVE_TIMEOUT)
    if result.status not in {
        PlanGenerationResultStatus.SOLVED_SATISFICING,
        PlanGenerationResultStatus.SOLVED_OPTIMALLY,
    } or result.plan is None:
        raise RuntimeError(f"planner failed: {result.status}")
    return result.plan


def run_episode(
    problem: Any,
    plan: Any,
    dynamic: Set[Any],
    universe: List[Any],
    rate: float,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    masked = {fexp for fexp in universe if rng.random() < rate}

    query_count = 0
    queried: Set[Any] = set()
    effect_updates = 0
    illegal_step = None
    incomplete_certificate_step = None
    support_miss_step = None
    initial_known = len(universe) - len(masked)

    with SequentialSimulator(problem) as simulator:
        state = simulator.get_initial_state()
        belief: Dict[Any, bool | None] = {
            fexp: (None if fexp in masked else bool_value(state, fexp)) for fexp in universe
        }

        for index, action_instance in enumerate(plan.actions):
            required = grounded_precondition_dynamic_leaves(action_instance, dynamic)
            if any(fexp not in belief for fexp in required):
                support_miss_step = index
                break
            for fexp in sorted(required, key=str):
                if belief[fexp] is None:
                    belief[fexp] = bool_value(state, fexp)
                    queried.add(fexp)
                    query_count += 1
            if any(belief[fexp] is None for fexp in required):
                incomplete_certificate_step = index
                break
            if not simulator.is_applicable(state, action_instance):
                illegal_step = index
                break
            new_state = simulator.apply(state, action_instance)
            if new_state is None:
                illegal_step = index
                break
            effect_updates += apply_effects_to_belief(action_instance, belief)
            state = new_state

        goal_required = goal_dynamic_leaves(problem, dynamic)
        if any(fexp not in belief for fexp in goal_required):
            support_miss_step = len(plan.actions)
        if illegal_step is None and incomplete_certificate_step is None and support_miss_step is None:
            for fexp in sorted(goal_required, key=str):
                if belief[fexp] is None:
                    belief[fexp] = bool_value(state, fexp)
                    queried.add(fexp)
                    query_count += 1
        goal_certificate_complete = support_miss_step is None and all(
            belief[fexp] is not None for fexp in goal_required
        )
        replay_goal = (
            illegal_step is None
            and incomplete_certificate_step is None
            and support_miss_step is None
            and simulator.is_goal(state)
        )

    return {
        "mask_rate": rate,
        "mask_seed": seed,
        "dynamic_ground_atom_count": len(universe),
        "initial_masked_count": len(masked),
        "initial_known_count": initial_known,
        "active_query_count": query_count,
        "unique_queried_count": len(queried),
        "query_fraction_of_all_dynamic": query_count / len(universe) if universe else 0.0,
        "query_fraction_of_initially_masked": query_count / len(masked) if masked else 0.0,
        "effect_updates": effect_updates,
        "plan_length": len(plan.actions),
        "illegal_step": illegal_step,
        "incomplete_certificate_step": incomplete_certificate_step,
        "support_miss_step": support_miss_step,
        "goal_certificate_complete": goal_certificate_complete,
        "replay_goal": bool(replay_goal),
        "pass": bool(
            illegal_step is None
            and incomplete_certificate_step is None
            and support_miss_step is None
            and goal_certificate_complete
            and replay_goal
        ),
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: real_pddl_partial_executor.py SIFT_ROOT OUTPUT_DIR")
    sift_root = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = choose_engine()

    report: Dict[str, Any] = {
        "source_repo": "https://github.com/JonasGoesgens/SIFT",
        "source_commit": SIFT_COMMIT,
        "engine": engine,
        "protocol": {
            "learning_state": "not tested in this stage",
            "execution_state": "50% and 70% masking over statically admissible action-support dynamic atoms",
            "planner": "full-state Fast Downward plan fixed before masking",
            "sensing": "only unknown dynamic atoms occurring in the next grounded action precondition or final goal",
            "belief_update": "grounded PDDL boolean effects only",
            "universe": "dynamic atoms in statically admissible grounded action preconditions/effects plus initial/goal atoms",
            "claim": "safe partial-observation execution, not autonomous partial-observation planning",
        },
        "cases": {},
    }

    for case_index, (name, (domain_rel, problem_rel)) in enumerate(CASES.items()):
        case_out: Dict[str, Any] = {
            "domain_file": domain_rel,
            "problem_file": problem_rel,
            "episodes": [],
        }
        report["cases"][name] = case_out
        try:
            domain_path = sift_root / domain_rel
            problem_path = sift_root / problem_rel
            problem = PDDLReader().parse_problem(str(domain_path), str(problem_path))
            dynamic = dynamic_fluents(problem)
            universe, universe_stats = build_semantic_dynamic_universe(
                problem, domain_path, problem_path, dynamic
            )
            case_out["universe_stats"] = universe_stats
            plan_start = time.time()
            plan = solve_once(problem, engine)
            case_out["plan_length"] = len(plan.actions)
            case_out["plan_solve_sec"] = round(time.time() - plan_start, 6)
            for rate in MASK_RATES:
                for base_seed in MASK_SEEDS:
                    episode_seed = base_seed + case_index * 1000 + int(rate * 100)
                    ep = run_episode(problem, plan, dynamic, universe, rate, episode_seed)
                    case_out["episodes"].append(ep)
                    print(
                        name,
                        rate,
                        episode_seed,
                        "PASS" if ep["pass"] else "FAIL",
                        ep["active_query_count"],
                        len(universe),
                        flush=True,
                    )
            case_out["pass"] = all(ep["pass"] for ep in case_out["episodes"])
        except Exception as exc:
            case_out["pass"] = False
            case_out["error"] = f"{type(exc).__name__}: {exc}"
            case_out["traceback"] = traceback.format_exc()
            print(name, "ERROR", case_out["error"], flush=True)
        (out_dir / f"partial_{name}.json").write_text(
            json.dumps(case_out, indent=2, sort_keys=True), encoding="utf-8"
        )

    episodes = [ep for case in report["cases"].values() for ep in case.get("episodes", [])]
    passed_cases = sum(1 for case in report["cases"].values() if case.get("pass"))
    passed_episodes = sum(1 for ep in episodes if ep.get("pass"))
    total_queries = sum(ep["active_query_count"] for ep in episodes)
    total_dynamic = sum(ep["dynamic_ground_atom_count"] for ep in episodes)
    total_masked = sum(ep["initial_masked_count"] for ep in episodes)
    total_plan_steps = sum(ep["plan_length"] for ep in episodes)
    report["summary"] = {
        "passed_cases": passed_cases,
        "total_cases": len(CASES),
        "passed_episodes": passed_episodes,
        "total_episodes": len(episodes),
        "illegal_action_count": sum(ep["illegal_step"] is not None for ep in episodes),
        "incomplete_certificate_count": sum(
            ep["incomplete_certificate_step"] is not None for ep in episodes
        ),
        "support_miss_count": sum(ep["support_miss_step"] is not None for ep in episodes),
        "goal_failure_count": sum(not ep["replay_goal"] for ep in episodes),
        "total_plan_steps_replayed": total_plan_steps,
        "total_active_queries": total_queries,
        "total_dynamic_ground_atoms_across_episodes": total_dynamic,
        "total_initially_masked_atoms": total_masked,
        "query_fraction_of_all_dynamic": total_queries / total_dynamic if total_dynamic else 0.0,
        "query_fraction_of_initially_masked": total_queries / total_masked if total_masked else 0.0,
        "all_passed": passed_cases == len(CASES) and passed_episodes == len(episodes),
    }

    (out_dir / "REAL_PDDL_PARTIAL_EXECUTOR.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    md = [
        "# Real public PDDL partial-observation executor",
        "",
        f"- Source commit: `{SIFT_COMMIT}`",
        f"- Cases: **{passed_cases}/{len(CASES)}**",
        f"- Episodes: **{passed_episodes}/{len(episodes)}**",
        f"- Illegal actions: **{report['summary']['illegal_action_count']}**",
        f"- Support misses: **{report['summary']['support_miss_count']}**",
        f"- Replayed plan steps: **{total_plan_steps}**",
        f"- Active queries: **{total_queries}**",
        f"- Query/all-semantic-dynamic: **{report['summary']['query_fraction_of_all_dynamic']:.6%}**",
        f"- Query/initially-masked: **{report['summary']['query_fraction_of_initially_masked']:.6%}**",
        "",
        "| Domain | Plan length | Semantic dynamic atoms | Admitted groundings | Episodes | Pass | Mean query |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, case in report["cases"].items():
        eps = case.get("episodes", [])
        stats = case.get("universe_stats", {})
        mean_q = sum(ep["active_query_count"] for ep in eps) / len(eps) if eps else 0.0
        md.append(
            f"| {name} | {case.get('plan_length', '')} | {stats.get('semantic_dynamic_atom_count', '')} | "
            f"{stats.get('statically_admitted_bindings', '')} | {len(eps)} | "
            f"{sum(ep.get('pass', False) for ep in eps)} | {mean_q:.3f} |"
        )
    md.extend(
        [
            "",
            "## Scope",
            "",
            "The full-state plan is generated before masking. This validates safe sensing and exact effect-based belief maintenance on real PDDL, not autonomous planning from a partial belief.",
        ]
    )
    (out_dir / "REAL_PDDL_PARTIAL_EXECUTOR.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0 if report["summary"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
