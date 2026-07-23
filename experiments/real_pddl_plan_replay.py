#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pddlpy
from unified_planning.engines import PlanGenerationResultStatus, ValidationResultStatus
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, PlanValidator, SequentialSimulator, get_environment

SIFT_COMMIT = "0f4518ffa609322a24521609f80d70728f07ac09"
CASES = {
    "gripper": ("pddl_files/gripper/gripper.pddl", "pddl_files/gripper/gripper_8_3.pddl"),
    "ferry": ("pddl_files/ferry/ferry.pddl", "pddl_files/ferry/ferry-6-5.pddl"),
    "delivery": ("pddl_files/delivery/delivery.pddl", "pddl_files/delivery/delivery-3-3-3-2.pddl"),
    "hanoi": ("pddl_files/hanoi/hanoi.pddl", "pddl_files/hanoi/hanoi-3-10.pddl"),
    "n-puzzle": ("pddl_files/npuzzle/npuzzle.pddl", "pddl_files/npuzzle/npuzzle-4-4.pddl"),
    "sokoban": ("pddl_files/sokoban/sokoban.pddl", "pddl_files/sokoban/sokoban-5-5_3.pddl"),
}
GROUND_CAP = 2_000_000
PER_CASE_TIMEOUT = 240


def safe_list(x: Iterable[Any]) -> List[Any]:
    return list(x)


def pddlpy_grounding(domain: Path, problem: Path) -> Dict[str, Any]:
    start = time.time()
    dp = pddlpy.DomainProblem(str(domain), str(problem))
    per_operator: Dict[str, int] = {}
    total = 0
    truncated = False
    applicable_initial = 0
    init = set(dp.initialstate())
    samples = []
    for op_name in sorted(dp.operators()):
        count = 0
        for g in dp.ground_operator(op_name):
            count += 1
            total += 1
            pos = set(getattr(g, "precondition_pos", []))
            neg = set(getattr(g, "precondition_neg", []))
            if pos.issubset(init) and not (neg & init):
                applicable_initial += 1
            if len(samples) < 12:
                samples.append({
                    "name": op_name,
                    "binding": {str(k): str(v) for k, v in dict(getattr(g, "variable_list", {})).items()},
                    "pre_pos": sorted(map(str, pos)),
                    "pre_neg": sorted(map(str, neg)),
                    "eff_pos": sorted(map(str, set(getattr(g, "effect_pos", [])))),
                    "eff_neg": sorted(map(str, set(getattr(g, "effect_neg", [])))),
                })
            if total >= GROUND_CAP:
                truncated = True
                break
        per_operator[op_name] = count
        if truncated:
            break
    return {
        "object_count": len(dp.worldobjects()),
        "initial_atom_count": len(init),
        "goal_atom_count": len(list(dp.goals())),
        "operator_names": sorted(dp.operators()),
        "ground_actions": total,
        "grounding_truncated_at": GROUND_CAP if truncated else None,
        "applicable_initial": applicable_initial,
        "per_operator": per_operator,
        "samples": samples,
        "elapsed_sec": round(time.time() - start, 6),
    }


def choose_engine() -> str:
    engines = sorted(get_environment().factory.engines)
    for name in ("fast-downward", "fast-downward-opt"):
        if name in engines:
            return name
    raise RuntimeError(f"Fast Downward engine unavailable; engines={engines}")


def up_solve_replay(domain: Path, problem_file: Path, engine_name: str) -> Dict[str, Any]:
    start = time.time()
    reader = PDDLReader()
    problem = reader.parse_problem(str(domain), str(problem_file))
    parsed_sec = time.time() - start

    result_info: Dict[str, Any] = {
        "problem_name": problem.name,
        "problem_kind": str(problem.kind),
        "action_schema_count": len(problem.actions),
        "fluent_count": len(problem.fluents),
        "object_count": len(problem.all_objects),
        "parser_elapsed_sec": round(parsed_sec, 6),
        "engine": engine_name,
    }

    solve_start = time.time()
    with OneshotPlanner(name=engine_name) as planner:
        result = planner.solve(problem, timeout=PER_CASE_TIMEOUT)
    result_info["solve_elapsed_sec"] = round(time.time() - solve_start, 6)
    result_info["solve_status"] = str(result.status)
    result_info["engine_metrics"] = {str(k): str(v) for k, v in (result.metrics or {}).items()}

    solved_statuses = {
        PlanGenerationResultStatus.SOLVED_SATISFICING,
        PlanGenerationResultStatus.SOLVED_OPTIMALLY,
    }
    if result.status not in solved_statuses or result.plan is None:
        result_info.update({
            "solved": False,
            "validated": False,
            "replay_goal": False,
            "plan_length": None,
            "plan": None,
        })
        return result_info

    plan = result.plan
    plan_text = [str(a) for a in plan.actions]
    result_info["solved"] = True
    result_info["plan_length"] = len(plan.actions)
    result_info["plan"] = plan_text

    with PlanValidator(problem_kind=problem.kind) as validator:
        validation = validator.validate(problem, plan)
    result_info["validation_status"] = str(validation.status)
    result_info["validated"] = validation.status == ValidationResultStatus.VALID

    replay_start = time.time()
    replay_steps = []
    illegal_step = None
    with SequentialSimulator(problem) as simulator:
        state = simulator.get_initial_state()
        for idx, action_instance in enumerate(plan.actions):
            applicable = simulator.is_applicable(state, action_instance)
            if not applicable:
                illegal_step = idx
                break
            before_hash = hash(state)
            next_state = simulator.apply(state, action_instance)
            if next_state is None:
                illegal_step = idx
                break
            if idx < 5 or idx >= len(plan.actions) - 5:
                replay_steps.append({
                    "index": idx,
                    "action": str(action_instance),
                    "state_hash_before": before_hash,
                    "state_hash_after": hash(next_state),
                })
            state = next_state
        replay_goal = illegal_step is None and simulator.is_goal(state)
    result_info["replay_elapsed_sec"] = round(time.time() - replay_start, 6)
    result_info["illegal_step"] = illegal_step
    result_info["replay_goal"] = bool(replay_goal)
    result_info["replay_samples"] = replay_steps
    return result_info


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: real_pddl_plan_replay.py SIFT_ROOT OUTPUT_DIR")
    sift_root = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = choose_engine()
    report: Dict[str, Any] = {
        "source_repo": "https://github.com/JonasGoesgens/SIFT",
        "source_commit": SIFT_COMMIT,
        "grounder": "pddlpy==1.2.1",
        "planner_stack": "unified-planning + up-fast-downward",
        "engine": engine,
        "per_case_timeout_sec": PER_CASE_TIMEOUT,
        "cases": {},
    }

    for name, (domain_rel, problem_rel) in CASES.items():
        domain = sift_root / domain_rel
        problem = sift_root / problem_rel
        case: Dict[str, Any] = {
            "domain_file": domain_rel,
            "problem_file": problem_rel,
            "exists": domain.is_file() and problem.is_file(),
        }
        report["cases"][name] = case
        try:
            if not case["exists"]:
                raise FileNotFoundError(f"missing {domain} or {problem}")
            case["pddlpy"] = pddlpy_grounding(domain, problem)
            case["unified_planning"] = up_solve_replay(domain, problem, engine)
            up = case["unified_planning"]
            case["pass"] = bool(up["solved"] and up["validated"] and up["replay_goal"] and up["illegal_step"] is None)
        except Exception as exc:
            case["pass"] = False
            case["error"] = f"{type(exc).__name__}: {exc}"
            case["traceback"] = traceback.format_exc()
        (out_dir / f"{name}.json").write_text(json.dumps(case, indent=2, sort_keys=True), encoding="utf-8")
        print(name, "PASS" if case["pass"] else "FAIL", case.get("error", ""), flush=True)

    passed = sum(1 for c in report["cases"].values() if c.get("pass"))
    report["summary"] = {
        "passed": passed,
        "total": len(CASES),
        "all_passed": passed == len(CASES),
        "plan_lengths": {
            name: case.get("unified_planning", {}).get("plan_length")
            for name, case in report["cases"].items()
        },
    }
    (out_dir / "REAL_PDDL_PLAN_REPLAY.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# Real public SIFT PDDL plan and replay",
        "",
        f"- Source commit: `{SIFT_COMMIT}`",
        f"- Passed: **{passed}/{len(CASES)}**",
        "",
        "| Domain | Ground actions | Solve | Plan length | Validator | Replay goal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, case in report["cases"].items():
        py = case.get("pddlpy", {})
        up = case.get("unified_planning", {})
        md.append(
            f"| {name} | {py.get('ground_actions', '')} | {up.get('solved', False)} | "
            f"{up.get('plan_length', '')} | {up.get('validated', False)} | {up.get('replay_goal', False)} |"
        )
    (out_dir / "REAL_PDDL_PLAN_REPLAY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0 if passed == len(CASES) else 2


if __name__ == "__main__":
    raise SystemExit(main())
