#!/usr/bin/env python3
from __future__ import annotations

import bisect
import collections
import json
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import real_pddl_trace_schema_equivalence as base

SIFT_COMMIT = base.SIFT_COMMIT
SEED = 908177
TRACE_LENGTH = 900
MAX_INTERVENTIONS = 40

Atom = base.Atom


@dataclass
class Event:
    index: int
    action_index: int
    success: bool
    before: frozenset[Atom]
    after: frozenset[Atom]
    guard: frozenset[Atom]
    writes: frozenset[Atom]
    objects: frozenset[str]


def event_objects(action: base.GroundAction) -> frozenset[str]:
    values: Set[str] = set(action.args)
    for atom in action.pre_pos | action.pre_neg | action.eff_add | action.eff_del:
        values.update(atom[1:])
    return frozenset(values)


def make_event(index: int, action_index: int, action: base.GroundAction, state: Set[Atom]) -> Event:
    before = frozenset(state)
    success = action.applicable(state)
    after_state = action.apply(state) if success else set(state)
    return Event(
        index=index,
        action_index=action_index,
        success=success,
        before=before,
        after=frozenset(after_state),
        guard=frozenset(action.pre_pos | action.pre_neg),
        writes=frozenset(action.eff_add | action.eff_del),
        objects=event_objects(action),
    )


def append_event(events: List[Event], action_index: int, actions: Sequence[base.GroundAction], state: Set[Atom]) -> Set[Atom]:
    event = make_event(len(events), action_index, actions[action_index], state)
    events.append(event)
    return set(event.after)


def build_attempt_trace(
    initial: Set[Atom],
    actions: Sequence[base.GroundAction],
    length: int,
    rng: random.Random,
) -> Tuple[List[Event], List[int]]:
    """Create success/failure streams with explicit emergent-event traps.

    A pivot success is followed, where possible, by an action that was applicable
    before the pivot but is inapplicable afterwards. Suppressing the pivot should
    therefore make the following failed attempt become successful.
    """
    events: List[Event] = []
    pivots: List[int] = []
    state = set(initial)

    while len(events) < length:
        applicable_indices = [i for i, action in enumerate(actions) if action.applicable(state)]
        if not applicable_indices:
            state = set(initial)
            continue

        pivot_action_index = rng.choice(applicable_indices)
        before = set(state)
        pivot_index = len(events)
        state = append_event(events, pivot_action_index, actions, state)
        if events[-1].success:
            pivots.append(pivot_index)

        if len(events) >= length:
            break

        shuffled = list(applicable_indices)
        rng.shuffle(shuffled)
        anti_index = None
        for candidate_index in shuffled[: min(len(shuffled), 256)]:
            if not actions[candidate_index].applicable(state):
                anti_index = candidate_index
                break
        if anti_index is None:
            for candidate_index in applicable_indices:
                if not actions[candidate_index].applicable(state):
                    anti_index = candidate_index
                    break
        if anti_index is None:
            failed = [i for i, action in enumerate(actions) if not action.applicable(state)]
            anti_index = rng.choice(failed) if failed else rng.randrange(len(actions))
        state = append_event(events, anti_index, actions, state)

        if len(events) >= length:
            break

        random_index = rng.randrange(len(actions))
        state = append_event(events, random_index, actions, state)

        if rng.random() < 0.02:
            state = set(initial)

    return events, pivots


def build_static_fact_children(events: Sequence[Event]) -> List[Set[int]]:
    children: List[Set[int]] = [set() for _ in events]
    latest_writer: Dict[Atom, int] = {}
    for event in events:
        for fact in event.guard:
            parent = latest_writer.get(fact)
            if parent is not None:
                children[parent].add(event.index)
        if event.success:
            for fact in event.writes:
                latest_writer[fact] = event.index
    return children


def descendants(children: Sequence[Set[int]], root: int) -> Set[int]:
    out = {root}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in children[node]:
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def object_overlap_cone(events: Sequence[Event], root: int) -> Set[int]:
    chosen = {root}
    affected = set(events[root].objects)
    for event in events[root + 1 :]:
        if affected.intersection(event.objects):
            chosen.add(event.index)
            affected.update(event.objects)
    return chosen


def replay_full(events: Sequence[Event], actions: Sequence[base.GroundAction], root: int) -> Dict[str, Any]:
    state = set(events[root].before)
    outcomes: List[bool] = []
    start = time.perf_counter()
    for event in events[root:]:
        action = actions[event.action_index]
        success = False if event.index == root else action.applicable(state)
        outcomes.append(success)
        if success:
            state = action.apply(state)
    return {
        "state": frozenset(state),
        "outcomes": outcomes,
        "recomputed": len(outcomes),
        "elapsed_sec": time.perf_counter() - start,
    }


def replay_selected(
    events: Sequence[Event],
    actions: Sequence[base.GroundAction],
    root: int,
    selected: Set[int],
) -> Dict[str, Any]:
    state = set(events[root].before)
    outcomes: List[bool] = []
    recomputed = 0
    start = time.perf_counter()
    for event in events[root:]:
        action = actions[event.action_index]
        if event.index == root:
            success = False
            recomputed += 1
        elif event.index in selected:
            success = action.applicable(state)
            recomputed += 1
        else:
            success = event.success
        outcomes.append(success)
        if success:
            state = action.apply(state)
    return {
        "state": frozenset(state),
        "outcomes": outcomes,
        "recomputed": recomputed,
        "elapsed_sec": time.perf_counter() - start,
    }


def replay_dynamic_guard_diff(
    events: Sequence[Event],
    actions: Sequence[base.GroundAction],
    root: int,
) -> Dict[str, Any]:
    """Exact candidate for deterministic STRIPS under a fixed attempt stream.

    An attempt's applicability can differ from the base run only if at least one
    of its positive or negative guard atoms differs at that time. Events whose
    guards are unchanged reuse the base success/failure result. Same effects may
    also erase the state divergence. Previously failed attempts are reconsidered
    whenever a blocker changes, so emergent successes are included.
    """
    state = set(events[root].before)
    outcomes: List[bool] = []
    recomputed = 0
    scanned = 0
    start = time.perf_counter()
    tail = events[root:]
    for offset, event in enumerate(tail):
        scanned += 1
        action = actions[event.action_index]
        if event.index == root:
            success = False
            recomputed += 1
        else:
            diff = state.symmetric_difference(event.before)
            if not diff:
                outcomes.extend(e.success for e in tail[offset:])
                state = set(events[-1].after)
                break
            if event.guard.intersection(diff):
                success = action.applicable(state)
                recomputed += 1
            else:
                success = event.success
        outcomes.append(success)
        if success:
            state = action.apply(state)
    return {
        "state": frozenset(state),
        "outcomes": outcomes,
        "recomputed": recomputed,
        "scanned": scanned,
        "elapsed_sec": time.perf_counter() - start,
    }


def compare_method(method: Mapping[str, Any], gold: Mapping[str, Any], base_outcomes: Sequence[bool]) -> Dict[str, Any]:
    method_outcomes = list(method["outcomes"])
    gold_outcomes = list(gold["outcomes"])
    exact_outcomes = method_outcomes == gold_outcomes
    exact_state = method["state"] == gold["state"]
    emergent_positions = [i for i, (base_value, gold_value) in enumerate(zip(base_outcomes, gold_outcomes)) if (not base_value) and gold_value]
    missed = sum(not method_outcomes[i] for i in emergent_positions)
    changed_positions = sum(a != b for a, b in zip(base_outcomes, gold_outcomes))
    return {
        "exact": bool(exact_outcomes and exact_state),
        "exact_outcomes": exact_outcomes,
        "exact_final_state": exact_state,
        "recomputed": int(method["recomputed"]),
        "recomputed_fraction": method["recomputed"] / len(gold_outcomes) if gold_outcomes else 0.0,
        "elapsed_sec": method["elapsed_sec"],
        "emergent_events": len(emergent_positions),
        "missed_emergent_events": missed,
        "changed_outcomes": changed_positions,
    }


def choose_interventions(events: Sequence[Event], pivots: Sequence[int], limit: int, rng: random.Random) -> List[int]:
    guard_positions: Dict[Atom, List[int]] = collections.defaultdict(list)
    for event in events:
        for fact in event.guard:
            guard_positions[fact].append(event.index)

    scored: List[Tuple[int, int]] = []
    for index in pivots:
        if index >= len(events) - 3 or not events[index].success or not events[index].writes:
            continue
        score = 0
        for fact in events[index].writes:
            positions = guard_positions.get(fact, [])
            score += len(positions) - bisect.bisect_right(positions, index)
        scored.append((score, index))
    scored.sort(reverse=True)
    top = [index for _, index in scored[: limit // 2]]
    rest = [index for _, index in scored[limit // 2 :] if index not in top]
    rng.shuffle(rest)
    return sorted(top + rest[: max(0, limit - len(top))])


def aggregate_method(rows: Sequence[Mapping[str, Any]], method: str) -> Dict[str, Any]:
    values = [row[method] for row in rows]
    n = len(values)
    return {
        "exact": sum(v["exact"] for v in values),
        "total": n,
        "exact_rate": sum(v["exact"] for v in values) / n if n else 0.0,
        "mean_recomputed_fraction": sum(v["recomputed_fraction"] for v in values) / n if n else 0.0,
        "total_recomputed": sum(v["recomputed"] for v in values),
        "total_emergent_events": sum(v["emergent_events"] for v in values),
        "missed_emergent_events": sum(v["missed_emergent_events"] for v in values),
        "total_elapsed_sec": sum(v["elapsed_sec"] for v in values),
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: event_cone_falsification.py SIFT_ROOT OUTPUT_DIR")
    root = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    master = random.Random(SEED)
    report: Dict[str, Any] = {
        "source_commit": SIFT_COMMIT,
        "protocol": {
            "domains": "six public SIFT large PDDL problems",
            "counterfactual": "suppress one successful attempted action and replay the fixed remaining attempt stream",
            "gold": "full original-STRIPS replay",
            "candidate": "dynamic guard-indexed state-difference cone",
            "baselines": ["static object-overlap cone", "static latest-writer fact dependency cone"],
            "operator_semantics": "original PDDL used in this stage to isolate event-index correctness; learned operator integration is not claimed here",
        },
        "cases": {},
    }

    all_rows: List[Dict[str, Any]] = []
    for case_index, (name, (domain_rel, _train_rel, test_rel)) in enumerate(base.CASES.items()):
        case: Dict[str, Any] = {"domain": domain_rel, "problem": test_rel}
        report["cases"][name] = case
        try:
            initial, actions, _ = base.load_grounded(root / domain_rel, root / test_rel)
            rng = random.Random(master.randrange(1 << 63))
            events, pivots = build_attempt_trace(initial, actions, TRACE_LENGTH, rng)
            children = build_static_fact_children(events)
            interventions = choose_interventions(events, pivots, MAX_INTERVENTIONS, rng)
            rows: List[Dict[str, Any]] = []
            for intervention in interventions:
                gold = replay_full(events, actions, intervention)
                base_outcomes = [event.success for event in events[intervention:]]
                object_method = replay_selected(events, actions, intervention, object_overlap_cone(events, intervention))
                fact_method = replay_selected(events, actions, intervention, descendants(children, intervention))
                dynamic_method = replay_dynamic_guard_diff(events, actions, intervention)
                row = {
                    "intervention": intervention,
                    "remaining_attempts": len(events) - intervention,
                    "gold_changed_outcomes": sum(a != b for a, b in zip(base_outcomes, gold["outcomes"])),
                    "gold_emergent_events": sum((not a) and b for a, b in zip(base_outcomes, gold["outcomes"])),
                    "object_overlap": compare_method(object_method, gold, base_outcomes),
                    "static_fact_graph": compare_method(fact_method, gold, base_outcomes),
                    "dynamic_guard_diff": compare_method(dynamic_method, gold, base_outcomes),
                }
                rows.append(row)
            case.update({
                "ground_actions": len(actions),
                "trace_attempts": len(events),
                "base_successes": sum(event.success for event in events),
                "base_failures": sum(not event.success for event in events),
                "pivot_candidates": len(pivots),
                "interventions": len(interventions),
                "effective_interventions": sum(row["gold_changed_outcomes"] > 0 for row in rows),
                "emergent_interventions": sum(row["gold_emergent_events"] > 0 for row in rows),
                "methods": {
                    method: aggregate_method(rows, method)
                    for method in ("object_overlap", "static_fact_graph", "dynamic_guard_diff")
                },
                "rows": rows,
            })
            case["pass"] = bool(
                case["methods"]["dynamic_guard_diff"]["exact"] == len(rows)
                and case["methods"]["dynamic_guard_diff"]["missed_emergent_events"] == 0
            )
            all_rows.extend(rows)
            print("EVENT_CONE_CASE=" + json.dumps({
                "domain": name,
                "pass": case["pass"],
                "interventions": len(rows),
                "effective": case["effective_interventions"],
                "emergent": case["emergent_interventions"],
                "object_exact": case["methods"]["object_overlap"]["exact"],
                "fact_exact": case["methods"]["static_fact_graph"]["exact"],
                "dynamic_exact": case["methods"]["dynamic_guard_diff"]["exact"],
                "dynamic_recompute": case["methods"]["dynamic_guard_diff"]["mean_recomputed_fraction"],
            }, sort_keys=True), flush=True)
        except Exception as exc:
            case["pass"] = False
            case["error"] = f"{type(exc).__name__}: {exc}"
            case["traceback"] = traceback.format_exc()
            print("EVENT_CONE_CASE=" + json.dumps({"domain": name, "pass": False, "error": case["error"]}, sort_keys=True), flush=True)
        (out_dir / f"event_cone_{name}.json").write_text(json.dumps(case, indent=2, sort_keys=True), encoding="utf-8")

    methods = {
        method: aggregate_method(all_rows, method)
        for method in ("object_overlap", "static_fact_graph", "dynamic_guard_diff")
    }
    summary = {
        "passed_cases": sum(bool(case.get("pass")) for case in report["cases"].values()),
        "total_cases": len(report["cases"]),
        "interventions": len(all_rows),
        "effective_interventions": sum(row["gold_changed_outcomes"] > 0 for row in all_rows),
        "emergent_interventions": sum(row["gold_emergent_events"] > 0 for row in all_rows),
        "total_emergent_events": sum(row["gold_emergent_events"] for row in all_rows),
        "methods": methods,
    }
    summary["all_passed"] = summary["passed_cases"] == summary["total_cases"]
    report["summary"] = summary
    (out_dir / "EVENT_CONE_FALSIFICATION.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# Event-cone counterfactual falsification on public PDDL",
        "",
        f"- Cases: **{summary['passed_cases']}/{summary['total_cases']}**",
        f"- Interventions: **{summary['interventions']}**",
        f"- Effective interventions: **{summary['effective_interventions']}**",
        f"- Emergent-event interventions: **{summary['emergent_interventions']}**",
        f"- Total emergent events: **{summary['total_emergent_events']}**",
        "",
        "| Method | Exact | Mean recomputed fraction | Missed emergent events |",
        "|---|---:|---:|---:|",
    ]
    for method, values in methods.items():
        md.append(f"| {method} | {values['exact']}/{values['total']} | {values['mean_recomputed_fraction']:.6f} | {values['missed_emergent_events']} |")
    md += [
        "",
        "The operator model is held fixed to the original public PDDL in this stage. This isolates whether the event-index/replay structure is exact; it is not yet a learned-operator end-to-end result.",
    ]
    (out_dir / "EVENT_CONE_FALSIFICATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("EVENT_CONE_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
