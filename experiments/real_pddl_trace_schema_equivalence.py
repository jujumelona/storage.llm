#!/usr/bin/env python3
from __future__ import annotations

import collections
import itertools
import json
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import pddlpy
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, get_environment

SIFT_COMMIT = "0f4518ffa609322a24521609f80d70728f07ac09"
CASES = {
    "gripper": (
        "pddl_files/gripper/gripper.pddl",
        "pddl_files/gripper/gripper_7_3.pddl",
        "pddl_files/gripper/gripper_8_3.pddl",
    ),
    "ferry": (
        "pddl_files/ferry/ferry.pddl",
        "pddl_files/ferry/ferry-5-5.pddl",
        "pddl_files/ferry/ferry-6-5.pddl",
    ),
    "delivery": (
        "pddl_files/delivery/delivery.pddl",
        "pddl_files/delivery/delivery-3-3-2-2.pddl",
        "pddl_files/delivery/delivery-3-3-3-2.pddl",
    ),
    "hanoi": (
        "pddl_files/hanoi/hanoi.pddl",
        "pddl_files/hanoi/hanoi-3-9.pddl",
        "pddl_files/hanoi/hanoi-3-10.pddl",
    ),
    "n-puzzle": (
        "pddl_files/npuzzle/npuzzle.pddl",
        "pddl_files/npuzzle/npuzzle-3-3.pddl",
        "pddl_files/npuzzle/npuzzle-4-4.pddl",
    ),
    "sokoban": (
        "pddl_files/sokoban/sokoban.pddl",
        "pddl_files/sokoban/sokoban-4-4_4.pddl",
        "pddl_files/sokoban/sokoban-5-5_3.pddl",
    ),
}

TRAIN_RANDOM_STATES = 90
TEST_RANDOM_STATES = 45
MAX_POS_PER_KEY = 1200
MAX_NEG_PER_KEY = 5000
SOLVE_TIMEOUT = 240
SEED = 731337

Atom = Tuple[str, ...]
Feature = Tuple[str, str, Tuple[int, ...]]  # sign, predicate, argument roles
Key = Tuple[str, Tuple[int, ...]]           # action name, equality partition


def atom_tuple(atom: Any) -> Atom:
    if hasattr(atom, "predicate"):
        raw = getattr(atom, "predicate")
    elif isinstance(atom, (tuple, list)):
        raw = atom
    else:
        raw = None
    if raw is None:
        text = str(atom).strip()
        match = re.match(r"^\(?\s*([^\s,()]+)(?:[,\s]+(.*?))?\)?$", text)
        if not match:
            raise TypeError(f"unsupported atom: {atom!r}")
        pred = match.group(1)
        tail = match.group(2)
        args = [] if not tail else [x for x in re.split(r"[,\s]+", tail.strip()) if x]
        return tuple([pred] + args)
    try:
        seq = list(raw)
    except TypeError:
        raise TypeError(f"unsupported atom payload: {raw!r}")
    if not seq:
        raise TypeError(f"empty atom: {atom!r}")
    return tuple(str(x) for x in seq)


def atom_set(items: Iterable[Any]) -> Set[Atom]:
    return {atom_tuple(x) for x in items}


def equality_partition(args: Sequence[str]) -> Tuple[int, ...]:
    classes: Dict[str, int] = {}
    out: List[int] = []
    for obj in args:
        if obj not in classes:
            classes[obj] = len(classes)
        out.append(classes[obj])
    return tuple(out)


def canonical_atom(atom: Atom, args: Sequence[str]) -> Tuple[str, Tuple[int, ...]] | None:
    positions: Dict[str, int] = {}
    for i, obj in enumerate(args):
        positions.setdefault(obj, i)
    roles: List[int] = []
    for obj in atom[1:]:
        if obj not in positions:
            return None
        roles.append(positions[obj])
    return atom[0], tuple(roles)


def instantiate_feature(feature: Feature, args: Sequence[str]) -> Atom | bool:
    sign, pred, roles = feature
    if pred == "__eq__":
        i, j = roles
        value = args[i] == args[j]
        return value if sign == "+" else not value
    return tuple([pred] + [args[i] for i in roles])


def feature_holds(feature: Feature, state: Set[Atom], args: Sequence[str]) -> bool:
    instantiated = instantiate_feature(feature, args)
    if isinstance(instantiated, bool):
        return instantiated
    if feature[0] == "+":
        return instantiated in state
    return instantiated not in state


@dataclass(frozen=True)
class GroundAction:
    name: str
    args: Tuple[str, ...]
    pre_pos: frozenset[Atom]
    pre_neg: frozenset[Atom]
    eff_add: frozenset[Atom]
    eff_del: frozenset[Atom]

    @property
    def key(self) -> Key:
        return self.name, equality_partition(self.args)

    def applicable(self, state: Set[Atom]) -> bool:
        return self.pre_pos.issubset(state) and not bool(self.pre_neg.intersection(state))

    def apply(self, state: Set[Atom]) -> Set[Atom]:
        return (state.difference(self.eff_del)).union(self.eff_add)


def choose_engine() -> str:
    engines = sorted(get_environment().factory.engines)
    for name in ("fast-downward", "fast-downward-opt"):
        if name in engines:
            return name
    raise RuntimeError(f"Fast Downward unavailable: {engines}")


def parameter_orders(domain_file: Path, problem_file: Path) -> Dict[str, List[str]]:
    problem = PDDLReader().parse_problem(str(domain_file), str(problem_file))
    return {
        action.name: [p.name if p.name.startswith("?") else "?" + p.name for p in action.parameters]
        for action in problem.actions
    }


def load_grounded(domain_file: Path, problem_file: Path) -> Tuple[Set[Atom], List[GroundAction], Dict[str, int]]:
    dom = pddlpy.DomainProblem(str(domain_file), str(problem_file))
    initial = atom_set(dom.initialstate())
    orders = parameter_orders(domain_file, problem_file)
    actions: List[GroundAction] = []
    arities: Dict[str, int] = {}
    for name in sorted(dom.operators()):
        order = orders[name]
        for op in dom.ground_operator(name):
            binding = {str(k): str(v) for k, v in dict(op.variable_list).items()}
            args = tuple(binding[p] for p in order)
            pre_pos = atom_set(op.precondition_pos)
            pre_neg = atom_set(op.precondition_neg)
            eff_add = atom_set(op.effect_pos)
            eff_del = atom_set(op.effect_neg)
            actions.append(GroundAction(name, args, frozenset(pre_pos), frozenset(pre_neg), frozenset(eff_add), frozenset(eff_del)))
            for atom in itertools.chain(pre_pos, pre_neg, eff_add, eff_del):
                arities[atom[0]] = len(atom) - 1
    actions.sort(key=lambda a: (a.name, a.args))
    return initial, actions, arities


def action_index(actions: Sequence[GroundAction]) -> Dict[Tuple[str, Tuple[str, ...]], GroundAction]:
    return {(a.name, a.args): a for a in actions}


def parse_plan_action(text: str) -> Tuple[str, Tuple[str, ...]]:
    text = text.strip()
    m = re.match(r"^([^\s(]+)\((.*)\)$", text)
    if not m:
        parts = text.replace("(", " ").replace(")", " ").replace(",", " ").split()
        return parts[0], tuple(parts[1:])
    name = m.group(1)
    args = tuple(x.strip() for x in m.group(2).split(",") if x.strip())
    return name, args


def solve_plan(domain_file: Path, problem_file: Path, engine: str) -> List[Tuple[str, Tuple[str, ...]]]:
    problem = PDDLReader().parse_problem(str(domain_file), str(problem_file))
    with OneshotPlanner(name=engine) as planner:
        result = planner.solve(problem, timeout=SOLVE_TIMEOUT)
    if result.status not in {PlanGenerationResultStatus.SOLVED_SATISFICING, PlanGenerationResultStatus.SOLVED_OPTIMALLY} or result.plan is None:
        raise RuntimeError(f"planner failed: {result.status}")
    return [parse_plan_action(str(ai)) for ai in result.plan.actions]


def collect_states(
    initial: Set[Atom],
    actions: Sequence[GroundAction],
    plan: Sequence[Tuple[str, Tuple[str, ...]]],
    random_count: int,
    rng: random.Random,
) -> List[Set[Atom]]:
    idx = action_index(actions)
    states: List[Set[Atom]] = [set(initial)]
    state = set(initial)
    for key in plan:
        action = idx.get(key)
        if action is None:
            raise KeyError(f"plan action absent from pddlpy grounding: {key}")
        if not action.applicable(state):
            raise RuntimeError(f"plan action inapplicable in local replay: {key}")
        state = action.apply(state)
        states.append(set(state))

    state = set(initial)
    seen = {frozenset(state)}
    attempts = 0
    while len(states) < len(plan) + 1 + random_count and attempts < random_count * 200:
        attempts += 1
        applicable = [a for a in actions if a.applicable(state)]
        if not applicable:
            state = set(initial)
            continue
        action = rng.choice(applicable)
        state = action.apply(state)
        frozen = frozenset(state)
        if frozen not in seen:
            seen.add(frozen)
            states.append(set(state))
        if rng.random() < 0.035:
            state = set(initial)
    return states


def predicate_arities(train_arities: Mapping[str, int], test_arities: Mapping[str, int]) -> Dict[str, int]:
    out = dict(train_arities)
    for p, a in test_arities.items():
        if p in out and out[p] != a:
            raise ValueError(f"predicate arity conflict: {p}")
        out[p] = a
    return out


def feature_universe(action_arity: int, pred_arities: Mapping[str, int]) -> List[Feature]:
    features: List[Feature] = []
    for pred, arity in sorted(pred_arities.items()):
        for roles in itertools.product(range(action_arity), repeat=arity):
            features.append(("+", pred, tuple(roles)))
            features.append(("-", pred, tuple(roles)))
    for i in range(action_arity):
        for j in range(i + 1, action_arity):
            features.append(("+", "__eq__", (i, j)))
            features.append(("-", "__eq__", (i, j)))
    return features


def candidate_intersection(features: Sequence[Feature], positives: Sequence[Tuple[Set[Atom], Tuple[str, ...]]]) -> List[Feature]:
    if not positives:
        return []
    candidates = [f for f in features if feature_holds(f, positives[0][0], positives[0][1])]
    for state, args in positives[1:]:
        candidates = [f for f in candidates if feature_holds(f, state, args)]
        if not candidates:
            break
    return candidates


def reduce_rule(candidates: Sequence[Feature], negatives: Sequence[Tuple[Set[Atom], Tuple[str, ...]]]) -> Tuple[List[Feature], int]:
    if not negatives:
        return [], 0
    cover: Dict[Feature, Set[int]] = {}
    for feature in candidates:
        rejected = {i for i, (state, args) in enumerate(negatives) if not feature_holds(feature, state, args)}
        if rejected:
            cover[feature] = rejected
    uncovered = set(range(len(negatives)))
    selected: List[Feature] = []
    while uncovered:
        best = max(cover, key=lambda f: (len(cover[f] & uncovered), str(f)), default=None)
        if best is None or not (cover[best] & uncovered):
            break
        selected.append(best)
        uncovered.difference_update(cover[best])
        cover.pop(best, None)
    # backward irredundancy
    changed = True
    while changed:
        changed = False
        for feature in list(selected):
            trial = [f for f in selected if f != feature]
            if all(any(not feature_holds(f, s, a) for f in trial) for s, a in negatives):
                selected = trial
                changed = True
                break
    return sorted(selected, key=str), len(uncovered)


def canonical_delta(before: Set[Atom], after: Set[Atom], args: Sequence[str]) -> Tuple[frozenset[Tuple[str, Tuple[int, ...]]], frozenset[Tuple[str, Tuple[int, ...]]]]:
    add: Set[Tuple[str, Tuple[int, ...]]] = set()
    delete: Set[Tuple[str, Tuple[int, ...]]] = set()
    for atom in after.difference(before):
        c = canonical_atom(atom, args)
        if c is None:
            raise ValueError(f"non-local add effect {atom} for args {args}")
        add.add(c)
    for atom in before.difference(after):
        c = canonical_atom(atom, args)
        if c is None:
            raise ValueError(f"non-local delete effect {atom} for args {args}")
        delete.add(c)
    return frozenset(add), frozenset(delete)


def original_canonical_effect(action: GroundAction) -> Tuple[frozenset[Tuple[str, Tuple[int, ...]]], frozenset[Tuple[str, Tuple[int, ...]]]]:
    add: Set[Tuple[str, Tuple[int, ...]]] = set()
    delete: Set[Tuple[str, Tuple[int, ...]]] = set()
    for atom in action.eff_add:
        c = canonical_atom(atom, action.args)
        if c is None:
            raise ValueError(f"non-local original add effect {atom}")
        add.add(c)
    for atom in action.eff_del:
        c = canonical_atom(atom, action.args)
        if c is None:
            raise ValueError(f"non-local original delete effect {atom}")
        delete.add(c)
    return frozenset(add), frozenset(delete)


def instantiate_effect(effect: Tuple[frozenset[Tuple[str, Tuple[int, ...]]], frozenset[Tuple[str, Tuple[int, ...]]]], args: Sequence[str]) -> Tuple[Set[Atom], Set[Atom]]:
    add_c, del_c = effect
    add = {tuple([pred] + [args[i] for i in roles]) for pred, roles in add_c}
    delete = {tuple([pred] + [args[i] for i in roles]) for pred, roles in del_c}
    return add, delete


def learn_model(
    states: Sequence[Set[Atom]],
    actions: Sequence[GroundAction],
    pred_arities: Mapping[str, int],
    rng: random.Random,
) -> Dict[str, Any]:
    positives: Dict[Key, List[Tuple[Set[Atom], Tuple[str, ...]]]] = collections.defaultdict(list)
    negatives: Dict[Key, List[Tuple[Set[Atom], Tuple[str, ...]]]] = collections.defaultdict(list)
    effects: Dict[Key, Set[Any]] = collections.defaultdict(set)
    key_arity: Dict[Key, int] = {}

    for state in states:
        for action in actions:
            key = action.key
            key_arity[key] = len(action.args)
            if action.applicable(state):
                if len(positives[key]) < MAX_POS_PER_KEY:
                    positives[key].append((state, action.args))
                after = action.apply(state)
                effects[key].add(canonical_delta(state, after, action.args))
            else:
                bucket = negatives[key]
                if len(bucket) < MAX_NEG_PER_KEY:
                    bucket.append((state, action.args))
                else:
                    j = rng.randrange(len(bucket) + 1)
                    if j < len(bucket):
                        bucket[j] = (state, action.args)

    keys = sorted(set(key_arity) | set(positives) | set(negatives), key=str)
    model: Dict[str, Any] = {"rules": {}, "effects": {}, "conflicts": [], "stats": {}}
    for key in keys:
        key_s = repr(key)
        pos = positives.get(key, [])
        neg = negatives.get(key, [])
        if not pos:
            model["rules"][key_s] = {"always_false": True, "features": []}
        else:
            universe = feature_universe(key_arity[key], pred_arities)
            candidates = candidate_intersection(universe, pos)
            selected, uncovered = reduce_rule(candidates, neg)
            model["rules"][key_s] = {
                "always_false": False,
                "features": selected,
                "candidate_count": len(candidates),
                "uncovered_training_negatives": uncovered,
            }
        signatures = effects.get(key, set())
        if len(signatures) == 1:
            model["effects"][key_s] = next(iter(signatures))
        elif len(signatures) > 1:
            model["conflicts"].append(key_s)
        model["stats"][key_s] = {"positive": len(pos), "negative": len(neg), "effect_signatures": len(signatures)}
    return model


def key_string(key: Key) -> str:
    return repr(key)


def predict_applicable(model: Mapping[str, Any], state: Set[Atom], action: GroundAction) -> bool | None:
    rule = model["rules"].get(key_string(action.key))
    if rule is None:
        return None
    if rule["always_false"]:
        return False
    return all(feature_holds(tuple(f), state, action.args) for f in rule["features"])


def predict_successor(model: Mapping[str, Any], state: Set[Atom], action: GroundAction) -> Set[Atom] | None:
    effect = model["effects"].get(key_string(action.key))
    if effect is None:
        return None
    add, delete = instantiate_effect(effect, action.args)
    return state.difference(delete).union(add)


def audit_effect_exact(model: Mapping[str, Any], test_actions: Sequence[GroundAction]) -> Dict[str, Any]:
    by_key: Dict[Key, GroundAction] = {}
    for action in test_actions:
        by_key.setdefault(action.key, action)
    rows = []
    for key, action in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        learned = model["effects"].get(key_string(key))
        if learned is None:
            rows.append({"key": repr(key), "status": "no_learned_effect"})
            continue
        original = original_canonical_effect(action)
        rows.append({
            "key": repr(key),
            "status": "exact" if learned == original else "mismatch",
            "learned_add": sorted(map(str, learned[0])),
            "learned_del": sorted(map(str, learned[1])),
            "original_add": sorted(map(str, original[0])),
            "original_del": sorted(map(str, original[1])),
        })
    exact = sum(r["status"] == "exact" for r in rows)
    relevant = sum(r["status"] in {"exact", "mismatch"} for r in rows)
    return {"rows": rows, "exact": exact, "relevant": relevant, "all_exact": exact == len(rows)}


def evaluate_model(model: Mapping[str, Any], states: Sequence[Set[Atom]], actions: Sequence[GroundAction]) -> Dict[str, Any]:
    tp = tn = fp = fn = abstain = successor_tested = successor_exact = successor_missing = 0
    first_errors: List[Dict[str, Any]] = []
    for state_index, state in enumerate(states):
        for action in actions:
            truth = action.applicable(state)
            pred = predict_applicable(model, state, action)
            if pred is None:
                abstain += 1
                continue
            if truth and pred:
                tp += 1
            elif (not truth) and (not pred):
                tn += 1
            elif pred:
                fp += 1
            else:
                fn += 1
            if truth and pred:
                successor_tested += 1
                predicted = predict_successor(model, state, action)
                if predicted is None:
                    successor_missing += 1
                elif predicted == action.apply(state):
                    successor_exact += 1
                elif len(first_errors) < 10:
                    first_errors.append({"type": "successor", "state_index": state_index, "action": [action.name, list(action.args)]})
            if pred != truth and len(first_errors) < 10:
                first_errors.append({"type": "applicability", "state_index": state_index, "action": [action.name, list(action.args)], "truth": truth, "pred": pred})
    total = tp + tn + fp + fn + abstain
    return {
        "total": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "abstain": abstain,
        "accuracy": (tp + tn) / total if total else 0.0,
        "successor_tested": successor_tested,
        "successor_exact": successor_exact,
        "successor_missing": successor_missing,
        "successor_accuracy": successor_exact / successor_tested if successor_tested else 0.0,
        "first_errors": first_errors,
    }


def rollout_audit(model: Mapping[str, Any], initial: Set[Atom], actions: Sequence[GroundAction], plan: Sequence[Tuple[str, Tuple[str, ...]]]) -> Dict[str, Any]:
    idx = action_index(actions)
    true_state = set(initial)
    learned_state = set(initial)
    mismatch = None
    for i, key in enumerate(plan):
        action = idx[key]
        true_app = action.applicable(true_state)
        pred_app = predict_applicable(model, learned_state, action)
        if not true_app or pred_app is not True:
            mismatch = {"step": i, "reason": "applicability", "truth": true_app, "pred": pred_app, "action": [action.name, list(action.args)]}
            break
        true_state = action.apply(true_state)
        predicted = predict_successor(model, learned_state, action)
        if predicted is None:
            mismatch = {"step": i, "reason": "missing_effect", "action": [action.name, list(action.args)]}
            break
        learned_state = predicted
        if learned_state != true_state:
            mismatch = {"step": i, "reason": "state_mismatch", "action": [action.name, list(action.args)]}
            break
    return {"plan_length": len(plan), "exact_steps": len(plan) if mismatch is None else mismatch["step"], "pass": mismatch is None, "mismatch": mismatch}


def jsonable_model(model: Mapping[str, Any]) -> Dict[str, Any]:
    out = {"rules": {}, "effects": {}, "conflicts": list(model["conflicts"]), "stats": model["stats"]}
    for key, rule in model["rules"].items():
        out["rules"][key] = dict(rule)
        out["rules"][key]["features"] = [list(f[:2]) + [list(f[2])] for f in rule.get("features", [])]
    for key, effect in model["effects"].items():
        out["effects"][key] = {
            "add": [[p, list(r)] for p, r in sorted(effect[0])],
            "delete": [[p, list(r)] for p, r in sorted(effect[1])],
        }
    return out


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: real_pddl_trace_schema_equivalence.py SIFT_ROOT OUTPUT_DIR")
    root = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = choose_engine()
    report: Dict[str, Any] = {
        "source_commit": SIFT_COMMIT,
        "protocol": {
            "learner_input": "full symbolic states, grounded action name/arguments, success/failure, successful successor",
            "learner_forbidden_input": "PDDL precondition/effect membership",
            "train": "public SIFT Table-1 small instance",
            "test": "public SIFT Table-1 larger instance",
            "precondition_claim": "behavioral equivalence on held-out sampled states and all raw ground actions",
            "effect_claim": "canonical literal exact match plus held-out successor exactness",
        },
        "cases": {},
    }
    rng_master = random.Random(SEED)

    for case_i, (name, (domain_rel, train_rel, test_rel)) in enumerate(CASES.items()):
        case: Dict[str, Any] = {"domain": domain_rel, "train_problem": train_rel, "test_problem": test_rel}
        report["cases"][name] = case
        try:
            domain_file = root / domain_rel
            train_file = root / train_rel
            test_file = root / test_rel
            train_initial, train_actions, train_arities = load_grounded(domain_file, train_file)
            test_initial, test_actions, test_arities = load_grounded(domain_file, test_file)
            arities = predicate_arities(train_arities, test_arities)
            train_plan = solve_plan(domain_file, train_file, engine)
            test_plan = solve_plan(domain_file, test_file, engine)
            rng = random.Random(rng_master.randrange(1 << 63))
            train_states = collect_states(train_initial, train_actions, train_plan, TRAIN_RANDOM_STATES, rng)
            test_states = collect_states(test_initial, test_actions, test_plan, TEST_RANDOM_STATES, rng)
            start = time.time()
            model = learn_model(train_states, train_actions, arities, rng)
            learn_sec = time.time() - start
            effect_audit = audit_effect_exact(model, test_actions)
            heldout = evaluate_model(model, test_states, test_actions)
            rollout = rollout_audit(model, test_initial, test_actions, test_plan)
            case.update({
                "train_ground_actions": len(train_actions),
                "test_ground_actions": len(test_actions),
                "train_states": len(train_states),
                "test_states": len(test_states),
                "train_plan_length": len(train_plan),
                "test_plan_length": len(test_plan),
                "learn_sec": round(learn_sec, 6),
                "model": jsonable_model(model),
                "effect_audit": effect_audit,
                "heldout": heldout,
                "rollout": rollout,
            })
            case["pass"] = bool(
                not model["conflicts"]
                and effect_audit["all_exact"]
                and heldout["fp"] == 0
                and heldout["fn"] == 0
                and heldout["abstain"] == 0
                and heldout["successor_missing"] == 0
                and heldout["successor_exact"] == heldout["successor_tested"]
                and rollout["pass"]
            )
            print("CEDC_SCHEMA_CASE=" + json.dumps({
                "domain": name,
                "pass": case["pass"],
                "effect_exact": effect_audit["all_exact"],
                "fp": heldout["fp"], "fn": heldout["fn"], "abstain": heldout["abstain"],
                "tested": heldout["total"],
                "successor_exact": heldout["successor_exact"],
                "successor_tested": heldout["successor_tested"],
                "rollout": rollout["pass"], "rollout_steps": rollout["plan_length"],
            }, sort_keys=True), flush=True)
        except Exception as exc:
            case["pass"] = False
            case["error"] = f"{type(exc).__name__}: {exc}"
            case["traceback"] = traceback.format_exc()
            print("CEDC_SCHEMA_CASE=" + json.dumps({"domain": name, "pass": False, "error": case["error"]}, sort_keys=True), flush=True)
        (out_dir / f"schema_{name}.json").write_text(json.dumps(case, indent=2, sort_keys=True), encoding="utf-8")

    cases = list(report["cases"].values())
    summary = {
        "passed_cases": sum(bool(c.get("pass")) for c in cases),
        "total_cases": len(cases),
        "effect_exact_cases": sum(bool(c.get("effect_audit", {}).get("all_exact")) for c in cases),
        "heldout_attempts": sum(c.get("heldout", {}).get("total", 0) for c in cases),
        "fp": sum(c.get("heldout", {}).get("fp", 0) for c in cases),
        "fn": sum(c.get("heldout", {}).get("fn", 0) for c in cases),
        "abstain": sum(c.get("heldout", {}).get("abstain", 0) for c in cases),
        "successor_tested": sum(c.get("heldout", {}).get("successor_tested", 0) for c in cases),
        "successor_exact": sum(c.get("heldout", {}).get("successor_exact", 0) for c in cases),
        "rollout_steps": sum(c.get("rollout", {}).get("plan_length", 0) for c in cases),
        "rollout_passed_cases": sum(bool(c.get("rollout", {}).get("pass")) for c in cases),
    }
    summary["all_passed"] = summary["passed_cases"] == summary["total_cases"]
    report["summary"] = summary
    (out_dir / "REAL_PDDL_TRACE_SCHEMA_EQUIVALENCE.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# Real PDDL trace-learned schema equivalence",
        "",
        f"- Passed: **{summary['passed_cases']}/{summary['total_cases']}**",
        f"- Effect exact cases: **{summary['effect_exact_cases']}/{summary['total_cases']}**",
        f"- Held-out attempts: **{summary['heldout_attempts']}**",
        f"- FP/FN/abstain: **{summary['fp']}/{summary['fn']}/{summary['abstain']}**",
        f"- Successor exact: **{summary['successor_exact']}/{summary['successor_tested']}**",
        f"- Long rollout: **{summary['rollout_passed_cases']}/{summary['total_cases']}**, {summary['rollout_steps']} steps",
        "",
        "| Domain | Pass | Test attempts | FP | FN | Abstain | Effect exact | Rollout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, case in report["cases"].items():
        h = case.get("heldout", {})
        md.append(f"| {name} | {case.get('pass', False)} | {h.get('total', 0)} | {h.get('fp', 0)} | {h.get('fn', 0)} | {h.get('abstain', 0)} | {case.get('effect_audit', {}).get('all_exact', False)} | {case.get('rollout', {}).get('pass', False)} |")
    (out_dir / "REAL_PDDL_TRACE_SCHEMA_EQUIVALENCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("CEDC_SCHEMA_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
