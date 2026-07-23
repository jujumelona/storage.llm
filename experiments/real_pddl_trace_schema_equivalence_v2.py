#!/usr/bin/env python3
from __future__ import annotations

import sys
from typing import Any, Dict, Mapping, Sequence

import real_pddl_trace_schema_equivalence as base


def audit_effect_exact(model: Mapping[str, Any], test_actions: Sequence[base.GroundAction]) -> Dict[str, Any]:
    by_key: Dict[base.Key, base.GroundAction] = {}
    for action in test_actions:
        by_key.setdefault(action.key, action)

    rows = []
    for key, action in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        key_s = base.key_string(key)
        learned = model["effects"].get(key_s)
        rule = model["rules"].get(key_s)
        always_false = bool(rule and rule.get("always_false"))
        original = base.original_canonical_effect(action)

        if learned is None and always_false:
            rows.append({
                "key": repr(key),
                "status": "semantically_inapplicable_partition",
                "reason": "no successful transition exists; effect is not identifiable or executable",
                "original_add": sorted(map(str, original[0])),
                "original_del": sorted(map(str, original[1])),
            })
            continue
        if learned is None:
            rows.append({
                "key": repr(key),
                "status": "missing_executable_effect",
                "original_add": sorted(map(str, original[0])),
                "original_del": sorted(map(str, original[1])),
            })
            continue
        rows.append({
            "key": repr(key),
            "status": "exact" if learned == original else "mismatch",
            "learned_add": sorted(map(str, learned[0])),
            "learned_del": sorted(map(str, learned[1])),
            "original_add": sorted(map(str, original[0])),
            "original_del": sorted(map(str, original[1])),
        })

    executable = [r for r in rows if r["status"] != "semantically_inapplicable_partition"]
    exact = sum(r["status"] == "exact" for r in executable)
    raw_exact = sum(r["status"] == "exact" for r in rows)
    return {
        "rows": rows,
        "exact": exact,
        "relevant": len(executable),
        "all_exact": exact == len(executable),
        "raw_syntactic_exact": raw_exact == len(rows),
        "raw_exact": raw_exact,
        "all_partitions": len(rows),
        "semantically_inapplicable_partitions": sum(r["status"] == "semantically_inapplicable_partition" for r in rows),
    }


base.audit_effect_exact = audit_effect_exact

if __name__ == "__main__":
    raise SystemExit(base.main())
