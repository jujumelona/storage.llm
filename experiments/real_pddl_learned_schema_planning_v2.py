#!/usr/bin/env python3
from __future__ import annotations

import ast
import inspect

import real_pddl_learned_schema_planning as base


def compiler_leak_audit():
    source = inspect.getsource(base.compile_propositional_problem) + "\n" + inspect.getsource(base.compiled_action_from_learned)
    tree = ast.parse(source)
    forbidden = {"pre_pos", "pre_neg", "eff_add", "eff_del"}
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr not in forbidden:
            continue
        root = node.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id == "action":
            hits.append(f"action.{node.attr}")
    return {
        "method": "python_ast_attribute_access",
        "forbidden_attribute_hits": sorted(set(hits)),
        "pass": not hits,
    }


base.compiler_leak_audit = compiler_leak_audit

if __name__ == "__main__":
    raise SystemExit(base.main())
