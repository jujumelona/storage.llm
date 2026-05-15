#!/usr/bin/env python3
"""Run JUJU generation contract tests without requiring pytest.

The contract test file uses plain test_* functions and assert statements.  This
runner makes the same checks available to CTest/CI even in minimal environments
where pytest is not installed.
"""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
import traceback


def main() -> int:
    test_path = pathlib.Path(__file__).with_name("test_juju_generation_contract_config.py")
    spec = importlib.util.spec_from_file_location("test_juju_generation_contract_config", test_path)
    if spec is None or spec.loader is None:
        print(f"failed to load test module spec: {test_path}", file=sys.stderr)
        return 2
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(test_path.parents[2]))
    spec.loader.exec_module(module)

    tests = [
        (name, obj)
        for name, obj in vars(module).items()
        if name.startswith("test_") and callable(obj)
    ]
    tests.sort(key=lambda item: item[0])
    if not tests:
        print("no JUJU generation contract tests found", file=sys.stderr)
        return 2

    failures = 0
    for name, fn in tests:
        try:
            sig = inspect.signature(fn)
            if sig.parameters:
                raise RuntimeError(f"{name} requires unsupported parameters: {list(sig.parameters)}")
            fn()
            print(f"PASS {name}")
        except Exception:
            failures += 1
            print(f"FAIL {name}", file=sys.stderr)
            traceback.print_exc()

    print(f"JUJU generation contract tests: {len(tests) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
