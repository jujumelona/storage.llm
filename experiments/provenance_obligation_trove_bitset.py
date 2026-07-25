#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAST_PATH = HERE / "provenance_obligation_trove_fast.py"
spec = importlib.util.spec_from_file_location("trove_fast", FAST_PATH)
f = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(f)


def exact_bitset_lcs_ratio(a, b):
    if not a or not b:
        return 0.0
    a = a[:180]
    b = b[:260]
    masks = {}
    for i, token in enumerate(b):
        masks[token] = masks.get(token, 0) | (1 << i)
    state = 0
    for token in a:
        union = state | masks.get(token, 0)
        state = union & ~(union - ((state << 1) | 1))
    return state.bit_count() / max(1, len(a))


f.b.lcs_ratio = exact_bitset_lcs_ratio

if __name__ == "__main__":
    f.main()
