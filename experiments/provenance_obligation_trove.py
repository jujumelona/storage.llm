#!/usr/bin/env python3
import base64
import hashlib
from pathlib import Path

_ROOT = Path(__file__).parent / "trove_src"
_PARTS = ["part00.b64", "part01.b64", "part02.b64"] + [f"part03_{i}.b64" for i in range(5)] + ["part04.b64"]
_B64 = "".join((_ROOT / name).read_text(encoding="utf-8").strip() for name in _PARTS)
_SOURCE = base64.b64decode(_B64)
_EXPECTED = "f835c5ee407626ae14423610462f51e2cc3d8232abec36905aa86b42bcd5735f"
_ACTUAL = hashlib.sha256(_SOURCE).hexdigest()
if _ACTUAL != _EXPECTED:
    raise RuntimeError(f"TROVE source integrity failure: {_ACTUAL}")
exec(compile(_SOURCE, __file__, "exec"))
