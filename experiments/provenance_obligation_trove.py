#!/usr/bin/env python3
import base64
import hashlib
from pathlib import Path

_ROOT = Path(__file__).parent / "trove_src"
_B64 = "".join((_ROOT / f"part{i:02d}.b64").read_text(encoding="utf-8").strip() for i in range(5))
_SOURCE = base64.b64decode(_B64)
_EXPECTED = "f835c5ee407626ae14423610462f51e2cc3d8232abec36905aa86b42bcd5735f"
_ACTUAL = hashlib.sha256(_SOURCE).hexdigest()
if _ACTUAL != _EXPECTED:
    raise RuntimeError(f"TROVE source integrity failure: {_ACTUAL}")
exec(compile(_SOURCE, __file__, "exec"))
