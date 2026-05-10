from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def tvm_importable() -> tuple[bool, str]:
    if importlib.util.find_spec("tvm") is None:
        return False, "tvm module not found"
    try:
        import tvm  # type: ignore
        return True, getattr(tvm, "__version__", "unknown")
    except Exception as exc:
        return False, str(exc)


def ensure_tvm(backend: str = "auto", status_out: str | None = None) -> dict:
    ok, version = tvm_importable()
    if ok:
        return {"ok": True, "version": version, "installed_by_script": False}
    here = Path(__file__).resolve()
    root = here.parents[3]
    script = root / "scripts" / "install_tvm_dependency.py"
    if not script.exists():
        return {"ok": False, "reason": f"installer missing: {script}"}
    cmd = [sys.executable, str(script), "--backend", backend]
    if status_out:
        cmd += ["--status-out", status_out]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ok, version = tvm_importable()
    status = {"ok": ok, "version": version, "installed_by_script": True, "returncode": proc.returncode, "log_tail": proc.stdout[-4000:]}
    return status
