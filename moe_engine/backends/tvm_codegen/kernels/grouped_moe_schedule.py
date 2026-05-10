from __future__ import annotations

import json
from pathlib import Path


def apply_default_schedule(module, target):
    """Apply TVM DLight default schedule when TVM and DLight are installed.

    The function is deliberately safe: if TVM is not installed or the API differs,
    it returns the input module unchanged and records no fake success.
    """
    try:
        import tvm  # noqa: F401
        from tvm import dlight as dl
    except Exception:
        return module

    try:
        with tvm.target.Target(target):
            return dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
            )(module)
    except Exception:
        return module


def write_tuning_record(work_dir: str, profile: dict, status: dict) -> None:
    out = Path(work_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "storage_llm_tvm_tuning_status.json").write_text(
        json.dumps({"profile": profile, "status": status}, indent=2),
        encoding="utf-8",
    )
