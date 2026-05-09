from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from kernels.grouped_moe_schedule import write_tuning_record
from tvm_dependency import ensure_tvm


def _build_proxy_ir(profile: dict):
    """Build a real TVM TIR workload for the current MoE shape.

    This is a fail-closed MetaSchedule workload: it tunes the dense gate/up/down
    compute pattern for the detected shape.  It does not claim that CUDA/Metal/
    Vulkan grouped dynamic dispatch is complete unless the downstream codegen
    successfully emits and validates that backend.
    """
    import tvm
    from tvm import te

    hidden = max(1, int(profile["hidden"]))
    intermediate = max(1, int(profile["intermediate"]))
    assignments = max(1, min(int(profile.get("max_assignments", 256)), 1024))

    x = te.placeholder((assignments, hidden), name="x", dtype="float32")
    gate_w = te.placeholder((intermediate, hidden), name="gate_w", dtype="float32")
    up_w = te.placeholder((intermediate, hidden), name="up_w", dtype="float32")
    down_w = te.placeholder((hidden, intermediate), name="down_w", dtype="float32")
    kh = te.reduce_axis((0, hidden), name="kh")
    ki = te.reduce_axis((0, intermediate), name="ki")
    gate = te.compute(
        (assignments, intermediate),
        lambda a, i: te.sum(x[a, kh] * gate_w[i, kh], axis=kh),
        name="gate",
    )
    up = te.compute(
        (assignments, intermediate),
        lambda a, i: te.sum(x[a, kh] * up_w[i, kh], axis=kh),
        name="up",
    )
    # Linear proxy for the activation multiply.  The generated C ABI bridge still
    # uses the exact activation; this workload exists so MetaSchedule has a real
    # measurable IRModule instead of an import-only placeholder.
    mid = te.compute((assignments, intermediate), lambda a, i: gate[a, i] * up[a, i], name="mid")
    out = te.compute(
        (assignments, hidden),
        lambda a, h: te.sum(mid[a, ki] * down_w[h, ki], axis=ki),
        name="out",
    )
    prim = te.create_prim_func([x, gate_w, up_w, down_w, out])
    return tvm.IRModule({"main": prim})


def try_tvm_metaschedule(profile: dict, work_dir: str, trials: int) -> dict:
    dep = ensure_tvm(str(profile.get("backend", "auto")), str(Path(work_dir) / "tvm_dependency_status.json"))
    if not dep.get("ok"):
        return {
            "metaschedule_available": False,
            "tuned": False,
            "reason": "TVM install/import failed",
            "dependency": dep,
        }
    try:
        import tvm
        import tvm.meta_schedule as ms
    except Exception as exc:
        return {
            "metaschedule_available": False,
            "tuned": False,
            "reason": f"TVM MetaSchedule import failed: {exc}",
            "dependency": dep,
        }

    target = str(profile.get("target") or ("cuda" if profile.get("backend") == "cuda" else "llvm"))
    out = Path(work_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        mod = _build_proxy_ir(profile)
        # DLight default schedule is applied first when available; MetaSchedule
        # then measures candidates for the actual target.
        try:
            from kernels.grouped_moe_schedule import apply_default_schedule
            mod = apply_default_schedule(mod, target)
        except Exception:
            pass
        tuned_db = ms.tune_tir(
            mod=mod,
            target=tvm.target.Target(target),
            work_dir=str(out / "meta_schedule_db"),
            max_trials_global=max(1, int(trials)),
            num_trials_per_iter=min(64, max(1, int(trials))),
        )
        return {
            "metaschedule_available": True,
            "tuned": True,
            "target": target,
            "workload": "dense_moe_shape_proxy_gate_up_down",
            "requested_trials": trials,
            "database": str(out / "meta_schedule_db"),
            "dependency": dep,
            "database_type": type(tuned_db).__name__,
        }
    except Exception as exc:
        return {
            "metaschedule_available": True,
            "tuned": False,
            "target": target,
            "reason": f"MetaSchedule run failed: {exc}",
            "requested_trials": trials,
            "dependency": dep,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--codegen-out", default=None)
    args = parser.parse_args()

    profile = json.loads(Path(args.profile).read_text(encoding="utf-8-sig"))
    status = try_tvm_metaschedule(profile, args.work_dir, args.trials)
    write_tuning_record(args.work_dir, profile, status)

    print("[storageLLM] tuning status:", json.dumps(status, ensure_ascii=False))

    if args.codegen_out:
        script = Path(__file__).with_name("tvm_grouped_moe_codegen.py")
        cmd = [
            sys.executable,
            str(script),
            "--profile",
            args.profile,
            "--out",
            args.codegen_out,
        ]
        print("[storageLLM] running codegen:", " ".join(cmd))
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
