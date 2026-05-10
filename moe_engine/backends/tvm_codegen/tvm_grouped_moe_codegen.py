from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from kernels.grouped_moe_ir import build_grouped_moe_ir
from tvm_dependency import ensure_tvm
from kernels.grouped_moe_targets import normalize_backend, shared_library_suffix


CPP_SOURCE = r"""
#include <stdint.h>
#include <math.h>
#include <vector>
#include <new>

#if defined(_WIN32)
#define STORAGELLM_TVM_EXPORT __declspec(dllexport)
#else
#define STORAGELLM_TVM_EXPORT __attribute__((visibility("default")))
#endif

enum {
    STORAGELLM_BACKEND_CPU = 1
};

typedef struct moe_grouped_expert_device_task {
    int32_t layer;
    int32_t expert;
    const void* gate_weight;
    const void* up_weight;
    const void* down_weight;
    const void* d_input;
    uint32_t input_stride;
    const uint32_t* d_token_indices;
    const float* d_token_weights;
    uint32_t assignment_offset;
    uint32_t assignment_count;
    void* d_accum;
    uint32_t accum_stride;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t activation_mode;
} moe_grouped_expert_device_task_t;

static inline float storagellm_silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float storagellm_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

static inline float storagellm_gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));
}

static inline float storagellm_apply_activation(float x, uint32_t mode) {
    if (mode == 1u) return storagellm_gelu(x);
    if (mode == 2u) return storagellm_gelu_tanh(x);
    return storagellm_silu(x);
}

extern "C" STORAGELLM_TVM_EXPORT int storagellm_tvm_grouped_moe_entry(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue
) {
    (void)stream_or_queue;

    // This generated library is a CPU/LLVM C ABI correctness bridge.
    // GPU backends must use target-specific TVM/CUDA/Metal wrappers.
    if (backend != STORAGELLM_BACKEND_CPU || !tasks || task_count == 0u) {
        return 0;
    }

    for (uint32_t t = 0; t < task_count; ++t) {
        const moe_grouped_expert_device_task_t& task = tasks[t];
        if (!task.gate_weight || !task.up_weight || !task.down_weight ||
            !task.d_input || !task.d_accum ||
            !task.d_token_indices || task.hidden_size == 0u ||
            task.intermediate_size == 0u || task.assignment_count == 0u) {
            return 0;
        }

        const uint32_t H = task.hidden_size;
        const uint32_t I = task.intermediate_size;
        const float* gate = static_cast<const float*>(task.gate_weight);
        const float* up = static_cast<const float*>(task.up_weight);
        const float* down = static_cast<const float*>(task.down_weight);
        const float* input = static_cast<const float*>(task.d_input);
        float* accum = static_cast<float*>(task.d_accum);

        std::vector<float> gate_buf;
        std::vector<float> up_buf;
        std::vector<float> mid_buf;
        try {
            gate_buf.resize(I);
            up_buf.resize(I);
            mid_buf.resize(I);
        } catch (const std::bad_alloc&) {
            return 0;
        }

        for (uint32_t a = 0; a < task.assignment_count; ++a) {
            const uint32_t global_a = task.assignment_offset + a;
            const uint32_t token = task.d_token_indices[global_a];
            const float token_weight = task.d_token_weights ? task.d_token_weights[global_a] : 1.0f;

            const float* x = input + (uint64_t)token * (uint64_t)task.input_stride;
            float* y = accum + (uint64_t)token * (uint64_t)task.accum_stride;

            for (uint32_t i = 0; i < I; ++i) {
                double g = 0.0;
                double u = 0.0;
                const float* wg = gate + (uint64_t)i * (uint64_t)H;
                const float* wu = up + (uint64_t)i * (uint64_t)H;
                for (uint32_t h = 0; h < H; ++h) {
                    g += (double)x[h] * (double)wg[h];
                    u += (double)x[h] * (double)wu[h];
                }
                gate_buf[i] = (float)g;
                up_buf[i] = (float)u;
                mid_buf[i] = storagellm_apply_activation(gate_buf[i], task.activation_mode) * up_buf[i];
            }

            for (uint32_t h = 0; h < H; ++h) {
                double v = 0.0;
                const float* wd = down + (uint64_t)h * (uint64_t)I;
                for (uint32_t i = 0; i < I; ++i) {
                    v += (double)mid_buf[i] * (double)wd[i];
                }
                y[h] += token_weight * (float)v;
            }
        }
    }

    return 1;
}
"""


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out
    except subprocess.CalledProcessError as exc:
        return exc.returncode, exc.output
    except FileNotFoundError as exc:
        return 127, str(exc)


def _compile_cpp(src: Path, out: Path) -> tuple[bool, str, list[str]]:
    system = platform.system().lower()
    out.parent.mkdir(parents=True, exist_ok=True)

    cxx = os.environ.get("CXX")
    if cxx:
        candidates = [cxx]
    elif system == "windows":
        candidates = ["cl", "g++", "clang++"]
    else:
        candidates = ["g++", "clang++"]

    logs: list[str] = []

    for compiler in candidates:
        if shutil.which(compiler) is None:
            logs.append(f"{compiler}: not found")
            continue

        if os.path.basename(compiler).lower() in ("cl.exe", "cl"):
            cmd = [
                compiler,
                "/nologo",
                "/LD",
                "/EHsc",
                "/std:c++17",
                str(src),
                f"/Fe:{out}",
            ]
        else:
            cmd = [
                compiler,
                "-std=c++17",
                "-O2",
                "-shared",
                "-fPIC",
                str(src),
                "-o",
                str(out),
            ]

        code, log = _run(cmd)
        logs.append("$ " + " ".join(cmd) + "\n" + log)
        if code == 0 and out.exists():
            return True, compiler, logs

    return False, "", logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-out", default=None)
    parser.add_argument("--source-out", default=None)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    profile = json.loads(Path(args.profile).read_text(encoding="utf-8-sig"))
    spec = build_grouped_moe_ir(profile)
    backend = normalize_backend(spec.backend)
    tvm_dependency = ensure_tvm(backend, str(Path(args.out).parent / "tvm_dependency_status.json"))

    out = Path(args.out)
    if out.suffix == "":
        out = out.with_suffix(shared_library_suffix())

    source_out = Path(args.source_out) if args.source_out else out.with_suffix(".cpp")
    source_out.parent.mkdir(parents=True, exist_ok=True)
    source_out.write_text(CPP_SOURCE, encoding="utf-8")

    metadata = {
        "backend": "tvm_codegen",
        "target_backend": backend,
        "target": spec.target,
        "hidden": spec.hidden,
        "intermediate": spec.intermediate,
        "dtype": spec.dtype,
        "max_experts": spec.max_experts,
        "max_assignments": spec.max_assignments,
        "entry_symbol": "storagellm_tvm_grouped_moe_entry",
        "source": str(source_out),
        "out": str(out),
        "compiled": False,
        "tvm_dependency": tvm_dependency,
        "load_env": {
            "cpu": "STORAGELLM_TVM_CPU_MOE_LIB",
            "cuda": "STORAGELLM_TVM_CUDA_MOE_LIB",
            "metal": "STORAGELLM_TVM_METAL_MOE_LIB",
            "vulkan": "STORAGELLM_TVM_VULKAN_MOE_LIB",
            "opencl": "STORAGELLM_TVM_OPENCL_MOE_LIB",
        }.get(backend, "STORAGELLM_TVM_MOE_LIB"),
        "important": (
            "CPU/LLVM path emits a working FP32 C ABI reference kernel. "
            "CUDA/Metal/Vulkan/OpenCL must not be marked available until a "
            "target-specific TVM/device kernel is generated and validated."
        ),
    }

    logs: list[str] = []
    if backend != "cpu":
        metadata["compiled"] = False
        metadata["reason"] = (
            f"{backend} TVM device wrapper generation is disabled in this source-complete build; "
            "engine-owned native backend adapters handle GPU fast paths."
        )
        print("[storageLLM] non-CPU TVM requested; wrote metadata only. Use engine-owned native GPU adapters for fast paths.")
    elif args.no_compile:
        metadata["compiled"] = False
        metadata["reason"] = "--no-compile"
    else:
        ok, compiler, logs = _compile_cpp(source_out, out)
        metadata["compiled"] = ok
        metadata["compiler"] = compiler
        metadata["compile_log"] = logs
        if ok:
            print("[storageLLM] generated C ABI TVM bridge library:", out)
            print("[storageLLM] set", metadata["load_env"], "=", out)
        else:
            print("[storageLLM] failed to compile generated C ABI library.")
            print("[storageLLM] source remains at:", source_out)

    metadata_out = Path(args.metadata_out) if args.metadata_out else Path(str(out) + ".json")
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[storageLLM] wrote metadata:", metadata_out)


if __name__ == "__main__":
    main()
