import os
import platform
import subprocess


def normalize_backend(name: str) -> str:
    value = (name or "").strip().lower()
    if value in ("cuda", "nvidia"):
        return "cuda"
    if value in ("metal", "apple"):
        return "metal"
    if value in ("vulkan", "vk"):
        return "vulkan"
    if value in ("opencl", "cl"):
        return "opencl"
    if value in ("cpu", "llvm", "x86", "x64"):
        return "cpu"
    return value or "cpu"


def target_from_backend(backend: str, arch: str | None = None) -> str:
    backend = normalize_backend(backend)
    if backend == "cuda":
        return f"cuda -arch={arch}" if arch else "cuda"
    if backend == "metal":
        return "metal"
    if backend == "vulkan":
        return "vulkan"
    if backend == "opencl":
        return "opencl"
    return "llvm"


def shared_library_suffix() -> str:
    if os.name == "nt":
        return ".dll"
    if platform.system().lower() == "darwin":
        return ".dylib"
    return ".so"


def detect_cuda_profile() -> dict | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    fields = [x.strip() for x in out.splitlines()[0].split(",")]
    if len(fields) < 2:
        return None
    name = fields[0]
    compute_cap = fields[1]
    sm = "sm_" + compute_cap.replace(".", "")
    return {
        "backend": "cuda",
        "target": f"cuda -arch={sm}",
        "gpu_name": name,
        "compute_cap": compute_cap,
        "arch": sm,
        "memory": fields[2] if len(fields) > 2 else "",
    }
