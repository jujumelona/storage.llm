import argparse
import json
import platform
from pathlib import Path

from kernels.grouped_moe_targets import detect_cuda_profile


def detect_default_target() -> dict:
    cuda = detect_cuda_profile()
    if cuda:
        return cuda

    if platform.system().lower() == "darwin":
        return {
            "backend": "metal",
            "target": "metal",
            "gpu_name": "apple_metal",
        }

    return {
        "backend": "cpu",
        "target": "llvm",
        "gpu_name": "cpu",
    }


def read_model_profile(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    intermediate = (
        data.get("expert_intermediate_size")
        or data.get("intermediate_size")
        or data.get("ffn_hidden_size")
    )
    if intermediate is None:
        raise ValueError("model metadata has no expert_intermediate_size/intermediate_size/ffn_hidden_size")
    return {
        "hidden": int(data["hidden_size"]),
        "intermediate": int(intermediate),
        "dtype": (
            data.get("expert_gpu_layout_dtype")
            or data.get("weight_dtype")
            or "fp32"
        ).lower(),
        "experts_per_layer": int(data.get("experts_per_layer", 0)),
        "top_k": int(data.get("top_k", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-meta", required=True)
    parser.add_argument("--runtime-telemetry", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    target = detect_default_target()
    model = read_model_profile(args.model_meta)

    profile = {
        **target,
        **model,
        "max_experts": max(1, min(8, model.get("experts_per_layer") or 8)),
        "max_assignments": 256,
        "source": "auto",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print("[storageLLM] wrote auto TVM profile:", args.out)


if __name__ == "__main__":
    main()
