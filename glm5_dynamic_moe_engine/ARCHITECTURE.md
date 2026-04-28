# Architecture

## Fixed Model Assumptions

This folder is for GLM-5.1 NVFP4 only.

Hardcode or precompute:

- layer count and MoE layer layout,
- expert id range per layer,
- `gate_proj`, `up_proj`, `down_proj` triplets,
- safetensors shard and byte offsets,
- scale4 entry offsets,
- multimodal block ids.

The hot path should operate on integer ids and offsets, not tensor-name strings.
Strings are acceptable in manifest/build tools only.

## Residency Tiers

### Tier 0: VRAM

Use for:

- always-needed runtime state,
- current layer selected experts,
- repeatedly hot experts,
- image/multimodal blocks while active,
- next-layer prefetch if space remains.

### Tier 1: RAM

Use for:

- mmap original safetensors or packed blob,
- pinned staging buffers,
- warm experts that do not fit in VRAM,
- blocks recently evicted from VRAM.

### Tier 2: DB/Blob

Use for:

- cold expert backing store,
- whole-model storage when RAM is insufficient,
- slow fallback streaming.

DB/blob is not a relational hot path. It is a page source. Reads should be large,
aligned, and keyed by fixed block id/offset.

## Platform Runtime Matrix

The manifest and cache policy are shared across platforms. Only the backend
adapter changes.

| Target | Backend | Transfer path | Notes |
|---|---|---|---|
| Windows PC + NVIDIA | CUDA | mmap/page-cache -> pinned host -> GPU copy stream; optional DirectStorage/GDS | Best target for async transfer overlap. |
| Linux PC + NVIDIA | CUDA | mmap/page-cache -> pinned host -> GPU copy stream; optional GDS | Same cache/prefetch API as Windows. |
| Windows/Linux PC + AMD | HIP/ROCm | mmap/page-cache -> pinned host -> GPU copy stream | Same cache/prefetch API as CUDA; Vulkan/OpenCL fallback. |
| Windows/Linux PC + Intel | Level Zero / SYCL / oneAPI | mmap/page-cache -> shared or pinned staging -> GPU queue | Prefer Level Zero/SYCL when compiled; DirectML/OpenCL/Vulkan fallback. |
| MacBook / Apple Silicon | Metal | mmap/page-cache -> unified memory / Metal buffer | No CUDA pinned-host path. Avoid DirectStorage/GDS assumptions. |
| Any CPU-only system | CPU | mmap/page-cache streaming | Correct fallback, slower. |
| Other PC GPU | Vulkan/DirectML/OpenCL/WebGPU | backend adapter decides | Keep API compatible, tune after profiling. |

The engine API exposes `glm5_backend_caps_t` so the application can choose
budgets and feature flags without guessing the platform. Static conversion
artifacts set model layout; runtime backend caps set transfer mechanics.

## Block Granularity

Cache units should match execution units:

```text
layer + expert + {gate_proj, up_proj, down_proj}
```

Do not manage every tensor independently unless the backend requires it. Router
selection uses experts, so expert triplets are the natural cache unit.

## Active-Set VRAM Strategy

The optimal MoE path is active-set residency:

```text
mandatory dense/runtime state
+ current router top-k expert triplets
+ lookahead expert triplets when prefetch is confident
```

The GLM-5.1 runtime constants are:

| Field | Value |
|---|---:|
| Hidden layers | 79 |
| MoE layers | 3 through 78 |
| Experts per MoE layer | 256 |
| Routed experts per token | 8 |
| Total expert triplet cache units | 19,456 |

Only the selected `top_k=8` experts are required for a token at a MoE layer.
Keeping those triplets and mandatory runtime state in VRAM is the fast-path
target. Loading every expert for every MoE layer is a capacity failure mode, not
an optimization target. Cold or low-score expert triplets remain in RAM or
storage until router telemetry makes them likely enough to promote.

Hugging Face is a provisioning source, not a runtime dependency. The converted
StorageLLM artifact repo is expected at:

```text
https://huggingface.co/storagejuju/GLM5.1-4q-storage
```

The official upstream base model reference is:

```text
https://huggingface.co/zai-org/GLM-5.1
```

## Admission Policy

1. Pinned mandatory blocks cannot be evicted.
2. Current selected experts have high priority.
3. Repeatedly used experts gain hot score and can stay in VRAM.
4. Unused experts decay and demote VRAM -> RAM -> DB/blob.
5. If a block is larger than available VRAM, compute from RAM/DB streaming path.

## Multimodal Policy

Vision blocks stay cold until image input is present.

When image input is detected:

1. request vision encoder blocks,
2. promote to VRAM if possible,
3. otherwise use RAM/DB streaming backend,
4. keep hot while repeated image requests arrive,
5. decay/demote when text-only traffic resumes.

## Compression Policy

Default enabled:

- packed NVFP4 weights,
- scale4 decoded inside dequant/matmul.

Runtime admission requires the full codec contract in
`COMPRESSION_CONTRACT.md`: tensor location, stored format, decode path,
residency ownership, and correctness proof. Ratio-only results are not enough.

Feature-flag only:

- local mixed-bit down-proj experiments,
- Tucker2/QAT/RPCA candidates after eval proof.

Rejected for default runtime:

- full fp16/bf16 materialized weights,
- loading all MoE experts into VRAM,
- generic database lookup per tensor per matmul.
