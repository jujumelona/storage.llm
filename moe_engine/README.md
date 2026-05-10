# StorageLLM: MoE-Native Entropy Decoding Engine

## 🎯 Project Goal
**"AI 모델의 1/100 스케일 구현을 위한 엔트로피 기반 디코딩 아키텍처 수립"**
**"Establishing an entropy-based decoding architecture for 1/100 scale AI model implementation"**

### 💡 Key Approach
1. **Entropy Rethinking:** Beyond simple quantization, we optimize information density by redefining weight (W) entropy.
2. **Decoder-Centric Logic:** A high-efficiency decoder reinterprets compressed data in real-time to restore inference performance.
3. **Extreme Efficiency:** Achieves a 99% reduction in model capacity compared to SOTA models while maintaining core domain performance.
4. **MoE-Exclusive:** This engine is specifically designed for Mixture-of-Experts structures, leveraging sparsity for high-performance storage offloading.

---

## Overview
Clean runtime folder for StorageLLM GGUF offload MoE engines.

The primary target is now the offload-native GGUF layout: normal `.gguf` files
with StorageLLM execution metadata embedded under the `offload.*` GGUF KV
namespace. This layout feeds the storage-first runtime:

```text
VRAM first -> RAM second -> DB/blob third
```

If there is enough VRAM, hot blocks stay there. If not, blocks are demoted to
RAM. If RAM is also tight, cold blocks are backed only by DB/blob pages and are
streamed tile-by-tile, even if that is slow.

## Model Layout

Place model files outside the repository and pass that folder as `model_root`.
The preferred dedicated GGUF layout is:

```text
<model_root>/
  tokenizer.json
  <quant_or_format_subdir>/
    <model>-00001-of-000NN.gguf
    <model>-00002-of-000NN.gguf
    ...
    <model>-000NN-of-000NN.gguf
```

The engine scans GGUF metadata headers only. If a shard contains
`offload.metadata_v2`, model-root validation uses the embedded
`offload.file_count` / `offload.file_index` contract instead of the hardcoded
legacy part table. QKV settings come from these GGUF KV entries when present:

```text
offload.weight_quant_family
offload.weight_kernel_family
offload.weight_bits
offload.weight_block_size
offload.qkv_k_bits
offload.qkv_v_bits
offload.qkv_group_size
offload.qkv_page_size_tokens
offload.qkv_sink_tokens
```

Weight quantization is dispatched by family, not by bit count alone. `UD-IQ2_M`
and other IQ formats need IQ-specific metadata and kernels; `MXFP4` must use an
MXFP4 decode table, not the NVFP4 table. If the runtime sees an unknown family,
it refuses the direct tensor dot fallback instead of silently decoding it as
FP4. The packed QKV cache remains the default KV cache path and is independently
configured by the `offload.qkv_*` fields.

When no legacy tensor index is present, `moe_pc_engine_server <model_root>`
falls back to this GGUF header contract. `/health` then exposes
`offloadGgufValid`, `offloadGgufFileCount`, `offloadGgufTensorHeaders`, and the
format-selected QKV fields so the runtime state is visible without pretending a
full executable tensor table was loaded.

### Hugging Face Sources

Hugging Face is the provisioning and distribution layer. It is not used as a
tensor lookup service inside the decode hot path. Download or sync the model
artifacts first, then pass the local folder as `model_root`.

| Purpose | Repository |
| --- | --- |
| Upstream base model | stored in the artifact metadata when provided |
| StorageLLM offload-native GGUF artifacts | target Hugging Face GGUF offload repo |
| Hugging Face model download docs | <https://huggingface.co/docs/hub/models-downloading> |

Provisioning example for downloading the ready-to-run model files:

```text
hf download <target Hugging Face GGUF offload repo> --local-dir <model_root>
moe_pc_engine_server <model_root>
```

Do not download only `*.juju`. A runnable StorageLLM model root needs the
weight package, the runtime assets, and the metadata sidecars:

```text
*.juju
*.juju.idx
*.juju.verify.json
verify/*.json
runtime_assets_manifest.json
storagellm_performance_metadata_manifest.json
metadata/**
README.md
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
added_tokens.json
chat_template.jinja
tokenizer.model
sentencepiece.bpe.model
tiktoken.model
vocab.json
merges.txt
processor_config.json
preprocessor_config.json
image_processor_config.json
feature_extractor.json
video_preprocessor_config.json
audio_config.json
tokenization_*.py
configuration_*.py
modeling_*.py
processing_*.py
*_processor.py
*_processing.py
*_utils.py
```

The engine consumes these sidecars during `model_root` load. Runtime manifest,
generation/tokenizer/processor config, graph/priority/prefetch/residency, QKV,
offload policy, validation, and metadata JSON are merged into the runtime
metadata path so attention, router, RoPE, embedding, graph hints, and planning
code can see them.

The expected StorageLLM Hugging Face repo must contain GGUF files with embedded
`offload.runtime_tensor_index_v1` and scalar `offload.*` metadata. The official
upstream model repo remains the source reference.

## Public API

Use `include/moe_pc_engine.h`.

Important entry points:

```c
moe_pc_engine_config_t cfg = moe_pc_default_config();
moe_pc_engine_t* engine = moe_pc_engine_create(&cfg);

moe_model_root_check_t check;
int ok = moe_storage_validate_model_root(model_root, &check);

moe_pc_engine_set_model_root(engine, model_root);
moe_pc_engine_get_forward_status(engine, &status);

moe_pc_engine_destroy(engine);
```

## Build Integration

Common include roots still come from the repository root:

```text
.
loader
engine_core/core
engine_core
```

Add this model runtime include root:

```text
moe_engine/include
```

Model-specific sources commonly needed by an embedding application:

```text
moe_engine/src/moe_pc_engine.cpp
moe_engine/src/model_shape.cpp
moe_engine/src/hardcode_constants.cpp
moe_engine/src/hardcode_parts.cpp
moe_engine/src/hardcode_raw_spans.cpp
moe_engine/src/hardcode_layers.cpp
moe_engine/src/hardcode_summary.cpp
moe_engine/src/hardcode_projection_shapes.cpp
moe_engine/src/tools/storage_f8.cpp
moe_engine/native/scale4.cpp
```

Add these root sources when this engine is embedded from C++:

```text
../engine_core/core/mmap_loader.cpp
../engine_core/kv/kv_qkv.cpp
../loader/*.cpp
```

## Runtime Rule

Always keep mandatory per-token state in the fastest available tier:

- current KV/scratch reservation
- token embedding and final head hot path
- current layer control metadata
- scale4 lookup tables and small fixed manifests

MoE expert weights are not all resident. They move dynamically:

```text
router selects experts
 -> selected expert gate/up/down blocks requested
 -> promote DB/RAM block to VRAM if capacity exists
 -> otherwise compute from RAM/DB streaming path
 -> update hot score
 -> demote unused experts over time
```

Multimodal blocks follow the same rule. The vision/image path is loaded only
when image input is active, then can remain hot if repeatedly used.

### Active-Set VRAM Contract

The intended fast path is active-set residency, not full-model residency. When
MoE is active, VRAM should contain only the mandatory dense/runtime state plus
the router-selected expert triplets needed now or in the near lookahead window.
Everything else remains in RAM, pinned staging, mmap/page cache, or DB/blob
backing store until promoted.

Current legacy fallback constants used when the format does not provide a
complete dynamic shape contract:

| Item | Value |
| --- | ---: |
| Hidden layers | 79 |
| MoE layers | 3 through 78 |
| MoE layer count | 76 |
| Experts per MoE layer | 256 |
| Routed experts per token (`top_k`) | 8 |
| Expert cache unit | `(layer, expert)` gate/up/down triplet |
| Total expert cache units | 19,456 |
| Hidden size | 6,144 |
| Expert intermediate size | 2,048 |
| Plain KV full preallocation | 745,537,536 bytes, about 0.69 GiB |

Optimal VRAM contents under enough budget:

- pinned mandatory state: KV cache when budget allows, decode scratch, scale4
  tables, tokenizer/embedding/lm-head hot metadata, current layer control data
- current MoE active set: at most `top_k=8` expert triplets for the current
  token/layer
- lookahead MoE set: predicted next-layer or nearby expert triplets when
  prefetch telemetry gives a stable signal
- current/next attention raw spans when the backend can overlap transfer with
  MLP work
- backend staging resources: pinned host buffers and GPU copy queues when the
  selected backend supports them

Non-goals:

- loading all 256 experts for a layer into VRAM by default
- loading all 19,456 expert triplets into VRAM
- querying Hugging Face, a database, or tensor-name strings inside every matmul
- expanding all packed NVFP4/BF16 weights into full FP16/FP32 buffers

This is the bottleneck-free target: router output selects a small expert set,
the prefetch planner promotes those triplets before compute reaches them, and
the backend executes fused dequant/matmul on the resident active set. If VRAM
is insufficient, correctness falls back to RAM or storage streaming, but that is
a slower operating mode rather than the target fast path.

## What This Folder Keeps

- `src/parts/residency_helpers.cpp.inc`: tiered VRAM/RAM/DB admission helpers.
- `src/parts/prefetch_plan*.cpp.inc`: Moe MoE prefetch planning.
- `src/parts/tensor_query.cpp.inc`: mmap packed FP4 row query and dot fallback.
- `src/parts/codec_table.cpp.inc`: offset manifest loader for original
  safetensors shards.
- `native/scale4.*`: runtime scale4 decoder copied from the working
  implementation.
- `native/fp4_decode.h`: native FP4 decode reference.
- `COMPRESSION_CONTRACT.md`: exact rules for what can become a runtime codec.

## What This Folder Uses From Root

- `../engine_core/core/mmap_loader.*`: shared mmap file mapping.
- `../engine_core/kv/kv_qkv.*`: default packed QKV cache quantization.
- `../loader/*`: manifest loading and path helpers.

## What This Folder Drops

- experiment/probe scripts,
- candidate lossy compression code not validated by eval,
- ratio-only compression claims without tensor location, metadata, decoder, and
  model-level validation,
- generic tensor-name lookup on the hot path,
- fp16/bf16 full-weight expansion,
- optimizer/training/autograd logic.

## KV Default

Default KV mode is packed QKV. Dedicated GGUF files can force this through the
embedded `engine_contract.qkv_packed_cache_required` flag, and the engine then
rejects attempts to switch back to persistent plain KV. The plain enum remains
in the C ABI only for legacy/debug roots that do not carry the offload-native
contract.

## Backend Target

The policy is backend-neutral:

- NVIDIA Windows/Linux PC + CUDA: promote hot expert blocks into VRAM, use
  pinned staging, backend-neutral async copy wrappers, and optional
  DirectStorage/GPUDirect Storage hooks only after the registered-buffer path
  is actually implemented. Large tensor uploads can bypass the pinned worker by
  reading the shard ranges directly into a pinned staging slot and queuing the
  GPU copy stage.
- AMD Windows/Linux PC + HIP/ROCm: same expert/cache API as CUDA, using HIP
  kernels once compiled. Until the HIP async adapter is filled, the runtime does
  not claim the CUDA stream path for HIP. Vulkan/OpenCL remain fallback routes.
- Intel Windows/Linux PC + Level Zero/SYCL/oneAPI: same expert/cache API, using
  unified/shared memory where available. Until a Level Zero async copy adapter
  is filled, this path remains host-visible streaming plus CPU fallback.
- MacBook / Apple Silicon + Metal: use unified memory and mmap/page-cache
  streaming first. The runtime now reports `zero_copy_host`, disables pinned
  staging/GPU upload workers, and avoids pretending there is a CUDA pinned-host
  path. A future Metal adapter can map the same `(layer, expert)` bundles into
  shared Metal buffers.
- CPU fallback: mmap/page-cache streaming path.
- Vulkan/DirectML/OpenCL/WebGPU: vendor-neutral fallback APIs. Same residency
  decisions, backend adapter decides whether a block can be promoted or must
  stream.

The engine must still run when only DB/blob capacity is sufficient. That mode is
slow, but it should be correct.

## Implemented Optimization Hooks

The C API exposes the Moe fast path directly:

- `moe_pc_Moe1_model_shape()`
  - format-derived or fallback constants for layer count, MoE layer range, expert count,
    hidden size, expert hidden size, vocab size, and projection count.
- `moe_pc_engine_get_optimization_plan(...)`
  - reports which runtime optimizations are active for the selected backend:
    static shape, static memory pool, expert prefetch, VRAM hot cache, pinned
    RAM cache, storage streaming, async copy, direct-to-GPU IO, fused FP4
    dequant+matmul target, scale4 fusion target, CUDA Graph target, Metal
    command-buffer target, paged KV target, and prefill/decode split.
- Backend capability flags now distinguish `supports_backend_async_api`,
  `supports_zero_copy_host`, `supports_fixed_read_staging`, and
  `supports_registered_io_buffers`. This prevents DirectStorage/GDS/io_uring
  style paths from being reported as active before a real registered-buffer
  implementation exists.
- `moe_pc_group_tokens_by_expert(...)`
  - sorts router assignments by `(layer, expert)` and emits compact expert
    batches so the runtime can load an expert once and process all matching
    tokens together.
- `moe_pc_engine_run_expert_triplet_f32(...)`
  - CPU correctness fallback for one Moe expert triplet. It executes
    gate/up/down matvec with caller-owned scratch, supports scale4 and raw
    scale tensors, and handles raw FP16/FP32 expert weights.

These hooks sit above the backend-specific native kernels. CPU stays the
correctness fallback. CUDA/HIP/Metal/Vulkan/DirectML/OpenCL adapters should
consume the same plan and replace the hot row/dot path with fused kernels.

## Public Backend API

The C API exposes backend capability detection before the engine hard-commits to
a path:

```c
moe_backend_caps_t caps;
moe_pc_detect_backend_caps(moe_BACKEND_AUTO, moe_PLATFORM_AUTO, &caps);
```

Use this to select budgets and feature flags:

- `moe_BACKEND_CUDA` for NVIDIA PC/server paths,
- `moe_BACKEND_HIP` for AMD ROCm/HIP paths,
- `moe_BACKEND_LEVEL_ZERO` or `moe_BACKEND_SYCL` for Intel oneAPI paths,
- `moe_BACKEND_METAL` for MacBook / Apple Silicon,
- `moe_BACKEND_CPU` for guaranteed fallback,
- `moe_BACKEND_VULKAN`, `moe_BACKEND_DIRECTML`, `moe_BACKEND_OPENCL`, or
  `moe_BACKEND_WEBGPU` as portable GPU fallbacks.

Do not bake runtime routing predictions into the static engine. The static
manifest supplies offsets and bundle sizes; router telemetry updates prefetch
and cache policy while the engine runs.

## Local Server

The C++ engine and the local API server are separate layers:

- `moe_pc_engine` is the in-process runtime API.
- `moe_pc_engine_server` is the local OpenAI-compatible `/v1` wrapper.

The server does not upload the model anywhere. It binds a local endpoint and
passes requests into the engine. Normal startup does not require backend,
platform, RAM, VRAM, performance, or OpenClaw mode flags; those are selected
from the local machine and model root.

### Quick Start

Start the local API with the model root:

```text
moe_pc_engine_server <model_root>
```

If `tensors.csv` and `moe_scale4.gsc4` are present under that root, they are
auto-detected. The server binds `127.0.0.1:8000`, exposes `/v1`, chooses the
available CPU/GPU/Metal-style runtime path, and sizes RAM/VRAM caches
automatically.

QKV is already the normal startup path. The old `qkv` selector is accepted for
compatibility but is no longer required.

Check the server:

```text
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/health
```

Send a minimal OpenAI Responses request:

```text
curl http://127.0.0.1:8000/v1/responses ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"<model-id-from-format>\",\"input\":\"hello\",\"max_output_tokens\":16}"
```

On Linux/macOS, replace `^` with `\`.

### OpenClaw Connection

OpenClaw should point at the same local `/v1` endpoint. The checked-in
`openclaw.storagellm.json` template uses this provider shape:

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "storagellm": {
        "baseUrl": "http://127.0.0.1:8000/v1",
        "apiKey": "sk-local",
        "api": "openai-responses"
      }
    }
  }
}
```

Copy or merge that JSON into OpenClaw's config file.

### Runtime Selection

| Runtime behavior | User action | Purpose |
| --- | --- | --- |
| Local API server | default | Start the OpenAI-compatible HTTP server on `127.0.0.1:8000`. |
| Backend and platform | automatic | Detect the best available local path and keep CPU correctness fallback. |
| RAM/VRAM budgets | automatic | Size cache and staging budgets from detected hardware. |
| QKV cache | default | Use packed QKV cache settings from the format metadata when present. |
| Plain KV mode | legacy/debug only | Available only when the loaded format does not require packed QKV. |
| Storage streaming fallback | default when needed | Stream cold blocks from local model-root storage; correct but slower. |

### Server Surface

Normal startup has two forms:

```text
moe_pc_engine_server <model_root>
```

Other command-line flags are for integration harnesses and debugging only. They
are intentionally not part of the normal user path.

Default endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/responses`
- `POST /v1/storagellm/eval`
- `POST /v1/storagellm/perplexity`
- `POST /v1/perplexity`
- `GET /openclaw/config` (returns the same OpenClaw provider config JSON)

Security rule: bind to `127.0.0.1` by default. Non-loopback hosts require the
explicit `--allow-remote` flag so model control is not exposed by accident.

The server mode provides the OpenAI-compatible API surface, automatic runtime
selection, forward-status reporting, tokenizer-backed text requests, generation,
and eval/perplexity routes. Generation calls the engine-native token loop,
which executes embedding, per-layer attention projectors, dense/MoE MLPs,
router-selected expert triplets, final norm, and `lm_head` sampling through the
C API. `/health` reports staging deficits, recommended staging bytes,
tensor-table readiness, tokenizer readiness, and decode/chat readiness so
deployment failures are visible instead of silently stalling.

### Request Shapes

The server accepts three OpenAI-style generation routes:

- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Eval/perplexity routes:

- `POST /v1/storagellm/eval`
- `POST /v1/storagellm/perplexity`
- `POST /v1/perplexity`

### Benchmarking

The benchmark helper targets the OpenAI-compatible HTTP surface and runs several
checks in one pass: system/GPU/RAM snapshots, direct `curl` latency/tok/s tests,
perplexity through `/v1/perplexity`, NVIDIA GenAI-Perf, NVIDIA AIPerf, and
GuideLLM if it is already installed:

```bash
cd /teamspace/studios/this_studio/storagellm_bench/storage.llm
bash benchmarks/run_openai_benchmarks.sh
```

Useful overrides:

```bash
PROMPT_TOKENS=512 OUTPUT_TOKENS=256 NUM_PROMPTS=50 CONCURRENCY=1 bash benchmarks/run_openai_benchmarks.sh
TOKENIZER=/teamspace/studios/this_studio/storagellm_bench/models/<model-root> bash benchmarks/run_openai_benchmarks.sh
```

Results are written under
`/teamspace/studios/this_studio/storagellm_bench/benchmark_results`. Key files
include `curl_latency.jsonl`, `ppl_results.jsonl`, `genai_perf_profile.json`,
`aiperf_profile*`, and `summary.jsonl`.

Text requests can use `input`, `prompt`, `messages[].content`, or collected
`text` fields. `input_ids` is accepted for clients that already tokenize.

Token limits:

- `max_output_tokens`, `max_completion_tokens`, and `max_tokens` are accepted.
- The server clamps generation to 4096 new tokens.

Streaming:

- `stream: true` is supported for chat/completions style responses.
- `/v1/responses` currently returns a normal JSON response.

### License

This runtime code is released under the repository MIT License. The converted
GGUF offload model files distributed on Hugging Face follow their upstream model
license. Keep the upstream model license and copyright notices with
redistributed model files.
