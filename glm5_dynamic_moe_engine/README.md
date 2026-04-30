# GLM5 Dynamic MoE Engine

Clean runtime folder for the GLM-5.1 NVFP4 StorageLLM engine.

The primary target is now the offload-native GGUF layout: normal `.gguf` files
with StorageLLM execution metadata embedded under the `offload.*` GGUF KV
namespace. The legacy JUJU layout remains supported for existing artifacts.
Both layouts feed the same storage-first runtime:

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
  MXFP4_MOE/
    GLM-5.1-MXFP4_MOE-00001-of-00011.gguf
    GLM-5.1-MXFP4_MOE-00002-of-00011.gguf
    ...
    GLM-5.1-MXFP4_MOE-00011-of-00011.gguf
```

The engine scans GGUF metadata headers only. If a shard contains
`offload.metadata_v2`, model-root validation uses the embedded
`offload.file_count` / `offload.file_index` contract instead of the hardcoded
JUJU part table. QKV settings come from these GGUF KV entries when present:

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

When no legacy tensor index is present, `glm5_pc_engine_server <model_root>`
falls back to this GGUF header contract. `/health` then exposes
`offloadGgufValid`, `offloadGgufFileCount`, `offloadGgufTensorHeaders`, and the
format-selected QKV fields so the runtime state is visible without pretending a
full executable tensor table was loaded.

The legacy JUJU layout is still accepted:

```text
<model_root>/
  tokenizer.json
  tensors.csv
  glm5_scale4.gsc4
  parts/
    glm5.1-storage-part01.juju
    glm5.1-storage-part02.juju
    ...
    glm5.1-storage-part21.juju
```

`tokenizer.json`, `tensors.csv`, and `glm5_scale4.gsc4` are runtime metadata.
They are not training artifacts and they are not regenerated during inference.
For legacy public distribution, keep those metadata assets beside the JUJU
parts.

The primary runtime layout uses manifest-relative paths such as
`parts/glm5.1-storage-part01.juju`. If a browser download folder contains the
part files directly, passing that folder as `model_root` is also accepted. The
runtime tries `<model_root>/parts/<file>` first and then `<model_root>/<file>`.

### Hugging Face Sources

Hugging Face is the provisioning and distribution layer. It is not used as a
tensor lookup service inside the decode hot path. Download or sync the model
artifacts first, then pass the local folder as `model_root`.

| Purpose | Repository |
| --- | --- |
| Official GLM-5.1 base model | <https://huggingface.co/zai-org/GLM-5.1> |
| StorageLLM converted JUJU artifacts | <https://huggingface.co/storagejuju/GLM5.1-4q-storage> |
| StorageLLM offload-native GGUF artifacts | <https://huggingface.co/storagejuju/GLM-5.1-GGUF-MXFP4-MOE-Offload> |
| Direct browser download page | <https://huggingface.co/storagejuju/GLM5.1-4q-storage/tree/main> |
| Hugging Face model download docs | <https://huggingface.co/docs/hub/models-downloading> |

Provisioning example for downloading the ready-to-run model files:

```text
hf download storagejuju/GLM5.1-4q-storage --local-dir <model_root>
glm5_pc_engine_server <model_root>
```

Dedicated GGUF provisioning:

```text
hf download storagejuju/GLM-5.1-GGUF-MXFP4-MOE-Offload --local-dir <model_root>
glm5_pc_engine_server <model_root>
```

Individual files can also be downloaded through Hugging Face's direct resolve
URL pattern:

```text
https://huggingface.co/storagejuju/GLM5.1-4q-storage/resolve/main/<relative-path>?download=true
```

For example, if the repo contains `parts/glm5.1-storage-part01.juju`, the direct
file URL is:

```text
https://huggingface.co/storagejuju/GLM5.1-4q-storage/resolve/main/parts/glm5.1-storage-part01.juju?download=true
```

The expected StorageLLM Hugging Face repo must contain either the dedicated
GGUF shards with embedded `offload.*` metadata or the legacy runtime assets:
`tokenizer.json`, `tensors.csv`, `glm5_scale4.gsc4`, and the
`glm5.1-storage-part*.juju` files. The official `zai-org/GLM-5.1` repo remains
the upstream model reference.

When using the shared root `loader/`, GLM-specific codec extras are carried in
generic aux slots:

```text
aux0_block -> GLM5 scale4 codebook block
aux1_block -> GLM5 scale4 index block
```

The GLM runtime keeps the slot interpretation model-local; the shared loader
keeps only the generic aux names.

Expected GLM-5.1 storage parts:

| Part | Expected size |
| --- | ---: |
| `glm5.1-storage-part01.juju` | 23,643,820,557 |
| `glm5.1-storage-part02.juju` | 19,369,187,509 |
| `glm5.1-storage-part03.juju` | 23,176,814,429 |
| `glm5.1-storage-part04.juju` | 21,367,618,930 |
| `glm5.1-storage-part05.juju` | 19,194,858,195 |
| `glm5.1-storage-part06.juju` | 23,175,639,236 |
| `glm5.1-storage-part07.juju` | 21,379,413,903 |
| `glm5.1-storage-part08.juju` | 19,198,789,856 |
| `glm5.1-storage-part09.juju` | 23,199,229,089 |
| `glm5.1-storage-part10.juju` | 21,380,594,405 |
| `glm5.1-storage-part11.juju` | 19,207,047,118 |
| `glm5.1-storage-part12.juju` | 23,185,076,178 |
| `glm5.1-storage-part13.juju` | 21,374,696,840 |
| `glm5.1-storage-part14.juju` | 19,198,790,663 |
| `glm5.1-storage-part15.juju` | 23,179,571,825 |
| `glm5.1-storage-part16.juju` | 21,513,876,253 |
| `glm5.1-storage-part17.juju` | 19,454,346,168 |
| `glm5.1-storage-part18.juju` | 23,524,768,312 |
| `glm5.1-storage-part19.juju` | 21,713,995,761 |
| `glm5.1-storage-part20.juju` | 16,166,004,403 |
| `glm5.1-storage-part21.juju` | 41,264,048,735 |

## Public API

Use `include/glm5_pc_engine.h`.

Important entry points:

```c
glm5_pc_engine_config_t cfg = glm5_pc_default_config();
glm5_pc_engine_t* engine = glm5_pc_engine_create(&cfg);

glm5_model_root_check_t check;
int ok = glm5_storage_validate_model_root(model_root, &check);

glm5_pc_engine_set_model_root(engine, model_root);
glm5_pc_engine_get_forward_status(engine, &status);

glm5_pc_engine_destroy(engine);
```

## Build Integration

Common include roots still come from the repository root:

```text
.
loader
engine_core/core
engine_core
```

Add this GLM folder's include root:

```text
glm5_dynamic_moe_engine/include
```

GLM-specific sources commonly needed by an embedding application:

```text
glm5_dynamic_moe_engine/src/glm5_pc_engine.cpp
glm5_dynamic_moe_engine/src/glm5_model_shape.cpp
glm5_dynamic_moe_engine/src/glm5_hardcode_constants.cpp
glm5_dynamic_moe_engine/src/glm5_hardcode_parts.cpp
glm5_dynamic_moe_engine/src/glm5_hardcode_raw_spans.cpp
glm5_dynamic_moe_engine/src/glm5_hardcode_layers.cpp
glm5_dynamic_moe_engine/src/glm5_hardcode_summary.cpp
glm5_dynamic_moe_engine/src/glm5_hardcode_projection_shapes.cpp
glm5_dynamic_moe_engine/src/tools/storage_f8.cpp
glm5_dynamic_moe_engine/native/glm5_scale4.cpp
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

Current GLM-5.1 constants used by this runtime:

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
- `src/parts/prefetch_plan*.cpp.inc`: GLM5 MoE prefetch planning.
- `src/parts/tensor_query.cpp.inc`: mmap packed FP4 row query and dot fallback.
- `src/parts/codec_table.cpp.inc`: offset manifest loader for original
  safetensors shards.
- `native/glm5_scale4.*`: runtime scale4 decoder copied from the working
  implementation.
- `native/fp4_decode.h`: native FP4 decode reference.
- `COMPRESSION_CONTRACT.md`: exact rules for what can become a runtime codec.

## What This Folder Uses From Root

- `../engine_core/core/mmap_loader.*`: shared mmap file mapping.
- `../engine_core/kv/kv_qkv.*`: default packed QKV cache quantization.
- `../loader/*`: JUJU/manifest loading and path helpers.

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

The C API exposes the GLM5 fast path directly:

- `glm5_pc_glm51_model_shape()`
  - hardcoded GLM-5.1 constants for layer count, MoE layer range, expert count,
    hidden size, expert hidden size, vocab size, and projection count.
- `glm5_pc_engine_get_optimization_plan(...)`
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
- `glm5_pc_group_tokens_by_expert(...)`
  - sorts router assignments by `(layer, expert)` and emits compact expert
    batches so the runtime can load an expert once and process all matching
    tokens together.
- `glm5_pc_engine_run_expert_triplet_f32(...)`
  - CPU correctness fallback for one GLM5 expert triplet. It executes
    gate/up/down matvec with caller-owned scratch, supports scale4 and raw
    scale tensors, and handles raw FP16/FP32 expert weights.

These hooks sit above the backend-specific native kernels. CPU stays the
correctness fallback. CUDA/HIP/Metal/Vulkan/DirectML/OpenCL adapters should
consume the same plan and replace the hot row/dot path with fused kernels.

## Public Backend API

The C API exposes backend capability detection before the engine hard-commits to
a path:

```c
glm5_backend_caps_t caps;
glm5_pc_detect_backend_caps(GLM5_BACKEND_AUTO, GLM5_PLATFORM_AUTO, &caps);
```

Use this to select budgets and feature flags:

- `GLM5_BACKEND_CUDA` for NVIDIA PC/server paths,
- `GLM5_BACKEND_HIP` for AMD ROCm/HIP paths,
- `GLM5_BACKEND_LEVEL_ZERO` or `GLM5_BACKEND_SYCL` for Intel oneAPI paths,
- `GLM5_BACKEND_METAL` for MacBook / Apple Silicon,
- `GLM5_BACKEND_CPU` for guaranteed fallback,
- `GLM5_BACKEND_VULKAN`, `GLM5_BACKEND_DIRECTML`, `GLM5_BACKEND_OPENCL`, or
  `GLM5_BACKEND_WEBGPU` as portable GPU fallbacks.

Do not bake runtime routing predictions into the static engine. The static
manifest supplies offsets and bundle sizes; router telemetry updates prefetch
and cache policy while the engine runs.

## Local Server

The C++ engine and the local API server are separate layers:

- `glm5_pc_engine` is the in-process runtime API.
- `glm5_pc_engine_server` is the local OpenAI-compatible `/v1` wrapper.

The server does not upload the model anywhere. It binds a local endpoint and
passes requests into the engine. Normal startup does not require backend,
platform, RAM, VRAM, performance, or OpenClaw mode flags; those are selected
from the local machine and model root.

### Quick Start

Start the local API with the model root:

```text
glm5_pc_engine_server <model_root>
```

If `tensors.csv` and `glm5_scale4.gsc4` are present under that root, they are
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
  -d "{\"model\":\"glm5.1-storage\",\"input\":\"hello\",\"max_output_tokens\":16}"
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
glm5_pc_engine_server <model_root>
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
TOKENIZER=/teamspace/studios/this_studio/storagellm_bench/models/GLM5.1-4q-storage bash benchmarks/run_openai_benchmarks.sh
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
GLM5.1 model files distributed on Hugging Face follow the upstream GLM-5.1 MIT
license. Keep the upstream model license and copyright notices with
redistributed model files.
