# StorageLLM

StorageLLM is a C++ runtime workspace for storage-first inference engines. It
keeps shared loading and mmap helpers at the repository root, while each model
folder owns its model-specific runtime, backend selection, residency policy,
numeric helpers, and model contracts.

This repository does not ship model weights.

## Layout

- `engine_core/core/` - shared mmap loader.
- `engine_core/kv/` - default packed QKV cache implementation.
- `loader/` - shared manifest loader and path helpers.
- `moe_engine/` - model runtime folder; see its README for
  model-specific API, metadata, numeric helpers, planning, and examples.

Common runtime code belongs at the repository root. A new model should add a
new model folder beside `moe_engine/` and reuse `engine_core/` and
`loader/` instead of copying them inside the model folder.

## Common Build Inputs

There is no repository-owned build generator. Add the required `.cpp` files to
the host application's build system.

Common include roots:

```text
.
loader
engine_core/core
engine_core
```

Common sources normally used by model runtimes:

```text
engine_core/core/mmap_loader.cpp
engine_core/kv/kv_qkv.cpp
```

Add `loader/*.cpp` when using the shared manifest loader from C++.
Model-specific source lists belong in each model folder's README.

## Runtime Policy

The shared runtime policy is storage-first:

```text
VRAM first -> RAM second -> DB/blob or mmap-backed storage third
```

If VRAM is limited, hot blocks can be promoted and cold blocks can be streamed
from RAM or storage. CPU execution remains the correctness fallback. Backend
selection and residency policy live in the model runtime folder.

Packed QKV append/cache/reuse is the normal KV contract. Plain float KV is kept
only as a debug/fallback path when the loaded format does not forbid it.

## GGUF Offload Runtime Usage

The GGUF offload model runtime lives in `moe_engine/`. Its README is the
source of truth for model layout, the local OpenAI-compatible API, automatic
runtime selection, the dedicated GGUF layout, QKV cache contract, and request
shapes:

```text
moe_engine/README.md
```

Model weights are distributed separately on Hugging Face. The dedicated GGUF
path keeps the `.gguf` file extension and embeds StorageLLM offload metadata in
`offload.*` GGUF KV entries:

```text
https://huggingface.co/storagejuju/gemma-4-26b-a4b-it-mxfp4-moe-storagellm-offload
```

The reusable Colab patcher is tracked at:

```text
notebooks/GGUF_Offload_Metadata_Patch_Stream.ipynb
```

For a new GGUF variant, edit only the notebook's `CHANGE HERE` block. In the
common case `SOURCE_HF_REPO_ID`, `SOURCE_SUBDIR`, and `TARGET_HF_OWNER` are
enough; shard prefix/count are auto-discovered from Hugging Face.

Download the converted StorageLLM GGUF artifact before starting the server:

```text
hf download storagejuju/gemma-4-26b-a4b-it-mxfp4-moe-storagellm-offload --local-dir <model_root>
```

The normal local API command is only the server binary plus the model root. The
server starts the local `/v1` API automatically, binds to `127.0.0.1:8000`,
detects the available CPU/GPU/Metal-style runtime path, and sizes RAM/VRAM
caches from the machine:

```text
moe_pc_engine_server <model_root>
```

QKV is now the default KV cache path. The old `qkv` selector is accepted as a
compatibility no-op.

For dedicated GGUF roots, the server does not require a separate `tensors.csv`
index just to accept the model root. It reads the `offload.*` GGUF header
contract first, forces packed QKV when the format requires it, and reports the
header-derived shard/tensor status through `/health`.

Dedicated GGUF variants also carry the source weight quantization family under
`offload.weight_*` keys. The runtime treats `UD-IQ2_M`, `MXFP4`, `Q4`, `Q8`,
and similar names as different kernel families; it no longer routes unknown
2-bit or 3-bit data through the old FP4 path just because the row byte count is
small. QKV cache bits and GGUF weight bits are separate contracts.

Platform I/O is selected by capability, not by CUDA names in the format. CUDA
uses pinned staging and backend-neutral async copy wrappers; CPU and
host-visible Metal-style paths use `zero_copy_host` and skip pinned/GPU upload
workers. DirectStorage, GPUDirect Storage, and io_uring registered-buffer paths
are kept behind explicit capability flags until their real adapters are wired,
so the engine does not silently route unsupported platforms through a fake CUDA
path.

Clients can use `http://127.0.0.1:8000/v1` directly. The checked-in
`openclaw.storagellm.json` template points OpenClaw at that same local API.

## Repository Rules

- Keep common runtime code in `engine_core/` or `loader/`.
- Keep model-specific code inside that model's folder.
- Keep runtime hot paths in C++.
- Do not add local-only absolute paths.
- Do not commit model weights, generated libraries, build output, or cache
  folders.
- Keep loader and runtime behavior aligned with manifest-relative model paths.

## License

This code repository is released under the MIT License. See `LICENSE`.

Converted StorageLLM model artifacts are distributed separately on Hugging
Face and keep the license of their upstream base model. Keep the upstream model
license and copyright notices with redistributed model files.
