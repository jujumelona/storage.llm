# StorageLLM

StorageLLM is a C++ runtime workspace for storage-first inference engines. It
keeps shared loading and mmap helpers at the repository root, while each model
folder owns its model-specific runtime, backend selection, residency policy,
numeric helpers, and model contracts.

This repository does not ship model weights.

## Layout

- `engine_core/core/` - shared mmap loader.
- `engine_core/kv/` - optional QKV implementation.
- `loader/` - shared JUJU/manifest loader and path helpers.
- `glm5_dynamic_moe_engine/` - model runtime folder; see its README for
  model-specific API, metadata, numeric helpers, planning, and examples.

Common runtime code belongs at the repository root. A new model should add a
new model folder beside `glm5_dynamic_moe_engine/` and reuse `engine_core/` and
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

Add `loader/*.cpp` when using the shared manifest/JUJU loader from C++.
Model-specific source lists belong in each model folder's README.

## Runtime Policy

The shared runtime policy is storage-first:

```text
VRAM first -> RAM second -> DB/blob or mmap-backed storage third
```

If VRAM is limited, hot blocks can be promoted and cold blocks can be streamed
from RAM or storage. CPU execution remains the correctness fallback. Backend
selection and residency policy live in the model runtime folder.

Plain KV append/cache/reuse is owned by the model runtime. QKV is the only
shared optional quantized KV implementation and lives under
`engine_core/kv/kv_qkv.*`.

## GLM5.1 Runtime Usage

The GLM5.1 model runtime lives in `glm5_dynamic_moe_engine/`. Its README is the
source of truth for model layout, the local OpenAI-compatible API, automatic
runtime selection, the optional QKV path, and request shapes:

```text
glm5_dynamic_moe_engine/README.md
```

Model weights are distributed separately on Hugging Face:

```text
https://huggingface.co/storagejuju/GLM5.1-4q-storage
```

Download the converted StorageLLM artifact before starting the server:

```text
hf download storagejuju/GLM5.1-4q-storage --local-dir <model_root>
```

The normal local API command is only the server binary plus the model root. The
server starts the local `/v1` API automatically, binds to `127.0.0.1:8000`,
detects the available CPU/GPU/Metal-style runtime path, and sizes RAM/VRAM
caches from the machine:

```text
glm5_pc_engine_server <model_root>
```

QKV is the only normal startup choice:

```text
glm5_pc_engine_server <model_root> qkv
```

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

The converted StorageLLM model artifacts are distributed separately on Hugging
Face and follow the upstream GLM-5.1 MIT license. Keep the upstream model
license and copyright notices with redistributed model files.
