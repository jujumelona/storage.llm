# Needed vs Removed

## Keep

Runtime core:

- fixed Moe tensor offset manifest,
- packed NVFP4 tile reader,
- scale4 decoder,
- fused FP4 + scale4 dequant/matmul backend hooks,
- MoE router result -> expert triplet planner,
- VRAM/RAM/DB residency manager,
- async prefetch and cold eviction policy,
- backend adapter boundary for CPU/NPU/TPU/GPU.

Always-hot state:

- KV/scratch reservation,
- current layer metadata,
- token embedding/output head path,
- scale4 lookup metadata,
- small block id -> offset tables.

Dynamic state:

- selected expert triplets,
- next-layer likely experts,
- multimodal vision blocks only when image input is active,
- repeated hot experts promoted to VRAM,
- cold experts demoted to RAM then DB/blob.

## Remove

Do not bring into the runtime hot path:

- generic Hugging Face loader,
- generic tensor string lookup per matmul,
- full fp16/bf16 expanded model weights,
- all-expert VRAM residency,
- training/autograd/optimizer logic,
- exploratory compression probes,
- unvalidated Tucker2/QAT/RPCA formats,
- DB row lookup per tiny tensor operation.

## Fallback Rule

The model must still run if only DB/blob capacity is sufficient:

```text
DB/blob page -> RAM staging -> backend tile compute
```

That mode is allowed to be slow. It must not require full model RAM or VRAM.

If VRAM exists, use it aggressively for hot blocks. If VRAM is small, do not
fail; use RAM or DB streaming instead.
