# Compression Contract

Compression is not accepted by ratio alone. A codec can enter the runtime only
when all of these are fixed:

1. Tensor location: layer, expert id, projection, shard file, byte offset, byte
   length, logical shape, and tile shape.
2. Stored format: dtype, packing order, scale source, codebook/centroid data,
   alignment, and version.
3. Decode path: exact CPU/GPU/NPU entry point that turns stored bytes into the
   tile consumed by matmul or attention.
4. Runtime ownership: which residency tier holds the packed bytes, temporary
   decoded tile, scales, and metadata.
5. Correctness proof: reconstruction error plus model-level logit/PPL or task
   validation on enough samples.

Without this full contract, the result is only a research signal. It is not a
runtime compression format.

## Runtime-Ready Codecs

### NVFP4 Packed Weights

Source of truth:

- `src/parts/codec_table.cpp.inc`
- `src/parts/tensor_query.cpp.inc`
- `native/fp4_decode.h`

Required metadata:

- safetensors shard file
- byte offset and byte length
- logical shape
- nibble order: high nibble first, then low nibble
- FP4 table used by the existing notebooks/runtime

Runtime rule:

```text
keep packed bytes in VRAM/RAM/DB
decode only the requested tile
never inflate the full model to fp16/bf16
```

### scale4

Source of truth:

- `native/glm5_scale4.h`
- `native/glm5_scale4.cpp`

Required metadata is carried by `glm5_scale4_entry_t`:

- key offset and length
- rows, groups, group size
- bit width and centroid count
- codebook offset
- packed index offset and byte count
- scale2 and max_abs_error

Runtime rule:

```text
load scale4 index by mmap
find the entry by fixed manifest/key during setup
decode row/group scales inside dequant/matmul
```

## Research-Only Signals

These are not runtime codecs yet:

- weight+scale entropy/remap
- Tucker2
- QAT correction
- RPCA
- local mixed-bit down-proj experiments

They become eligible only after they provide the full contract above and pass
model-level validation. A high estimated ratio or a small probe table is not
enough.

## KV Modes

Default KV is plain/original append/cache/reuse.

QKV is the only optional KV quantization mode in this tree:

- `../engine_core/kv/kv_qkv.h`
- `../engine_core/kv/kv_qkv.cpp`

If QKV is not explicitly selected, the runtime stays on plain KV. No archived
KV compression variant may silently replace plain KV.
