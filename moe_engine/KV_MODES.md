# KV Contract

QKV is the normal StorageLLM KV cache contract.

The legacy `moe_KV_MODE_PLAIN` enum remains in the C ABI for debug and old
callers, but the runtime default is `moe_KV_MODE_QKV`. Offload-native GGUF
models that embed `offload.metadata_v2` and require
`qkv_packed_cache_required` force QKV; after such a model root is loaded,
attempting to switch back to plain KV is rejected.

## Runtime Mapping

```text
default config      -> moe_KV_MODE_QKV
offload-native GGUF -> force moe_KV_MODE_QKV
legacy plain enum   -> debug/fallback only when the loaded format allows it
```

`qkv_set_mode(true)` is now the default process state. The server and probe
still accept `qkv` / `--qkv` as compatibility no-ops.

## Format Source Of Truth

Dedicated GGUF files keep the `.gguf` container name and store the StorageLLM
contract under `offload.*` KV entries:

```text
offload.metadata_v2
offload.qkv_k_bits
offload.qkv_v_bits
offload.qkv_group_size
offload.qkv_page_size_tokens
offload.qkv_sink_tokens
```

The engine reads those keys from the GGUF metadata header without parsing tensor
payloads. The contract configures the packed QKV cache and marks persistent
plain KV storage as disabled for that model.

Server startup accepts this as a metadata/header index when no legacy
`tensors.csv` or binary tensor index is found. The stats surface keeps
`offload_gguf_tensor_count` separate from `offload_gguf_executable_tensor_count`
so header discovery is not confused with a fully materialized tensor execution
table.
