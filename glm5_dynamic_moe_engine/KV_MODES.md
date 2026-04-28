# KV Modes

Default stays plain KV. QKV is the explicit quantized KV mode.

## 1. Plain KV

Default mode. This is the GLM runtime float K/V path. It is not backed by a
separate root cache-manager subsystem.

## 2. QKV

QKV uses the implementation named by the code:

- `../engine_core/kv/kv_qkv.h`
- `../engine_core/kv/kv_qkv.cpp`

Mode wiring:

```text
GLM5_KV_MODE_PLAIN -> qkv_set_mode(false)
GLM5_KV_MODE_QKV   -> qkv_set_mode(true)
```

The server/probe CLI flag is `--qkv`.

`glm5_pc_engine_attention_decode_f32()` takes `const float*` K/V cache inputs.
That path must stay plain float attention. It must not quantize the float input
and immediately dequantize it again inside the attention loop.

If QKV is wired into runtime cache storage, it needs a quantized-KV storage/read
path and a separate attention entry point that accepts QKV state or packed QKV
buffers. It should not be hidden inside the `_f32` API.

Runtime rule:

```text
default = plain KV
optional = QKV
```
