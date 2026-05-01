import hashlib
import io
import json
import threading
import bisect
import os
import struct
import time
from pathlib import Path

import requests


GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12
GGUF_TYPE_SIZE = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}

JUJU_HEADER_BYTES = 4096
JUJU_SECTION_ENTRY_BYTES = 96
JUJU_SECTION_MODEL_META = 0x0001
JUJU_SECTION_SHARED_WEIGHTS = 0x0010
JUJU_SECTION_HOT_EXPERTS = 0x0011
JUJU_SECTION_WARM_EXPERTS = 0x0012
JUJU_SECTION_COLD_EXPERTS = 0x0013
JUJU_SECTION_LAYER_ORDER_INDEX = 0x0020
JUJU_SECTION_QKV_POLICY = 0x0021
HF_INDIVIDUAL_FILE_LIMIT_BYTES = 50 * 1024 * 1024 * 1024
DEFAULT_JUJU_UPLOAD_FILE_LIMIT_BYTES = 45 * 1024 * 1024 * 1024
JUJU_SPLIT_METADATA_RESERVE_BYTES = 512 * 1024 * 1024


def juju_artifact_names(source_name):
    stem = Path(source_name).stem
    return {
        "weights": f"{stem}.juju",
        "index": f"{stem}.juju.idx",
        "verify": f"{stem}.juju.verify.json",
    }


def juju_upload_file_limit_bytes():
    raw = str(os.environ.get("JUJU_MAX_UPLOAD_FILE_BYTES", "")).strip()
    if not raw:
        return DEFAULT_JUJU_UPLOAD_FILE_LIMIT_BYTES
    value = int(raw)
    if value <= 0:
        raise ValueError("JUJU_MAX_UPLOAD_FILE_BYTES must be positive")
    return value


def juju_split_source_name(source_name, split_index, split_count):
    path = Path(source_name)
    suffix = path.suffix or ".gguf"
    return f"{path.stem}.split{int(split_index):02d}-of-{int(split_count):02d}{suffix}"


def align_up(value, alignment=4096):
    rem = int(value) % int(alignment)
    return int(value) if rem == 0 else int(value) + int(alignment) - rem


def fixed_bytes(value, size):
    raw = str(value or "").encode("utf-8")[:size]
    return raw + (b"\x00" * (size - len(raw)))


def read_exact(handle, size):
    data = handle.read(size)
    if len(data) != size:
        raise EOFError("unexpected EOF while reading GGUF")
    return data


def read_u32(handle):
    return struct.unpack("<I", read_exact(handle, 4))[0]


def read_u64(handle):
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_string(handle):
    size = read_u64(handle)
    return read_exact(handle, size).decode("utf-8")


def skip_value(handle, value_type):
    if value_type == GGUF_TYPE_STRING:
        handle.seek(read_u64(handle), 1)
        return
    if value_type == GGUF_TYPE_ARRAY:
        elem_type = read_u32(handle)
        count = read_u64(handle)
        if elem_type == GGUF_TYPE_STRING:
            for _ in range(count):
                handle.seek(read_u64(handle), 1)
            return
        elem_size = GGUF_TYPE_SIZE.get(elem_type)
        if elem_size is None:
            raise ValueError(f"unsupported GGUF array element type: {elem_type}")
        handle.seek(elem_size * count, 1)
        return
    size = GGUF_TYPE_SIZE.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF value type: {value_type}")
    handle.seek(size, 1)


def file_size(session, url, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = session.head(url, allow_redirects=True, headers=headers, timeout=120)
    if resp.ok and resp.headers.get("Content-Length"):
        return int(resp.headers["Content-Length"])
    headers["Range"] = "bytes=0-0"
    resp = session.get(url, headers=headers, stream=True, timeout=120)
    resp.close()
    cr = resp.headers.get("Content-Range", "")
    if "/" in cr:
        return int(cr.rsplit("/", 1)[1])
    raise RuntimeError(f"could not determine remote file size: {url}")


def fetch_range(session, url, start, end=None, token=None, stream=True):
    headers = {"Range": f"bytes={int(start)}-" + (str(int(end)) if end is not None else "")}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = session.get(url, headers=headers, stream=stream, timeout=120)
    resp.raise_for_status()
    return resp


def parse_gguf_directory(prefix, total_bytes):
    handle = io.BytesIO(prefix)
    if read_exact(handle, 4) != b"GGUF":
        raise ValueError("source is not GGUF")
    version = read_u32(handle)
    tensor_count = read_u64(handle)
    kv_count = read_u64(handle)
    alignment = 32
    for _ in range(kv_count):
        key = read_string(handle)
        value_type = read_u32(handle)
        if key == "general.alignment" and value_type == GGUF_TYPE_UINT32:
            alignment = read_u32(handle)
        else:
            skip_value(handle, value_type)
    tensors = []
    for _ in range(tensor_count):
        name = read_string(handle)
        dims = read_u32(handle)
        shape = [read_u64(handle) for _ in range(dims)]
        tensor_type = read_u32(handle)
        rel_offset = read_u64(handle)
        tensors.append({
            "name": name,
            "dims": dims,
            "shape": shape,
            "type": tensor_type,
            "relative_offset": rel_offset,
        })
    data_start = align_up(handle.tell(), alignment)
    if data_start > len(prefix):
        raise EOFError(f"GGUF tensor table needs {data_start} bytes, got {len(prefix)}")
    order = sorted(range(len(tensors)), key=lambda i: tensors[i]["relative_offset"])
    for pos, idx in enumerate(order):
        cur = tensors[idx]["relative_offset"]
        nxt = tensors[order[pos + 1]]["relative_offset"] if pos + 1 < len(order) else min(total_bytes - data_start, cur + tensors[idx].get("dims", 1) * 8)
        tensors[idx]["source_offset"] = data_start + cur
        tensors[idx]["bytes"] = max(0, nxt - cur)
        tensors[idx]["bucket"] = tensor_bucket(tensors[idx]["name"])
    return {
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "alignment": alignment,
        "data_start": data_start,
        "tensors": tensors,
    }


def read_remote_directory(session, url, token=None, initial_range_bytes=8 * 1024 * 1024):
    total = file_size(session, url, token=token)
    size = min(initial_range_bytes, total)
    while True:
        resp = fetch_range(session, url, 0, size - 1, token=token, stream=False)
        prefix = resp.content
        resp.close()
        try:
            return parse_gguf_directory(prefix, total), total
        except EOFError:
            if size >= total:
                raise
            if size >= 256 * 1024 * 1024:
                raise RuntimeError(f"Max fetch exceeded: {size}")
            size = min(size * 2, total)


def juju_estimated_tensor_payload_bytes(tensors, is_first_shard=True):
    total = JUJU_SPLIT_METADATA_RESERVE_BYTES if is_first_shard else 32 * 1024 * 1024
    for tensor in tensors:
        total = align_up(total, 4096)
        total += int(tensor.get("bytes") or 0)
    return total


def plan_juju_tensor_splits(directory, max_file_bytes=None):
    limit = int(max_file_bytes or juju_upload_file_limit_bytes())
    if limit >= HF_INDIVIDUAL_FILE_LIMIT_BYTES:
        limit = HF_INDIVIDUAL_FILE_LIMIT_BYTES - (256 * 1024 * 1024)
    payload_limit_first = limit - JUJU_SPLIT_METADATA_RESERVE_BYTES
    payload_limit_sub = limit - 32 * 1024 * 1024
    if payload_limit_first <= 0:
        raise ValueError("JUJU upload file limit is too small after metadata reserve")

    tensors = [
        tensor for tensor in sorted(directory["tensors"], key=lambda item: (int(item.get("source_offset") or 0), str(item.get("name") or "")))
        if int(tensor.get("bytes") or 0) > 0
    ]
    if not tensors:
        return [{
            "enabled": False,
            "split_index": 1,
            "split_count": 1,
            "tensor_names": [],
            "tensor_bytes": 0,
            "max_file_bytes": limit,
        }]

    groups = []
    current = []
    current_bytes = 0
    for tensor in tensors:
        tensor_bytes = int(tensor["bytes"])
        aligned_tensor_bytes = align_up(tensor_bytes, 4096)
        current_limit = payload_limit_first if len(groups) == 0 else payload_limit_sub
        if aligned_tensor_bytes > current_limit:
            raise RuntimeError(
                f"single tensor exceeds upload-safe JUJU split limit: {tensor['name']} "
                f"bytes={tensor_bytes} limit={payload_limit}"
            )
        if current and current_bytes + aligned_tensor_bytes > current_limit:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(tensor)
        current_bytes += aligned_tensor_bytes
    if current:
        groups.append(current)

    split_count = len(groups)
    planned = []
    for idx, group in enumerate(groups, start=1):
        planned.append({
            "enabled": split_count > 1,
            "split_index": idx,
            "split_count": split_count,
            "tensor_names": [tensor["name"] for tensor in group],
            "tensor_bytes": sum(int(tensor["bytes"]) for tensor in group),
            "estimated_file_bytes": juju_estimated_tensor_payload_bytes(group, is_first_shard=(idx == 1)),
            "max_file_bytes": limit,
        })
    return planned


def tensor_bucket(name):
    lower = str(name).lower()
    if "shared_expert" in lower or "shared.expert" in lower:
        return "shared_weights"
    if "attn" in lower or "attention" in lower:
        return "shared_weights"
    if (
        "expert" in lower or
        "_exps" in lower or
        "ffn_gate_exps" in lower or
        "ffn_up_exps" in lower or
        "ffn_down_exps" in lower or
        "gate_proj" in lower or
        "up_proj" in lower or
        "down_proj" in lower
    ):
        return "cold_experts"
    return "shared_weights"


def section_type_for_bucket(bucket):
    if bucket == "shared_weights":
        return JUJU_SECTION_SHARED_WEIGHTS
    if bucket == "hot_experts":
        return JUJU_SECTION_HOT_EXPERTS
    if bucket == "warm_experts":
        return JUJU_SECTION_WARM_EXPERTS
    return JUJU_SECTION_COLD_EXPERTS


def pack_section(entry):
    payload = struct.pack(
        "<IIQQQII4B16s32s",
        int(entry["type"]),
        int(entry.get("flags", 0)),
        int(entry["offset"]),
        int(entry["size"]),
        int(entry.get("uncompressed_size", entry["size"])),
        int(entry.get("sequential_block_size", 4096)),
        int(entry.get("random_block_size", 4096)),
        int(entry.get("compression", 0)),
        int(entry.get("prefetch_distance", 0)),
        int(entry.get("mmap_friendly", 1)),
        0,
        bytes.fromhex(entry.get("sha256", "0" * 64))[:16].ljust(16, b"\x00"),
        fixed_bytes(entry.get("name", ""), 32),
    )
    return payload + (b"\x00" * (JUJU_SECTION_ENTRY_BYTES - len(payload)))


def write_padding(out, alignment=4096):
    pad = align_up(out.tell(), alignment) - out.tell()
    if pad:
        out.write(b"\x00" * pad)


def stream_range(session, url, start, size, out, token, digest, chunk_size=16 * 1024 * 1024):
    if size <= 0:
        return
    resp = fetch_range(session, url, start, start + size - 1, token=token, stream=True)
    written = 0
    try:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            out.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    finally:
        resp.close()
    if written != size:
        raise EOFError(f"short tensor range read: expected {size}, got {written}")


def u32(value):
    try:
        if value is None:
            return 0
        return max(0, min(int(value), 0xFFFFFFFF))
    except Exception:
        return 0


def make_header(contract, source_name, file_size_value, sections, section_sizes):
    header = bytearray(JUJU_HEADER_BYTES)
    header[0:8] = b"JUJU\x00\x01\x00\x00"
    struct.pack_into("<I", header, 8, 1)
    struct.pack_into("<I", header, 12, 0)
    struct.pack_into("<Q", header, 16, int(time.time()))
    struct.pack_into("<Q", header, 24, int(file_size_value))
    struct.pack_into("<Q", header, 32, JUJU_HEADER_BYTES)
    struct.pack_into("<Q", header, 40, len(sections) * JUJU_SECTION_ENTRY_BYTES)
    struct.pack_into("<Q", header, 48, section_sizes.get(JUJU_SECTION_SHARED_WEIGHTS, 0))
    struct.pack_into("<Q", header, 56, section_sizes.get(JUJU_SECTION_HOT_EXPERTS, 0))
    struct.pack_into("<Q", header, 64, section_sizes.get(JUJU_SECTION_WARM_EXPERTS, 0))
    struct.pack_into("<Q", header, 72, section_sizes.get(JUJU_SECTION_COLD_EXPERTS, 0))
    model_name = contract.get("model_name") or contract.get("model_id") or Path(source_name).stem
    header[88:152] = fixed_bytes(model_name, 64)
    arch = contract.get("arch_meta") or {}
    struct.pack_into("<I", header, 152, len(sections))
    struct.pack_into("<I", header, 156, JUJU_HEADER_BYTES)
    struct.pack_into("<I", header, 160, u32(arch.get("n_layers") or arch.get("num_hidden_layers")))
    struct.pack_into("<I", header, 164, u32(arch.get("experts_per_moe_layer") or arch.get("n_experts")))
    struct.pack_into("<I", header, 168, u32(arch.get("routed_experts_per_token") or arch.get("top_k")))
    struct.pack_into("<I", header, 172, u32(arch.get("hidden_dim") or arch.get("hidden_size")))
    struct.pack_into("<I", header, 176, u32(arch.get("expert_intermediate_dim") or arch.get("expert_intermediate_size")))
    struct.pack_into("<I", header, 180, u32(contract.get("source_weight_bits")))
    struct.pack_into("<I", header, 188, 2)
    struct.pack_into("<I", header, 192, 1)
    struct.pack_into("<I", header, 196, 4096)
    struct.pack_into("<I", header, 200, 8)
    return bytes(header)


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_section_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _juju_layer_id_from_name(name):
    match = __import__("re").search(r"(?:^|[.])blk\.(\d+)\.", str(name or ""))
    return int(match.group(1)) if match else None


def _juju_first_tensor(tensors, *names):
    wanted = {str(x).lower() for x in names if x}
    for tensor in tensors:
        name = str(tensor.get("name") or "")
        if name.lower() in wanted:
            return name
    return ""


def _juju_tensors_by_prefix(tensors, prefix):
    prefix = str(prefix or "").lower()
    return [
        str(t.get("name") or "")
        for t in tensors
        if str(t.get("name") or "").lower().startswith(prefix)
    ]


def _juju_tensor_shape_map(tensors):
    return {
        str(t.get("name") or ""): list(t.get("shape") or [])
        for t in tensors
        if t.get("name")
    }


def tensor_runtime_priority(name, bucket, size):
    lower = str(name or "").lower()
    bucket = str(bucket or "")
    role = "weight"
    priority = 50
    prefetch = 50
    residency = "SLOW_MEM"
    prefetch_class = "stream"
    if lower in {"token_embd.weight", "output.weight", "output_norm.weight", "rope_freqs.weight"}:
        role = "shared_core"
        priority = 100
        prefetch = 100
        residency = "FAST_MEM"
        prefetch_class = "startup_hot"
    elif ".attn_" in lower or ".attn" in lower:
        role = "attention"
        priority = 90
        prefetch = 90
        residency = "FAST_MEM"
        prefetch_class = "layer_hot"
    elif "ffn_gate_inp" in lower or "router" in lower:
        role = "router"
        priority = 95
        prefetch = 95
        residency = "FAST_MEM"
        prefetch_class = "router_hot"
    elif "_exps" in lower or "expert" in lower:
        role = "expert"
        priority = 65
        prefetch = 80
        residency = "SLOW_MEM"
        prefetch_class = "expert_stream"
    elif ".ffn_" in lower:
        role = "dense_ffn"
        priority = 75
        prefetch = 75
        residency = "FAST_MEM" if bucket == "shared_weights" else "SLOW_MEM"
        prefetch_class = "layer_warm"
    elif "norm" in lower:
        role = "norm"
        priority = 85
        prefetch = 85
        residency = "FAST_MEM"
        prefetch_class = "layer_hot"
    if int(size or 0) > 512 * 1024 * 1024 and residency == "FAST_MEM" and role != "attention":
        residency = "FAST_MEM_STREAMABLE"
    return {
        "graph_role": role,
        "runtime_priority": priority,
        "prefetch_priority": prefetch,
        "prefetch_class": prefetch_class,
        "residency_hint": residency,
    }


def infer_juju_graph_family(contract, tensors):
    text = json.dumps(contract, ensure_ascii=False).lower()
    names = {str(t.get("name") or "").lower() for t in tensors}
    if "gemma" in text:
        return "gemma_moe"
    if any("ffn_gate_up_exps.weight" in n for n in names):
        return "combined_gate_up_moe"
    if "qwen" in text:
        return "qwen"
    if "llama" in text or "mistral" in text:
        return "llama"
    if "glm" in text:
        return "glm"
    return "generic_transformer"


def build_layer_graph_ir(layer, tensors):
    prefix = f"blk.{layer}."
    names = set(_juju_tensors_by_prefix(tensors, prefix))

    def bind(*suffixes):
        out = []
        for suffix in suffixes:
            name = prefix + suffix
            if name in names:
                out.append(name)
        return out

    ops = [
        {"op": "rms_norm", "name": "attention_input_norm", "inputs": ["hidden"], "weights": bind("attn_norm.weight", "input_layernorm.weight"), "required": False},
        {"op": "linear", "name": "q_projection", "inputs": ["attention_norm"], "weights": bind("attn_q.weight", "attention.wq.weight"), "output": "q", "required": False},
        {"op": "linear", "name": "k_projection", "inputs": ["attention_norm"], "weights": bind("attn_k.weight", "attention.wk.weight"), "output": "k", "required": False},
        {"op": "linear", "name": "v_projection", "inputs": ["attention_norm"], "weights": bind("attn_v.weight", "attention.wv.weight"), "output": "v", "required": False},
        {"op": "rms_norm", "name": "q_norm", "inputs": ["q"], "weights": bind("attn_q_norm.weight"), "required": False},
        {"op": "rms_norm", "name": "k_norm", "inputs": ["k"], "weights": bind("attn_k_norm.weight"), "required": False},
        {"op": "rope", "name": "rotary_embedding", "inputs": ["q", "k"], "weights": bind("rope_freqs.weight"), "required": False},
        {"op": "attention", "name": "attention", "inputs": ["q", "k", "v"], "kv_cache": "quantized_qkv_cache", "required": False},
        {"op": "linear", "name": "attention_output", "inputs": ["attention"], "weights": bind("attn_output.weight", "attention.wo.weight"), "output": "attention_out", "required": False},
        {"op": "residual", "name": "attention_residual", "inputs": ["hidden", "attention_out"], "required": True},
        {"op": "rms_norm", "name": "ffn_norm", "inputs": ["hidden"], "weights": bind("ffn_norm.weight", "post_attention_norm.weight", "pre_ffw_norm.weight"), "required": False},
        {"op": "hidden_snapshot", "name": "fate_gate_input_snapshot", "inputs": ["ffn_norm"], "target": "engine_state.gate_input_snapshots[layer]", "required": False},
        {"op": "linear", "name": "moe_router", "inputs": ["ffn_norm"], "weights": bind("ffn_gate_inp.weight", "router.weight"), "scale": bind("ffn_gate_inp.scale"), "output": "expert_scores", "required": False},
        {"op": "topk", "name": "expert_select", "inputs": ["expert_scores"], "config_key": "adaptive_seq_topk_entropy", "required": False},
        {"op": "moe_expert_mlp", "name": "moe_experts", "inputs": ["ffn_norm", "selected_experts"], "weights": bind("ffn_gate_up_exps.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"), "required": bool(3 <= int(layer) <= 78)},
        {"op": "dense_mlp", "name": "dense_ffn_fallback", "inputs": ["ffn_norm"], "weights": bind("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"), "required": not bool(3 <= int(layer) <= 78)},
        {"op": "residual", "name": "ffn_residual", "inputs": ["hidden", "ffn_out"], "required": True},
        {"op": "scale", "name": "layer_output_scale", "inputs": ["hidden"], "weights": bind("layer_output_scale.weight"), "required": False},
    ]
    return {
        "layer": int(layer),
        "tensor_prefix": prefix,
        "available_tensors": sorted(names),
        "ops": ops,
    }


def build_juju_graph_ir(*, contract, tensor_records, sections, source_name, source_path, source_repo_id, weight_file, index_file):
    arch = dict(contract.get("arch_meta") or {})
    shape_map = _juju_tensor_shape_map(tensor_records)
    layers = sorted({
        layer for layer in (_juju_layer_id_from_name(t.get("name")) for t in tensor_records)
        if layer is not None
    })
    token_embd = _juju_first_tensor(tensor_records, "token_embd.weight")
    lm_head = _juju_first_tensor(tensor_records, "output.weight") or token_embd
    output_norm = _juju_first_tensor(tensor_records, "output_norm.weight", "norm.weight")
    priority_rules = [
        {"match": "token_embd.weight|output.weight|output_norm.weight|rope_freqs.weight", "priority": 100, "residency": "FAST_MEM", "prefetch": "startup_hot"},
        {"match": "attention/norm/router tensors", "priority": 85, "residency": "FAST_MEM", "prefetch": "layer_hot"},
        {"match": "expert tensors", "priority": 65, "residency": "SLOW_MEM", "prefetch": "expert_stream"},
        {"match": "large FAST_MEM tensors", "priority": "keep but streamable", "residency": "FAST_MEM_STREAMABLE", "prefetch": "bounded"},
    ]
    return {
        "format": "JUJU_GRAPH_IR_V1",
        "schema_version": 1,
        "required": True,
        "fail_closed_if_missing": True,
        "graph_id": f"{source_repo_id}:{source_path}:{weight_file}",
        "source": {
            "repo_id": source_repo_id,
            "source_path": source_path,
            "source_name": source_name,
            "weight_file": weight_file,
            "index_file": index_file,
        },
        "architecture": {
            "family": infer_juju_graph_family(contract, tensor_records),
            "declared_architecture": contract.get("architecture") or arch.get("architecture") or "",
            "model_id": contract.get("model_id") or contract.get("model_name") or "",
            "num_hidden_layers": arch.get("n_layers") or arch.get("num_hidden_layers") or len(layers),
            "hidden_size": arch.get("hidden_dim") or arch.get("hidden_size") or (shape_map.get(token_embd, [0])[0] if token_embd else 0),
            "vocab_size": arch.get("vocab_size") or (shape_map.get(token_embd, [0, 0])[1] if token_embd and len(shape_map.get(token_embd, [])) > 1 else 0),
            "head_dim": arch.get("head_dim"),
            "num_attention_heads": arch.get("n_heads") or arch.get("num_attention_heads"),
            "num_key_value_heads": arch.get("n_kv_heads") or arch.get("num_key_value_heads"),
            "experts_per_moe_layer": arch.get("experts_per_moe_layer") or arch.get("n_experts"),
            "routed_experts_per_token": arch.get("routed_experts_per_token") or arch.get("top_k"),
            "norm_eps": arch.get("norm_eps") or arch.get("rms_norm_eps"),
            "rope": {
                "type": arch.get("rope_type") or arch.get("rope_scaling_type") or "runtime_from_source_metadata",
                "theta": arch.get("rope_theta"),
                "scaling": arch.get("rope_scaling"),
            },
        },
        "tokenizer_contract": {
            "tokenizer_files": ["tokenizer.json", "tokenizer.model"],
            "chat_template_source": "tokenizer_config_or_model_card",
            "missing_tokenizer_behavior": "text_api_returns_tokenizer_unavailable_input_ids_still_allowed",
        },
        "quantization": {
            "weight": contract.get("weight_quant_schema", {}),
            "qkv_cache": contract.get("qkv_cache_schema", {}),
            "source_weight_bits": contract.get("source_weight_bits"),
            "source_weight_family": contract.get("source_weight_quant_family"),
            "source_weight_kernel_family": contract.get("source_weight_kernel_family"),
        },
        "tensor_bindings": {
            "token_embedding": token_embd,
            "lm_head": lm_head,
            "lm_head_tied_to_token_embedding": bool(lm_head and token_embd and lm_head == token_embd),
            "final_norm": output_norm,
            "rope_freqs": _juju_first_tensor(tensor_records, "rope_freqs.weight"),
            "layer_tensor_prefix": "blk.{layer}.",
            "shape_map": shape_map,
        },
        "ops": [
            {"op": "input_tokens", "output": "token_ids", "required": True},
            {"op": "embedding_lookup", "weights": [token_embd], "output": "hidden", "required": bool(token_embd)},
            {"op": "for_each_layer", "layers": [int(x) for x in layers], "body_ref": "layers"},
            {"op": "rms_norm", "name": "final_norm", "weights": [output_norm] if output_norm else [], "required": bool(output_norm)},
            {"op": "lm_head", "weights": [lm_head] if lm_head else [], "tied_to_embedding": bool(lm_head == token_embd), "required": bool(lm_head)},
            {"op": "sampler", "inputs": ["logits"], "required": True},
        ],
        "layers": [build_layer_graph_ir(layer, tensor_records) for layer in layers],
        "runtime_policy": {
            "execution": "graph_ir_executor_required",
            "unknown_op": "fail_closed",
            "unknown_tensor": "fail_closed_for_required_optional_skip",
            "kv_cache": "quantized_qkv_cache_required" if contract.get("qkv_cache_schema") else "runtime_default",
            "residency_policy": contract.get("residency_policy", {}),
            "prefetch_plan_hints": contract.get("prefetch_plan_hints", {}),
            "kernel_hints": contract.get("kernel_hints", {}),
            "execution_hints": contract.get("execution_hints", {}),
            "memory_management_hints": contract.get("memory_management_hints", {}),
            "hard_defaults": {
                "vram_double_admission_guard": {"enabled": True, "counter": "pending_vram_reservation"},
                "macos_available_ram": {"count_inactive_pages": False},
                "metal_unified_memory_budget_percent": 60,
                "router_seq_topk_entropy": {
                    "enabled": True,
                    "base_k": 8,
                    "low_entropy_threshold": 0.30,
                    "low_entropy_k_multiplier": 0.50,
                    "high_entropy_threshold": 0.70,
                    "high_entropy_max_k": 12,
                },
                "duoserve_prefill_decode_split": {
                    "enabled": True,
                    "disable_lookahead_during_prefill": True,
                    "prefill_phase_source": "engine.generation_phase == PREFILL",
                },
                "expertflow_adaptive_prediction_depth": {
                    "enabled": True,
                    "entropy_over": 0.70,
                    "max_prefetch_depth": 1,
                },
                "fate_hidden_snapshot": {
                    "enabled": True,
                    "capture": "gate_input_before_router",
                    "storage": "engine_state.gate_input_snapshots[layer]",
                },
            },
        },
        "priority_tables": {
            "tensor_priority_fields": ["runtime_priority", "prefetch_priority", "prefetch_class", "residency_hint", "graph_role"],
            "rules": priority_rules,
            "section_priorities": [
                {
                    "name": s.get("name", ""),
                    "type": s.get("type", 0),
                    "prefetch_distance": s.get("prefetch_distance", 0),
                    "mmap_friendly": s.get("mmap_friendly", 0),
                    "priority": 100 if s.get("name") == "SHARED_WEIGHTS" else 70,
                }
                for s in sections
            ],
        },
        "moe_offload_policy": {
            "enabled": True,
            "router_first": True,
            "expert_tensor_patterns": [
                "blk.{layer}.ffn_gate_up_exps.weight",
                "blk.{layer}.ffn_gate_exps.weight",
                "blk.{layer}.ffn_up_exps.weight",
                "blk.{layer}.ffn_down_exps.weight",
            ],
            "dense_fallback_patterns": [
                "blk.{layer}.ffn_gate.weight",
                "blk.{layer}.ffn_up.weight",
                "blk.{layer}.ffn_down.weight",
            ],
            "tier_names": ["COMPUTE_MEM", "FAST_MEM", "SLOW_MEM"],
            "bucket_mapping": {
                "shared_weights": "FAST_MEM",
                "hot_experts": "FAST_MEM",
                "warm_experts": "FAST_MEM_STREAMABLE",
                "cold_experts": "SLOW_MEM",
            },
            "admission_priority": {
                "router": 100,
                "attention": 95,
                "norm": 90,
                "dense_ffn": 80,
                "expert": 70,
                "cold_expert": 60,
            },
            "prefetch": {
                "unit": "layer_expert_tensor",
                "trigger": "router_topk_and_previous_layer_coactivation",
                "lookahead_layers": [1, 2],
                "coactivation_table": "mutable_runtime_index",
                "fallback_when_no_history": "router_scores",
                "bounded_by": ["ram_budget", "vram_budget", "staging_slots", "io_queue_depth"],
                "priority_field": "prefetch_priority",
                "score_filter": {
                    "enabled": True,
                    "vram_percentile": 0.70,
                    "ram_percentile": 0.50,
                    "drop_below_percentile": 0.50,
                },
            },
            "eviction": {
                "policy": "least_stale_predicted_next_use_max_heap",
                "protect_roles": ["router", "attention", "norm", "token_embedding", "lm_head"],
                "demote_order": ["cold_experts", "warm_experts", "large_shared_streamable"],
                "primary_key": "predicted_next_use_epoch",
                "tie_breakers": ["hot_score", "last_touch_epoch"],
                "rollback_required": True,
            },
            "cpu_hot_miss": {
                "enabled": True,
                "condition": "expert_in_ram_and_decode_batch_le_4",
                "decision": "cpu_ms < pcie_transfer_ms",
                "cpu_gflops_default": 100.0,
            },
            "score_aware_precision": {
                "enabled": True,
                "low_score_load_bits": 4,
                "fallback_when_nvfp4_unavailable": "int4",
                "requires_decoder": "engine_int4_or_scale4_decode",
            },
            "streaming": {
                "expert_streaming_required": True,
                "direct_io_alignment": 4096,
                "split_combined_gate_up": True,
                "allow_partial_expert_segments": True,
            },
            "telemetry": {
                "record_expert_hits": True,
                "record_layer_latency": True,
                "record_io_wait": True,
                "record_cache_promotions": True,
                "record_cache_evictions": True,
                "record_coactivation": True,
            },
            "source_contracts": {
                "residency_policy": contract.get("residency_policy", {}),
                "prefetch_plan_hints": contract.get("prefetch_plan_hints", {}),
                "dynamic_swap_triggers": contract.get("dynamic_swap_triggers", {}),
                "activation_stats_schema": contract.get("activation_stats_schema", {}),
                "sparsity_schema": contract.get("sparsity_schema", {}),
                "runtime_monitoring_hints": contract.get("runtime_monitoring_hints", {}),
            },
        },
        "validation": {
            "require_all_required_ops_bound": True,
            "require_tensor_shape_match": True,
            "require_quant_schema_match": True,
            "require_qkv_policy_match": bool(contract.get("qkv_cache_schema")),
            "allow_optional_ops_missing": True,
            "tensor_count": len(tensor_records),
            "section_count": len(sections),
        },
        "compatibility": {
            "min_engine_graph_ir_version": 1,
            "endianness": "little",
            "alignment": 4096,
            "portable_backend_terms_only": True,
        },
    }


def build_juju_shard_plan_from_hf_url(
    *,
    source_url,
    source_name,
    source_path,
    contract,
    token=None,
    source_repo_id="",
    chunk_size=16 * 1024 * 1024,
    artifact_source_name=None,
    tensor_name_allowlist=None,
    split_info=None,
):
    artifact_source_name = artifact_source_name or source_name
    artifact_names = juju_artifact_names(artifact_source_name)
    fixed_segments = []
    source_segments = []
    sections = []
    section_sizes = {}
    tensor_records = []

    def add_fixed(offset, data):
        if data:
            fixed_segments.append({"offset": int(offset), "size": len(data), "data": data})

    def add_json_section_at(pos, section_type, name, payload):
        raw = json_section_bytes(payload)
        pos = align_up(pos, 4096)
        offset = pos
        add_fixed(offset, raw)
        entry = {
            "type": section_type,
            "name": name,
            "offset": offset,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "mmap_friendly": 0,
        }
        sections.append(entry)
        section_sizes[section_type] = section_sizes.get(section_type, 0) + len(raw)
        return pos + len(raw)

    with requests.Session() as session:
        directory, total_bytes = read_remote_directory(session, source_url, token=token)
        allowset = set(tensor_name_allowlist or [])
        if allowset:
            known = {tensor["name"] for tensor in directory["tensors"]}
            missing = sorted(allowset - known)
            if missing:
                raise RuntimeError(f"JUJU split references missing tensors: {missing[:8]}")
        active_tensors = [
            tensor for tensor in directory["tensors"]
            if int(tensor.get("bytes") or 0) > 0 and (not allowset or tensor["name"] in allowset)
        ]
        split_meta = split_info or {
            "enabled": False,
            "split_index": 1,
            "split_count": 1,
            "parent_source_name": source_name,
            "artifact_source_name": artifact_source_name,
            "tensor_count": len(active_tensors),
        }
        pos = JUJU_HEADER_BYTES
        table_offset = pos
        pos += 8 * JUJU_SECTION_ENTRY_BYTES
        pos = align_up(pos, 4096)

        meta = {
            "format": "JUJU_SHARDED_CONTAINER_V1",
            "source_format": "GGUF",
            "source_role": "conversion_source_only",
            "source_repo_id": source_repo_id,
            "source_path": source_path,
            "source_name": source_name,
            "artifact_source_name": artifact_source_name,
            "weight_file": artifact_names["weights"],
            "index_file": artifact_names["index"],
            "tensor_payload_layout": "4kb_aligned_tensor_sections",
            "artifact_name_policy": "preserve_original_shard_stem_change_extension_only",
            "graph_ir_format": "JUJU_GRAPH_IR_V1",
            "graph_ir_required": True,
            "split": split_meta,
            "gguf_directory": {
                "version": directory["version"],
                "tensor_count": directory["tensor_count"],
                "emitted_tensor_count": len(active_tensors),
                "kv_count": directory["kv_count"],
                "alignment": directory["alignment"],
                "data_start": directory["data_start"],
                "source_bytes": total_bytes,
            },
            "contract": contract,
        }
        pos = add_json_section_at(pos, JUJU_SECTION_MODEL_META, "MODEL_META", meta)
        qkv_schema = contract.get("qkv_cache_schema")
        if qkv_schema:
            pos = add_json_section_at(pos, JUJU_SECTION_QKV_POLICY, "QKV_POLICY", qkv_schema)

        for bucket in ("shared_weights", "hot_experts", "warm_experts", "cold_experts"):
            group = [t for t in active_tensors if t["bucket"] == bucket and t["bytes"] > 0]
            if not group:
                continue
            pos = align_up(pos, 4096)
            section_offset = pos
            for tensor in group:
                pos = align_up(pos, 4096)
                tensor_offset = pos
                source_segments.append({
                    "offset": tensor_offset,
                    "size": tensor["bytes"],
                    "source_offset": tensor["source_offset"],
                })
                runtime_priority = tensor_runtime_priority(tensor["name"], bucket, tensor["bytes"])
                tensor_records.append({
                    "name": tensor["name"],
                    "bucket": bucket,
                    "dims": tensor["dims"],
                    "shape": tensor["shape"],
                    "gguf_type": tensor["type"],
                    "source_offset": tensor["source_offset"],
                    "source_bytes": tensor["bytes"],
                    "juju_offset": tensor_offset,
                    "juju_bytes": tensor["bytes"],
                    "alignment": 4096,
                    **runtime_priority,
                })
                pos += tensor["bytes"]
            size = pos - section_offset
            section_type = section_type_for_bucket(bucket)
            sections.append({
                "type": section_type,
                "name": bucket.upper()[:32],
                "offset": section_offset,
                "size": size,
                "sha256": "0" * 64,
                "prefetch_distance": 2 if bucket != "shared_weights" else 0,
                "mmap_friendly": 1,
            })
            section_sizes[section_type] = section_sizes.get(section_type, 0) + size

        graph_ir = build_juju_graph_ir(
            contract=contract,
            tensor_records=tensor_records,
            sections=list(sections),
            source_name=source_name,
            source_path=source_path,
            source_repo_id=source_repo_id,
            weight_file=artifact_names["weights"],
            index_file=artifact_names["index"],
        )
        idx = {
            "format": "JUJU_IDX_JSON_V1",
            "schema_version": 2,
            "mutable_runtime_index": True,
            "weight_file": artifact_names["weights"],
            "source_repo_id": source_repo_id,
            "source_path": source_path,
            "source_name": source_name,
            "artifact_source_name": artifact_source_name,
            "split": split_meta,
            "graph_ir_format": graph_ir["format"],
            "graph_ir_required": True,
            "graph_ir": graph_ir,
            "priority_tables": graph_ir["priority_tables"],
            "moe_offload_policy": graph_ir["moe_offload_policy"],
            "tensor_count": len(tensor_records),
            "tensors": tensor_records,
            "sections": list(sections),
        }
        pos = add_json_section_at(pos, JUJU_SECTION_LAYER_ORDER_INDEX, "TENSOR_INDEX", idx)
        file_size_value = pos
        if len(sections) > 8:
            raise RuntimeError(f"too many JUJU sections: {len(sections)}")

        table = b"".join(pack_section(entry) for entry in sections)
        table = table + (b"\x00" * ((8 * JUJU_SECTION_ENTRY_BYTES) - len(table)))
        header = make_header(contract, artifact_source_name, file_size_value, sections, section_sizes)
        add_fixed(0, header)
        add_fixed(table_offset, table)

    idx["sections"] = sections
    return {
        "format": "juju_sharded_container_v1",
        "source_url": source_url,
        "source_name": source_name,
        "artifact_source_name": artifact_source_name,
        "source_path": source_path,
        "source_repo_id": source_repo_id,
        "weight_file": artifact_names["weights"],
        "index_file": artifact_names["index"],
        "verify_file": artifact_names["verify"],
        "split": split_meta,
        "bytes": file_size_value,
        "source_bytes": total_bytes,
        "tensor_count": len(tensor_records),
        "section_count": len(sections),
        "storage_mode": "remote_range_to_streamed_4kb_aligned_juju_sections",
        "artifact_name_policy": "original_shard_stem_with_juju_extension",
        "fixed_segments": fixed_segments,
        "source_segments": source_segments,
        "index_json": idx,
        "chunk_size": int(chunk_size),
        "token": token,
    }


class JujuVirtualFile(io.BufferedIOBase):
    def __init__(self, plan):
        super().__init__()
        self._lock = threading.RLock()
        self._plan = plan
        self._size = int(plan["bytes"])
        self._pos = 0
        self._session = requests.Session()
        self._remote_chunk = max(1, int(plan.get("chunk_size") or (16 * 1024 * 1024)))
        self._cache_start = -1
        self._cache_end = -1
        self._cache_data = b""
        segments = []
        for segment in plan["fixed_segments"]:
            segments.append({
                "kind": "fixed",
                "offset": int(segment["offset"]),
                "size": int(segment["size"]),
                "data": segment["data"],
            })
        for segment in plan["source_segments"]:
            segments.append({
                "kind": "source",
                "offset": int(segment["offset"]),
                "size": int(segment["size"]),
                "source_offset": int(segment["source_offset"]),
            })
        self._segments = sorted(segments, key=lambda item: item["offset"])
        self._offsets = [item["offset"] for item in self._segments]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        with self._lock:
            return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        with self._lock:
            if whence == io.SEEK_SET:
            pos = int(offset)
        elif whence == io.SEEK_CUR:
            pos = self._pos + int(offset)
        elif whence == io.SEEK_END:
            pos = self._size + int(offset)
        else:
                raise ValueError(f"unsupported whence: {whence}")
            if pos < 0:
                raise ValueError("negative seek position")
            self._pos = min(pos, self._size)
            return self._pos

    def readinto(self, buffer):
        with self._lock:
            data = self.read(len(buffer))
            n = len(data)
            buffer[:n] = data
            return n

    def read(self, size=-1):
        with self._lock:
            if self.closed or self._pos >= self._size:
            return b""
            if size is None or size < 0:
                end = min(self._size, self._pos + self._remote_chunk)
            else:
                end = min(self._size, self._pos + int(size))
            chunks = []
            while self._pos < end:
                idx = bisect.bisect_right(self._offsets, self._pos) - 1
                segment = self._segments[idx] if idx >= 0 else None
                if segment and self._pos < segment["offset"] + segment["size"]:
                    rel = self._pos - segment["offset"]
                    take = min(end - self._pos, segment["size"] - rel)
                    if segment["kind"] == "fixed":
                        chunks.append(segment["data"][rel:rel + take])
                    else:
                        chunks.append(self._read_source_segment(segment, rel, take))
                    self._pos += take
                    continue
                next_offset = self._segments[idx + 1]["offset"] if idx + 1 < len(self._segments) else self._size
                take = min(end - self._pos, next_offset - self._pos)
                chunks.append(b"\x00" * take)
                self._pos += take
            return b"".join(chunks)

    def _read_source_segment(self, segment, rel, size):
        out = []
        remaining = int(size)
        source_abs = int(segment["source_offset"]) + int(rel)
        segment_end = int(segment["source_offset"]) + int(segment["size"])
        while remaining > 0:
            if self._cache_start <= source_abs < self._cache_end:
                cache_rel = source_abs - self._cache_start
                take = min(remaining, self._cache_end - source_abs)
                out.append(self._cache_data[cache_rel:cache_rel + take])
                source_abs += take
                remaining -= take
                continue
            fetch_end = min(segment_end, source_abs + max(self._remote_chunk, remaining)) - 1
            resp = fetch_range(
                self._session,
                self._plan["source_url"],
                source_abs,
                fetch_end,
                token=self._plan.get("token"),
                stream=False,
            )
            try:
                data = resp.content
            finally:
                resp.close()
            if not data:
                raise EOFError("empty source range while streaming JUJU upload")
            self._cache_start = source_abs
            self._cache_end = source_abs + len(data)
            self._cache_data = data
        return b"".join(out)

    def close(self):
        try:
            self._session.close()
        finally:
            super().close()


def prepare_juju_shard_upload_from_hf_url(
    *,
    source_url,
    source_name,
    source_path,
    index_path,
    contract,
    token=None,
    source_repo_id="",
    chunk_size=16 * 1024 * 1024,
    artifact_source_name=None,
    tensor_name_allowlist=None,
    split_info=None,
):
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    plan = build_juju_shard_plan_from_hf_url(
        source_url=source_url,
        source_name=source_name,
        source_path=source_path,
        contract=contract,
        token=token,
        source_repo_id=source_repo_id,
        chunk_size=chunk_size,
        artifact_source_name=artifact_source_name,
        tensor_name_allowlist=tensor_name_allowlist,
        split_info=split_info,
    )
    index_path.write_text(json.dumps(plan["index_json"], ensure_ascii=False, indent=2), encoding="utf-8")
    info = {
        "format": plan["format"],
        "path": f"<stream:{plan['weight_file']}>",
        "index_path": str(index_path),
        "source_name": plan["source_name"],
        "artifact_source_name": plan["artifact_source_name"],
        "weight_file": plan["weight_file"],
        "index_file": plan["index_file"],
        "verify_file": plan["verify_file"],
        "split": plan["split"],
        "bytes": plan["bytes"],
        "index_bytes": index_path.stat().st_size,
        "index_sha256": sha256_file(index_path),
        "source_bytes": plan["source_bytes"],
        "source_sha256": "",
        "tensor_count": plan["tensor_count"],
        "section_count": plan["section_count"],
        "storage_mode": plan["storage_mode"],
        "artifact_name_policy": plan["artifact_name_policy"],
    }
    return info, JujuVirtualFile(plan)


def prepare_juju_shard_upload_parts_from_hf_url(
    *,
    source_url,
    source_name,
    source_path,
    index_path,
    contract,
    token=None,
    source_repo_id="",
    chunk_size=16 * 1024 * 1024,
    max_file_bytes=None,
):
    with requests.Session() as session:
        directory, total_bytes = read_remote_directory(session, source_url, token=token)
    split_plan = plan_juju_tensor_splits(directory, max_file_bytes=max_file_bytes)
    base_index_path = Path(index_path)
    parts = []
    for split in split_plan:
        if split["enabled"]:
            artifact_source_name = juju_split_source_name(source_name, split["split_index"], split["split_count"])
            child_index_path = base_index_path.parent / juju_artifact_names(artifact_source_name)["index"]
        else:
            artifact_source_name = source_name
            child_index_path = base_index_path
        split_info = {
            "enabled": bool(split["enabled"]),
            "parent_source_name": source_name,
            "artifact_source_name": artifact_source_name,
            "split_index": int(split["split_index"]),
            "split_count": int(split["split_count"]),
            "source_tensor_count": int(directory["tensor_count"]),
            "tensor_count": len(split["tensor_names"]),
            "tensor_bytes": int(split["tensor_bytes"]),
            "estimated_file_bytes": int(split.get("estimated_file_bytes") or 0),
            "max_file_bytes": int(split["max_file_bytes"]),
        }
        info, stream = prepare_juju_shard_upload_from_hf_url(
            source_url=source_url,
            source_name=source_name,
            source_path=source_path,
            index_path=child_index_path,
            contract=contract,
            token=token,
            source_repo_id=source_repo_id,
            chunk_size=chunk_size,
            artifact_source_name=artifact_source_name,
            tensor_name_allowlist=split["tensor_names"],
            split_info=split_info,
        )
        info["source_bytes"] = total_bytes
        parts.append((info, stream))
    return parts


def write_juju_shard_from_hf_url(
    *,
    source_url,
    source_name,
    source_path,
    output_path,
    index_path,
    contract,
    token=None,
    source_repo_id="",
    chunk_size=16 * 1024 * 1024,
    artifact_source_name=None,
    tensor_name_allowlist=None,
    split_info=None,
):
    artifact_source_name = artifact_source_name or source_name
    output_path = Path(output_path)
    index_path = Path(index_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    section_sizes = {}
    tensor_records = []

    def add_json_section(out, section_type, name, payload):
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        write_padding(out, 4096)
        offset = out.tell()
        out.write(raw)
        entry = {
            "type": section_type,
            "name": name,
            "offset": offset,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "mmap_friendly": 0,
        }
        sections.append(entry)
        section_sizes[section_type] = section_sizes.get(section_type, 0) + len(raw)

    with requests.Session() as session:
        directory, total_bytes = read_remote_directory(session, source_url, token=token)
        allowset = set(tensor_name_allowlist or [])
        if allowset:
            known = {tensor["name"] for tensor in directory["tensors"]}
            missing = sorted(allowset - known)
            if missing:
                raise RuntimeError(f"JUJU split references missing tensors: {missing[:8]}")
        active_tensors = [
            tensor for tensor in directory["tensors"]
            if int(tensor.get("bytes") or 0) > 0 and (not allowset or tensor["name"] in allowset)
        ]
        split_meta = split_info or {
            "enabled": False,
            "split_index": 1,
            "split_count": 1,
            "parent_source_name": source_name,
            "artifact_source_name": artifact_source_name,
            "tensor_count": len(active_tensors),
        }
        with output_path.open("wb") as out:
            out.write(b"\x00" * JUJU_HEADER_BYTES)
            table_offset = out.tell()
            out.write(b"\x00" * (8 * JUJU_SECTION_ENTRY_BYTES))
            write_padding(out, 4096)

            meta = {
                "format": "JUJU_SHARDED_CONTAINER_V1",
                "source_format": "GGUF",
                "source_role": "conversion_source_only",
                "source_repo_id": source_repo_id,
                "source_path": source_path,
                "source_name": source_name,
                "artifact_source_name": artifact_source_name,
                "weight_file": output_path.name,
                "index_file": index_path.name,
                "tensor_payload_layout": "4kb_aligned_tensor_sections",
                "artifact_name_policy": "preserve_original_shard_stem_change_extension_only",
                "graph_ir_format": "JUJU_GRAPH_IR_V1",
                "graph_ir_required": True,
                "split": split_meta,
                "gguf_directory": {
                    "version": directory["version"],
                    "tensor_count": directory["tensor_count"],
                    "emitted_tensor_count": len(active_tensors),
                    "kv_count": directory["kv_count"],
                    "alignment": directory["alignment"],
                    "data_start": directory["data_start"],
                    "source_bytes": total_bytes,
                },
                "contract": contract,
            }
            add_json_section(out, JUJU_SECTION_MODEL_META, "MODEL_META", meta)
            qkv_schema = contract.get("qkv_cache_schema")
            if qkv_schema:
                add_json_section(out, JUJU_SECTION_QKV_POLICY, "QKV_POLICY", qkv_schema)

            for bucket in ("shared_weights", "hot_experts", "warm_experts", "cold_experts"):
                group = [t for t in active_tensors if t["bucket"] == bucket and t["bytes"] > 0]
                if not group:
                    continue
                write_padding(out, 4096)
                offset = out.tell()
                digest = hashlib.sha256()
                for tensor in group:
                    write_padding(out, 4096)
                    tensor_offset = out.tell()
                    stream_range(
                        session,
                        source_url,
                        tensor["source_offset"],
                        tensor["bytes"],
                        out,
                        token,
                        digest,
                        chunk_size=chunk_size,
                    )
                    runtime_priority = tensor_runtime_priority(tensor["name"], bucket, tensor["bytes"])
                    tensor_records.append({
                        "name": tensor["name"],
                        "bucket": bucket,
                        "dims": tensor["dims"],
                        "shape": tensor["shape"],
                        "gguf_type": tensor["type"],
                        "source_offset": tensor["source_offset"],
                        "source_bytes": tensor["bytes"],
                        "juju_offset": tensor_offset,
                        "juju_bytes": tensor["bytes"],
                        "alignment": 4096,
                        **runtime_priority,
                    })
                size = out.tell() - offset
                section_type = section_type_for_bucket(bucket)
                sections.append({
                    "type": section_type,
                    "name": bucket.upper()[:32],
                    "offset": offset,
                    "size": size,
                    "sha256": digest.hexdigest(),
                    "prefetch_distance": 2 if bucket != "shared_weights" else 0,
                    "mmap_friendly": 1,
                })
                section_sizes[section_type] = section_sizes.get(section_type, 0) + size

            graph_ir = build_juju_graph_ir(
                contract=contract,
                tensor_records=tensor_records,
                sections=list(sections),
                source_name=source_name,
                source_path=source_path,
                source_repo_id=source_repo_id,
                weight_file=output_path.name,
                index_file=index_path.name,
            )
            idx = {
                "format": "JUJU_IDX_JSON_V1",
                "schema_version": 2,
                "mutable_runtime_index": True,
                "weight_file": output_path.name,
                "source_repo_id": source_repo_id,
                "source_path": source_path,
                "source_name": source_name,
                "artifact_source_name": artifact_source_name,
                "split": split_meta,
                "graph_ir_format": graph_ir["format"],
                "graph_ir_required": True,
                "graph_ir": graph_ir,
                "priority_tables": graph_ir["priority_tables"],
                "moe_offload_policy": graph_ir["moe_offload_policy"],
                "tensor_count": len(tensor_records),
                "tensors": tensor_records,
                "sections": sections,
            }
            add_json_section(out, JUJU_SECTION_LAYER_ORDER_INDEX, "TENSOR_INDEX", idx)
            file_size_value = out.tell()
            if len(sections) > 8:
                raise RuntimeError(f"too many JUJU sections: {len(sections)}")
            out.seek(table_offset)
            for entry in sections:
                out.write(pack_section(entry))
            out.seek(0)
            out.write(make_header(contract, artifact_source_name, file_size_value, sections, section_sizes))

    index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "format": "juju_sharded_container_v1",
        "path": str(output_path),
        "index_path": str(index_path),
        "source_name": source_name,
        "artifact_source_name": artifact_source_name,
        "weight_file": output_path.name,
        "index_file": index_path.name,
        "verify_file": juju_artifact_names(artifact_source_name)["verify"],
        "split": split_meta,
        "bytes": output_path.stat().st_size,
        "index_bytes": index_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "index_sha256": sha256_file(index_path),
        "source_bytes": total_bytes,
        "source_sha256": "",
        "tensor_count": len(tensor_records),
        "section_count": len(sections),
        "storage_mode": "remote_range_to_4kb_aligned_juju_sections",
        "artifact_name_policy": "original_shard_stem_with_juju_extension",
    }


def write_juju_shard_parts_from_hf_url(
    *,
    source_url,
    source_name,
    source_path,
    output_path,
    index_path,
    contract,
    token=None,
    source_repo_id="",
    chunk_size=16 * 1024 * 1024,
    max_file_bytes=None,
):
    with requests.Session() as session:
        directory, total_bytes = read_remote_directory(session, source_url, token=token)
    split_plan = plan_juju_tensor_splits(directory, max_file_bytes=max_file_bytes)
    base_output_path = Path(output_path)
    base_index_path = Path(index_path)
    infos = []
    for split in split_plan:
        if split["enabled"]:
            artifact_source_name = juju_split_source_name(source_name, split["split_index"], split["split_count"])
            child_output_path = base_output_path.parent / juju_artifact_names(artifact_source_name)["weights"]
            child_index_path = base_index_path.parent / juju_artifact_names(artifact_source_name)["index"]
        else:
            artifact_source_name = source_name
            child_output_path = base_output_path
            child_index_path = base_index_path
        split_info = {
            "enabled": bool(split["enabled"]),
            "parent_source_name": source_name,
            "artifact_source_name": artifact_source_name,
            "split_index": int(split["split_index"]),
            "split_count": int(split["split_count"]),
            "source_tensor_count": int(directory["tensor_count"]),
            "tensor_count": len(split["tensor_names"]),
            "tensor_bytes": int(split["tensor_bytes"]),
            "estimated_file_bytes": int(split.get("estimated_file_bytes") or 0),
            "max_file_bytes": int(split["max_file_bytes"]),
        }
        info = write_juju_shard_from_hf_url(
            source_url=source_url,
            source_name=source_name,
            source_path=source_path,
            output_path=child_output_path,
            index_path=child_index_path,
            contract=contract,
            token=token,
            source_repo_id=source_repo_id,
            chunk_size=chunk_size,
            artifact_source_name=artifact_source_name,
            tensor_name_allowlist=split["tensor_names"],
            split_info=split_info,
        )
        info["source_bytes"] = total_bytes
        infos.append(info)
    return infos
