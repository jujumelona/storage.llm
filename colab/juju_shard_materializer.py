import hashlib
import io
import json
import threading
import bisect
import os
import re
import struct
import time
from pathlib import Path

import requests


STRICT_GGUF_EXACT_BYTES = os.environ.get("STRICT_GGUF_EXACT_BYTES", "0") != "0"

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

GGUF_RUNTIME_KV_KEY_HINTS = (
    "attention",
    "attn",
    "rope",
    "rotary",
    "norm",
    "scale",
    "softcap",
    "sliding",
    "expert",
    "moe",
    "router",
    "lora",
    "head",
    "context",
    "embedding",
    "feed_forward",
    "ffn",
    "block_count",
    "vocab_size",
    "file_type",
)

GGUF_RUNTIME_KV_EXACT_KEYS = {
    "general.architecture",
    "general.name",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
}

GGUF_RUNTIME_KV_ALIAS_MAP = {
    "general.architecture": ("architecture", "declared_architecture", "model_type"),
    "general.name": ("model_name",),
    "block_count": ("num_hidden_layers", "n_layers"),
    "embedding_length": ("hidden_size", "hidden_dim"),
    "vocab_size": ("vocab_size",),
    "context_length": ("context_length", "max_position_embeddings"),
    "attention.head_count": ("num_attention_heads", "n_heads", "head_count"),
    "attention.head_count_kv": ("num_key_value_heads", "n_kv_heads", "head_count_kv"),
    "attention.head_count_global_kv": ("num_global_key_value_heads", "global_head_count_kv"),
    "attention.key_length": ("head_dim", "key_length"),
    "attention.value_length": ("value_head_dim", "v_head_dim", "value_length"),
    "attention.global_key_length": ("global_head_dim", "global_key_length"),
    "attention.global_value_length": ("global_value_head_dim", "global_value_length"),
    "attention.layer_norm_rms_epsilon": ("rms_norm_eps", "norm_eps"),
    "attention.q_lora_rank": ("q_lora_rank",),
    "attention.kv_lora_rank": ("kv_lora_rank",),
    "attention.qk_nope_head_dim": ("qk_nope_head_dim",),
    "attention.qk_rope_head_dim": ("qk_rope_head_dim",),
    "rope.freq_base": ("rope_theta", "theta"),
    "rope.dimension_count": ("qk_rope_head_dim", "rope_dimension_count"),
    "expert_count": ("experts_per_moe_layer", "n_experts"),
    "expert_used_count": ("routed_experts_per_token", "top_k", "num_experts_per_tok"),
    "expert_feed_forward_length": ("expert_intermediate_size", "expert_intermediate_dim"),
    "feed_forward_length": ("intermediate_size", "ffn_intermediate_size"),
    "final_logit_softcap": ("final_logit_softcap", "final_logit_softcapping", "logit_softcap"),
    "final_logit_softcapping": ("final_logit_softcapping", "final_logit_softcap", "logit_softcap"),
    "logit_softcap": ("logit_softcap", "final_logit_softcap"),
    "embedding_scale": ("embedding_scale", "embed_scale", "scale_emb"),
    "embed_scale": ("embed_scale", "embedding_scale", "scale_emb"),
    "scale_emb": ("scale_emb", "embedding_scale", "embed_scale"),
    "scale_embedding": ("scale_embedding", "embedding_scale", "embed_scale"),
    "partial_rotary_factor": ("partial_rotary_factor",),
    "full_rope_theta": ("full_rope_theta", "full_attention_rope_theta"),
    "sliding_rope_theta": ("sliding_rope_theta", "sliding_attention_rope_theta"),
    "full_attention_rope_theta": ("full_attention_rope_theta", "full_rope_theta"),
    "sliding_attention_rope_theta": ("sliding_attention_rope_theta", "sliding_rope_theta"),
    "routed_scaling_factor": ("routed_scaling_factor", "route_scale", "moe_routed_scaling_factor"),
    "route_scale": ("route_scale", "routed_scaling_factor"),
    "scoring_func": ("scoring_func", "score_func", "router_score_func"),
    "score_func": ("score_func", "scoring_func", "router_score_func"),
    "norm_topk_prob": ("norm_topk_prob", "normalize_topk_prob"),
    "normalize_topk_prob": ("normalize_topk_prob", "norm_topk_prob"),
    "sliding_window": ("sliding_window",),
    "full_attention_interval": ("full_attention_interval", "global_attention_interval"),
    "global_attention_interval": ("global_attention_interval", "full_attention_interval"),
    "full_attention_offset": ("full_attention_offset", "global_attention_offset"),
    "global_attention_offset": ("global_attention_offset", "full_attention_offset"),
}

JUJU_HEADER_BYTES = 4096
JUJU_SECTION_ENTRY_BYTES = 96
JUJU_SECTION_TABLE_RESERVED_ENTRIES = 32
JUJU_SECTION_MODEL_META = 0x0001
JUJU_SECTION_SHARED_WEIGHTS = 0x0010
JUJU_SECTION_HOT_EXPERTS = 0x0011
JUJU_SECTION_WARM_EXPERTS = 0x0012
JUJU_SECTION_COLD_EXPERTS = 0x0013
JUJU_SECTION_LAYER_ORDER_INDEX = 0x0020
JUJU_SECTION_QKV_POLICY = 0x0021
JUJU_SECTION_VISION_ENCODER = 0x0030
JUJU_SECTION_VISION_PROJ = 0x0031
JUJU_SECTION_AUDIO_ENCODER = 0x0040
JUJU_SECTION_VIDEO_ENCODER = 0x0050
JUJU_SECTION_DOCUMENT_ENCODER = 0x0060
JUJU_MODALITY_TEXT = 0x01
JUJU_MODALITY_IMAGE = 0x02
JUJU_MODALITY_AUDIO = 0x04
JUJU_MODALITY_VIDEO = 0x08
JUJU_MODALITY_DOCUMENT = 0x10
JUJU_TENSOR_BUCKET_ORDER = (
    "shared_weights",
    "hot_experts",
    "warm_experts",
    "cold_experts",
    "vision_encoder",
    "vision_projector",
    "audio_encoder",
    "video_encoder",
    "document_encoder",
)
HF_INDIVIDUAL_FILE_LIMIT_BYTES = 50 * 1024 * 1024 * 1024
DEFAULT_JUJU_UPLOAD_FILE_LIMIT_BYTES = 45 * 1024 * 1024 * 1024
JUJU_SPLIT_METADATA_RESERVE_BYTES = 512 * 1024 * 1024
JUJU_FORMAT_CONTRACT_VERSION = 2
JUJU_BINARY_WIRE_ID = "JUJU_V1_HEADER4096_SECTION96_ABS_OFFSETS"
JUJU_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "tiktoken.model",
    "vocab.json",
    "merges.txt",
    "tokenization_kimi.py",
    "tool_declaration_ts.py",
    "generation_config.json",
    "config.json",
    "configuration_deepseek.py",
    "configuration_kimi_k25.py",
    "modeling_deepseek.py",
    "modeling_kimi_k25.py",
    "kimi_k25_processor.py",
    "kimi_k25_vision_processing.py",
    "media_utils.py",
    "processor_config.json",
    "preprocessor_config.json",
    "image_processor_config.json",
    "feature_extractor.json",
    "video_preprocessor_config.json",
    "audio_config.json",
]
JUJU_REQUIRED_TOKENIZER_FILES = ["config.json"]
JUJU_REQUIRED_TOKENIZER_ANY_OF = ["tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model", "tiktoken.model", "vocab.json"]


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


def juju_target_tensor_splits():
    raw = str(os.environ.get("JUJU_TARGET_TENSOR_SPLITS", "")).strip()
    if not raw:
        return 0
    value = int(raw)
    if value < 0:
        raise ValueError("JUJU_TARGET_TENSOR_SPLITS must be non-negative")
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


def skip_array_payload(handle, elem_type, count):
    if elem_type == GGUF_TYPE_STRING:
        for _ in range(count):
            handle.seek(read_u64(handle), 1)
        return
    elem_size = GGUF_TYPE_SIZE.get(elem_type)
    if elem_size is None:
        raise ValueError(f"unsupported GGUF array element type: {elem_type}")
    handle.seek(elem_size * count, 1)


def skip_value(handle, value_type):
    if value_type == GGUF_TYPE_STRING:
        handle.seek(read_u64(handle), 1)
        return
    if value_type == GGUF_TYPE_ARRAY:
        elem_type = read_u32(handle)
        count = read_u64(handle)
        skip_array_payload(handle, elem_type, count)
        return
    size = GGUF_TYPE_SIZE.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF value type: {value_type}")
    handle.seek(size, 1)


def read_gguf_scalar_value(handle, value_type):
    if value_type == GGUF_TYPE_UINT8:
        return struct.unpack("<B", read_exact(handle, 1))[0]
    if value_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", read_exact(handle, 1))[0]
    if value_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", read_exact(handle, 2))[0]
    if value_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", read_exact(handle, 2))[0]
    if value_type == GGUF_TYPE_UINT32:
        return read_u32(handle)
    if value_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", read_exact(handle, 4))[0]
    if value_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", read_exact(handle, 4))[0]
    if value_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack("<?", read_exact(handle, 1))[0])
    if value_type == GGUF_TYPE_STRING:
        return read_string(handle)
    if value_type == GGUF_TYPE_UINT64:
        return read_u64(handle)
    if value_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", read_exact(handle, 8))[0]
    if value_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", read_exact(handle, 8))[0]
    return None


def read_gguf_array_value(handle):
    elem_type = read_u32(handle)
    count = read_u64(handle)
    limit = int(os.environ.get("GGUF_RUNTIME_ARRAY_CAPTURE_LIMIT", "4096"))
    if count > limit:
        skip_array_payload(handle, elem_type, count)
        return None
    values = []
    if elem_type == GGUF_TYPE_STRING:
        for _ in range(count):
            values.append(read_string(handle))
        return values
    if elem_type not in GGUF_TYPE_SIZE:
        skip_array_payload(handle, elem_type, count)
        return None
    for _ in range(count):
        values.append(read_gguf_scalar_value(handle, elem_type))
    return values


def should_capture_gguf_runtime_kv(key):
    lower = str(key or "").lower()
    if lower in GGUF_RUNTIME_KV_EXACT_KEYS:
        return True
    return any(hint in lower for hint in GGUF_RUNTIME_KV_KEY_HINTS)


def gguf_runtime_aliases_for_key(key):
    lower = str(key or "").lower()
    aliases = []
    for suffix, mapped in GGUF_RUNTIME_KV_ALIAS_MAP.items():
        if lower == suffix or lower.endswith("." + suffix) or lower.endswith(suffix):
            aliases.extend(mapped)
    return aliases


def gguf_tensor_row_bytes(tensor_type, cols):
    t = u32(tensor_type)
    cols = int(cols or 0)
    if cols <= 0:
        return 0
    block32 = (cols + 31) // 32
    block256 = (cols + 255) // 256
    if t == 0:
        return cols * 4
    if t in {1, 30}:
        return cols * 2
    if t == 2:
        return block32 * 18
    if t == 3:
        return block32 * 20
    if t == 6:
        return block32 * 22
    if t == 7:
        return block32 * 24
    if t == 8:
        return block32 * 34
    if t == 9:
        return block32 * 36
    if t == 10:
        return block256 * 84
    if t == 11:
        return block256 * 110
    if t == 12:
        return block256 * 144
    if t == 13:
        return block256 * 176
    if t == 14:
        return block256 * 210
    if t == 15:
        return block256 * 292
    if t == 16:
        return block256 * 66
    if t == 17:
        return block256 * 74
    if t == 18:
        return block256 * 98
    if t == 19:
        return block256 * 50
    if t == 20:
        return block32 * 18
    if t == 21:
        return block256 * 110
    if t == 22:
        return block256 * 82
    if t == 23:
        return block256 * 136
    if t == 24:
        return cols
    if t == 25:
        return cols * 2
    if t == 26:
        return cols * 4
    if t == 27:
        return cols * 8
    if t == 28:
        return cols * 8
    if t == 29:
        return block256 * 56
    if t in {31, 32, 33}:
        return block32 * 18
    if t == 34:
        return block256 * 54
    if t == 35:
        return block256 * 66
    if t in {36, 37, 38}:
        return 0
    if t == 39:
        return block32 * 17
    return 0


def gguf_tensor_exact_bytes(tensor_type, shape):
    shape = [int(v or 0) for v in (shape or [])]
    if not shape or shape[0] <= 0:
        return 0
    cols = shape[0]
    rows = 1
    for dim in shape[1:]:
        rows *= max(0, dim)
    row_bytes = gguf_tensor_row_bytes(tensor_type, cols)
    return row_bytes * rows if row_bytes and rows > 0 else 0


def gguf_tensor_byte_diagnostics(tensors, limit=32):
    mismatches = []
    type_stats = {}
    for tensor in tensors or []:
        t = u32(tensor.get("type"))
        key = str(t)
        exact = int(tensor.get("exact_bytes") or 0)
        storage = int(tensor.get("source_storage_bytes") or 0)
        emitted = int(tensor.get("bytes") or 0)
        padding = int(tensor.get("source_padding_bytes") or 0)
        stats = type_stats.setdefault(key, {
            "gguf_type": t,
            "gguf_type_name": gguf_type_name(t),
            "count": 0,
            "exact_bytes": 0,
            "storage_bytes": 0,
            "emitted_bytes": 0,
            "padding_bytes": 0,
            "unknown_exact_count": 0,
        })
        stats["count"] += 1
        stats["exact_bytes"] += exact
        stats["storage_bytes"] += storage
        stats["emitted_bytes"] += emitted
        stats["padding_bytes"] += padding
        if not exact:
            stats["unknown_exact_count"] += 1
        if storage and (
            not exact or
            exact > storage or
            emitted != exact or
            exact * 100 < storage * 95 or
            exact * 100 > storage * 105
        ):
            mismatches.append({
                "name": tensor.get("name", ""),
                "gguf_type": t,
                "gguf_type_name": gguf_type_name(t),
                "shape": tensor.get("shape", []),
                "exact_bytes": exact,
                "source_storage_bytes": storage,
                "emitted_bytes": emitted,
                "source_padding_bytes": padding,
                "exact_to_storage": (exact / storage) if exact and storage else 0,
            })
    return {
        "tensor_count": len(tensors or []),
        "mismatch_count": len(mismatches),
        "unknown_exact_count": sum(stat["unknown_exact_count"] for stat in type_stats.values()),
        "mismatches": mismatches[:int(limit or 32)],
        "type_stats": [
            type_stats[key] for key in sorted(type_stats, key=lambda value: int(value))
        ],
    }


def print_gguf_byte_diagnostics(directory, label=""):
    if os.environ.get("JUJU_PRINT_GGUF_BYTE_DIAGNOSTICS", "1") == "0":
        return
    diag = (directory or {}).get("byte_diagnostics") or {}
    prefix = f"[GGUF bytes:{label}]" if label else "[GGUF bytes]"
    print(
        f"{prefix} tensors={diag.get('tensor_count', 0)} "
        f"mismatch={diag.get('mismatch_count', 0)} "
        f"unknown_exact={diag.get('unknown_exact_count', 0)}"
    )
    for stat in diag.get("type_stats", []):
        print(
            f"{prefix} type={stat.get('gguf_type')}({stat.get('gguf_type_name')}) "
            f"count={stat.get('count')} exact={stat.get('exact_bytes')} "
            f"storage={stat.get('storage_bytes')} emitted={stat.get('emitted_bytes')} "
            f"padding={stat.get('padding_bytes')} unknown={stat.get('unknown_exact_count')}"
        )
    for item in diag.get("mismatches", [])[:16]:
        print(
            f"{prefix} mismatch name={item.get('name')} "
            f"type={item.get('gguf_type')}({item.get('gguf_type_name')}) "
            f"exact={item.get('exact_bytes')} storage={item.get('source_storage_bytes')} "
            f"emitted={item.get('emitted_bytes')} padding={item.get('source_padding_bytes')}"
        )


def _hex_preview(data, limit=256):
    data = bytes(data or b"")[:int(limit or 256)]
    return " ".join(f"{byte:02x}" for byte in data)


def print_gguf_tensor_layout_probes(session, url, directory, token=None, label="", probe_types=None):
    if os.environ.get("JUJU_PRINT_GGUF_LAYOUT_PROBES", "1") == "0":
        return
    probe_types = set(probe_types or (15, 22, 29, 36, 37, 38, 39))
    tensors = list((directory or {}).get("tensors") or [])
    present_types = sorted({u32(tensor.get("type")) for tensor in tensors})
    prefix = f"[GGUF probe:{label}]" if label else "[GGUF probe]"
    print(f"{prefix} present_types={present_types}")
    selected = []
    seen = {}
    for tensor in tensors:
        t = u32(tensor.get("type"))
        if t not in probe_types and int(tensor.get("exact_bytes") or 0) > 0:
            continue
        if seen.get(t, 0) >= 1:
            continue
        seen[t] = seen.get(t, 0) + 1
        selected.append(tensor)
    for tensor in selected:
        t = u32(tensor.get("type"))
        size = min(256, int(tensor.get("source_storage_bytes") or tensor.get("bytes") or 0))
        if size <= 0:
            preview = ""
        else:
            resp = fetch_range(session, url, int(tensor["source_offset"]), int(tensor["source_offset"]) + size - 1, token=token, stream=False)
            try:
                preview = _hex_preview(resp.content, limit=size)
            finally:
                resp.close()
        print(
            f"{prefix} tensor={tensor.get('name')} "
            f"type={t}({gguf_type_name(t)}) shape={tensor.get('shape')} "
            f"source_offset={tensor.get('source_offset')} exact={tensor.get('exact_bytes')} "
            f"storage={tensor.get('source_storage_bytes')} emitted={tensor.get('bytes')} "
            f"padding={tensor.get('source_padding_bytes')} first{size}={preview}"
        )


def validate_gguf_byte_diagnostics(directory):
    diag = (directory or {}).get("byte_diagnostics") or {}
    fatal = list(diag.get("fatal_errors") or [])
    if fatal:
        raise RuntimeError(
            "GGUF tensor byte layout has unsupported or inconsistent entries; "
            f"first={json.dumps(fatal[0], ensure_ascii=False)} "
            f"fatal_count={diag.get('fatal_error_count', len(fatal))}"
        )


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
    gguf_kv = {}
    gguf_kv_aliases = {}
    for _ in range(kv_count):
        key = read_string(handle)
        value_type = read_u32(handle)
        if key == "general.alignment" and value_type == GGUF_TYPE_UINT32:
            value = read_u32(handle)
            alignment = value
            gguf_kv[key] = value
            gguf_kv_aliases["alignment"] = value
        elif should_capture_gguf_runtime_kv(key):
            value = read_gguf_array_value(handle) if value_type == GGUF_TYPE_ARRAY else read_gguf_scalar_value(handle, value_type)
            if value is None:
                if value_type != GGUF_TYPE_ARRAY:
                    skip_value(handle, value_type)
                continue
            gguf_kv[key] = value
            gguf_kv_aliases[key] = value
            for alias in gguf_runtime_aliases_for_key(key):
                gguf_kv_aliases.setdefault(alias, value)
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
    byte_fatal_errors = []
    for pos, idx in enumerate(order):
        is_last_tensor = pos == len(order) - 1
        cur = tensors[idx]["relative_offset"]
        nxt = tensors[order[pos + 1]]["relative_offset"] if pos + 1 < len(order) else total_bytes - data_start
        storage_bytes = max(0, nxt - cur)
        exact_bytes = gguf_tensor_exact_bytes(tensors[idx]["type"], tensors[idx]["shape"])
        if storage_bytes and not exact_bytes:
            byte_fatal_errors.append({
                "reason": "unsupported_gguf_tensor_byte_size",
                "name": tensors[idx]["name"],
                "type": tensors[idx]["type"],
                "type_name": gguf_type_name(tensors[idx]["type"]),
                "shape": tensors[idx]["shape"],
                "storage": storage_bytes,
            })
        if storage_bytes and exact_bytes > storage_bytes:
            status = "unknown" if not exact_bytes else "larger_than_storage"
            byte_fatal_errors.append({
                "reason": "inconsistent_gguf_tensor_byte_size",
                "name": tensors[idx]["name"],
                "type": tensors[idx]["type"],
                "type_name": gguf_type_name(tensors[idx]["type"]),
                "shape": tensors[idx]["shape"],
                "exact": exact_bytes,
                "storage": storage_bytes,
                "status": status,
            })
        tensors[idx]["source_offset"] = data_start + cur
        tensors[idx]["source_storage_bytes"] = storage_bytes
        tensors[idx]["exact_bytes"] = exact_bytes
        tensors[idx]["bytes"] = exact_bytes if exact_bytes and exact_bytes <= storage_bytes else 0
        tensors[idx]["source_padding_bytes"] = max(0, storage_bytes - tensors[idx]["bytes"])
        if STRICT_GGUF_EXACT_BYTES and tensors[idx]["bytes"] and tensors[idx]["source_padding_bytes"] >= alignment and not is_last_tensor:
            byte_fatal_errors.append({
                "reason": "impossible_alignment_padding",
                "name": tensors[idx]["name"],
                "type": tensors[idx]["type"],
                "type_name": gguf_type_name(tensors[idx]["type"]),
                "shape": tensors[idx]["shape"],
                "exact": exact_bytes,
                "storage": storage_bytes,
                "padding": tensors[idx]["source_padding_bytes"],
                "alignment": alignment,
            })
        tensors[idx]["bucket"] = tensor_bucket(tensors[idx]["name"])
    byte_diagnostics = gguf_tensor_byte_diagnostics(tensors)
    byte_diagnostics["fatal_error_count"] = len(byte_fatal_errors)
    byte_diagnostics["fatal_errors"] = byte_fatal_errors[:32]
    return {
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "alignment": alignment,
        "data_start": data_start,
        "gguf_kv": gguf_kv,
        "gguf_runtime": gguf_kv_aliases,
        "gguf_kv_floats": {
            k: v for k, v in gguf_kv_aliases.items()
            if isinstance(v, float)
        },
        "byte_diagnostics": byte_diagnostics,
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


def juju_aligned_tensor_bytes(tensor):
    return align_up(int(tensor.get("bytes") or 0), 4096)


def juju_groups_fit_upload_limits(groups, payload_limit_first, payload_limit_sub):
    for idx, group in enumerate(groups):
        limit = payload_limit_first if idx == 0 else payload_limit_sub
        if sum(juju_aligned_tensor_bytes(tensor) for tensor in group) > limit:
            return False
    return True


def balance_juju_tensor_groups(tensors, split_count):
    tensors = list(tensors)
    split_count = min(max(1, int(split_count)), len(tensors))
    if split_count <= 1:
        return [tensors]
    total = sum(juju_aligned_tensor_bytes(tensor) for tensor in tensors)
    target = max(1, (total + split_count - 1) // split_count)
    groups = []
    current = []
    current_bytes = 0
    remaining_groups = split_count
    for idx, tensor in enumerate(tensors):
        tensor_bytes = juju_aligned_tensor_bytes(tensor)
        remaining_tensors = len(tensors) - idx
        if (
            current
            and len(groups) < split_count - 1
            and current_bytes + tensor_bytes > target
            and remaining_tensors >= remaining_groups
        ):
            groups.append(current)
            current = []
            current_bytes = 0
            remaining_groups -= 1
        current.append(tensor)
        current_bytes += tensor_bytes
    if current:
        groups.append(current)

    while len(groups) < split_count:
        split_at = max(range(len(groups)), key=lambda i: len(groups[i]))
        group = groups[split_at]
        if len(group) <= 1:
            break
        half_bytes = sum(juju_aligned_tensor_bytes(tensor) for tensor in group) // 2
        running = 0
        cut = 1
        for idx, tensor in enumerate(group[:-1], start=1):
            running += juju_aligned_tensor_bytes(tensor)
            cut = idx
            if running >= half_bytes:
                break
        groups[split_at:split_at + 1] = [group[:cut], group[cut:]]
    return groups


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
        aligned_tensor_bytes = juju_aligned_tensor_bytes(tensor)
        current_limit = payload_limit_first if len(groups) == 0 else payload_limit_sub
        if aligned_tensor_bytes > current_limit:
            raise RuntimeError(
                f"single tensor exceeds upload-safe JUJU split limit: {tensor['name']} "
                f"bytes={tensor_bytes} limit={current_limit}"
            )
        if current and current_bytes + aligned_tensor_bytes > current_limit:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(tensor)
        current_bytes += aligned_tensor_bytes
    if current:
        groups.append(current)

    split_strategy = "limit_tensor_groups"
    target_split_count = juju_target_tensor_splits()
    if target_split_count > 1 and len(groups) > 1:
        balanced_groups = balance_juju_tensor_groups(tensors, max(target_split_count, len(groups)))
        if juju_groups_fit_upload_limits(balanced_groups, payload_limit_first, payload_limit_sub):
            groups = balanced_groups
            split_strategy = "balanced_tensor_groups"

    split_count = len(groups)
    planned = []
    for idx, group in enumerate(groups, start=1):
        planned.append({
            "enabled": split_count > 1,
            "split_index": idx,
            "split_count": split_count,
            "split_strategy": split_strategy,
            "target_split_count": target_split_count,
            "tensor_names": [tensor["name"] for tensor in group],
            "tensor_bytes": sum(int(tensor["bytes"]) for tensor in group),
            "estimated_file_bytes": juju_estimated_tensor_payload_bytes(group, is_first_shard=(idx == 1)),
            "max_file_bytes": limit,
        })
    return planned


def tensor_bucket(name):
    lower = str(name).lower()
    if any(k in lower for k in (
        "mm_projector",
        "multi_modal_projector",
        "vision_projector",
        "image_projector",
    )):
        return "vision_projector"
    if any(k in lower for k in (
        "vision_model.",
        "vit.",
        "visual_encoder.",
        "image_encoder.",
        "moonvit.",
        "siglip.",
    )):
        return "vision_encoder"
    if any(k in lower for k in (
        "audio_model.",
        "whisper.",
        "audio_encoder.",
    )):
        return "audio_encoder"
    if any(k in lower for k in (
        "video_model.",
        "video_encoder.",
        "temporal_encoder.",
        "timesformer.",
    )):
        return "video_encoder"
    if any(k in lower for k in (
        "document_encoder.",
        "pdf_encoder.",
        "ocr_encoder.",
    )):
        return "document_encoder"
    if is_shared_expert_tensor_name(lower):
        return "shared_weights"
    if "attn" in lower or "attention" in lower:
        return "shared_weights"
    if is_routed_expert_tensor_name(lower):
        return "cold_experts"
    return "shared_weights"


def is_shared_expert_tensor_name(name):
    text = str(name or "").lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not normalized:
        return False
    if re.search(r"(?:^|_)shared_(?:expert|experts|exps)(?:_|$)", normalized):
        return True
    if re.search(r"(?:^|_)(?:expert|experts|exps)_shared(?:_|$)", normalized):
        return True
    return False


def is_routed_expert_tensor_name(name):
    text = str(name or "").lower()
    if not text or is_shared_expert_tensor_name(text):
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if "_exps" in normalized:
        return True
    if re.search(r"(?:^|_)(?:expert|experts)(?:_|$)", normalized):
        return True
    return False


def section_type_for_bucket(bucket):
    if bucket == "shared_weights":
        return JUJU_SECTION_SHARED_WEIGHTS
    if bucket == "hot_experts":
        return JUJU_SECTION_HOT_EXPERTS
    if bucket == "warm_experts":
        return JUJU_SECTION_WARM_EXPERTS
    if bucket == "vision_encoder":
        return JUJU_SECTION_VISION_ENCODER
    if bucket == "vision_projector":
        return JUJU_SECTION_VISION_PROJ
    if bucket == "audio_encoder":
        return JUJU_SECTION_AUDIO_ENCODER
    if bucket == "video_encoder":
        return JUJU_SECTION_VIDEO_ENCODER
    if bucket == "document_encoder":
        return JUJU_SECTION_DOCUMENT_ENCODER
    return JUJU_SECTION_COLD_EXPERTS


def _juju_json_dict(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def juju_metadata_files_from_contract(contract):
    files = {}
    for key in (
        "hf_metadata_files",
        "runtime_metadata_files",
        "runtime_assets",
        "runtime_asset_files",
        "tokenizer_assets",
    ):
        raw = contract.get(key)
        if isinstance(raw, dict):
            for name, value in raw.items():
                files[str(name)] = value
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    name = item.get("path") or item.get("name") or item.get("file")
                    if name:
                        files[str(name)] = item.get("content", item.get("json", item.get("raw", item)))
                elif isinstance(item, str):
                    files[item] = {}
    return files


def detect_vision_config(hf_metadata_files):
    cfg = {}
    raw = (
        hf_metadata_files.get("image_processor_config.json") or
        hf_metadata_files.get("preprocessor_config.json") or
        hf_metadata_files.get("processor_config.json") or
        hf_metadata_files.get("tokenizer/image_processor_config.json")
    )
    data = _juju_json_dict(raw)
    if data:
        cfg["image_token_id"] = u32(data.get("image_token_id") or data.get("image_token_index"))
        cfg["patch_size"] = u32(data.get("patch_size") or data.get("vision_patch_size") or 14)
        cfg["encoder_hidden_dim"] = u32(data.get("hidden_size") or data.get("encoder_hidden_dim"))
    return cfg


def juju_modality_flags_from_buckets(buckets, hf_metadata_files=None):
    flags = JUJU_MODALITY_TEXT
    bucket_set = set(buckets or [])
    if bucket_set.intersection({"vision_encoder", "vision_projector"}):
        flags |= JUJU_MODALITY_IMAGE
    if "audio_encoder" in bucket_set:
        flags |= JUJU_MODALITY_AUDIO
    if "video_encoder" in bucket_set:
        flags |= JUJU_MODALITY_VIDEO
    if "document_encoder" in bucket_set:
        flags |= JUJU_MODALITY_DOCUMENT
    for name in (hf_metadata_files or {}).keys():
        lower = str(name).lower()
        if "image_processor_config.json" in lower or "preprocessor_config.json" in lower or "processor_config.json" in lower:
            flags |= JUJU_MODALITY_IMAGE
        if "audio_config.json" in lower:
            flags |= JUJU_MODALITY_AUDIO
        if "video_preprocessor_config.json" in lower or "video_config.json" in lower:
            flags |= JUJU_MODALITY_VIDEO
        if "document" in lower or "pdf" in lower or "ocr" in lower:
            flags |= JUJU_MODALITY_DOCUMENT
    return flags


def juju_modality_metadata(contract, tensors):
    metadata_files = juju_metadata_files_from_contract(contract)
    buckets = [t.get("bucket", "") for t in tensors or []]
    flags = juju_modality_flags_from_buckets(buckets, metadata_files)
    return {
        "modality_flags": flags,
        "modalities": {
            "text": bool(flags & JUJU_MODALITY_TEXT),
            "image": bool(flags & JUJU_MODALITY_IMAGE),
            "audio": bool(flags & JUJU_MODALITY_AUDIO),
            "video": bool(flags & JUJU_MODALITY_VIDEO),
            "document": bool(flags & JUJU_MODALITY_DOCUMENT),
        },
        "vision_config": detect_vision_config(metadata_files),
        "section_types": {
            "vision_encoder": JUJU_SECTION_VISION_ENCODER,
            "vision_projector": JUJU_SECTION_VISION_PROJ,
            "audio_encoder": JUJU_SECTION_AUDIO_ENCODER,
            "video_encoder": JUJU_SECTION_VIDEO_ENCODER,
            "document_encoder": JUJU_SECTION_DOCUMENT_ENCODER,
        },
        "section_policy": "write_only_nonempty_modality_sections",
    }


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
    if len(payload) > JUJU_SECTION_ENTRY_BYTES:
        raise ValueError(f"JUJU section entry is too large: {len(payload)} > {JUJU_SECTION_ENTRY_BYTES}")
    return payload + (b"\x00" * (JUJU_SECTION_ENTRY_BYTES - len(payload)))


def write_padding(out, alignment=4096, digest=None):
    pad = align_up(out.tell(), alignment) - out.tell()
    if pad:
        data = b"\x00" * pad
        out.write(data)
        if digest is not None:
            digest.update(data)


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


def sha256_juju_section_ranges(session, url, section_offset, section_size, ranges, token=None, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    cursor = int(section_offset)
    section_end = int(section_offset) + int(section_size)
    for item in sorted(ranges or [], key=lambda value: int(value["offset"])):
        item_offset = int(item["offset"])
        if item_offset > cursor:
            digest.update(b"\x00" * (item_offset - cursor))
            cursor = item_offset
        start = int(item["source_offset"])
        size = int(item["size"])
        if size <= 0:
            continue
        remaining = size
        pos = start
        while remaining > 0:
            take = min(int(chunk_size), remaining)
            resp = fetch_range(session, url, pos, pos + take - 1, token=token, stream=False)
            try:
                data = resp.content
            finally:
                resp.close()
            if len(data) != take:
                raise EOFError(f"short checksum range read: expected {take}, got {len(data)}")
            digest.update(data)
            pos += take
            remaining -= take
        cursor += size
    if section_end > cursor:
        digest.update(b"\x00" * (section_end - cursor))
    return digest.hexdigest()


def u32(value):
    try:
        if value is None:
            return 0
        return max(0, min(int(value), 0xFFFFFFFF))
    except Exception:
        return 0


def contract_value(contract, *keys, default=None):
    for key in keys:
        cur = contract
        ok = True
        for part in str(key).split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default


def mb_from_bytes(value):
    try:
        n = int(value or 0)
    except Exception:
        return 0
    if n <= 0:
        return 0
    return max(1, n // (1024 * 1024))


def juju_arch_type(contract, source_name):
    text = " ".join(
        str(x or "")
        for x in (
            contract.get("architecture"),
            contract_value(contract, "arch_meta.architecture"),
            contract.get("model_id"),
            contract.get("model_name"),
            source_name,
        )
    ).lower()
    if "glm" in text:
        return 1
    if "kimi" in text or "moonshot" in text:
        return 6
    if "gemma" in text:
        return 2
    if "qwen" in text:
        return 3
    if "llama" in text:
        return 4
    if "mistral" in text:
        return 5
    return 0


def juju_weight_bits(contract):
    return u32(contract_value(
        contract,
        "source_weight_bits",
        "weight_bits",
        "weight_quant_schema.bits",
        "weight_quant_schema.weight_bits",
        default=0,
    ))


def juju_weight_encoding(contract):
    explicit = contract_value(contract, "source_weight_encoding", "weight_encoding", "weight_quant_schema.encoding", default=0)
    if explicit:
        return u32(explicit)
    family = str(contract_value(
        contract,
        "source_weight_quant_family",
        "weight_quant_family",
        "weight_quant_schema.family",
        default="",
    ) or "").lower()
    if "iq2_xxs" in family:
        return 19
    if "iq3_xxs" in family:
        return 20
    if "iq1_s" in family:
        return 32
    if "iq1_m" in family:
        return 33
    if "iq2_xs" in family:
        return 29
    if "iq2_s" in family:
        return 30
    if "iq3_s" in family:
        return 31
    if "iq4_nl" in family:
        return 27
    if "iq4_xs" in family:
        return 28
    if "bf16" in family or "bfloat16" in family:
        return 21
    if "q5_0" in family:
        return 24
    if "q5_1" in family:
        return 12
    if "q4_0" in family:
        return 22
    if "q4_1" in family:
        return 23
    if "q8_1" in family:
        return 25
    if "q8_0" in family:
        return 13
    if "iq2" in family or "ud-iq2" in family:
        return 9
    if "iq3" in family:
        return 10
    if "iq4" in family:
        return 11
    if "mxfp4" in family:
        return 4
    if "tq1_0" in family or "ternary_tq1" in family:
        return 35
    if "tq2_0" in family or "ternary_tq2" in family:
        return 36
    if "nvfp4" in family or "fp4" in family:
        return 3
    if "q8" in family:
        return 8
    if "q4" in family:
        return 7
    if "q3" in family:
        return 6
    if "q2" in family:
        return 5
    return 0


def weight_encoding_from_gguf_type(tensor_type, contract=None):
    t = u32(tensor_type)
    mapping = {
        0: 2,
        1: 1,
        2: 22,
        3: 23,
        6: 24,
        7: 12,
        8: 13,
        9: 25,
        10: 15,
        11: 16,
        12: 17,
        13: 14,
        14: 18,
        15: 34,
        16: 19,
        18: 20,
        19: 32,
        17: 29,
        22: 30,
        29: 33,
        21: 31,
        20: 27,
        23: 28,
        30: 21,
        31: 22,
        32: 22,
        33: 22,
        34: 35,
        35: 36,
        39: 4,
    }
    enc = mapping.get(t, 0)
    if enc:
        return enc
    return juju_weight_encoding(contract or {})


def gguf_type_name(tensor_type):
    names = {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        6: "Q5_0",
        7: "Q5_1",
        8: "Q8_0",
        9: "Q8_1",
        10: "Q2_K",
        11: "Q3_K",
        12: "Q4_K",
        13: "Q5_K",
        14: "Q6_K",
        15: "Q8_K",
        16: "IQ2_XXS",
        17: "IQ2_XS",
        18: "IQ3_XXS",
        19: "IQ1_S",
        20: "IQ4_NL",
        21: "IQ3_S",
        22: "IQ2_S",
        23: "IQ4_XS",
        24: "I8",
        25: "I16",
        26: "I32",
        27: "I64",
        28: "F64",
        29: "IQ1_M",
        30: "BF16",
        31: "Q4_0_4_4",
        32: "Q4_0_4_8",
        33: "Q4_0_8_8",
        34: "TQ1_0",
        35: "TQ2_0",
        36: "REMOVED_IQ4_NL_4_4",
        37: "REMOVED_IQ4_NL_4_8",
        38: "REMOVED_IQ4_NL_8_8",
        39: "MXFP4",
    }
    return names.get(u32(tensor_type), f"GGUF_TYPE_{u32(tensor_type)}")


def quant_family_from_gguf_type(tensor_type, contract=None):
    t = u32(tensor_type)
    if t in {0, 1, 24, 25, 26, 27, 28, 30}:
        return "raw_scalar_or_integer"
    if t in {2, 3, 6, 7, 8, 9}:
        return "legacy_ggml_quant"
    if t in {31, 32, 33}:
        return "legacy_ggml_interleaved_quant"
    if t in {10, 11, 12, 13, 14, 15}:
        return "k_quant"
    if t in {16, 17, 18, 19, 20, 21, 22, 23, 29}:
        return "importance_quant"
    if t in {34, 35}:
        return "ternary_quant"
    if t == 39:
        return "mxfp4"
    explicit = contract_value(contract or {}, "source_weight_quant_family", "weight_quant_family", "weight_quant_schema.family", default="")
    if explicit:
        return str(explicit)
    return "unknown_preserved_source_type"


def kernel_key_from_gguf_type(tensor_type, contract=None):
    return f"{quant_family_from_gguf_type(tensor_type, contract)}:{gguf_type_name(tensor_type)}"


def juju_qkv_policy(contract):
    if contract.get("qkv_cache_schema") or contract.get("qkv_policy_contract"):
        return 1
    if contract_value(contract, "qkv_packed_cache_required", default=False):
        return 1
    return 0


def juju_format_extension_contract(contract):
    return {
        "contract_version": JUJU_FORMAT_CONTRACT_VERSION,
        "binary_wire_id": JUJU_BINARY_WIRE_ID,
        "binary_wire_frozen": True,
        "header_bytes": JUJU_HEADER_BYTES,
        "section_entry_bytes": JUJU_SECTION_ENTRY_BYTES,
        "section_table_reserved_entries": JUJU_SECTION_TABLE_RESERVED_ENTRIES,
        "section_table_offset": JUJU_HEADER_BYTES,
        "offset_unit": "absolute_file_byte_offset",
        "length_unit": "exact_payload_byte_length",
        "alignment_bytes": 4096,
        "endianness": "little",
        "tensor_payload_layout": "source_bytes_preserved_without_requantization",
        "json_sections_are_extension_surface": True,
        "additive_json_fields_allowed": True,
        "unknown_json_field_policy": "engine_ignore_if_not_required",
        "unknown_required_feature_policy": "fail_closed",
        "engine_update_without_repack": [
            "new_cpu_quant_kernel",
            "new_gpu_quant_kernel",
            "new_attention_kernel",
            "new_qkv_cache_backend",
            "new_prefetch_scheduler",
            "new_residency_policy",
            "new_graph_ir_executor",
            "new_adapter_runtime",
            "new_tokenizer_loader_policy",
            "new_sampler",
            "new_validation_probe",
            "new_multimodal_executor",
        ],
        "repack_required_only_for": [
            "model_weights_changed",
            "tensor_payload_bytes_changed",
            "tensor_order_or_offsets_changed",
            "new_required_tokenizer_asset_contents",
            "new_section_compression_requiring_reencoded_payload",
            "file_checksum_or_payload_corruption",
        ],
        "reserved_extension_namespaces": [
            "MODEL_META.format_extension_contract",
            "MODEL_META.kernel_registry_contract",
            "MODEL_META.adapter_registry_contract",
            "MODEL_META.validation_contract",
            "TENSOR_INDEX.tensors[].extension",
            "TENSOR_INDEX.tensors[].kernel_contract",
            "GRAPH_IR.runtime_policy",
            "GRAPH_IR.execution_plan",
            "GRAPH_IR.priority_tables",
            "GRAPH_IR.performance_research_slots",
            "MODEL_META.multimodal_contract",
            "MODEL_META.modality_flags",
        ],
        "reserved_section_types": {
            "vision_encoder": JUJU_SECTION_VISION_ENCODER,
            "vision_projector": JUJU_SECTION_VISION_PROJ,
            "audio_encoder": JUJU_SECTION_AUDIO_ENCODER,
            "video_encoder": JUJU_SECTION_VIDEO_ENCODER,
            "document_encoder": JUJU_SECTION_DOCUMENT_ENCODER,
        },
        "modality_flags": {
            "text": JUJU_MODALITY_TEXT,
            "image": JUJU_MODALITY_IMAGE,
            "audio": JUJU_MODALITY_AUDIO,
            "video": JUJU_MODALITY_VIDEO,
            "document": JUJU_MODALITY_DOCUMENT,
        },
        "compatibility_rule": "binary_header_and_section_table_remain_stable; add new behavior through JSON sections and engine code",
    }


def juju_kernel_registry_contract(contract):
    return {
        "selection_key_order": ["weight_encoding", "gguf_type", "gguf_type_name", "quant_family", "kernel_key"],
        "required_behavior": "engine_must_execute_or_fail_closed_never_silent_zero",
        "source_type_preserved": True,
        "supported_source_families_declared": [
            "raw_fp32",
            "raw_fp16",
            "bf16",
            "legacy_q4_q5_q8",
            "k_quant_q2_q3_q4_q5_q6_q8",
            "iq1_iq2_iq3_iq4",
            "ternary_tq",
            "mxfp4",
            "vendor_dynamic_quant",
        ],
        "row_layout_rule": "preserve_source_quant_block_layout_until_kernel_decode",
        "mixed_quant_per_tensor_allowed": True,
        "per_tensor_weight_encoding_required": True,
        "per_tensor_source_type_required": True,
        "contract_weight_encoding": juju_weight_encoding(contract),
        "contract_weight_bits": juju_weight_bits(contract),
    }


def juju_tokenizer_contract():
    return {
        "tokenizer_files": list(JUJU_TOKENIZER_FILES),
        "required_files": list(JUJU_REQUIRED_TOKENIZER_FILES),
        "required_any_of": list(JUJU_REQUIRED_TOKENIZER_ANY_OF),
        "target_subdirs": ["", "tokenizer"],
        "chat_template_source": "tokenizer_config_or_model_card",
        "missing_tokenizer_behavior": "fail_text_api_if_required_tokenizer_missing",
        "input_ids_api_allowed_without_tokenizer": True,
    }


def juju_adapter_registry_contract():
    return {
        "adapter_metadata_slots_reserved": True,
        "supported_adapter_classes": [
            "lora",
            "qlora",
            "dora",
            "ia3",
            "prompt_tuning",
            "prefix_tuning",
            "runtime_delta_weight",
            "router_override",
            "expert_bias_or_scale",
        ],
        "storage_policy": "adapters_external_or_json_declared; base_tensor_payload_not_repacked",
        "merge_policy": "engine_runtime_merge_or_sidecar_cache",
        "compatibility_key_fields": ["target_tensor", "rank", "alpha", "dtype", "quant_compatibility"],
    }


def juju_validation_contract():
    return {
        "load_time_checks": [
            "magic",
            "header_size",
            "section_table_size",
            "section_offsets",
            "tensor_offsets",
            "tensor_lengths",
            "tensor_sha256_if_present",
            "tokenizer_required_any_of",
            "kernel_support_for_all_required_tensors",
        ],
        "correctness_checks": [
            "no_required_tensor_silent_zero",
            "dense_mlp_not_classified_as_expert_stream",
            "all_required_graph_ops_bound",
            "logits_finite",
            "ppl_probe_supported",
        ],
        "failure_policy": "fail_closed_with_actionable_error",
    }


def juju_research_offload_contract():
    return {
        "goal": "maximize_moe_offload_without_repacking_base_model",
        "phase_aware_execution": {
            "prefill": {
                "expected_pattern": "many_experts_active",
                "required_slots": [
                    "separate_prefill_scheduler",
                    "non_moe_compute_overlap_window",
                    "bounded_expert_residency",
                    "bulk_prefetch_stream",
                    "token_reordering_optional",
                ],
            },
            "decode": {
                "expected_pattern": "few_experts_active_per_token",
                "required_slots": [
                    "layer_level_expert_predictor",
                    "cross_layer_gate_predictor",
                    "activation_trace_predictor",
                    "semantic_prompt_hint_predictor",
                    "speculative_expert_prefetch",
                    "cache_hit_rate_feedback",
                ],
            },
        },
        "expert_cache_policy_inputs": [
            "layer_id",
            "expert_id",
            "token_position",
            "sequence_id",
            "router_topk",
            "router_score",
            "router_entropy",
            "local_routing_consistency",
            "previous_layer_experts",
            "previous_token_experts",
            "expert_hit_rate",
            "expert_load_latency_us",
            "expert_compute_latency_us",
            "pcie_bandwidth_bytes_per_s",
            "disk_bandwidth_bytes_per_s",
            "gpu_free_bytes",
            "cpu_free_bytes",
            "pinned_staging_bytes",
        ],
        "bottleneck_breaker_slots": {
            "critical_path_io": [
                "prefetch_before_router_consumer",
                "overlap_dma_with_attention_or_dense_compute",
                "two_stream_copy_compute_pipeline",
                "bounded_retry_queue",
                "io_priority_by_graph_role",
            ],
            "expert_cache_miss": [
                "proactive_cache",
                "activation_aware_cache",
                "fine_grained_expert_segments",
                "semantic_hint_cache_seed",
                "local_routing_consistency_score",
            ],
            "gpu_memory_pressure": [
                "hot_shared_residency",
                "expert_lru_or_score_eviction",
                "prefill_decode_different_budget",
                "qkv_page_eviction",
                "compressed_kv_cache",
            ],
            "token_scheduling": [
                "dynamic_token_ordering",
                "expert_batching",
                "router_entropy_adaptive_topk",
                "decode_microbatch_policy",
            ],
            "storage_path": [
                "mmap_tensor_spans",
                "direct_io_alignment",
                "pinned_cpu_stage",
                "async_read_ahead",
                "checksum_after_stream",
            ],
        },
        "research_method_slots": {
            "moe_infinity": ["sequence_level_activation_trace", "activation_aware_prefetch", "activation_aware_cache"],
            "promoe": ["proactive_expert_cache", "intermediate_result_prediction"],
            "fmoe": ["fine_grained_expert_offload", "expert_selection_patterns", "semantic_prompt_hints"],
            "duoserve_moe": ["prefill_decode_split", "dual_phase_expert_prefetch", "cache_scheduling"],
            "expertflow": ["adaptive_expert_scheduling", "memory_coordination", "dynamic_token_ordering"],
            "fate_cross_layer_gate": ["cross_layer_expert_prediction", "prediction_confidence"],
            "local_routing_consistency": ["routing_locality_metric", "offload_suitability_score"],
            "moe_speq": ["speculative_quantized_decode", "proactive_expert_prefetch"],
            "flexgen": ["gpu_cpu_disk_placement", "offload_policy_search", "weight_and_cache_compression"],
        },
        "kv_cache_research_slots": {
            "paged_attention": ["page_size_tokens", "block_table", "fragmentation_control"],
            "vattention": ["virtual_memory_backed_kv", "demand_paging_policy"],
            "infinigen": ["essential_kv_prefetch", "cpu_kv_pool", "counter_based_eviction"],
            "kivi": ["key_per_channel_quant", "value_per_token_quant", "residual_window"],
            "kvquant": ["sub_4bit_kv_quant", "outlier_aware_quant"],
            "turboquant": ["polarquant", "qjl_residual_correction", "online_vector_quantization"],
        },
        "metrics_required": [
            "expert_hit_rate",
            "expert_miss_latency_us",
            "tokens_per_second",
            "time_to_first_token_ms",
            "inter_token_latency_ms",
            "gpu_resident_expert_bytes",
            "cpu_resident_expert_bytes",
            "disk_read_bytes",
            "pcie_copy_bytes",
            "kv_cache_bytes",
            "prefetch_waste_ratio",
            "prediction_accuracy",
            "logits_finite_rate",
            "ppl_probe",
        ],
    }


def juju_contract_metadata(contract, source_name, source_repo_id):
    arch = dict(contract.get("arch_meta") or {})
    qkv = dict(contract.get("qkv_cache_schema") or {})
    model_id = contract.get("model_id") or contract.get("source_model_id") or source_repo_id
    model_name = contract.get("model_name") or model_id or Path(source_name).stem
    out = {
        "format_version": 1,
        "backend_neutral": True,
        "model_id": model_id,
        "model_name": model_name,
        "architecture": contract.get("architecture") or arch.get("architecture") or "",
        "source_weight_bits": juju_weight_bits(contract),
        "source_weight_encoding": juju_weight_encoding(contract),
        "source_weight_quant_family": contract.get("source_weight_quant_family") or contract.get("weight_quant_family") or contract_value(contract, "weight_quant_schema.family", default=""),
        "source_weight_kernel_family": contract.get("source_weight_kernel_family") or contract.get("weight_kernel_family") or contract_value(contract, "weight_quant_schema.kernel_family", default=""),
        "source_weight_block_size": u32(contract_value(contract, "source_weight_block_size", "weight_block_size", "weight_quant_schema.block_size", default=0)),
        "qkv_packed_cache_required": bool(qkv),
        "persistent_plain_kv_cache_allowed": not bool(qkv),
        "final_model_structure_contract": contract.get("final_model_structure_contract", {}),
        "pipeline_budget_contract": contract.get("pipeline_budget_contract", {}),
        "execution_path_contract": contract.get("execution_path_contract", {}),
        "expert_segmentation_contract": contract.get("expert_segmentation_contract", {}),
        "chunk_io_contract": contract.get("chunk_io_contract", {}),
        "universal_tier_contract": contract.get("universal_tier_contract", {}),
        "qkv_policy_contract": contract.get("qkv_policy_contract", qkv),
        "format_extension_contract": juju_format_extension_contract(contract),
        "kernel_registry_contract": juju_kernel_registry_contract(contract),
        "tokenizer_contract": juju_tokenizer_contract(),
        "adapter_registry_contract": juju_adapter_registry_contract(),
        "validation_contract": juju_validation_contract(),
        "research_offload_contract": juju_research_offload_contract(),
        "runtime_adapter_contract": {
            "weight_source": "juju_tensor_index",
            "offset_unit": "absolute_file_byte_offset",
            "row_layout": "source_quant_row_layout_preserved",
            "section_entry_bytes": JUJU_SECTION_ENTRY_BYTES,
            "section_table_reserved_entries": JUJU_SECTION_TABLE_RESERVED_ENTRIES,
            "header_bytes": JUJU_HEADER_BYTES,
            "alignment": 4096,
            "fail_closed": True,
        },
        "performance_contract": {
            "startup_prefetch_roles": ["shared_core", "router", "attention", "norm"],
            "streaming_roles": ["expert", "dense_ffn"],
            "direct_io_alignment": 4096,
            "mmap_friendly_sections": True,
            "split_large_uploads": True,
            "tokenizer_required_at_repo_root": True,
        },
    }
    for src, dst in (
        ("k_bits", "k_bits"),
        ("v_bits", "v_bits"),
        ("group_size", "group_size"),
        ("page_size_tokens", "page_size_tokens"),
        ("sink_tokens", "sink_tokens"),
        ("rotation_seed", "rotation_seed"),
        ("qjl_seed", "qjl_seed"),
        ("enable_qjl", "enable_qjl"),
        ("enable_rotation", "enable_rotation"),
    ):
        if src in qkv:
            out[dst] = qkv[src]
    return out


def make_header(contract, source_name, file_size_value, sections, section_sizes, index_checksum=0, modality_flags=JUJU_MODALITY_TEXT):
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
    struct.pack_into("<Q", header, 80, int(index_checksum or 0) & 0xFFFFFFFFFFFFFFFF)
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
    struct.pack_into("<I", header, 180, juju_weight_bits(contract))
    struct.pack_into("<I", header, 184, juju_arch_type(contract, source_name))
    struct.pack_into("<I", header, 188, u32(contract_value(contract, "segment_policy", "expert_segmentation_contract.segment_policy", default=2)))
    struct.pack_into("<I", header, 192, juju_qkv_policy(contract))
    struct.pack_into("<I", header, 196, u32(contract_value(contract, "preferred_segment_bytes", "chunk_io_contract.preferred_segment_bytes", default=4096)))
    struct.pack_into("<I", header, 200, u32(contract_value(contract, "max_segments_per_expert", "expert_segmentation_contract.max_segments_per_expert", default=8)))
    struct.pack_into("<I", header, 204, u32(contract_value(contract, "recommended_vram_mb", default=mb_from_bytes(contract_value(contract, "recommended_vram_bytes", "pipeline_budget_contract.recommended_vram_bytes", default=0)))))
    struct.pack_into("<I", header, 208, u32(contract_value(contract, "recommended_ram_mb", default=mb_from_bytes(contract_value(contract, "recommended_ram_bytes", "pipeline_budget_contract.recommended_ram_bytes", default=0)))))
    struct.pack_into("<I", header, 212, u32(modality_flags))
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


JUJU_LAYER_NAME_PATTERNS = (
    re.compile(r"(?:^|[.])blk\.(\d+)\.(.+)$"),
    re.compile(r"(?:^|[.])blocks\.(\d+)\.(.+)$"),
    re.compile(r"(?:^|[.])layers\.(\d+)\.(.+)$"),
    re.compile(r"(?:^|[.])model\.layers\.(\d+)\.(.+)$"),
    re.compile(r"(?:^|[.])transformer\.h\.(\d+)\.(.+)$"),
    re.compile(r"(?:^|[.])h\.(\d+)\.(.+)$"),
)


def _juju_layer_match(name):
    text = str(name or "")
    for pattern in JUJU_LAYER_NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1)), match.group(2)
    return None, ""


def _juju_layer_id_from_name(name):
    layer, _ = _juju_layer_match(name)
    return layer


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


def _juju_tensors_by_layer(tensors, layer):
    out = []
    for tensor in tensors:
        name = str(tensor.get("name") or "")
        layer_id, _ = _juju_layer_match(name)
        if layer_id == int(layer):
            out.append(name)
    return out


def _juju_layer_suffix(name):
    _, suffix = _juju_layer_match(name)
    return suffix.lower()


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
    if bucket in {"vision_encoder", "vision_projector", "audio_encoder", "video_encoder", "document_encoder"}:
        role = bucket
        priority = 45
        prefetch = 20
        residency = "SLOW_MEM"
        prefetch_class = "stream"
    elif lower in {"token_embd.weight", "output.weight", "output_norm.weight", "rope_freqs.weight"}:
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
    elif is_shared_expert_tensor_name(lower):
        role = "shared_core"
        priority = 90
        prefetch = 90
        residency = "FAST_MEM"
        prefetch_class = "layer_hot"
    elif is_routed_expert_tensor_name(lower):
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
    if "kimi" in text or "moonshot" in text:
        return "kimi_moe"
    if "qwen" in text:
        return "qwen"
    if "llama" in text or "mistral" in text:
        return "llama"
    if "glm" in text:
        return "glm"
    return "generic_transformer"


def first_present(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def juju_runtime_arch_metadata(contract, directory=None):
    arch = dict(contract.get("arch_meta") or {})
    runtime = dict((directory or {}).get("gguf_runtime") or {})
    out = dict(runtime)

    fields = {
        "declared_architecture": first_present(contract.get("architecture"), arch.get("architecture"), runtime.get("declared_architecture"), runtime.get("architecture")),
        "model_id": first_present(contract.get("model_id"), contract.get("model_name"), runtime.get("model_id")),
        "model_name": first_present(contract.get("model_name"), runtime.get("model_name")),
        "num_hidden_layers": first_present(arch.get("n_layers"), arch.get("num_hidden_layers"), runtime.get("num_hidden_layers"), runtime.get("n_layers")),
        "hidden_size": first_present(arch.get("hidden_dim"), arch.get("hidden_size"), runtime.get("hidden_size"), runtime.get("hidden_dim")),
        "vocab_size": first_present(arch.get("vocab_size"), runtime.get("vocab_size")),
        "head_dim": first_present(arch.get("head_dim"), runtime.get("head_dim"), runtime.get("key_length")),
        "value_head_dim": first_present(arch.get("value_head_dim"), arch.get("v_head_dim"), runtime.get("value_head_dim"), runtime.get("v_head_dim")),
        "global_head_dim": first_present(arch.get("global_head_dim"), runtime.get("global_head_dim")),
        "num_attention_heads": first_present(arch.get("n_heads"), arch.get("num_attention_heads"), runtime.get("num_attention_heads"), runtime.get("n_heads")),
        "num_key_value_heads": first_present(arch.get("n_kv_heads"), arch.get("num_key_value_heads"), runtime.get("num_key_value_heads"), runtime.get("n_kv_heads")),
        "num_global_key_value_heads": first_present(arch.get("num_global_key_value_heads"), runtime.get("num_global_key_value_heads")),
        "kv_lora_rank": first_present(arch.get("kv_lora_rank"), runtime.get("kv_lora_rank")),
        "q_lora_rank": first_present(arch.get("q_lora_rank"), runtime.get("q_lora_rank")),
        "qk_nope_head_dim": first_present(arch.get("qk_nope_head_dim"), runtime.get("qk_nope_head_dim")),
        "qk_rope_head_dim": first_present(arch.get("qk_rope_head_dim"), runtime.get("qk_rope_head_dim")),
        "experts_per_moe_layer": first_present(arch.get("experts_per_moe_layer"), arch.get("n_experts"), runtime.get("experts_per_moe_layer"), runtime.get("n_experts")),
        "routed_experts_per_token": first_present(arch.get("routed_experts_per_token"), arch.get("top_k"), runtime.get("routed_experts_per_token"), runtime.get("top_k")),
        "expert_intermediate_size": first_present(arch.get("expert_intermediate_size"), arch.get("expert_intermediate_dim"), runtime.get("expert_intermediate_size"), runtime.get("expert_intermediate_dim")),
        "rms_norm_eps": first_present(arch.get("rms_norm_eps"), arch.get("norm_eps"), runtime.get("rms_norm_eps"), runtime.get("norm_eps")),
        "norm_eps": first_present(arch.get("norm_eps"), arch.get("rms_norm_eps"), runtime.get("norm_eps"), runtime.get("rms_norm_eps")),
        "rope_theta": first_present(arch.get("rope_theta"), runtime.get("rope_theta"), runtime.get("theta")),
        "theta": first_present(arch.get("rope_theta"), runtime.get("theta"), runtime.get("rope_theta")),
        "sliding_window": first_present(arch.get("sliding_window"), runtime.get("sliding_window")),
        "embedding_scale": first_present(arch.get("embedding_scale"), arch.get("scale_emb"), runtime.get("embedding_scale"), runtime.get("scale_emb")),
        "scale_emb": first_present(arch.get("scale_emb"), arch.get("embedding_scale"), runtime.get("scale_emb"), runtime.get("embedding_scale")),
        "final_logit_softcap": first_present(arch.get("final_logit_softcap"), arch.get("final_logit_softcapping"), runtime.get("final_logit_softcap"), runtime.get("final_logit_softcapping"), runtime.get("logit_softcap")),
        "final_logit_softcapping": first_present(arch.get("final_logit_softcapping"), arch.get("final_logit_softcap"), runtime.get("final_logit_softcapping"), runtime.get("final_logit_softcap"), runtime.get("logit_softcap")),
        "partial_rotary_factor": first_present(arch.get("partial_rotary_factor"), runtime.get("partial_rotary_factor")),
        "full_rope_theta": first_present(arch.get("full_rope_theta"), arch.get("full_attention_rope_theta"), runtime.get("full_rope_theta"), runtime.get("full_attention_rope_theta")),
        "sliding_rope_theta": first_present(arch.get("sliding_rope_theta"), arch.get("sliding_attention_rope_theta"), runtime.get("sliding_rope_theta"), runtime.get("sliding_attention_rope_theta")),
        "full_attention_interval": first_present(arch.get("full_attention_interval"), arch.get("global_attention_interval"), runtime.get("full_attention_interval"), runtime.get("global_attention_interval")),
        "global_attention_interval": first_present(arch.get("global_attention_interval"), arch.get("full_attention_interval"), runtime.get("global_attention_interval"), runtime.get("full_attention_interval")),
        "full_attention_offset": first_present(arch.get("full_attention_offset"), arch.get("global_attention_offset"), runtime.get("full_attention_offset"), runtime.get("global_attention_offset")),
        "global_attention_offset": first_present(arch.get("global_attention_offset"), arch.get("full_attention_offset"), runtime.get("global_attention_offset"), runtime.get("full_attention_offset")),
        "routed_scaling_factor": first_present(arch.get("routed_scaling_factor"), arch.get("route_scale"), runtime.get("routed_scaling_factor"), runtime.get("route_scale")),
        "norm_topk_prob": first_present(arch.get("norm_topk_prob"), arch.get("normalize_topk_prob"), runtime.get("norm_topk_prob"), runtime.get("normalize_topk_prob")),
        "scoring_func": first_present(arch.get("scoring_func"), arch.get("score_func"), runtime.get("scoring_func"), runtime.get("score_func")),
    }
    for key, value in fields.items():
        if value is not None:
            out[key] = value
    return out


def build_layer_graph_ir(layer, tensors):
    prefix = f"blk.{layer}."
    names = set(_juju_tensors_by_layer(tensors, layer))

    def bind(*suffixes):
        out = []
        wanted = {str(suffix or "").lower() for suffix in suffixes if suffix}
        for name in sorted(names):
            suffix = _juju_layer_suffix(name)
            if suffix in wanted:
                out.append(name)
        return out

    moe_weights = sorted(name for name in names if is_routed_expert_tensor_name(name))
    shared_expert_weights = sorted(name for name in names if is_shared_expert_tensor_name(name))
    dense_weights = bind("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")

    ops = [
        {"op": "rms_norm", "name": "attention_input_norm", "inputs": ["hidden"], "weights": bind("attn_norm.weight", "input_layernorm.weight"), "required": False},
        {"op": "linear", "name": "q_projection", "inputs": ["attention_norm"], "weights": bind("attn_q.weight", "attention.wq.weight", "attn_q_a_proj.weight", "attn_q_b_proj.weight"), "output": "q", "required": False},
        {"op": "linear", "name": "k_projection", "inputs": ["attention_norm"], "weights": bind("attn_k.weight", "attention.wk.weight", "attn_kv_a_proj_with_mqa.weight"), "output": "k", "required": False},
        {"op": "linear", "name": "v_projection", "inputs": ["attention_norm"], "weights": bind("attn_v.weight", "attention.wv.weight", "attn_kv_b_proj.weight"), "output": "v", "required": False},
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
        {"op": "moe_expert_mlp", "name": "moe_experts", "inputs": ["ffn_norm", "selected_experts"], "weights": moe_weights, "required": bool(moe_weights)},
        {"op": "shared_expert_mlp", "name": "shared_experts", "inputs": ["ffn_norm"], "weights": shared_expert_weights, "required": bool(shared_expert_weights)},
        {"op": "dense_mlp", "name": "dense_ffn_fallback", "inputs": ["ffn_norm"], "weights": dense_weights, "required": bool(dense_weights)},
        {"op": "residual", "name": "ffn_residual", "inputs": ["hidden", "ffn_out"], "required": True},
        {"op": "scale", "name": "layer_output_scale", "inputs": ["hidden"], "weights": bind("layer_output_scale.weight"), "required": False},
    ]
    return {
        "layer": int(layer),
        "tensor_prefix": prefix,
        "layer_name_parser": "common_gguf_layer_prefixes",
        "available_tensors": sorted(names),
        "ops": ops,
    }


def juju_expert_tensor_diagnostics(tensor_records):
    diagnostics = {
        "routed_expert_tensor_count": 0,
        "shared_expert_tensor_count": 0,
        "routed_expert_layers": [],
        "shared_expert_layers": [],
        "format_errors": [],
    }
    routed_layers = set()
    shared_layers = set()
    for rec in tensor_records or []:
        name = str(rec.get("name") or "")
        bucket = str(rec.get("bucket") or "")
        role = str(rec.get("graph_role") or "")
        prefetch_class = str(rec.get("prefetch_class") or "")
        layer = _juju_layer_id_from_name(name)
        is_shared = is_shared_expert_tensor_name(name)
        is_routed = is_routed_expert_tensor_name(name)
        if is_shared:
            diagnostics["shared_expert_tensor_count"] += 1
            if layer is not None:
                shared_layers.add(int(layer))
            if bucket != "shared_weights":
                diagnostics["format_errors"].append({
                    "name": name,
                    "error": "shared_expert_not_in_shared_weights",
                    "bucket": bucket,
                })
            if role == "expert" or prefetch_class == "expert_stream":
                diagnostics["format_errors"].append({
                    "name": name,
                    "error": "shared_expert_marked_as_routed_expert",
                    "graph_role": role,
                    "prefetch_class": prefetch_class,
                })
        if is_routed:
            diagnostics["routed_expert_tensor_count"] += 1
            if layer is not None:
                routed_layers.add(int(layer))
            if bucket not in {"hot_experts", "warm_experts", "cold_experts"}:
                diagnostics["format_errors"].append({
                    "name": name,
                    "error": "routed_expert_not_in_expert_bucket",
                    "bucket": bucket,
                })
            if role != "expert":
                diagnostics["format_errors"].append({
                    "name": name,
                    "error": "routed_expert_graph_role_not_expert",
                    "graph_role": role,
                })
    diagnostics["routed_expert_layers"] = sorted(routed_layers)
    diagnostics["shared_expert_layers"] = sorted(shared_layers)
    if diagnostics["format_errors"]:
        raise RuntimeError("JUJU expert tensor format validation failed: " + json.dumps(
            diagnostics["format_errors"][:16], ensure_ascii=False
        ))
    return diagnostics


def build_juju_graph_ir(*, contract, tensor_records, sections, source_name, source_path, source_repo_id, weight_file, index_file, directory=None):
    arch = dict(contract.get("arch_meta") or {})
    runtime_arch = juju_runtime_arch_metadata(contract, directory)
    expert_diagnostics = juju_expert_tensor_diagnostics(tensor_records)
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
    role_counts = {}
    bucket_counts = {}
    encoding_counts = {}
    for rec in tensor_records:
        role_counts[str(rec.get("graph_role") or "unknown")] = role_counts.get(str(rec.get("graph_role") or "unknown"), 0) + 1
        bucket_counts[str(rec.get("bucket") or "unknown")] = bucket_counts.get(str(rec.get("bucket") or "unknown"), 0) + 1
        encoding_key = str(rec.get("weight_encoding") or rec.get("gguf_type") or 0)
        encoding_counts[encoding_key] = encoding_counts.get(encoding_key, 0) + 1
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
        "format_extension_contract": juju_format_extension_contract(contract),
        "kernel_registry_contract": juju_kernel_registry_contract(contract),
        "adapter_registry_contract": juju_adapter_registry_contract(),
        "validation_contract": juju_validation_contract(),
        "research_offload_contract": juju_research_offload_contract(),
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
            **runtime_arch,
        },
        "tokenizer_contract": juju_tokenizer_contract(),
        "quantization": {
            "weight": contract.get("weight_quant_schema", {}),
            "qkv_cache": contract.get("qkv_cache_schema", {}),
            "source_weight_bits": juju_weight_bits(contract),
            "source_weight_encoding": juju_weight_encoding(contract),
            "source_weight_family": contract.get("source_weight_quant_family"),
            "source_weight_kernel_family": contract.get("source_weight_kernel_family"),
            "tensor_weight_encoding_counts": encoding_counts,
            "kernel_requirement": "engine_must_support_every_tensor_weight_encoding_or_fail_closed",
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
        "tensor_index_contract": {
            "binary_schema_version": 3,
            "offsets": "absolute_file_offsets",
            "lengths": "exact_payload_bytes",
            "alignment_bytes": 4096,
            "weight_encoding_field": "weight_encoding",
            "gguf_type_field": "gguf_type",
            "binary_required_fields": ["gguf_type", "weight_encoding"],
            "gguf_type_name_field": "gguf_type_name",
            "quant_family_field": "quant_family",
            "kernel_key_field": "kernel_key",
            "kernel_contract_field": "kernel_contract",
            "row_layout_field": "row_layout",
            "role_counts": role_counts,
            "bucket_counts": bucket_counts,
            "expert_diagnostics": expert_diagnostics,
            "sections_embedded_in_header_table": True,
            "paired_idx_required": True,
            "external_adapter_required": False,
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
            "model_load": "eager_validate_header_sections_idx_tokenizer_and_kernel_support",
            "weight_decode": "juju_weight_encoding_and_gguf_type_exact_dispatch",
            "residency_policy": contract.get("residency_policy", {}),
            "prefetch_plan_hints": contract.get("prefetch_plan_hints", {}),
            "kernel_hints": contract.get("kernel_hints", {}),
            "execution_hints": contract.get("execution_hints", {}),
            "memory_management_hints": contract.get("memory_management_hints", {}),
            "adapter_contract": {
                "dense_mlp_uses_shared_path": True,
                "expert_mlp_uses_streaming_path": True,
                "required_quant_decode": "all_tensor_index_weight_encodings",
                "prefetch_must_respect_graph_role": True,
                "tokenizer_assets_must_exist": True,
            },
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
        "execution_plan": {
            "input": ["tokenizer_assets", "token_ids"],
            "prefill": ["embedding", "layer_loop", "kv_write", "logits"],
            "decode": ["next_token_embedding", "layer_loop", "kv_read_write", "logits"],
            "offload_units": ["shared_tensor", "expert_tensor", "dense_ffn_tensor", "qkv_page", "vision_tensor", "audio_tensor", "video_tensor", "document_tensor"],
            "io_policy": {
                "read_unit": "tensor_span",
                "alignment": 4096,
                "mmap": True,
                "stream_large_slow_mem_tensors": True,
                "protect_fastmem_roles": ["shared_core", "attention", "router", "norm"],
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
                "vision_encoder": "SLOW_MEM",
                "vision_projector": "SLOW_MEM",
                "audio_encoder": "SLOW_MEM",
                "video_encoder": "SLOW_MEM",
                "document_encoder": "SLOW_MEM",
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
                "demote_order": ["video_encoder", "document_encoder", "audio_encoder", "vision_encoder", "cold_experts", "warm_experts", "large_shared_streamable"],
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
        print_gguf_byte_diagnostics(directory, source_name)
        print_gguf_tensor_layout_probes(session, source_url, directory, token=token, label=source_name)
        validate_gguf_byte_diagnostics(directory)
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
        modality_meta = juju_modality_metadata(contract, active_tensors)
        modality_flags = int(modality_meta["modality_flags"])
        pos = JUJU_HEADER_BYTES
        table_offset = pos
        pos += JUJU_SECTION_TABLE_RESERVED_ENTRIES * JUJU_SECTION_ENTRY_BYTES
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
                "gguf_kv": directory.get("gguf_kv", {}),
                "gguf_runtime": directory.get("gguf_runtime", {}),
                "gguf_kv_floats": directory.get("gguf_kv_floats", {}),
                "byte_diagnostics": directory.get("byte_diagnostics", {}),
            },
            "contract": contract,
            "modality_flags": modality_flags,
            "multimodal_contract": modality_meta,
            **juju_contract_metadata(contract, source_name, source_repo_id),
        }
        pos = add_json_section_at(pos, JUJU_SECTION_MODEL_META, "MODEL_META", meta)
        qkv_schema = contract.get("qkv_cache_schema")
        if qkv_schema:
            pos = add_json_section_at(pos, JUJU_SECTION_QKV_POLICY, "QKV_POLICY", qkv_schema)

        for bucket in JUJU_TENSOR_BUCKET_ORDER:
            group = [t for t in active_tensors if t["bucket"] == bucket and t["bytes"] > 0]
            if not group:
                continue
            pos = align_up(pos, 4096)
            section_offset = pos
            section_source_ranges = []
            for tensor in group:
                pos = align_up(pos, 4096)
                tensor_offset = pos
                source_segment = {
                    "offset": tensor_offset,
                    "size": tensor["bytes"],
                    "source_offset": tensor["source_offset"],
                }
                source_segments.append(source_segment)
                section_source_ranges.append(source_segment)
                runtime_priority = tensor_runtime_priority(tensor["name"], bucket, tensor["bytes"])
                tensor_records.append({
                    "name": tensor["name"],
                    "bucket": bucket,
                    "dims": tensor["dims"],
                    "shape": tensor["shape"],
                    "gguf_type": tensor["type"],
                    "gguf_type_name": gguf_type_name(tensor["type"]),
                    "weight_encoding": weight_encoding_from_gguf_type(tensor["type"], contract),
                    "quant_family": quant_family_from_gguf_type(tensor["type"], contract),
                    "kernel_key": kernel_key_from_gguf_type(tensor["type"], contract),
                    "row_layout": "source_gguf_quant_block_layout_preserved",
                    "source_offset": tensor["source_offset"],
                    "source_bytes": tensor["bytes"],
                    "juju_offset": tensor_offset,
                    "juju_bytes": tensor["bytes"],
                    "alignment": 4096,
                    "kernel_contract": {
                        "must_have_dot_kernel": True,
                        "must_not_return_silent_zero": True,
                        "decode_key": kernel_key_from_gguf_type(tensor["type"], contract),
                        "source_type_preserved": True,
                    },
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
                "sha256": sha256_juju_section_ranges(session, source_url, section_offset, size, section_source_ranges, token=token, chunk_size=chunk_size),
                "hash_semantics": "juju_section_bytes_including_alignment_padding",
                "prefetch_distance": 2 if bucket != "shared_weights" else 0,
                "mmap_friendly": 1,
            })
            section_sizes[section_type] = section_sizes.get(section_type, 0) + size

        runtime_arch = juju_runtime_arch_metadata(contract, directory)
        graph_ir = build_juju_graph_ir(
            contract=contract,
            tensor_records=tensor_records,
            sections=list(sections),
            source_name=source_name,
            source_path=source_path,
            source_repo_id=source_repo_id,
            weight_file=artifact_names["weights"],
            index_file=artifact_names["index"],
            directory=directory,
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
            "modality_flags": modality_flags,
            "multimodal_contract": modality_meta,
            "graph_ir_format": graph_ir["format"],
            "graph_ir_required": True,
            "graph_ir": graph_ir,
            "priority_tables": graph_ir["priority_tables"],
            "moe_offload_policy": graph_ir["moe_offload_policy"],
            "tensor_count": len(tensor_records),
            "tensors": tensor_records,
            "sections": list(sections),
            **runtime_arch,
        }
        pos = add_json_section_at(pos, JUJU_SECTION_LAYER_ORDER_INDEX, "TENSOR_INDEX", idx)
        index_checksum = int(sections[-1].get("sha256", "0" * 64)[:16], 16) if sections else 0
        file_size_value = pos
        if len(sections) > JUJU_SECTION_TABLE_RESERVED_ENTRIES:
            raise RuntimeError(f"too many JUJU sections: {len(sections)}")

        table = b"".join(pack_section(entry) for entry in sections)
        table_capacity = JUJU_SECTION_TABLE_RESERVED_ENTRIES * JUJU_SECTION_ENTRY_BYTES
        table = table + (b"\x00" * (table_capacity - len(table)))
        header = make_header(contract, artifact_source_name, file_size_value, sections, section_sizes, index_checksum=index_checksum, modality_flags=modality_flags)
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
        print_gguf_byte_diagnostics(directory, source_name)
        print_gguf_tensor_layout_probes(session, source_url, directory, token=token, label=source_name)
        validate_gguf_byte_diagnostics(directory)
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
            "split_strategy": str(split.get("split_strategy") or "limit_tensor_groups"),
            "target_split_count": int(split.get("target_split_count") or 0),
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
        print_gguf_byte_diagnostics(directory, source_name)
        print_gguf_tensor_layout_probes(session, source_url, directory, token=token, label=source_name)
        validate_gguf_byte_diagnostics(directory)
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
        modality_meta = juju_modality_metadata(contract, active_tensors)
        modality_flags = int(modality_meta["modality_flags"])
        with output_path.open("wb") as out:
            out.write(b"\x00" * JUJU_HEADER_BYTES)
            table_offset = out.tell()
            out.write(b"\x00" * (JUJU_SECTION_TABLE_RESERVED_ENTRIES * JUJU_SECTION_ENTRY_BYTES))
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
                    "gguf_kv": directory.get("gguf_kv", {}),
                    "gguf_runtime": directory.get("gguf_runtime", {}),
                    "gguf_kv_floats": directory.get("gguf_kv_floats", {}),
                    "byte_diagnostics": directory.get("byte_diagnostics", {}),
                },
                "contract": contract,
                "modality_flags": modality_flags,
                "multimodal_contract": modality_meta,
                **juju_contract_metadata(contract, source_name, source_repo_id),
            }
            add_json_section(out, JUJU_SECTION_MODEL_META, "MODEL_META", meta)
            qkv_schema = contract.get("qkv_cache_schema")
            if qkv_schema:
                add_json_section(out, JUJU_SECTION_QKV_POLICY, "QKV_POLICY", qkv_schema)

            for bucket in JUJU_TENSOR_BUCKET_ORDER:
                group = [t for t in active_tensors if t["bucket"] == bucket and t["bytes"] > 0]
                if not group:
                    continue
                write_padding(out, 4096)
                offset = out.tell()
                digest = hashlib.sha256()
                for tensor in group:
                    write_padding(out, 4096, digest=digest)
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
                        "gguf_type_name": gguf_type_name(tensor["type"]),
                        "weight_encoding": weight_encoding_from_gguf_type(tensor["type"], contract),
                        "quant_family": quant_family_from_gguf_type(tensor["type"], contract),
                        "kernel_key": kernel_key_from_gguf_type(tensor["type"], contract),
                        "row_layout": "source_gguf_quant_block_layout_preserved",
                        "source_offset": tensor["source_offset"],
                        "source_bytes": tensor["bytes"],
                        "juju_offset": tensor_offset,
                        "juju_bytes": tensor["bytes"],
                        "alignment": 4096,
                        "kernel_contract": {
                            "must_have_dot_kernel": True,
                            "must_not_return_silent_zero": True,
                            "decode_key": kernel_key_from_gguf_type(tensor["type"], contract),
                            "source_type_preserved": True,
                        },
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
                    "hash_semantics": "juju_section_bytes_including_alignment_padding",
                    "prefetch_distance": 2 if bucket != "shared_weights" else 0,
                    "mmap_friendly": 1,
                })
                section_sizes[section_type] = section_sizes.get(section_type, 0) + size

            runtime_arch = juju_runtime_arch_metadata(contract, directory)
            graph_ir = build_juju_graph_ir(
                contract=contract,
                tensor_records=tensor_records,
                sections=list(sections),
                source_name=source_name,
                source_path=source_path,
                source_repo_id=source_repo_id,
                weight_file=output_path.name,
                index_file=index_path.name,
                directory=directory,
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
                "modality_flags": modality_flags,
                "multimodal_contract": modality_meta,
                "graph_ir_format": graph_ir["format"],
                "graph_ir_required": True,
                "graph_ir": graph_ir,
                "priority_tables": graph_ir["priority_tables"],
                "moe_offload_policy": graph_ir["moe_offload_policy"],
                "tensor_count": len(tensor_records),
                "tensors": tensor_records,
                "sections": sections,
                **runtime_arch,
            }
            add_json_section(out, JUJU_SECTION_LAYER_ORDER_INDEX, "TENSOR_INDEX", idx)
            index_checksum = int(sections[-1].get("sha256", "0" * 64)[:16], 16) if sections else 0
            file_size_value = out.tell()
            if len(sections) > JUJU_SECTION_TABLE_RESERVED_ENTRIES:
                raise RuntimeError(f"too many JUJU sections: {len(sections)}")
            out.seek(table_offset)
            for entry in sections:
                out.write(pack_section(entry))
            out.seek(0)
            out.write(make_header(contract, artifact_source_name, file_size_value, sections, section_sizes, index_checksum=index_checksum, modality_flags=modality_flags))

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
        print_gguf_byte_diagnostics(directory, source_name)
        print_gguf_tensor_layout_probes(session, source_url, directory, token=token, label=source_name)
        validate_gguf_byte_diagnostics(directory)
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
            "split_strategy": str(split.get("split_strategy") or "limit_tensor_groups"),
            "target_split_count": int(split.get("target_split_count") or 0),
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
