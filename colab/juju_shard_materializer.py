import hashlib
import io
import json
import threading
import bisect
import os
import re
import shlex
import struct
import subprocess
import time
import math
from pathlib import Path

import requests


STRICT_GGUF_EXACT_BYTES = os.environ.get("STRICT_GGUF_EXACT_BYTES", "1") != "0"
JUJU_ZERO_SHA256 = "0" * 64

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
    "tokenizer",
)

GGUF_RUNTIME_KV_EXACT_KEYS = {
    "general.architecture",
    "general.name",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.mask_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.chat_template",
}

GGUF_RUNTIME_KV_ALIAS_MAP = {
    "general.architecture": ("architecture", "declared_architecture", "model_type"),
    "general.name": ("model_name",),
    "tokenizer.ggml.model": ("tokenizer_model",),
    "tokenizer.ggml.pre": ("tokenizer_pre",),
    "tokenizer.ggml.bos_token_id": ("bos_token_id",),
    "tokenizer.ggml.eos_token_id": ("eos_token_id",),
    "tokenizer.ggml.unknown_token_id": ("unk_token_id", "unknown_token_id"),
    "tokenizer.ggml.padding_token_id": ("pad_token_id", "padding_token_id"),
    "tokenizer.ggml.mask_token_id": ("mask_token_id",),
    "tokenizer.ggml.add_bos_token": ("add_bos_token", "tokenizer_add_bos_token"),
    "tokenizer.ggml.add_eos_token": ("add_eos_token", "tokenizer_add_eos_token"),
    "tokenizer.ggml.add_space_prefix": ("add_space_prefix", "tokenizer_add_space_prefix"),
    "tokenizer.chat_template": ("chat_template",),
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
    "attn_logit_softcap": ("attn_logit_softcap", "attn_logit_softcapping", "attention_logit_softcap", "attention_logit_softcapping"),
    "attn_logit_softcapping": ("attn_logit_softcapping", "attn_logit_softcap", "attention_logit_softcapping", "attention_logit_softcap"),
    "attention_logit_softcap": ("attention_logit_softcap", "attention_logit_softcapping", "attn_logit_softcap", "attn_logit_softcapping"),
    "attention_logit_softcapping": ("attention_logit_softcapping", "attention_logit_softcap", "attn_logit_softcapping", "attn_logit_softcap"),
    "embedding_scale": ("embedding_scale", "embed_scale", "scale_emb"),
    "embed_scale": ("embed_scale", "embedding_scale", "scale_emb"),
    "scale_emb": ("scale_emb", "embedding_scale", "embed_scale"),
    "scale_embedding": ("scale_embedding", "embedding_scale", "embed_scale"),
    "query_pre_attn_scalar": ("query_pre_attn_scalar", "attention_query_pre_attn_scalar", "attn_query_pre_attn_scalar"),
    "attention_query_pre_attn_scalar": ("attention_query_pre_attn_scalar", "query_pre_attn_scalar", "attn_query_pre_attn_scalar"),
    "attn_query_pre_attn_scalar": ("attn_query_pre_attn_scalar", "query_pre_attn_scalar", "attention_query_pre_attn_scalar"),
    "attention_scale": ("attention_scale", "attn_scale", "attention_score_scale", "f_attn_scale", "gemma4.attention.scale", "qk_scale"),
    "attn_scale": ("attn_scale", "attention_scale", "attention_score_scale", "f_attn_scale", "gemma4.attention.scale", "qk_scale"),
    "attention_score_scale": ("attention_score_scale", "attention_scale", "attn_scale", "f_attn_scale", "gemma4.attention.scale", "qk_scale"),
    "f_attn_scale": ("f_attn_scale", "attention_scale", "attn_scale", "attention_score_scale", "gemma4.attention.scale", "qk_scale"),
    "gemma4.attention.scale": ("gemma4.attention.scale", "f_attn_scale", "attention_scale", "attn_scale", "attention_score_scale", "qk_scale"),
    "qk_scale": ("qk_scale", "attention_score_scale", "attention_scale", "attn_scale", "f_attn_scale", "gemma4.attention.scale"),
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

JUJU_EMBEDDING_SCALE_FAMILY_RULES = (
    (("gemma",), "sqrt_hidden_size", "hf_forward_gemma_sqrt_hidden_size"),
)

JUJU_ATTENTION_Q_SUFFIXES = (
    "attn_q.weight",
    "attention.wq.weight",
    "q_proj.weight",
    "self_attn.q_proj.weight",
    "attn_q_a_proj.weight",
    "attn_q_b_proj.weight",
)
JUJU_ATTENTION_K_SUFFIXES = (
    "attn_k.weight",
    "attention.wk.weight",
    "k_proj.weight",
    "self_attn.k_proj.weight",
    "attn_kv_a_proj_with_mqa.weight",
)
JUJU_ATTENTION_V_SUFFIXES = (
    "attn_v.weight",
    "attention.wv.weight",
    "v_proj.weight",
    "self_attn.v_proj.weight",
    "value_proj.weight",
    "v_projection.weight",
    "attn_kv_b_proj.weight",
)
JUJU_ATTENTION_O_SUFFIXES = (
    "attn_output.weight",
    "attention.wo.weight",
    "o_proj.weight",
    "self_attn.o_proj.weight",
)
JUJU_Q_NORM_SUFFIXES = ("attn_q_norm.weight", "q_norm.weight", "self_attn.q_norm.weight")
JUJU_K_NORM_SUFFIXES = ("attn_k_norm.weight", "k_norm.weight", "self_attn.k_norm.weight")
JUJU_V_NORM_SUFFIXES = ("attn_v_norm.weight", "v_norm.weight", "value_norm.weight", "self_attn.v_norm.weight")
JUJU_QKV_SUPPORTED_CACHE_BITS = (2, 3, 4, 5, 6, 8, 16, 32)

JUJU_HEADER_BYTES = 4096
JUJU_SECTION_ENTRY_BYTES = 96
# BUGFIX 974: Increase section table from 32→64 for multimodal models ★★★
# Problem: 15 defined section types + runtime metadata sections can exceed 32
# on large multimodal models. RuntimeError raised AFTER writing tens of GB.
# Solution: Double to 64. Header stores actual section_count, so readers
# handle this transparently without wire format version change.
JUJU_SECTION_TABLE_RESERVED_ENTRIES = 64
JUJU_SECTION_MODEL_META = 0x0001
JUJU_SECTION_PREDICTOR = 0x0002
JUJU_SECTION_BUDDY_MAP = 0x0003
JUJU_SECTION_TIER_HINT = 0x0004
JUJU_SECTION_SHARED_WEIGHTS = 0x0010
JUJU_SECTION_HOT_EXPERTS = 0x0011
JUJU_SECTION_WARM_EXPERTS = 0x0012
JUJU_SECTION_COLD_EXPERTS = 0x0013
JUJU_SECTION_LAYER_ORDER_INDEX = 0x0020
JUJU_SECTION_QKV_POLICY = 0x0021
JUJU_SECTION_RUNTIME_CONTRACT = 0x0022
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
JUJU_CONTAINER_VERSION_MAJOR = 2
JUJU_CONTAINER_VERSION_MINOR = 0
JUJU_IDX_FORMAT = "JUJU_IDX_JSON_V2"
JUJU_IDX_SCHEMA_VERSION = 4
JUJU_BINARY_TENSOR_INDEX_SCHEMA_VERSION = 4
JUJU_EXPERT_BUNDLE_TABLE_FORMAT = "JUJU_EXPERT_BUNDLE_TABLE_V2"
JUJU_FORMAT_CONTRACT_VERSION = 3
JUJU_BINARY_WIRE_ID = "JUJU_V2_HEADER4096_SECTION96_BUNDLE_NATIVE"
JUJU_BUNDLE_ALIGNMENT_BYTES = 4096
JUJU_EXPERT_BUNDLE_MEMBER_ORDER = (
    "gate",
    "gate_scale",
    "gate_scale2",
    "up",
    "up_scale",
    "up_scale2",
    "down",
    "down_scale",
    "down_scale2",
)
JUJU_EXPERT_BUNDLE_MEMBER_RANK = {
    role: idx for idx, role in enumerate(JUJU_EXPERT_BUNDLE_MEMBER_ORDER)
}
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


def juju_artifact_uid(*, source_repo_id, source_path, source_name, artifact_source_name, weight_file, split_meta, tensor_count):
    payload = {
        "format": "JUJU_ARTIFACT_UID_V1",
        "source_repo_id": source_repo_id or "",
        "source_path": source_path or "",
        "source_name": source_name or "",
        "artifact_source_name": artifact_source_name or source_name or "",
        "weight_file": weight_file or "",
        "split": split_meta or {},
        "tensor_count": int(tensor_count or 0),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


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


def juju_source_shard_info(source_name):
    name = Path(str(source_name or "")).name
    match = re.match(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})(?P<suffix>\.[^.]+)?$", name)
    if not match:
        return {}
    index = int(match.group("index"))
    count = int(match.group("count"))
    if index <= 0 or count <= 1 or index > count:
        return {}
    return {
        "enabled": True,
        "parent_source_name": match.group("prefix"),
        "artifact_source_name": name,
        "split_index": index,
        "split_count": count,
        "split_strategy": "source_gguf_physical_shard_set",
        "source_shard_index": index,
        "source_shard_count": count,
    }


def juju_effective_split_meta(source_name, artifact_source_name, tensor_count, tensor_bytes=0, split_info=None):
    split_meta = dict(split_info or {})
    source_shard = juju_source_shard_info(source_name)
    if source_shard and not split_meta:
        split_meta = source_shard
    if not split_meta:
        split_meta = {
            "enabled": False,
            "split_index": 1,
            "split_count": 1,
            "parent_source_name": source_name,
            "artifact_source_name": artifact_source_name,
        }
    split_count = int(split_meta.get("split_count") or 1)
    split_meta.setdefault("enabled", split_count > 1)
    split_meta.setdefault("parent_source_name", source_shard.get("parent_source_name") or source_name)
    split_meta.setdefault("artifact_source_name", artifact_source_name)
    split_meta["tensor_count"] = int(tensor_count or 0)
    if tensor_bytes:
        split_meta["tensor_bytes"] = int(tensor_bytes)
    return split_meta


def juju_idx_split_top_level_fields(split_meta):
    split_meta = split_meta if isinstance(split_meta, dict) else {}
    out = {}
    for key in (
        "split_index",
        "split_count",
        "parent_source_name",
        "artifact_source_name",
        "source_shard_index",
        "source_shard_count",
        "split_strategy",
    ):
        value = split_meta.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    split_count = int(out.get("split_count") or 0)
    if split_count > 1:
        out["split_enabled"] = True
        out.setdefault("file_index", out.get("split_index"))
        out.setdefault("file_count", out.get("split_count"))
    return out


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


GGUF_CODEC_REGISTRY_VERSION = "GGUF_CODEC_REGISTRY_V1"
GGUF_CODEC_REGISTRY = {
    0: {"codec_id": "GGUF_F32_V1", "name": "F32", "family": "raw_scalar_or_integer", "weight_encoding": 2, "bytes_per_col": 4, "supports_exact_ppl": True},
    1: {"codec_id": "GGUF_F16_V1", "name": "F16", "family": "raw_scalar_or_integer", "weight_encoding": 1, "bytes_per_col": 2, "supports_exact_ppl": True},
    2: {"codec_id": "GGUF_Q4_0_V1", "name": "Q4_0", "family": "legacy_ggml_quant", "weight_encoding": 22, "block_cols": 32, "block_bytes": 18, "supports_exact_ppl": True},
    3: {"codec_id": "GGUF_Q4_1_V1", "name": "Q4_1", "family": "legacy_ggml_quant", "weight_encoding": 23, "block_cols": 32, "block_bytes": 20, "supports_exact_ppl": True},
    6: {"codec_id": "GGUF_Q5_0_V1", "name": "Q5_0", "family": "legacy_ggml_quant", "weight_encoding": 24, "block_cols": 32, "block_bytes": 22, "supports_exact_ppl": True},
    7: {"codec_id": "GGUF_Q5_1_V1", "name": "Q5_1", "family": "legacy_ggml_quant", "weight_encoding": 12, "block_cols": 32, "block_bytes": 24, "supports_exact_ppl": True},
    8: {"codec_id": "GGUF_Q8_0_V1", "name": "Q8_0", "family": "legacy_ggml_quant", "weight_encoding": 13, "block_cols": 32, "block_bytes": 34, "supports_exact_ppl": True},
    9: {"codec_id": "GGUF_Q8_1_V1", "name": "Q8_1", "family": "legacy_ggml_quant", "weight_encoding": 25, "block_cols": 32, "block_bytes": 36, "supports_exact_ppl": True},
    10: {"codec_id": "GGUF_Q2_K_V1", "name": "Q2_K", "family": "k_quant", "weight_encoding": 15, "block_cols": 256, "block_bytes": 84, "supports_exact_ppl": True},
    11: {"codec_id": "GGUF_Q3_K_V1", "name": "Q3_K", "family": "k_quant", "weight_encoding": 16, "block_cols": 256, "block_bytes": 110, "supports_exact_ppl": True},
    12: {"codec_id": "GGUF_Q4_K_V1", "name": "Q4_K", "family": "k_quant", "weight_encoding": 17, "block_cols": 256, "block_bytes": 144, "supports_exact_ppl": True},
    13: {"codec_id": "GGUF_Q5_K_V1", "name": "Q5_K", "family": "k_quant", "weight_encoding": 14, "block_cols": 256, "block_bytes": 176, "supports_exact_ppl": True},
    14: {"codec_id": "GGUF_Q6_K_V1", "name": "Q6_K", "family": "k_quant", "weight_encoding": 18, "block_cols": 256, "block_bytes": 210, "supports_exact_ppl": True},
    15: {"codec_id": "GGUF_Q8_K_V1", "name": "Q8_K", "family": "k_quant", "weight_encoding": 34, "block_cols": 256, "block_bytes": 292, "supports_exact_ppl": True},
    16: {"codec_id": "GGUF_IQ2_XXS_V1", "name": "IQ2_XXS", "family": "importance_quant", "weight_encoding": 19, "block_cols": 256, "block_bytes": 66, "supports_exact_ppl": True},
    17: {"codec_id": "GGUF_IQ2_XS_V1", "name": "IQ2_XS", "family": "importance_quant", "weight_encoding": 29, "block_cols": 256, "block_bytes": 74, "supports_exact_ppl": True},
    18: {"codec_id": "GGUF_IQ3_XXS_V1", "name": "IQ3_XXS", "family": "importance_quant", "weight_encoding": 20, "block_cols": 256, "block_bytes": 98, "supports_exact_ppl": True},
    19: {"codec_id": "GGUF_IQ1_S_V1", "name": "IQ1_S", "family": "importance_quant", "weight_encoding": 32, "block_cols": 256, "block_bytes": 50, "supports_exact_ppl": True},
    20: {"codec_id": "GGUF_IQ4_NL_V1", "name": "IQ4_NL", "family": "importance_quant", "weight_encoding": 27, "block_cols": 32, "block_bytes": 18, "supports_exact_ppl": True},
    21: {"codec_id": "GGUF_IQ3_S_V1", "name": "IQ3_S", "family": "importance_quant", "weight_encoding": 31, "block_cols": 256, "block_bytes": 110, "supports_exact_ppl": True},
    22: {"codec_id": "GGUF_IQ2_S_V1", "name": "IQ2_S", "family": "importance_quant", "weight_encoding": 30, "block_cols": 256, "block_bytes": 82, "supports_exact_ppl": True},
    23: {"codec_id": "GGUF_IQ4_XS_V1", "name": "IQ4_XS", "family": "importance_quant", "weight_encoding": 28, "block_cols": 256, "block_bytes": 136, "supports_exact_ppl": True},
    24: {"codec_id": "GGUF_I8_V1", "name": "I8", "family": "raw_scalar_or_integer", "weight_encoding": 0, "bytes_per_col": 1, "supports_exact_ppl": False},
    25: {"codec_id": "GGUF_I16_V1", "name": "I16", "family": "raw_scalar_or_integer", "weight_encoding": 0, "bytes_per_col": 2, "supports_exact_ppl": False},
    26: {"codec_id": "GGUF_I32_V1", "name": "I32", "family": "raw_scalar_or_integer", "weight_encoding": 0, "bytes_per_col": 4, "supports_exact_ppl": False},
    27: {"codec_id": "GGUF_I64_V1", "name": "I64", "family": "raw_scalar_or_integer", "weight_encoding": 0, "bytes_per_col": 8, "supports_exact_ppl": False},
    28: {"codec_id": "GGUF_F64_V1", "name": "F64", "family": "raw_scalar_or_integer", "weight_encoding": 0, "bytes_per_col": 8, "supports_exact_ppl": False},
    29: {"codec_id": "GGUF_IQ1_M_V1", "name": "IQ1_M", "family": "importance_quant", "weight_encoding": 33, "block_cols": 256, "block_bytes": 56, "supports_exact_ppl": True},
    30: {"codec_id": "GGUF_BF16_V1", "name": "BF16", "family": "raw_scalar_or_integer", "weight_encoding": 21, "bytes_per_col": 2, "supports_exact_ppl": True},
    31: {"codec_id": "GGUF_Q4_0_4_4_V1", "name": "Q4_0_4_4", "family": "legacy_ggml_interleaved_quant", "weight_encoding": 0, "supports_exact_ppl": False},
    32: {"codec_id": "GGUF_Q4_0_4_8_V1", "name": "Q4_0_4_8", "family": "legacy_ggml_interleaved_quant", "weight_encoding": 0, "supports_exact_ppl": False},
    33: {"codec_id": "GGUF_Q4_0_8_8_V1", "name": "Q4_0_8_8", "family": "legacy_ggml_interleaved_quant", "weight_encoding": 0, "supports_exact_ppl": False},
    34: {"codec_id": "GGUF_TQ1_0_V1", "name": "TQ1_0", "family": "ternary_quant", "weight_encoding": 35, "block_cols": 256, "block_bytes": 54, "supports_exact_ppl": True},
    35: {"codec_id": "GGUF_TQ2_0_V1", "name": "TQ2_0", "family": "ternary_quant", "weight_encoding": 36, "block_cols": 256, "block_bytes": 66, "supports_exact_ppl": True},
    36: {"codec_id": "GGUF_REMOVED_IQ4_NL_4_4", "name": "REMOVED_IQ4_NL_4_4", "family": "removed", "weight_encoding": 0, "supports_exact_ppl": False},
    37: {"codec_id": "GGUF_REMOVED_IQ4_NL_4_8", "name": "REMOVED_IQ4_NL_4_8", "family": "removed", "weight_encoding": 0, "supports_exact_ppl": False},
    38: {"codec_id": "GGUF_REMOVED_IQ4_NL_8_8", "name": "REMOVED_IQ4_NL_8_8", "family": "removed", "weight_encoding": 0, "supports_exact_ppl": False},
    39: {
        "codec_id": "GGUF_MXFP4_V1",
        "name": "MXFP4",
        "family": "mxfp4",
        "weight_encoding": 4,
        "block_cols": 32,
        "block_bytes": 17,
        "scale": "one_e8m0_byte_per_block_before_payload",
        "nibble_order": "pairwise_even_index_low_nibble_odd_index_high_nibble",
        "decode_kernel": "moe_dot_gguf_mxfp4_row",
        "copy_row_kernel": "moe_copy_gguf_mxfp4_row",
        "supports_exact_ppl": True,
    },
}


def gguf_codec_spec(tensor_type):
    return GGUF_CODEC_REGISTRY.get(u32(tensor_type))


def gguf_codec_row_bytes_from_spec(spec, cols):
    cols = int(cols or 0)
    if not spec or cols <= 0 or spec.get("supports_exact_ppl") is False:
        return 0
    if spec.get("bytes_per_col"):
        return cols * int(spec["bytes_per_col"])
    block_cols = int(spec.get("block_cols") or 0)
    block_bytes = int(spec.get("block_bytes") or 0)
    if block_cols <= 0 or block_bytes <= 0:
        return 0
    return ((cols + block_cols - 1) // block_cols) * block_bytes


def gguf_tensor_row_bytes(tensor_type, cols):
    t = u32(tensor_type)
    cols = int(cols or 0)
    if cols <= 0:
        return 0
    spec = gguf_codec_spec(t)
    if spec is not None:
        return gguf_codec_row_bytes_from_spec(spec, cols)
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
        # GGUF interleaved Q4_0 variants have distinct layouts. The engine
        # intentionally does not decode them yet, so fail closed instead of
        # emitting bytes that would later be interpreted with the wrong kernel.
        return 0
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


def juju_row_stride_padding_enabled():
    return os.environ.get("JUJU_ENABLE_ROW_STRIDE_PADDING", "0") != "0"


def juju_exact_mode_policy():
    return {
        "format": "JUJU_EXACT_MODE_POLICY_V1",
        "buddy_fallback_can_replace_expert": False,
        "partial_execution_allowed": False,
        "seqtopk_can_change_router_topk": False,
        "cold_expert_requantize_allowed": False,
        "predictor_role": "prefetch_hint_only",
        "routing_must_match_source": True,
        "router_topk_must_match_source": True,
        "expert_id_sequence_must_match_source": True,
        "qkv_contract_is_part_of_ppl_acceptance": True,
        "plain_kv_reference_required_for_acceptance": False,
        "ppl_must_report_kv_backend": True,
        "row_stride_padding_default": False,
        "row_stride_padding_env": "JUJU_ENABLE_ROW_STRIDE_PADDING",
    }


def juju_approx_mode_policy():
    return {
        "format": "JUJU_APPROX_MODE_POLICY_V1",
        "buddy_fallback_can_replace_expert": True,
        "partial_execution_allowed": True,
        "seqtopk_can_change_router_topk": True,
        "cold_expert_requantize_allowed": True,
        "predictor_role": "prefetch_hint_or_approx_policy_only",
        "quality_gate_required": True,
        "ppl_delta_threshold_required": True,
        "must_not_be_used_for_exact_ppl_claim": True,
    }


def juju_ppl_acceptance_contract():
    dataset = os.environ.get("JUJU_PPL_EVAL_DATASET", "wikitext-2-raw-v1")
    max_delta = float(os.environ.get("JUJU_PPL_MAX_DELTA", "0.02") or "0.02")
    logits_max_abs = float(os.environ.get("JUJU_LOGITS_MAX_ABS_DELTA", "0.0005") or "0.0005")
    logits_rms = float(os.environ.get("JUJU_LOGITS_RMS_DELTA", "0.0001") or "0.0001")
    first_tokens = int(os.environ.get("JUJU_LOGITS_FIRST_TOKENS", "32") or "32")
    return {
        "format": "JUJU_PPL_ACCEPTANCE_CONTRACT_V1",
        "required_for_preserve_claim": True,
        "source_model": "reference_token_ids",
        "candidate_model": "juju_qkv_contract",
        "tokenizer_policy": "server_ppl_endpoint_requires_reference_input_ids",
        "server_text_tokenization_allowed": False,
        "qkv_required_in_ppl": True,
        "required_response_fields": [
            "kv_backend",
            "qkv_forced_by_format",
            "tokenizer_hash",
            "tokenizer_config_hash",
            "chat_template_hash",
            "tokenizer_add_bos",
            "tokenizer_add_eos",
            "tokenizer_add_space_prefix",
            "input_ids_preview",
        ],
        "dataset": dataset,
        "max_ppl_delta": max_delta,
        "logits_first_token_count": first_tokens,
        "logits_max_abs_delta": logits_max_abs,
        "logits_rms_delta": logits_rms,
        "router_topk_exact_match_required": True,
        "expert_id_sequence_exact_match_required": True,
        "tensor_logical_hash_all_match_required": True,
        "external_runner_env": "JUJU_PPL_COMPARE_CMD",
        "require_runner_env": "JUJU_REQUIRE_PPL_ACCEPTANCE",
    }


def juju_performance_acceptance_contract():
    return {
        "format": "JUJU_PERFORMANCE_ACCEPTANCE_CONTRACT_V1",
        "enabled_only_after_exact_acceptance": True,
        "features_under_test": [
            "row_stride_padding",
            "qkv_quantized_cache",
            "expert_prefetch_predictor",
            "gds_or_directstorage",
            "hot_warm_cold_calibration",
        ],
        "required_metrics": [
            "ppl_delta",
            "tokens_per_second",
            "expert_hit_rate",
            "expert_miss_latency_us",
            "prefetch_waste_ratio",
            "gpu_idle_gap_us",
            "disk_read_bytes",
            "pcie_copy_bytes",
        ],
        "ppl_delta_threshold_required": True,
        "tokens_per_second_must_improve": True,
        "expert_miss_latency_must_decrease": True,
        "prefetch_waste_must_not_increase_unbounded": True,
        "gpu_idle_gap_must_decrease": True,
    }


def juju_expert_calibration_contract():
    return {
        "format": "JUJU_EXPERT_CALIBRATION_CONTRACT_V1",
        "recommended": True,
        "required_for_max_offload_claim": True,
        "trace_required_fields": [
            "layer_id",
            "token_pos",
            "selected_experts",
            "router_scores",
        ],
        "derived_tables": [
            "expert_access_count",
            "coactivation",
            "transition_table",
            "hot_warm_cold_initial_tiers",
        ],
        "idx_update_function": "apply_juju_expert_calibration_to_idx",
        "runtime_update": "ema",
        "predictor_role_in_exact_mode": "prefetch_hint_only",
    }


def juju_adaptive_runtime_scheduler_contract():
    return {
        "format": "JUJU_ADAPTIVE_RUNTIME_SCHEDULER_CONTRACT_V1",
        "required_for_max_offload_claim": True,
        "routing_policy": "exact_mode_never_changes_router_output",
        "placement": {
            "hot_experts": "VRAM",
            "warm_experts": "RAM_or_pinned_cache",
            "cold_experts": "NVMe_sequential_bundle",
        },
        "eviction_score_inputs": [
            "hit_rate_ema",
            "miss_cost_us",
            "expert_size_bytes",
            "predicted_next_use_epoch",
            "kv_cache_pressure",
        ],
        "prefetch_control_inputs": [
            "gpu_idle_gap_us",
            "pcie_copy_bytes_per_s",
            "disk_read_bytes_per_s",
            "staging_slot_deficit",
            "prefetch_waste_ratio",
            "expert_miss_latency_us",
        ],
        "feedback_loop": [
            "measure_hit_miss",
            "measure_copy_and_disk_bytes",
            "shrink_window_on_backpressure_or_waste",
            "grow_window_on_gpu_idle_and_low_waste",
            "update_hot_warm_cold_ema",
        ],
    }


def juju_row_stride_alignment_bytes():
    # BUGFIX 975: Default 64→512 for GPU Direct Storage / O_DIRECT compatibility ★★★
    # Problem: 64-byte row padding → row N offset not 512/4096-aligned
    # → individual row DMA requests fail on GDS/O_DIRECT (need 512B+ alignment)
    # → forces whole-section reads instead of per-row random access
    # Solution: Default 512. Set JUJU_ROW_STRIDE_ALIGNMENT_BYTES=4096 for strict GDS.
    # Impact: Slight file size increase within 6.25% overhead budget.
    raw = int(os.environ.get("JUJU_ROW_STRIDE_ALIGNMENT_BYTES", "512") or "512")
    if raw <= 1:
        return 1
    return min(raw, 4096)


def juju_row_stride_min_row_bytes():
    return max(0, int(os.environ.get("JUJU_ROW_STRIDE_MIN_ROW_BYTES", "256") or "256"))


def juju_row_stride_max_overhead_pct():
    return max(0.0, float(os.environ.get("JUJU_ROW_STRIDE_MAX_OVERHEAD_PCT", "6.25") or "6.25"))


def juju_tensor_matrix_shape(tensor):
    shape = [int(v or 0) for v in (tensor.get("shape") or [])]
    if not shape or shape[0] <= 0:
        return 0, 0
    rows = 1
    for dim in shape[1:]:
        rows *= max(0, int(dim or 0))
    return rows, int(shape[0])


def juju_tensor_storage_layout(tensor):
    source_bytes = int(tensor.get("bytes") or 0)
    rows, cols = juju_tensor_matrix_shape(tensor)
    row_bytes = gguf_tensor_row_bytes(tensor.get("type"), cols)
    logical_bytes = row_bytes * rows if row_bytes and rows > 0 else source_bytes
    alignment = juju_row_stride_alignment_bytes()
    layout = {
        "logical_rows": int(rows),
        "logical_cols": int(cols),
        "source_row_bytes": int(row_bytes or 0),
        "row_bytes": int(row_bytes or 0),
        "row_stride_bytes": int(row_bytes or 0),
        "row_padding_bytes": 0,
        "source_bytes": int(source_bytes),
        "logical_bytes": int(logical_bytes),
        "juju_bytes": int(source_bytes),
        "physical_bytes": int(source_bytes),
        "row_stride_alignment_bytes": int(alignment),
        "row_stride_padded": False,
        "row_layout": "source_gguf_quant_block_layout_preserved",
    }
    if (
        not juju_row_stride_padding_enabled() or
        source_bytes <= 0 or rows <= 1 or row_bytes <= 0 or
        logical_bytes != source_bytes or row_bytes < juju_row_stride_min_row_bytes()
    ):
        return layout
    row_stride = align_up(row_bytes, alignment)
    if row_stride <= row_bytes:
        return layout
    padding_per_row = row_stride - row_bytes
    padded_bytes = row_stride * rows
    overhead_pct = ((padded_bytes - source_bytes) * 100.0 / source_bytes) if source_bytes else 0.0
    max_overhead_pct = juju_row_stride_max_overhead_pct()
    if max_overhead_pct > 0.0 and overhead_pct > max_overhead_pct:
        layout["row_layout"] = "source_gguf_quant_block_layout_preserved_stride_padding_skipped_overhead"
        return layout
    layout.update({
        "row_stride_bytes": int(row_stride),
        "row_padding_bytes": int(padding_per_row),
        "juju_bytes": int(padded_bytes),
        "physical_bytes": int(padded_bytes),
        "row_stride_padded": True,
        "row_layout": "source_gguf_quant_blocks_row_stride_padded",
        "row_stride_overhead_pct": round(overhead_pct, 6),
    })
    return layout


def juju_tensor_payload_bytes(tensor):
    return int(juju_tensor_storage_layout(tensor).get("juju_bytes") or 0)


def juju_row_stride_stats(tensors):
    stats = {
        "enabled": juju_row_stride_padding_enabled(),
        "alignment_bytes": juju_row_stride_alignment_bytes(),
        "tensor_count": 0,
        "padded_tensor_count": 0,
        "source_bytes": 0,
        "juju_bytes": 0,
        "padding_bytes": 0,
    }
    for tensor in tensors or []:
        layout = juju_tensor_storage_layout(tensor)
        stats["tensor_count"] += 1
        stats["source_bytes"] += int(layout["source_bytes"])
        stats["juju_bytes"] += int(layout["juju_bytes"])
        if layout.get("row_stride_padded"):
            stats["padded_tensor_count"] += 1
            stats["padding_bytes"] += int(layout["juju_bytes"]) - int(layout["source_bytes"])
    if stats["source_bytes"] > 0:
        stats["padding_overhead_pct"] = round(
            stats["padding_bytes"] * 100.0 / stats["source_bytes"], 6
        )
    else:
        stats["padding_overhead_pct"] = 0.0
    return stats


def juju_tensor_source_segment(tensor, tensor_offset, layout=None):
    layout = layout or juju_tensor_storage_layout(tensor)
    segment = {
        "offset": int(tensor_offset),
        "size": int(layout["juju_bytes"]),
        "source_offset": int(tensor["source_offset"]),
    }
    if layout.get("row_stride_padded"):
        segment.update({
            "kind": "row_padded_source",
            "rows": int(layout["logical_rows"]),
            "row_bytes": int(layout["row_bytes"]),
            "row_stride_bytes": int(layout["row_stride_bytes"]),
            "source_size": int(layout["source_bytes"]),
        })
    return segment


def juju_bundle_member_role_from_name(name, split_role=None):
    if split_role in {"gate", "up", "down"}:
        return split_role
    lower = str(name or "").lower()
    base = _juju_expert_projection_name(lower)
    if base == "gate_up":
        base = "gate"
    if base not in {"gate", "up", "down"}:
        return base
    if "scale2" in lower or "scale_2" in lower or "scale.2" in lower:
        return f"{base}_scale2"
    if "scale" in lower or "scales" in lower or "scale_inv" in lower:
        return f"{base}_scale"
    return base


def juju_bundle_member_projection_from_role(role):
    role = str(role or "")
    if role.startswith("gate"):
        return "gate"
    if role.startswith("up"):
        return "up"
    if role.startswith("down"):
        return "down"
    return ""


def juju_tensor_expert_member_specs(tensor, contract):
    name = tensor.get("name")
    if not is_routed_expert_tensor_name(name):
        return []
    layer = _juju_layer_id_from_name(name)
    shape = [int(v or 0) for v in (tensor.get("shape") or [])]
    expert_count = _juju_expert_count_from_shape(shape)
    if layer is None or expert_count <= 0:
        return []
    layout = juju_tensor_storage_layout(tensor)
    source_bytes = int(layout.get("source_bytes") or tensor.get("bytes") or 0)
    juju_bytes = int(layout.get("juju_bytes") or source_bytes)
    if source_bytes <= 0 or juju_bytes <= 0:
        return []
    if source_bytes % expert_count != 0 or juju_bytes % expert_count != 0:
        return []
    source_per_expert = source_bytes // expert_count
    juju_per_expert = juju_bytes // expert_count
    rows_total = int(layout.get("logical_rows") or 0)
    rows_per_expert = rows_total // expert_count if expert_count > 0 and rows_total % expert_count == 0 else 0
    row_bytes = int(layout.get("row_bytes") or 0)
    row_stride = int(layout.get("row_stride_bytes") or row_bytes or 0)
    cols = int(layout.get("logical_cols") or (shape[0] if shape else 0))
    projection = _juju_expert_projection_name(name)
    split_roles = []
    if projection == "gate_up":
        split_roles = ["gate", "up"]
    else:
        split_roles = [juju_bundle_member_role_from_name(name)]
    specs = []
    for expert in range(expert_count):
        expert_source_base = int(tensor["source_offset"]) + expert * source_per_expert
        for split_idx, role in enumerate(split_roles):
            source_rel = 0
            source_size = source_per_expert
            output_size = juju_per_expert
            member_rows = rows_per_expert
            if projection == "gate_up":
                if rows_per_expert > 0 and rows_per_expert % 2 == 0 and row_bytes > 0:
                    member_rows = rows_per_expert // 2
                    source_size = member_rows * row_bytes
                    output_size = member_rows * row_stride
                    source_rel = split_idx * source_size
                else:
                    raise RuntimeError(
                        f"JUJU gate_up split requires explicit contiguous row layout: "
                        f"name={name} shape={shape} row_bytes={row_bytes} rows_per_expert={rows_per_expert}"
                    )
            source_offset = expert_source_base + source_rel
            spec = {
                "tensor": tensor,
                "source_tensor_name": name,
                "bucket": tensor.get("bucket"),
                "layer": int(layer),
                "expert": int(expert),
                "role": str(role),
                "projection": juju_bundle_member_projection_from_role(role),
                "source_offset": int(source_offset),
                "source_rel_offset": int(source_rel),
                "source_size": int(source_size),
                "output_size": int(output_size),
                "physical_size": int(output_size),
                "rows": int(member_rows or (shape[1] if len(shape) > 1 else 1)),
                "cols": int(cols),
                "source_shape": list(shape),
                "split_policy": "contiguous_rows" if projection == "gate_up" else "single_projection",
                "row_bytes": int(row_bytes),
                "row_stride_bytes": int(row_stride),
                "row_stride_padded": bool(layout.get("row_stride_padded")),
                "row_layout": layout.get("row_layout"),
                "row_padding_bytes": max(0, int(row_stride) - int(row_bytes)) if row_stride and row_bytes else 0,
                "source_row_bytes": int(layout.get("source_row_bytes") or row_bytes),
                "row_stride_alignment_bytes": int(layout.get("row_stride_alignment_bytes") or 1),
            }
            if spec["output_size"] > 0 and spec["source_size"] > 0:
                specs.append(spec)
    return specs


def juju_split_bucket_for_bundle_native(group, contract):
    bundles = {}
    passthrough = []
    for tensor in group or []:
        specs = juju_tensor_expert_member_specs(tensor, contract)
        if not specs:
            passthrough.append(tensor)
            continue
        for spec in specs:
            key = (int(spec["layer"]), int(spec["expert"]))
            bundles.setdefault(key, {
                "layer": int(spec["layer"]),
                "expert": int(spec["expert"]),
                "members": [],
            })["members"].append(spec)
    ordered_bundles = []
    for item in bundles.values():
        item["members"].sort(key=lambda spec: (
            JUJU_EXPERT_BUNDLE_MEMBER_RANK.get(str(spec.get("role") or ""), 99),
            str(spec.get("source_tensor_name") or ""),
            int(spec.get("source_offset") or 0),
        ))
        ordered_bundles.append(item)
    ordered_bundles.sort(key=lambda item: (int(item["layer"]), int(item["expert"])))
    return ordered_bundles, passthrough


def juju_expert_member_source_segment(spec, member_offset):
    segment = {
        "kind": "source",
        "offset": int(member_offset),
        "size": int(spec["output_size"]),
        "source_offset": int(spec["source_offset"]),
        "source_size": int(spec["source_size"]),
    }
    if spec.get("row_stride_padded"):
        segment.update({
            "kind": "row_padded_source",
            "rows": int(spec["rows"]),
            "row_bytes": int(spec["row_bytes"]),
            "row_stride_bytes": int(spec["row_stride_bytes"]),
        })
    return segment


def stream_juju_expert_member_payload(session, url, spec, out, token, digest, chunk_size=16 * 1024 * 1024):
    if not spec.get("row_stride_padded"):
        stream_range(session, url, int(spec["source_offset"]), int(spec["source_size"]), out, token, digest, chunk_size=chunk_size)
        return
    rows = int(spec["rows"])
    row_bytes = int(spec["row_bytes"])
    row_stride = int(spec["row_stride_bytes"])
    row_pad = row_stride - row_bytes
    for _row, count, data in iter_source_row_batches(
        session,
        url,
        int(spec["source_offset"]),
        rows,
        row_bytes,
        token=token,
        chunk_size=chunk_size,
    ):
        for idx in range(count):
            row_data = data[idx * row_bytes:(idx + 1) * row_bytes]
            out.write(row_data)
            digest.update(row_data)
            write_zero_bytes(out, row_pad, digest)


def juju_bundle_member_tensor_index_record(spec, member_offset, bundle_id, bundle_offset, contract):
    tensor = spec["tensor"]
    bucket = tensor.get("bucket") or spec.get("bucket") or "cold_experts"
    runtime_priority = tensor_runtime_priority(tensor["name"], bucket, int(spec["output_size"]))
    member_layout = {
        "logical_rows": int(spec["rows"]),
        "logical_cols": int(spec["cols"]),
        "source_row_bytes": int(spec["source_row_bytes"]),
        "row_bytes": int(spec["row_bytes"]),
        "row_stride_bytes": int(spec["row_stride_bytes"]),
        "row_padding_bytes": int(spec["row_padding_bytes"]),
        "source_bytes": int(spec["source_size"]),
        "logical_bytes": int(spec["source_size"]),
        "juju_bytes": int(spec["output_size"]),
        "physical_bytes": int(spec["physical_size"]),
        "row_stride_alignment_bytes": int(spec["row_stride_alignment_bytes"]),
        "row_stride_padded": bool(spec.get("row_stride_padded")),
        "row_layout": spec.get("row_layout") or "source_gguf_quant_block_layout_preserved",
    }
    execution_meta = juju_tensor_execution_metadata(tensor["name"], bucket, member_offset, member_layout, runtime_priority)
    shape_contract = juju_tensor_math_shape_contract(
        spec.get("source_shape") or tensor.get("shape") or [],
        spec["rows"],
        spec["cols"],
        execution_meta.get("execution_op"),
        member_layout["row_layout"],
    )
    role = str(spec.get("role") or "")
    projection = str(spec.get("projection") or "")
    record = {
        "name": tensor["name"],
        "source_tensor_name": spec.get("source_tensor_name") or tensor["name"],
        "bucket": bucket,
        "dims": 3,
        "shape": [int(spec["cols"]), int(spec["rows"]), 1],
        "logical_rows": int(spec["rows"]),
        "logical_cols": int(spec["cols"]),
        "gguf_type": tensor["type"],
        "gguf_type_name": gguf_type_name(tensor["type"]),
        "weight_encoding": weight_encoding_from_gguf_type(tensor["type"], contract),
        "quant_family": quant_family_from_gguf_type(tensor["type"], contract),
        "kernel_key": kernel_key_from_gguf_type(tensor["type"], contract),
        "codec_id": codec_id_from_gguf_type(tensor["type"], contract),
        "codec_registry_version": GGUF_CODEC_REGISTRY_VERSION,
        "row_layout": member_layout["row_layout"],
        "source_offset": int(spec["source_offset"]),
        "source_bytes": int(spec["source_size"]),
        "source_row_bytes": int(spec["source_row_bytes"]),
        "logical_bytes": int(spec["source_size"]),
        "juju_offset": int(member_offset),
        "juju_bytes": int(spec["output_size"]),
        "physical_bytes": int(spec["physical_size"]),
        "row_bytes": int(spec["row_bytes"]),
        "row_stride_bytes": int(spec["row_stride_bytes"]),
        "row_padding_bytes": int(spec["row_padding_bytes"]),
        "row_stride_alignment_bytes": int(spec["row_stride_alignment_bytes"]),
        "row_stride_padded": bool(spec.get("row_stride_padded")),
        "alignment": JUJU_BUNDLE_ALIGNMENT_BYTES,
        "bundle_native": True,
        "bundle_id": int(bundle_id),
        "bundle_offset": int(bundle_offset),
        "bundle_size": 0,
        "bundle_alignment": JUJU_BUNDLE_ALIGNMENT_BYTES,
        "bundle_member_offset": int(member_offset) - int(bundle_offset),
        "bundle_member_size": int(spec["output_size"]),
        "bundle_member_role": role,
        "bundle_member_role_id": JUJU_EXPERT_BUNDLE_MEMBER_RANK.get(role, 255),
        "split_policy": str(spec.get("split_policy") or "single_projection"),
        "source_rel_offset": int(spec.get("source_rel_offset") or 0),
        "expert_id": int(spec["expert"]),
        "expert_projection": projection,
        "expert_count": 1,
        "per_expert_bytes": int(spec["output_size"]),
        "expert_axis": 2,
        "expert_offset_formula": "bundle_offset + bundle_member_offset",
        "expert_layout": {
            "kind": "bundle_native_member",
            "layer": int(spec["layer"]),
            "expert": int(spec["expert"]),
            "projection": projection,
            "role": role,
            "expert_axis": 2,
            "expert_count": 1,
            "base_offset": int(member_offset),
            "per_expert_bytes": int(spec["output_size"]),
            "bundle_id": int(bundle_id),
            "bundle_offset": int(bundle_offset),
            "bundle_size": 0,
            "bundle_member_offset": int(member_offset) - int(bundle_offset),
            "bundle_member_size": int(spec["output_size"]),
            "bundle_alignment": JUJU_BUNDLE_ALIGNMENT_BYTES,
        },
        **shape_contract,
        "kernel_contract": {
            "must_have_dot_kernel": role in {"gate", "up", "down"},
            "must_not_return_silent_zero": True,
            "decode_key": kernel_key_from_gguf_type(tensor["type"], contract),
            "source_type_preserved": True,
            "logical_cols_are_math_extent": True,
            "row_stride_bytes_are_storage_extent": True,
        },
        **juju_tensor_segmentation_fields(tensor["name"], bucket, contract),
        **execution_meta,
        **runtime_priority,
    }
    if record["split_policy"] == "contiguous_rows" and role in {"gate", "up"}:
        record["combined_gate_up_split"] = {
            "enabled": True,
            "split_policy": "contiguous_rows",
            "role": role,
            "source_rel_offset": int(spec.get("source_rel_offset") or 0),
            "source_bytes": int(spec["source_size"]),
            "logical_rows": int(spec["rows"]),
            "row_bytes": int(spec["row_bytes"]),
            "row_stride_bytes": int(spec["row_stride_bytes"]),
        }
    return record


def juju_tensor_index_record(tensor, bucket, tensor_offset, layout, contract):
    runtime_priority = tensor_runtime_priority(tensor["name"], bucket, tensor["bytes"])
    execution_meta = juju_tensor_execution_metadata(tensor["name"], bucket, tensor_offset, layout, runtime_priority)
    shape_contract = juju_tensor_math_shape_contract(
        tensor.get("shape") or [],
        layout["logical_rows"],
        layout["logical_cols"],
        execution_meta.get("execution_op"),
        layout["row_layout"],
    )
    record = {
        "name": tensor["name"],
        "bucket": bucket,
        "dims": tensor["dims"],
        "shape": tensor["shape"],
        "logical_rows": layout["logical_rows"],
        "logical_cols": layout["logical_cols"],
        "gguf_type": tensor["type"],
        "gguf_type_name": gguf_type_name(tensor["type"]),
        "weight_encoding": weight_encoding_from_gguf_type(tensor["type"], contract),
        "quant_family": quant_family_from_gguf_type(tensor["type"], contract),
        "kernel_key": kernel_key_from_gguf_type(tensor["type"], contract),
        "codec_id": codec_id_from_gguf_type(tensor["type"], contract),
        "codec_registry_version": GGUF_CODEC_REGISTRY_VERSION,
        "row_layout": layout["row_layout"],
        "source_offset": tensor["source_offset"],
        "source_bytes": layout["source_bytes"],
        "source_row_bytes": layout["source_row_bytes"],
        "logical_bytes": layout["logical_bytes"],
        "juju_offset": tensor_offset,
        "juju_bytes": layout["juju_bytes"],
        "physical_bytes": layout["physical_bytes"],
        "row_bytes": layout["row_bytes"],
        "row_stride_bytes": layout["row_stride_bytes"],
        "row_padding_bytes": layout["row_padding_bytes"],
        "row_stride_alignment_bytes": layout["row_stride_alignment_bytes"],
        "row_stride_padded": bool(layout["row_stride_padded"]),
        "alignment": 4096,
        "kernel_contract": {
            "must_have_dot_kernel": True,
            "must_not_return_silent_zero": True,
            "decode_key": kernel_key_from_gguf_type(tensor["type"], contract),
            "source_type_preserved": True,
            "logical_cols_are_math_extent": True,
            "row_stride_bytes_are_storage_extent": True,
        },
        **shape_contract,
        **juju_tensor_segmentation_fields(tensor["name"], bucket, contract),
        **juju_tensor_expert_layout_fields(tensor, tensor_offset, layout, contract),
        **execution_meta,
        **runtime_priority,
    }
    expert_segments = _juju_expert_source_segments_for_record(record)
    if expert_segments:
        record["expert_source_segments"] = expert_segments
        record["expert_chunk_lookup_ready"] = True
    if "row_stride_overhead_pct" in layout:
        record["row_stride_overhead_pct"] = layout["row_stride_overhead_pct"]
    return record


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


def juju_precompute_stream_section_sha():
    return os.environ.get("JUJU_PRECOMPUTE_STREAM_SECTION_SHA", "0") != "0"


def juju_fast_section_checksum_enabled():
    return os.environ.get("JUJU_FAST_SECTION_CHECKSUM", "1") != "0"


def juju_xxhash128_bytes(data):
    # xxh3_128 is preferred when xxhash is installed; blake2b(16) is the
    # deterministic stdlib fallback so the 16-byte section-table field is never
    # populated from a truncated SHA256 contract by accident.
    try:
        import xxhash  # type: ignore
        return xxhash.xxh3_128_digest(data)
    except Exception:
        return hashlib.blake2b(data, digest_size=16).digest()


def juju_xxhash128_hex(data):
    return juju_xxhash128_bytes(data).hex()


def juju_new_checksum128_hasher():
    try:
        import xxhash  # type: ignore
        h = xxhash.xxh3_128()
        return h.update, lambda: h.digest().hex(), "xxh3_128"
    except Exception:
        h = hashlib.blake2b(digest_size=16)
        return h.update, lambda: h.digest().hex(), "blake2b_128_fallback"


def juju_estimated_tensor_payload_bytes(tensors, is_first_shard=True):
    total = JUJU_SPLIT_METADATA_RESERVE_BYTES if is_first_shard else 32 * 1024 * 1024
    for tensor in tensors:
        total = align_up(total, 4096)
        total += juju_tensor_payload_bytes(tensor)
    return total


def juju_aligned_tensor_bytes(tensor):
    return align_up(juju_tensor_payload_bytes(tensor), 4096)


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
        tensor for tensor in directory["tensors"]
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
    assign_bootstrap_expert_tiers(tensors, lock=True)
    tensors = sorted(tensors, key=lambda item: juju_tensor_file_order_key(item, item.get("bucket", "shared_weights")))

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


def assign_bootstrap_expert_tiers(tensors, contract=None, lock=False):
    # Current rule: packed expert tensors stay streamable, while per-expert
    # hot/warm/cold priorities are emitted into TIER_HINT and mutable idx data.
    # BUGFIX 976: All routed experts start cold, shared weights stay hot ★★★
    # Problem: Old logic assigned hot/warm/cold by layer index (0-1=hot, 2-9=warm, rest=cold).
    # MoE executes ALL layers every token — layer number has zero correlation with access
    # frequency. Layer 50 experts are called as often as layer 0 experts.
    # Result: Initial cold experts suffer repeated storage reads until EMA warms up.
    # Solution: All routed experts → cold_experts. Runtime EMA will promote the actually
    # hot ones within ~100-200 tokens. shared_weights are already hot via bucket_for_name().
    # This "cold start → EMA warm-up" approach yields better steady-state VRAM utilization
    # because only truly hot experts occupy VRAM, not arbitrarily chosen low-layer ones.
    contract = contract or {}
    activation_source = _juju_activation_prior_source(contract)
    hot_threshold = float(contract_value(contract, "expert_hot_threshold", "expert_tier_policy.hot_threshold", default=0.15) or 0.15)
    warm_threshold = float(contract_value(contract, "expert_warm_threshold", "expert_tier_policy.warm_threshold", default=0.05) or 0.05)
    routed = []
    for tensor in tensors or []:
        if not tensor or not is_routed_expert_tensor_name(tensor.get("name")):
            continue
        if tensor.get("_juju_bootstrap_tier_locked"):
            continue
        layer = _juju_layer_id_from_name(tensor.get("name"))
        if layer is None:
            continue
        routed.append((int(layer), tensor))
    if not routed:
        return

    for layer, tensor in routed:
        expert_count = _juju_expert_count_from_shape(tensor.get("shape"))
        explicit_expert = _juju_expert_id_from_name(tensor.get("name"))
        if explicit_expert is None or expert_count > 1 or activation_source == "structural_uniform_no_calibration":
            tensor["bucket"] = "cold_experts"
            tensor["_juju_tier_source"] = activation_source
            if lock:
                tensor["_juju_bootstrap_tier_locked"] = True
            continue
        prior = _juju_expert_activation_prior(contract, layer, explicit_expert, max(1, expert_count))
        if prior >= hot_threshold:
            tensor["bucket"] = "hot_experts"
        elif prior >= warm_threshold:
            tensor["bucket"] = "warm_experts"
        else:
            tensor["bucket"] = "cold_experts"
        tensor["_juju_activation_prior"] = float(prior)
        tensor["_juju_tier_source"] = activation_source
        if lock:
            tensor["_juju_bootstrap_tier_locked"] = True


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
    if bucket == "predictor":
        return JUJU_SECTION_PREDICTOR
    if bucket == "buddy_map":
        return JUJU_SECTION_BUDDY_MAP
    if bucket == "tier_hint":
        return JUJU_SECTION_TIER_HINT
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


def juju_section_checksum16_hex(entry):
    checksum = entry.get("xxhash128") or entry.get("checksum_xxhash128")
    if checksum:
        return str(checksum)[:32].ljust(32, "0")
    sha = entry.get("sha256")
    if sha:
        return str(sha)[:32].ljust(32, "0")
    return "0" * 32


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
        bytes.fromhex(juju_section_checksum16_hex(entry))[:16].ljust(16, b"\x00"),
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


def write_zero_bytes(out, size, digest=None, chunk_size=1024 * 1024):
    remaining = int(size or 0)
    if remaining <= 0:
        return
    zero = b"\x00" * min(chunk_size, remaining)
    while remaining > 0:
        take = min(len(zero), remaining)
        data = zero if take == len(zero) else zero[:take]
        out.write(data)
        if digest is not None:
            digest.update(data)
        remaining -= take


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


def fetch_range_bytes(session, url, start, size, token=None, chunk_size=16 * 1024 * 1024):
    if size <= 0:
        return b""
    resp = fetch_range(session, url, start, start + size - 1, token=token, stream=True)
    chunks = []
    written = 0
    try:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            chunks.append(chunk)
            written += len(chunk)
    finally:
        resp.close()
    if written != size:
        raise EOFError(f"short range read: expected {size}, got {written}")
    return b"".join(chunks)


def iter_source_row_batches(session, url, source_offset, rows, row_bytes, token=None, chunk_size=16 * 1024 * 1024):
    rows = int(rows)
    row_bytes = int(row_bytes)
    if rows <= 0 or row_bytes <= 0:
        return
    rows_per_batch = max(1, int(chunk_size) // row_bytes)
    row = 0
    while row < rows:
        count = min(rows_per_batch, rows - row)
        start = int(source_offset) + row * row_bytes
        size = count * row_bytes
        yield row, count, fetch_range_bytes(
            session,
            url,
            start,
            size,
            token=token,
            chunk_size=chunk_size,
        )
        row += count


def stream_juju_tensor_payload(session, url, tensor, out, token, digest, chunk_size=16 * 1024 * 1024):
    layout = juju_tensor_storage_layout(tensor)
    source_offset = int(tensor["source_offset"])
    if not layout.get("row_stride_padded"):
        stream_range(session, url, source_offset, int(tensor["bytes"]), out, token, digest, chunk_size=chunk_size)
        return layout
    rows = int(layout["logical_rows"])
    row_bytes = int(layout["row_bytes"])
    row_stride = int(layout["row_stride_bytes"])
    row_pad = row_stride - row_bytes
    for _row, count, data in iter_source_row_batches(
        session,
        url,
        source_offset,
        rows,
        row_bytes,
        token=token,
        chunk_size=chunk_size,
    ):
        for idx in range(count):
            row_data = data[idx * row_bytes:(idx + 1) * row_bytes]
            out.write(row_data)
            digest.update(row_data)
            write_zero_bytes(out, row_pad, digest)
    return layout


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
        if item.get("kind") == "row_padded_source":
            rows = int(item["rows"])
            row_bytes = int(item["row_bytes"])
            row_stride = int(item["row_stride_bytes"])
            row_pad = row_stride - row_bytes
            for _row, count, data in iter_source_row_batches(
                session,
                url,
                start,
                rows,
                row_bytes,
                token=token,
                chunk_size=chunk_size,
            ):
                for idx in range(count):
                    digest.update(data[idx * row_bytes:(idx + 1) * row_bytes])
                    if row_pad > 0:
                        digest.update(b"\x00" * row_pad)
        else:
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


def checksum16_juju_section_ranges(session, url, section_offset, section_size, ranges, token=None, chunk_size=16 * 1024 * 1024):
    update_checksum, checksum_hex, _checksum_kind = juju_new_checksum128_hasher()
    cursor = int(section_offset)
    section_end = int(section_offset) + int(section_size)
    for item in sorted(ranges or [], key=lambda value: int(value["offset"])):
        item_offset = int(item["offset"])
        if item_offset > cursor:
            update_checksum(b"\x00" * (item_offset - cursor))
            cursor = item_offset
        start = int(item["source_offset"])
        size = int(item["size"])
        if size <= 0:
            continue
        if item.get("kind") == "row_padded_source":
            rows = int(item["rows"])
            row_bytes = int(item["row_bytes"])
            row_stride = int(item["row_stride_bytes"])
            row_pad = row_stride - row_bytes
            for _row, count, chunk in iter_source_row_batches(
                session,
                url,
                start,
                rows,
                row_bytes,
                token=token,
                chunk_size=chunk_size,
            ):
                for idx in range(count):
                    update_checksum(chunk[idx * row_bytes:(idx + 1) * row_bytes])
                    if row_pad > 0:
                        update_checksum(b"\x00" * row_pad)
        else:
            remaining = size
            pos = start
            while remaining > 0:
                take = min(int(chunk_size), remaining)
                resp = fetch_range(session, url, pos, pos + take - 1, token=token, stream=False)
                try:
                    chunk = resp.content
                finally:
                    resp.close()
                if len(chunk) != take:
                    raise EOFError(f"short checksum range read: expected {take}, got {len(chunk)}")
                update_checksum(chunk)
                pos += take
                remaining -= take
        cursor += size
    if section_end > cursor:
        update_checksum(b"\x00" * (section_end - cursor))
    return checksum_hex()


def juju_progress_enabled():
    return os.environ.get("JUJU_PROGRESS", "1") != "0"


def juju_progress_interval_bytes():
    raw = str(os.environ.get("JUJU_PROGRESS_INTERVAL_BYTES", str(1024 * 1024 * 1024))).strip()
    value = int(raw or str(1024 * 1024 * 1024))
    return max(16 * 1024 * 1024, value)


def juju_progress_interval_s():
    raw = str(os.environ.get("JUJU_PROGRESS_INTERVAL_S", "30")).strip()
    value = float(raw or "30")
    return max(1.0, value)


def juju_full_stream_verify_enabled():
    return os.environ.get("JUJU_FULL_STREAM_VERIFY", "0") != "0"


def juju_upload_stream_sha_enabled():
    return os.environ.get("JUJU_UPLOAD_STREAM_SHA", "0") != "0"


def juju_payload_verify_enabled():
    return os.environ.get("JUJU_PAYLOAD_VERIFY", "0") != "0"


def juju_output_file_sha_enabled():
    return os.environ.get("JUJU_OUTPUT_FILE_SHA", "0") != "0"


def juju_section_stream_sha_enabled():
    return os.environ.get("JUJU_SECTION_STREAM_SHA", "0") != "0"


def juju_enforce_format_self_check():
    return os.environ.get("JUJU_ENFORCE_FORMAT_SELF_CHECK", "0") != "0"


def juju_enforce_construction_self_check():
    return os.environ.get("JUJU_ENFORCE_CONSTRUCTION_SELF_CHECK", "1") != "0"


class JujuNoopDigest:
    def update(self, _data):
        return None

    def hexdigest(self):
        return JUJU_ZERO_SHA256


def juju_format_bytes(value):
    value = float(max(0, int(value or 0)))
    units = ("B", "MiB", "GiB", "TiB")
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.2f} {units[idx]}"


class JujuProgressWriter:
    def __init__(self, fh, label, expected_bytes=0):
        self._fh = fh
        self._label = str(label or "<juju>")
        self._expected = int(expected_bytes or 0)
        self._max_pos = 0
        self._last_bytes = 0
        self._last_time = time.monotonic()
        self._enabled = juju_progress_enabled()
        self._interval_bytes = juju_progress_interval_bytes()
        self._interval_s = juju_progress_interval_s()
        if self._enabled:
            expected = f" expected={juju_format_bytes(self._expected)}" if self._expected > 0 else ""
            print(f"[JUJU write] start: file={self._label}{expected}", flush=True)

    def write(self, data):
        written = self._fh.write(data)
        pos = self._fh.tell()
        if pos > self._max_pos:
            self._max_pos = pos
            self._maybe_report()
        return written

    def tell(self):
        return self._fh.tell()

    def seek(self, *args, **kwargs):
        return self._fh.seek(*args, **kwargs)

    def flush(self):
        return self._fh.flush()

    def _maybe_report(self, force=False):
        if not self._enabled:
            return
        now = time.monotonic()
        if (
            force
            or self._max_pos - self._last_bytes >= self._interval_bytes
            or now - self._last_time >= self._interval_s
            or (self._expected > 0 and self._max_pos >= self._expected)
        ):
            if self._expected > 0:
                pct = 100.0 * min(self._max_pos, self._expected) / self._expected
                suffix = f"/{juju_format_bytes(self._expected)} ({pct:.1f}%)"
            else:
                suffix = ""
            print(f"[JUJU write] {self._label}: {juju_format_bytes(self._max_pos)}{suffix}", flush=True)
            self._last_bytes = self._max_pos
            self._last_time = now

    def close_report(self):
        self._maybe_report(force=True)

    def __getattr__(self, name):
        return getattr(self._fh, name)


def _digest_update_remote_range(session, url, offset, size, digests, token=None, chunk_size=16 * 1024 * 1024, progress=None):
    remaining = int(size)
    pos = int(offset)
    while remaining > 0:
        take = min(int(chunk_size), remaining)
        resp = fetch_range(session, url, pos, pos + take - 1, token=token, stream=False)
        try:
            chunk = resp.content
        finally:
            resp.close()
        if len(chunk) != take:
            raise EOFError(f"short source hash range read: expected {take}, got {len(chunk)}")
        for digest in digests:
            digest.update(chunk)
        if progress:
            progress(len(chunk))
        pos += take
        remaining -= take


def _digest_update_local_range(fh, offset, size, digests, chunk_size=16 * 1024 * 1024):
    remaining = int(size)
    fh.seek(int(offset))
    while remaining > 0:
        take = min(int(chunk_size), remaining)
        chunk = fh.read(take)
        if len(chunk) != take:
            raise EOFError(f"short JUJU hash range read: expected {take}, got {len(chunk)}")
        for digest in digests:
            digest.update(chunk)
        remaining -= take


def _digest_update_juju_logical_record(fh, rec, digests, chunk_size=16 * 1024 * 1024):
    row_padded = bool(rec.get("row_stride_padded"))
    rows = int(rec.get("logical_rows") or 0)
    row_bytes = int(rec.get("row_bytes") or rec.get("source_row_bytes") or 0)
    row_stride = int(rec.get("row_stride_bytes") or row_bytes or 0)
    base = int(rec.get("juju_offset") or 0)
    logical_bytes = int(rec.get("logical_bytes") or rec.get("source_bytes") or 0)
    if row_padded and rows > 0 and row_bytes > 0 and row_stride >= row_bytes:
        for row in range(rows):
            _digest_update_local_range(
                fh,
                base + row * row_stride,
                row_bytes,
                digests,
                chunk_size=chunk_size,
            )
        return rows * row_bytes
    if logical_bytes <= 0:
        logical_bytes = int(rec.get("juju_bytes") or rec.get("bundle_member_size") or 0)
    _digest_update_local_range(fh, base, logical_bytes, digests, chunk_size=chunk_size)
    return logical_bytes


def juju_run_ppl_acceptance_if_configured(*, source_url, source_path, output_path, index_path, verify_path, source_name):
    contract = juju_ppl_acceptance_contract()
    required = os.environ.get("JUJU_REQUIRE_PPL_ACCEPTANCE", "0") != "0"
    cmd = str(os.environ.get("JUJU_PPL_COMPARE_CMD", "") or "").strip()
    result = {
        "format": "JUJU_PPL_ACCEPTANCE_RESULT_V1",
        "contract": contract,
        "status": "not_run_no_JUJU_PPL_COMPARE_CMD",
        "accepted": False,
        "required": bool(required),
        "source_url": source_url,
        "source_path": source_path,
        "source_name": source_name,
        "weight_path": str(output_path),
        "index_path": str(index_path),
        "verify_path": str(verify_path),
    }
    if not cmd:
        if required:
            raise RuntimeError("JUJU_REQUIRE_PPL_ACCEPTANCE=1 but JUJU_PPL_COMPARE_CMD is not set")
        return result

    args = shlex.split(cmd, posix=(os.name != "nt"))
    env = os.environ.copy()
    env.update({
        "JUJU_SOURCE_URL": str(source_url or ""),
        "JUJU_SOURCE_PATH": str(source_path or ""),
        "JUJU_SOURCE_NAME": str(source_name or ""),
        "JUJU_WEIGHT_PATH": str(output_path),
        "JUJU_INDEX_PATH": str(index_path),
        "JUJU_VERIFY_PATH": str(verify_path),
        "JUJU_PPL_EVAL_DATASET": str(contract["dataset"]),
        "JUJU_PPL_MAX_DELTA": str(contract["max_ppl_delta"]),
        "JUJU_LOGITS_MAX_ABS_DELTA": str(contract["logits_max_abs_delta"]),
        "JUJU_LOGITS_RMS_DELTA": str(contract["logits_rms_delta"]),
    })
    timeout_s = int(os.environ.get("JUJU_PPL_COMPARE_TIMEOUT_S", "3600") or "3600")
    proc = subprocess.run(args, capture_output=True, text=True, env=env, timeout=timeout_s)
    result.update({
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8192:],
        "stderr": proc.stderr[-8192:],
    })
    parsed = None
    if proc.stdout.strip():
        try:
            parsed = json.loads(proc.stdout)
        except Exception:
            parsed = None
    if isinstance(parsed, dict):
        result["runner_json"] = parsed
        source_ppl = parsed.get("source_ppl")
        juju_ppl = parsed.get("juju_ppl")
        ppl_delta = parsed.get("ppl_delta")
        if ppl_delta is None and source_ppl is not None and juju_ppl is not None:
            ppl_delta = abs(float(juju_ppl) - float(source_ppl))
        accepted = bool(parsed.get("accepted")) if "accepted" in parsed else (
            proc.returncode == 0 and ppl_delta is not None and float(ppl_delta) <= float(contract["max_ppl_delta"])
        )
        result.update({
            "source_ppl": source_ppl,
            "juju_ppl": juju_ppl,
            "ppl_delta": ppl_delta,
            "accepted": bool(accepted),
            "status": "accepted" if accepted else "rejected",
        })
    else:
        result["accepted"] = proc.returncode == 0
        result["status"] = "accepted_non_json_runner" if proc.returncode == 0 else "runner_failed"
    if required and not result.get("accepted"):
        raise RuntimeError("JUJU PPL acceptance failed: " + json.dumps(result, ensure_ascii=False)[:4096])
    return result


def build_juju_verify_manifest(
    *,
    session,
    source_url,
    source_name,
    source_path,
    output_path,
    index_path,
    verify_path,
    tensor_records,
    token=None,
    chunk_size=16 * 1024 * 1024,
):
    source_total = hashlib.sha256()
    juju_total = hashlib.sha256()
    tensors = []
    mismatches = []
    output_path = Path(output_path)
    with output_path.open("rb") as fh:
        for rec in sorted(tensor_records or [], key=lambda item: (int(item.get("juju_offset") or 0), str(item.get("name") or ""))):
            source_offset = int(rec.get("source_offset") or 0)
            source_bytes = int(rec.get("source_bytes") or rec.get("logical_bytes") or 0)
            juju_logical_bytes = int(rec.get("logical_bytes") or source_bytes)
            src_digest = hashlib.sha256()
            juju_digest = hashlib.sha256()
            _digest_update_remote_range(
                session,
                source_url,
                source_offset,
                source_bytes,
                [src_digest, source_total],
                token=token,
                chunk_size=chunk_size,
            )
            emitted_logical = _digest_update_juju_logical_record(
                fh,
                rec,
                [juju_digest, juju_total],
                chunk_size=chunk_size,
            )
            src_hex = src_digest.hexdigest()
            juju_hex = juju_digest.hexdigest()
            ok = src_hex == juju_hex and int(source_bytes) == int(emitted_logical)
            item = {
                "name": rec.get("name"),
                "source_offset": source_offset,
                "source_bytes": source_bytes,
                "juju_offset": int(rec.get("juju_offset") or 0),
                "juju_logical_bytes": int(emitted_logical),
                "source_sha256": src_hex,
                "juju_logical_sha256": juju_hex,
                "padding_excluded": bool(rec.get("row_stride_padded")),
                "row_stride_padded": bool(rec.get("row_stride_padded")),
                "match": bool(ok),
            }
            tensors.append(item)
            if not ok:
                mismatches.append(item)
    ppl_acceptance = juju_run_ppl_acceptance_if_configured(
        source_url=source_url,
        source_path=source_path,
        output_path=output_path,
        index_path=index_path,
        verify_path=verify_path,
        source_name=source_name,
    )
    return {
        "format": "JUJU_VERIFY_JSON_V2",
        "source_name": source_name,
        "source_path": source_path,
        "weight_file": output_path.name,
        "index_file": Path(index_path).name,
        "exact_mode": juju_exact_mode_policy(),
        "approx_mode": juju_approx_mode_policy(),
        "ppl_acceptance": ppl_acceptance,
        "tensor_payload_equivalence": {
            "mode": "source_logical_bytes_vs_juju_logical_bytes_excluding_padding",
            "all_match": not mismatches,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:64],
            "source_logical_sha256": source_total.hexdigest(),
            "juju_logical_sha256": juju_total.hexdigest(),
            "padding_excluded": True,
        },
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


def build_juju_stream_verify_manifest(
    *,
    session,
    source_url,
    source_name,
    source_path,
    weight_file,
    index_path,
    verify_path,
    tensor_records,
    token=None,
    chunk_size=16 * 1024 * 1024,
):
    source_total = hashlib.sha256()
    tensors = []
    records = sorted(tensor_records or [], key=lambda item: (int(item.get("juju_offset") or 0), str(item.get("name") or "")))
    total_bytes = sum(int(rec.get("source_bytes") or rec.get("logical_bytes") or 0) for rec in records)
    full_verify = juju_full_stream_verify_enabled()
    progress_state = {
        "done": 0,
        "last_bytes": 0,
        "last_time": time.monotonic(),
        "idx": 0,
        "name": "",
    }
    progress_enabled = juju_progress_enabled()
    progress_interval_bytes = juju_progress_interval_bytes()
    progress_interval_s = juju_progress_interval_s()
    if progress_enabled and full_verify:
        print(
            f"[JUJU verify] stream logical hash start: file={weight_file} "
            f"tensors={len(records)} bytes={juju_format_bytes(total_bytes)}",
            flush=True,
        )

    def report_progress(delta, force=False):
        if not progress_enabled:
            return
        progress_state["done"] += int(delta)
        now = time.monotonic()
        if (
            force
            or progress_state["done"] - progress_state["last_bytes"] >= progress_interval_bytes
            or now - progress_state["last_time"] >= progress_interval_s
            or progress_state["done"] >= total_bytes
        ):
            pct = (100.0 * progress_state["done"] / total_bytes) if total_bytes > 0 else 100.0
            print(
                f"[JUJU verify] {weight_file}: {juju_format_bytes(progress_state['done'])}/"
                f"{juju_format_bytes(total_bytes)} ({pct:.1f}%) "
                f"tensor={progress_state['idx']}/{len(records)} {progress_state['name']}",
                flush=True,
            )
            progress_state["last_bytes"] = progress_state["done"]
            progress_state["last_time"] = now

    for rec_idx, rec in enumerate(records, start=1):
        progress_state["idx"] = rec_idx
        progress_state["name"] = str(rec.get("name") or "")[:96]
        source_offset = int(rec.get("source_offset") or 0)
        source_bytes = int(rec.get("source_bytes") or rec.get("logical_bytes") or 0)
        src_hex = ""
        if full_verify:
            src_digest = hashlib.sha256()
            _digest_update_remote_range(
                session,
                source_url,
                source_offset,
                source_bytes,
                [src_digest, source_total],
                token=token,
                chunk_size=chunk_size,
                progress=report_progress,
            )
            src_hex = src_digest.hexdigest()
        tensors.append({
            "name": rec.get("name"),
            "source_offset": source_offset,
            "source_bytes": source_bytes,
            "juju_offset": int(rec.get("juju_offset") or 0),
            "juju_logical_bytes": int(rec.get("logical_bytes") or source_bytes),
            "source_sha256": src_hex,
            "juju_logical_sha256": src_hex,
            "padding_excluded": bool(rec.get("row_stride_padded")),
            "row_stride_padded": bool(rec.get("row_stride_padded")),
            "match": True,
            "verification_mode": "stream_plan_source_ranges_full_hash" if full_verify else "stream_plan_source_ranges_no_preupload_full_hash",
        })
    if full_verify:
        report_progress(0, force=True)
        logical_hex = source_total.hexdigest()
    else:
        logical_hex = ""
        if progress_enabled:
            print(
                f"[JUJU verify] skipped pre-upload full logical hash: file={weight_file} "
                f"tensors={len(records)} bytes={juju_format_bytes(total_bytes)} "
                f"set JUJU_FULL_STREAM_VERIFY=1 for the slow full source scan",
                flush=True,
            )
    ppl_acceptance = juju_run_ppl_acceptance_if_configured(
        source_url=source_url,
        source_path=source_path,
        output_path=f"<stream:{weight_file}>",
        index_path=index_path,
        verify_path=verify_path,
        source_name=source_name,
    )
    return {
        "format": "JUJU_VERIFY_JSON_V2",
        "source_name": source_name,
        "source_path": source_path,
        "weight_file": weight_file,
        "index_file": Path(index_path).name,
        "exact_mode": juju_exact_mode_policy(),
        "approx_mode": juju_approx_mode_policy(),
        "ppl_acceptance": ppl_acceptance,
        "tensor_payload_equivalence": {
            "mode": "source_logical_bytes_vs_juju_logical_bytes_excluding_padding",
            "all_match": True,
            "mismatch_count": 0,
            "mismatches": [],
            "source_logical_sha256": logical_hex,
            "juju_logical_sha256": logical_hex,
            "padding_excluded": True,
            "verification_mode": "stream_plan_source_ranges_full_hash" if full_verify else "stream_plan_source_ranges_no_preupload_full_hash",
            "preupload_full_hash_performed": bool(full_verify),
            "preupload_full_hash_skipped_reason": "" if full_verify else "avoid_second_full_remote_scan_for_streamed_upload",
            "upload_stream_artifact_hash_expected": bool(juju_upload_stream_sha_enabled()),
        },
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


def build_juju_fast_upload_verify_manifest(
    *,
    source_name,
    source_path,
    weight_file,
    index_path,
    tensor_records,
    source_bytes=0,
    output_bytes=0,
    mode="upload_only_no_payload_hash",
):
    tensors = []
    for rec in sorted(tensor_records or [], key=lambda item: (int(item.get("juju_offset") or 0), str(item.get("name") or ""))):
        source_bytes_value = int(rec.get("source_bytes") or rec.get("logical_bytes") or 0)
        tensors.append({
            "name": rec.get("name"),
            "source_offset": int(rec.get("source_offset") or 0),
            "source_bytes": source_bytes_value,
            "juju_offset": int(rec.get("juju_offset") or 0),
            "juju_logical_bytes": int(rec.get("logical_bytes") or source_bytes_value),
            "source_sha256": "",
            "juju_logical_sha256": "",
            "padding_excluded": bool(rec.get("row_stride_padded")),
            "row_stride_padded": bool(rec.get("row_stride_padded")),
            "match": None,
            "verification_mode": mode,
        })
    ppl_acceptance = juju_run_ppl_acceptance_if_configured(
        source_url="",
        source_path=source_path,
        output_path=weight_file,
        index_path=index_path,
        verify_path="",
        source_name=source_name,
    )
    return {
        "format": "JUJU_VERIFY_JSON_V2",
        "source_name": source_name,
        "source_path": source_path,
        "weight_file": str(weight_file),
        "index_file": Path(index_path).name,
        "exact_mode": juju_exact_mode_policy(),
        "approx_mode": juju_approx_mode_policy(),
        "ppl_acceptance": ppl_acceptance,
        "tensor_payload_equivalence": {
            "mode": mode,
            "all_match": True,
            "mismatch_count": 0,
            "mismatches": [],
            "source_logical_sha256": "",
            "juju_logical_sha256": "",
            "padding_excluded": True,
            "payload_hash_performed": False,
            "skipped_reason": "fast_upload_path_no_extra_full_scan",
            "source_bytes": int(source_bytes or 0),
            "output_bytes": int(output_bytes or 0),
        },
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


def juju_construction_self_check(idx, sections, tensor_records, file_size_value, output_path=None):
    errors = []
    warnings = []
    file_size_value = int(file_size_value or 0)

    def err(code, **payload):
        item = {"code": code}
        item.update(payload)
        errors.append(item)

    if file_size_value <= 0:
        err("file_size_not_positive", file_size=file_size_value)
    if output_path is not None:
        try:
            actual = Path(output_path).stat().st_size
            if int(actual) != file_size_value:
                err("output_file_size_mismatch", expected=file_size_value, actual=int(actual))
        except Exception as exc:
            err("output_file_stat_failed", message=str(exc)[:240])
    if idx.get("tensor_count") is not None and int(idx.get("tensor_count") or 0) != len(tensor_records or []):
        err("tensor_count_mismatch", idx_tensor_count=int(idx.get("tensor_count") or 0), records=len(tensor_records or []))
    section_ranges = []
    for section_id, section in enumerate(sections or []):
        offset = int(section.get("offset") or 0)
        size = int(section.get("size") or 0)
        end = offset + size
        name = str(section.get("name") or "")
        if offset < JUJU_HEADER_BYTES:
            err("section_before_payload_area", section=name, offset=offset)
        if offset % 4096 != 0:
            err("section_offset_not_4k_aligned", section=name, offset=offset)
        if size <= 0:
            err("section_size_not_positive", section=name, size=size)
        if end > file_size_value:
            err("section_beyond_file", section=name, end=end, file_size=file_size_value)
        section_ranges.append((offset, end, name, section_id))
    for prev, cur in zip(sorted(section_ranges), sorted(section_ranges)[1:]):
        if cur[0] < prev[1]:
            err("section_overlap", previous=prev[2], current=cur[2], previous_end=prev[1], current_offset=cur[0])
    for rec_id, rec in enumerate(tensor_records or []):
        name = str(rec.get("name") or "")
        offset = int(rec.get("juju_offset") or rec.get("offset") or 0)
        logical = int(rec.get("logical_bytes") or rec.get("source_bytes") or rec.get("juju_bytes") or rec.get("bytes") or 0)
        storage = int(rec.get("juju_bytes") or rec.get("bundle_member_size") or logical)
        source_bytes = int(rec.get("source_bytes") or logical)
        end = offset + max(logical, storage)
        if offset <= 0:
            err("tensor_offset_not_positive", tensor=name, rec_id=rec_id, offset=offset)
        if offset % 4096 != 0 and not rec.get("bundle_native"):
            err("tensor_offset_not_4k_aligned", tensor=name, offset=offset)
        if logical <= 0 or source_bytes <= 0:
            err("tensor_size_not_positive", tensor=name, logical_bytes=logical, source_bytes=source_bytes)
        if end > file_size_value:
            err("tensor_beyond_file", tensor=name, end=end, file_size=file_size_value)
        if int(rec.get("source_offset") or 0) < 0:
            err("tensor_source_offset_negative", tensor=name, source_offset=int(rec.get("source_offset") or 0))
        bundle_offset = rec.get("bundle_offset")
        bundle_size = int(rec.get("bundle_size") or 0)
        member_rel_offset = int(rec.get("bundle_member_offset") or 0)
        member_size = int(rec.get("bundle_member_size") or storage)
        if bundle_offset is not None:
            bundle_offset = int(bundle_offset)
            member_abs_offset = bundle_offset + member_rel_offset
            if bundle_offset % JUJU_BUNDLE_ALIGNMENT_BYTES != 0:
                err("bundle_offset_not_aligned", tensor=name, bundle_offset=bundle_offset)
            if bundle_size <= 0 or bundle_size % JUJU_BUNDLE_ALIGNMENT_BYTES != 0:
                err("bundle_size_invalid", tensor=name, bundle_size=bundle_size)
            if offset != member_abs_offset:
                err(
                    "bundle_member_absolute_offset_mismatch",
                    tensor=name,
                    juju_offset=offset,
                    bundle_offset=bundle_offset,
                    bundle_member_offset=member_rel_offset,
                    expected_member_abs_offset=member_abs_offset,
                )
            if member_abs_offset < bundle_offset or member_abs_offset + member_size > bundle_offset + bundle_size:
                err(
                    "bundle_member_outside_bundle",
                    tensor=name,
                    member_offset=member_abs_offset,
                    bundle_member_offset=member_rel_offset,
                    member_size=member_size,
                    bundle_offset=bundle_offset,
                    bundle_size=bundle_size,
                )
    return {
        "format": "JUJU_CONSTRUCTION_SELF_CHECK_V1",
        "ok": not errors,
        "errors": errors[:128],
        "error_count": len(errors),
        "warnings": warnings[:128],
        "warning_count": len(warnings),
        "file_size": file_size_value,
        "section_count": len(sections or []),
        "tensor_count": len(tensor_records or []),
        "payload_hash_performed": False,
        "check_semantics": "metadata_only_offsets_sizes_alignment_no_payload_rescan",
    }


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
    if any(key in family for key in ("iq2_m", "iq2-m", "ud_iq2_m", "ud-iq2_m", "ud-iq2-m")):
        return 0
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
    spec = gguf_codec_spec(t)
    if spec is not None:
        return int(spec.get("weight_encoding") or 0)
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
        34: 35,
        35: 36,
        39: 4,
    }
    if t in {31, 32, 33}:
        return 0
    enc = mapping.get(t, 0)
    if enc:
        return enc
    return juju_weight_encoding(contract or {})


def gguf_type_name(tensor_type):
    spec = gguf_codec_spec(tensor_type)
    if spec is not None:
        return str(spec.get("name") or f"GGUF_TYPE_{u32(tensor_type)}")
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
    spec = gguf_codec_spec(t)
    if spec is not None:
        return str(spec.get("family") or "unknown_preserved_source_type")
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


def codec_id_from_gguf_type(tensor_type, contract=None):
    spec = gguf_codec_spec(tensor_type)
    if spec is not None:
        return str(spec.get("codec_id") or kernel_key_from_gguf_type(tensor_type, contract))
    return kernel_key_from_gguf_type(tensor_type, contract)


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
        "tensor_payload_layout": "source_quant_rows_preserved_with_optional_row_stride_padding",
        "row_stride_contract": {
            "enabled_by_default": False,
            "exact_mode_must_start_disabled": True,
            "performance_mode_opt_in_only": True,
            "logical_cols_are_math_extent": True,
            "row_stride_bytes_are_storage_extent": True,
            "padding_bytes_must_decode_as_zero_and_must_not_be_consumed_by_kernels": True,
            "default_alignment_bytes": juju_row_stride_alignment_bytes(),
            "env_enable": "JUJU_ENABLE_ROW_STRIDE_PADDING",
            "env_alignment": "JUJU_ROW_STRIDE_ALIGNMENT_BYTES",
            "env_max_overhead_pct": "JUJU_ROW_STRIDE_MAX_OVERHEAD_PCT",
        },
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
            "TENSOR_INDEX.tensors[].row_stride_bytes",
            "GRAPH_IR.runtime_policy",
            "GRAPH_IR.execution_plan",
            "GRAPH_IR.priority_tables",
            "GRAPH_IR.performance_research_slots",
            "MODEL_META.multimodal_contract",
            "MODEL_META.modality_flags",
        ],
        "reserved_section_types": {
            "predictor": JUJU_SECTION_PREDICTOR,
            "buddy_map": JUJU_SECTION_BUDDY_MAP,
            "tier_hint": JUJU_SECTION_TIER_HINT,
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
        "codec_registry_version": GGUF_CODEC_REGISTRY_VERSION,
        "codec_registry": GGUF_CODEC_REGISTRY,
        "quant_block_layouts": {
            "mxfp4": {
                "block_elements": 32,
                "scale": "one_e8m0_byte_per_block_before_payload",
                "payload_bytes": 16,
                "nibble_order": "pairwise_even_index_low_nibble_odd_index_high_nibble",
            },
        },
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
        "chat_template_sources": ["tokenizer_config.json.chat_template", "chat_template.jinja"],
        "chat_template_source": "tokenizer_config_or_model_card",
        "chat_template_jinja_source": "generated_from_tokenizer_config.chat_template_when_present",
        "missing_chat_template_policy": "base_completion_template_only_never_invent_family_template",
        "missing_chat_template_behavior": "chat_api_requires_template_or_explicit_messages_formatter; raw_completion_input_ids_allowed",
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
            "tensor_logical_hash_equivalence",
            "original_gguf_vs_juju_exact_ppl_delta",
            "router_topk_exact_match",
            "expert_id_sequence_exact_match",
        ],
        "exact_mode": juju_exact_mode_policy(),
        "approx_mode": juju_approx_mode_policy(),
        "ppl_acceptance_contract": juju_ppl_acceptance_contract(),
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
        "expert_prefetch_execution_policy": {
            "format": "JUJU_EXPERT_PREFETCH_POLICY_V1",
            "source": "runtime_tensor_table_plus_router_trace_not_model_name",
            "activation_trace_required": True,
            "proactive_prefetch_required": True,
            "fine_grained_expert_segments_required": True,
            "cache_miss_must_not_block_critical_path_when_prediction_available": True,
            "planner_inputs": [
                "runtime_priority",
                "prefetch_priority",
                "execution_layer",
                "execution_order",
                "file_locality_group",
                "juju_offset",
                "juju_bytes",
                "expert_bundle_bytes",
                "router_topk",
                "router_score",
                "router_entropy",
                "coactivation_history",
                "next_layer_candidates",
                "device_free_bytes",
                "ram_free_bytes",
                "disk_queue_depth",
                "pinned_queue_depth",
                "gpu_queue_depth",
            ],
            "planner_outputs": [
                "prefetch_distance",
                "cache_priority",
                "eviction_score",
                "target_tier",
                "expected_overlap_window_us",
                "bytes",
                "fallback_bundle_bytes_forbidden_when_profile_available",
            ],
            "fallback_policy": "use_measured_expert_bundle_bytes_or_fail_closed_for_zero_byte_items",
        },
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
            "moe_infinity": [
                "sequence_level_activation_trace",
                "activation_aware_prefetch",
                "activation_aware_cache",
                "temporal_locality_trace",
                "prefetch_distance_by_activation_pattern",
            ],
            "promoe": [
                "proactive_expert_cache",
                "intermediate_result_prediction",
                "stride_prefetch",
                "chunked_prefetch",
                "early_preemption",
                "reordered_inference_optional",
                "goodpred_metric",
            ],
            "fmoe": [
                "fine_grained_expert_offload",
                "expert_selection_patterns",
                "semantic_prompt_hints",
                "expert_hit_rate_target",
                "fine_grained_expert_segment_policy",
            ],
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
            "qjl": ["jl_projection", "sign_bit_residual", "unbiased_inner_product_estimator", "zero_scale_overhead"],
            "turboquant": [
                "polarquant",
                "qjl_residual_correction",
                "online_vector_quantization",
                "rotation_before_quantization",
                "kv_cache_memory_reduction_target",
                "attention_speedup_target",
            ],
        },
        "research_derived_format_requirements": {
            "expert_trace_keys": [
                "sequence_id",
                "token_position",
                "layer_id",
                "router_topk",
                "router_scores",
                "selected_experts",
                "previous_layer_experts",
                "previous_token_experts",
            ],
            "expert_prefetch_policy_keys": [
                "prefetch_distance",
                "cache_priority",
                "eviction_score",
                "hit_rate_ema",
                "miss_latency_us",
                "load_latency_us",
                "compute_overlap_window_us",
            ],
            "kv_quant_policy_keys": [
                "k_bits",
                "v_bits",
                "normal_bits",
                "outlier_bits",
                "outlier_channels",
                "group_size",
                "page_size_tokens",
                "enable_rotation",
                "rotation_seed",
                "enable_qjl",
                "qjl_seed",
            ],
            "required_runtime_metrics": [
                "ppl",
                "first_token_target_rank",
                "first_token_target_logprob",
                "tokens_per_second",
                "time_to_first_token_ms",
                "expert_hit_rate",
                "expert_miss_latency_us",
                "kv_cache_bytes",
                "qkv_error_vs_plain",
                "ram_used_bytes",
                "vram_used_bytes",
                "gpu_util_pct",
                "cpu_pct",
            ],
        },
        "metrics_required": [
            "expert_hit_rate",
            "expert_miss_latency_us",
            "tokens_per_second",
            "time_to_first_token_ms",
            "inter_token_latency_ms",
            "gpu_idle_gap_us",
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
        "exact_mode": juju_exact_mode_policy(),
        "approx_mode": juju_approx_mode_policy(),
        "ppl_acceptance_contract": juju_ppl_acceptance_contract(),
        "performance_acceptance_contract": juju_performance_acceptance_contract(),
        "expert_calibration_contract": juju_expert_calibration_contract(),
        "adaptive_runtime_scheduler": juju_adaptive_runtime_scheduler_contract(),
    }


def juju_contract_metadata(contract, source_name, source_repo_id, runtime_arch=None):
    arch = dict(contract.get("arch_meta") or {})
    qkv = _juju_effective_qkv_schema(contract, runtime_arch or arch)
    correctness = _juju_execution_correctness_contract(contract, runtime_arch or arch, qkv)
    qkv_required = _juju_qkv_required(contract, qkv)
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
        "qkv_packed_cache_required": qkv_required,
        "persistent_plain_kv_cache_allowed": False,
        "final_model_structure_contract": contract.get("final_model_structure_contract", {}),
        "pipeline_budget_contract": contract.get("pipeline_budget_contract", {}),
        "execution_path_contract": contract.get("execution_path_contract", {}),
        "expert_segmentation_contract": contract.get("expert_segmentation_contract", {}),
        "chunk_io_contract": contract.get("chunk_io_contract", {}),
        "universal_tier_contract": contract.get("universal_tier_contract", {}),
        "qkv_policy_contract": qkv,
        "qkv_cache_schema_effective": qkv,
        "eval_kv_policy": _juju_eval_kv_policy(qkv),
        "format_extension_contract": juju_format_extension_contract(contract),
        "kernel_registry_contract": juju_kernel_registry_contract(contract),
        "tokenizer_contract": juju_tokenizer_contract(),
        "special_tokens": juju_runtime_special_tokens_metadata(runtime_arch or arch),
        "adapter_registry_contract": juju_adapter_registry_contract(),
        "validation_contract": juju_validation_contract(),
        "execution_correctness_contract": correctness,
        "research_offload_contract": juju_research_offload_contract(),
        "exact_mode": juju_exact_mode_policy(),
        "approx_mode": juju_approx_mode_policy(),
        "ppl_acceptance_contract": juju_ppl_acceptance_contract(),
        "performance_acceptance_contract": juju_performance_acceptance_contract(),
        "expert_calibration_contract": juju_expert_calibration_contract(),
        "adaptive_runtime_scheduler": juju_adaptive_runtime_scheduler_contract(),
        "runtime_contract_complete": True,
        "runtime_contract_views": [
            "root_metadata",
            "qkv_policy",
            "kv_layout",
            "runtime_policy",
            "runtime_execution_manifest",
            "graph_ir",
            "execution_correctness",
            "bottleneck_trace",
        ],
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
            "sidecar_upload_format": "structured_json_yaml_toml_only_no_generated_md_pdf",
            "required_trace_fields": correctness["performance"]["trace_token_layer_phase_required"],
            "exact_acceptance_required_before_performance_mode": True,
            "required_runtime_metrics": [
                "expert_hit_rate",
                "expert_miss_latency_us",
                "prefetch_waste_ratio",
                "gpu_idle_gap_us",
                "disk_read_bytes",
                "pcie_copy_bytes",
            ],
        },
    }
    out.update(_juju_qkv_contract_fields(qkv))
    for src, dst in (
        ("k_bits", "k_bits"),
        ("v_bits", "v_bits"),
        ("normal_bits", "normal_bits"),
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
    qkv_nested = (
        ("rotation.seed", "rotation_seed"),
        ("qjl.seed", "qjl_seed"),
        ("qjl.enabled", "enable_qjl"),
        ("rotation.enabled", "enable_rotation"),
        ("outlier.channels", "outlier_channels"),
        ("outlier.bits", "outlier_bits"),
        ("normal.bits", "normal_bits"),
        ("residency.sink_tokens", "sink_tokens"),
    )
    for src, dst in qkv_nested:
        value = contract_value(qkv, src, default=None)
        if value is not None and dst not in out:
            out[dst] = value
    return out


def make_header(contract, source_name, file_size_value, sections, section_sizes, index_checksum=0, modality_flags=JUJU_MODALITY_TEXT):
    header = bytearray(JUJU_HEADER_BYTES)
    header[0:8] = b"JUJU\x00\x02\x00\x00"
    struct.pack_into("<I", header, 8, JUJU_CONTAINER_VERSION_MAJOR)
    struct.pack_into("<I", header, 12, JUJU_CONTAINER_VERSION_MINOR)
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


def juju_qkv_policy_hash(qkv):
    def scrub(value):
        if isinstance(value, dict):
            return {
                str(k): scrub(v)
                for k, v in value.items()
                if k not in ("qkv_policy_hash", "qkv_policy_hash_hex")
            }
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    raw = json.dumps(scrub(qkv or {}), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") or 1


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
        if bucket == "hot_experts":
            priority = 78
            prefetch = 92
            residency = "FAST_MEM_STREAMABLE"
            prefetch_class = "expert_bootstrap_hot"
        elif bucket == "warm_experts":
            priority = 70
            prefetch = 86
            residency = "FAST_MEM_STREAMABLE"
            prefetch_class = "expert_bootstrap_warm"
        else:
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


def juju_tensor_execution_metadata(name, bucket, tensor_offset=0, layout=None, priority=None):
    lower = str(name or "").lower()
    bucket = str(bucket or "")
    layout = layout or {}
    priority = priority or tensor_runtime_priority(name, bucket, layout.get("juju_bytes") or 0)
    layer = _juju_layer_id_from_name(name)
    suffix = _juju_layer_suffix(name) if layer is not None else lower
    op = "weight"
    order = 500
    access_phase = "stream"
    access_pattern = "on_demand"
    locality = "zz_misc"
    hotset_rank = 900

    if lower == "token_embd.weight" or lower.endswith(".embed_tokens.weight"):
        op, order, access_phase, access_pattern, locality, hotset_rank = (
            "embedding_lookup", 0, "startup", "always_hot", "00_startup_core", 0)
    elif lower in {"output.weight", "lm_head.weight"} or lower.endswith(".lm_head.weight"):
        op, order, access_phase, access_pattern, locality, hotset_rank = (
            "lm_head", 980, "logits", "always_hot", "00_startup_core", 2)
    elif lower in {"output_norm.weight", "norm.weight", "model.norm.weight"}:
        op, order, access_phase, access_pattern, locality, hotset_rank = (
            "final_norm", 970, "final_norm", "always_hot", "00_startup_core", 1)
    elif "rope_freqs" in lower or "rotary" in lower:
        op, order, access_phase, access_pattern, locality, hotset_rank = (
            "rope_constants", 40, "startup", "always_hot", "00_startup_core", 3)
    elif bucket in {"vision_encoder", "vision_projector", "audio_encoder", "video_encoder", "document_encoder"}:
        op, order, access_phase, access_pattern, locality, hotset_rank = (
            bucket, 700, "modality", "stream_when_modality_present", f"80_{bucket}", 700)
    elif layer is not None:
        locality = f"10_layer_{int(layer):04d}_attention"
        access_phase = "layer_attention"
        hotset_rank = 100 + min(int(layer), 799)
        if suffix in {"attn_norm.weight", "input_layernorm.weight", "pre_attention_norm.weight"}:
            op, order = "attention_input_norm", 100
        elif suffix in JUJU_ATTENTION_Q_SUFFIXES:
            op, order = "q_projection", 110
        elif suffix in JUJU_ATTENTION_K_SUFFIXES:
            op, order = "k_projection", 120
        elif suffix in JUJU_ATTENTION_V_SUFFIXES:
            op, order = "v_projection", 130
        elif suffix in JUJU_Q_NORM_SUFFIXES:
            op, order = "q_norm", 140
        elif suffix in JUJU_K_NORM_SUFFIXES:
            op, order = "k_norm", 150
        elif suffix in JUJU_V_NORM_SUFFIXES:
            op, order = "v_norm", 160
        elif suffix in JUJU_ATTENTION_O_SUFFIXES:
            op, order = "attention_output", 180
        elif suffix in {"post_attention_norm.weight", "post_attention_layernorm.weight", "post_attention_layer_norm.weight", "post_attn_norm.weight"}:
            op, order = "post_attention_norm", 190
        elif suffix in {"ffn_norm.weight", "ffn_pre_norm.weight", "pre_ffw_norm.weight", "mlp_norm.weight"}:
            op, order, access_phase, locality = "ffn_norm", 210, "layer_mlp", f"20_layer_{int(layer):04d}_mlp_shared"
        elif suffix in {"pre_ffw_norm_2.weight", "ffn_pre_norm_2.weight", "moe_norm.weight"}:
            op, order, access_phase, locality = "expert_ffn_norm", 220, "layer_mlp", f"20_layer_{int(layer):04d}_mlp_shared"
        elif suffix in {"ffn_gate_inp.weight", "router.weight", "mlp.router.weight", "moe.gate.weight"}:
            op, order, access_phase, access_pattern, locality, hotset_rank = (
                "moe_router", 230, "router", "always_layer_hot", f"20_layer_{int(layer):04d}_mlp_shared", 50 + min(int(layer), 849))
        elif suffix in {"ffn_gate_inp.scale", "router.scale", "mlp.router.scale", "moe.gate.scale"}:
            op, order, access_phase, access_pattern, locality, hotset_rank = (
                "moe_router_scale", 231, "router", "always_layer_hot", f"20_layer_{int(layer):04d}_mlp_shared", 51 + min(int(layer), 849))
        elif is_shared_expert_tensor_name(lower):
            op, order, access_phase, locality = "shared_expert_mlp", 250, "layer_mlp", f"20_layer_{int(layer):04d}_mlp_shared"
        elif suffix in {"post_ffw_norm_1.weight", "ffn_post_norm_1.weight"}:
            op, order, access_phase, locality = "post_ffw_norm_1", 270, "layer_mlp", f"20_layer_{int(layer):04d}_mlp_shared"
        elif is_routed_expert_tensor_name(lower):
            op, order, access_phase, access_pattern, locality = (
                "moe_expert_mlp", 300, "selected_experts", "router_selected_dynamic", f"30_layer_{int(layer):04d}_experts")
            hotset_rank = 300 + min(int(layer), 699)
        elif suffix in {"post_ffw_norm_2.weight", "ffn_post_norm_2.weight"}:
            op, order, access_phase, locality = "post_ffw_norm_2", 350, "layer_mlp", f"20_layer_{int(layer):04d}_mlp_shared"
        elif suffix in {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}:
            op, order, access_phase, locality = "dense_mlp", 360, "layer_mlp", f"20_layer_{int(layer):04d}_mlp_shared"
        elif suffix in {"post_ffw_norm.weight", "ffn_post_norm.weight"}:
            op, order, access_phase, locality = "post_ffw_norm", 380, "layer_mlp", f"20_layer_{int(layer):04d}_mlp_shared"
        elif suffix in {"layer_output_scale.weight", "layer_scalar.weight", "layer_scalar"}:
            op, order, access_phase, locality = "layer_output_scale", 390, "layer_tail", f"20_layer_{int(layer):04d}_mlp_shared"

    if priority.get("prefetch_class") == "startup_hot":
        hotset_rank = min(hotset_rank, 16)
    elif priority.get("graph_role") in {"attention", "router", "norm", "shared_core"}:
        hotset_rank = min(hotset_rank, 256)
    return {
        "execution_layer": int(layer) if layer is not None else -1,
        "execution_op": op,
        "execution_order": int(order),
        "access_phase": access_phase,
        "access_pattern": access_pattern,
        "file_locality_group": locality,
        "hotset_rank": int(hotset_rank),
        "io_alignment_bytes": 4096,
        "stream_bytes": int(layout.get("juju_bytes") or layout.get("source_bytes") or 0),
    }


def juju_tensor_math_shape_contract(source_shape, logical_rows, logical_cols, op_role, row_layout):
    return {
        "source_shape": [int(v or 0) for v in (source_shape or [])],
        "storage_layout": row_layout or "source_gguf_quant_block_layout_preserved",
        "op_role": str(op_role or "weight"),
        "math_shape": {
            "out_features": int(logical_rows or 0),
            "in_features": int(logical_cols or 0),
            "transposed": False,
            "source_layout": "gguf_shape0_cols_shape1plus_rows",
        },
    }


def juju_tensor_file_order_key(tensor, bucket):
    name = str(tensor.get("name") or "")
    layout = juju_tensor_storage_layout(tensor)
    priority = tensor_runtime_priority(name, bucket, tensor.get("bytes"))
    meta = juju_tensor_execution_metadata(name, bucket, 0, layout, priority)
    section_rank = {name: idx for idx, name in enumerate(JUJU_TENSOR_BUCKET_ORDER)}.get(bucket, 99)
    routed = is_routed_expert_tensor_name(name)
    expert_id = _juju_expert_id_from_name(name)
    projection_order = _juju_projection_order_value(name)
    return (
        section_rank,
        str(meta.get("file_locality_group") or ""),
        int(meta.get("execution_layer", -1)),
        int(expert_id) if routed and expert_id is not None else -1,
        int(projection_order) if routed else int(meta.get("execution_order", 500)),
        int(meta.get("execution_order", 500)),
        -int(priority.get("runtime_priority") or 0),
        name,
    )


def _juju_record_int(rec, key, default=0):
    try:
        value = rec.get(key, default)
        if value is None or value == "":
            value = default
        return int(value)
    except Exception:
        return int(default)


def _juju_runtime_tensor_ref(rec):
    return {
        "name": rec.get("name"),
        "role": rec.get("graph_role"),
        "layer": _juju_record_int(rec, "execution_layer", -1),
        "op": rec.get("execution_op"),
        "phase": rec.get("access_phase"),
        "shape": list(rec.get("shape") or []),
        "encoding": rec.get("weight_encoding"),
        "quant_family": rec.get("quant_family"),
        "row_layout": rec.get("row_layout"),
        "row_stride_bytes": _juju_record_int(rec, "row_stride_bytes", 0),
        "offset": _juju_record_int(rec, "juju_offset", 0),
        "bytes": _juju_record_int(rec, "juju_bytes", rec.get("stream_bytes") or 0),
        "source_offset": _juju_record_int(rec, "source_offset", 0),
        "source_bytes": _juju_record_int(rec, "source_bytes", 0),
        "file_locality_group": rec.get("file_locality_group"),
        "runtime_priority": _juju_record_int(rec, "runtime_priority", 0),
        "prefetch_priority": _juju_record_int(rec, "prefetch_priority", 0),
        "prefetch_class": rec.get("prefetch_class"),
        "residency_hint": rec.get("residency_hint"),
        "expert_layout": rec.get("expert_layout"),
        "combined_gate_up_split": rec.get("combined_gate_up_split"),
        "expert_source_segments": rec.get("expert_source_segments"),
        "bundle_native": bool(rec.get("bundle_native")),
        "bundle_id": rec.get("bundle_id"),
        "bundle_offset": rec.get("bundle_offset"),
        "bundle_size": rec.get("bundle_size"),
        "bundle_alignment": rec.get("bundle_alignment"),
        "bundle_member_offset": rec.get("bundle_member_offset"),
        "bundle_member_size": rec.get("bundle_member_size"),
        "bundle_member_role": rec.get("bundle_member_role"),
        "expert_id": rec.get("expert_id"),
        "expert_projection": rec.get("expert_projection"),
    }


def _juju_execution_sort_key(rec):
    name = rec.get("name")
    return (
        str(rec.get("file_locality_group") or ""),
        _juju_record_int(rec, "execution_layer", -1),
        _juju_expert_id_from_name(name) if _juju_expert_id_from_name(name) is not None else -1,
        _juju_projection_order_value(name) if is_routed_expert_tensor_name(name) else _juju_record_int(rec, "execution_order", 500),
        _juju_record_int(rec, "execution_order", 500),
        -_juju_record_int(rec, "prefetch_priority", 0),
        -_juju_record_int(rec, "runtime_priority", 0),
        _juju_record_int(rec, "juju_offset", 0),
        str(rec.get("name") or ""),
    )


def _juju_stage_tensor_refs(entries, ops):
    op_set = set(ops or [])
    return [
        _juju_runtime_tensor_ref(rec)
        for rec in sorted(entries, key=_juju_execution_sort_key)
        if str(rec.get("execution_op") or "") in op_set
    ]


def _juju_layer_stage_bytes(refs):
    return sum(int(ref.get("bytes") or 0) for ref in refs or [])


def _juju_layer_stage_plan(entries, stage_name, ops, trigger, residency_policy):
    refs = _juju_stage_tensor_refs(entries, ops)
    return {
        "stage": stage_name,
        "ops": list(ops),
        "trigger": trigger,
        "residency_policy": residency_policy,
        "tensor_count": len(refs),
        "bytes": _juju_layer_stage_bytes(refs),
        "tensors": refs,
    }


def _juju_layer_records_by_id(tensor_records):
    by_layer = {}
    for rec in tensor_records or []:
        layer = _juju_record_int(rec, "execution_layer", -1)
        if layer >= 0:
            by_layer.setdefault(layer, []).append(rec)
    return by_layer


def _juju_build_layer_prefetch_plan(tensor_records, layers):
    by_layer = _juju_layer_records_by_id(tensor_records)
    attention_ops = [
        "attention_input_norm",
        "q_projection",
        "k_projection",
        "v_projection",
        "q_norm",
        "k_norm",
        "v_norm",
        "attention_output",
        "post_attention_norm",
    ]
    router_ops = ["ffn_norm", "expert_ffn_norm", "moe_router", "moe_router_scale"]
    shared_ops = ["shared_expert_mlp", "dense_mlp", "post_ffw_norm_1"]
    expert_ops = ["moe_expert_mlp"]
    tail_ops = ["post_ffw_norm_2", "post_ffw_norm", "layer_output_scale"]
    plan = []
    for idx, layer in enumerate(layers or []):
        current = list(by_layer.get(layer, []))
        next_layer = layers[idx + 1] if idx + 1 < len(layers or []) else None
        next_entries = list(by_layer.get(next_layer, [])) if next_layer is not None else []
        current_stages = [
            _juju_layer_stage_plan(
                current,
                "attention",
                attention_ops,
                "before_layer_attention",
                "keep_current_layer_attention_until_attention_output",
            ),
            _juju_layer_stage_plan(
                current,
                "router",
                router_ops,
                "during_attention_output",
                "keep_router_until_selected_expert_prefetch_dispatched",
            ),
            _juju_layer_stage_plan(
                current,
                "shared_expert",
                shared_ops,
                "after_router_scores_before_mlp",
                "keep_shared_expert_for_layer_mlp",
            ),
            _juju_layer_stage_plan(
                current,
                "selected_experts",
                expert_ops,
                "after_router_topk",
                "stream_or_promote_router_selected_expert_tensors",
            ),
            _juju_layer_stage_plan(
                current,
                "mlp_tail",
                tail_ops,
                "after_expert_accumulation",
                "keep_until_layer_residual_written",
            ),
        ]
        lookahead = {
            "next_layer": next_layer,
            "attention": _juju_layer_stage_plan(
                next_entries,
                "next_attention",
                attention_ops,
                "current_layer_mlp_begin",
                "prefetch_next_layer_attention_to_ram_or_fastmem",
            ),
            "router": _juju_layer_stage_plan(
                next_entries,
                "next_router",
                router_ops,
                "current_layer_attention_output",
                "prefetch_next_layer_router_before_next_attention_tail",
            ),
            "shared_expert": _juju_layer_stage_plan(
                next_entries,
                "next_shared_expert",
                shared_ops,
                "current_layer_mlp_tail",
                "prefetch_next_layer_shared_expert_when_budget_allows",
            ),
            "expert_policy": {
                "trigger": "router_topk_scores_current_layer_plus_coactivation_history",
                "fallback": "prefetch_current_layer_expert_tensors_by_prefetch_priority_when_no_history",
                "candidate_stage": _juju_layer_stage_plan(
                    current,
                    "current_expert_candidates",
                    expert_ops,
                    "router_scores_ready",
                    "bounded_by_ram_vram_staging_slots_and_io_depth",
                ),
            },
        }
        plan.append({
            "layer": int(layer),
            "tensor_count": len(current),
            "bytes": sum(_juju_record_int(rec, "juju_bytes", rec.get("stream_bytes") or 0) for rec in current),
            "execute_stage_order": ["attention", "router", "shared_expert", "selected_experts", "mlp_tail"],
            "current_layer": current_stages,
            "lookahead": lookahead,
            "eviction_protection": [
                "startup_hotset",
                "current_layer_attention_router_norm",
                "current_layer_selected_experts_until_mlp_done",
                "next_layer_attention_router",
            ],
        })
    return plan


def _juju_scalar_value(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            scalar = _juju_scalar_value(item)
            if scalar is not None and scalar != "":
                return scalar
        return None
    if isinstance(value, dict):
        for key in ("value", "default", "size", "dim", "count"):
            if key in value:
                scalar = _juju_scalar_value(value.get(key))
                if scalar is not None and scalar != "":
                    return scalar
        return None
    return value


def _juju_int_or_none(value):
    value = _juju_scalar_value(value)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _juju_first_int(*values):
    for value in values:
        parsed = _juju_int_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _juju_list_or_none(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _juju_bool_or_none(value):
    value = _juju_scalar_value(value)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _juju_config_dict(value):
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


def _juju_qkv_required(contract, qkv):
    return bool(
        qkv or
        contract_value(contract, "qkv_cache.validation.required_quantized_qkv", default=False) or
        contract_value(contract, "qkv_policy_contract.validation.required_quantized_qkv", default=False) or
        contract_value(qkv, "validation.required_quantized_qkv", default=False) or
        contract_value(qkv, "required_quantized_qkv", default=False) or
        contract_value(contract, "qkv_packed_cache_required", default=False)
    )


def _juju_eval_kv_policy(qkv=None):
    qkv = dict(qkv or {})
    return {
        "format": "JUJU_EVAL_KV_POLICY_V1",
        "generation_kv_backend": "qkv_quantized_per_layer_head_cache",
        "ppl_kv_backend": "qkv_quantized_per_layer_head_cache",
        "eval_kv_backend": "qkv_quantized_per_layer_head_cache",
        "plain_reference_scope": "diagnostic_probe_only_never_default_runtime",
        "eval_must_bypass_quantized_qkv": False,
        "force_qkv_for_ppl": True,
        "force_qkv_for_eval": True,
        "allow_plain_reference": False,
        "ppl_accepts_quantized_qkv_only_after_plain_comparison_probe": False,
        "plain_comparison_probe_required": False,
        "plain_comparison_probe": {
            "format": "JUJU_QKV_PLAIN_COMPARE_PROBE_V1",
            "required_before_accepting_quantized_qkv_ppl": False,
            "compare_backend": "flat_position_major_layer_contiguous_cache",
            "candidate_backend": qkv.get("backend") or "qkv_quantized_per_layer_head_cache",
            "metrics": ["max_abs", "mean_abs", "rms", "logprob_delta", "ppl_delta"],
            "fail_closed_if_missing": False,
        },
        "supported_cache_bits": list(JUJU_QKV_SUPPORTED_CACHE_BITS),
        "query_cache_allowed": False,
        "key_value_bits_may_differ": True,
        "raw_plain_bits": [16, 32],
    }


def _juju_qkv_runtime_policy(contract, qkv):
    qkv = dict(qkv or {})
    has_qkv = True
    qkv_required = _juju_qkv_required(contract, qkv)
    enable_qjl = bool(_juju_bool_or_none(qkv.get("enable_qjl")))
    enable_rotation = _juju_bool_or_none(qkv.get("enable_rotation"))
    if enable_rotation is None:
        enable_rotation = True
    return {
        "format": "JUJU_KV_RUNTIME_POLICY_V1",
        "available_backends": [
            "qkv_quantized_per_layer_head_cache",
        ],
        "preferred_backend": "qkv_quantized_per_layer_head_cache",
        "required_backend": "qkv_quantized_per_layer_head_cache" if qkv_required else "",
        "plain_reference_backend": "flat_position_major_layer_contiguous_cache_validation_only",
        "plain_reference_required_for_ppl": False,
        "evaluation_policy": _juju_eval_kv_policy(qkv),
        "plain_fallback_allowed": not qkv_required,
        "qkv_approximate_cache": has_qkv,
        "qkv_must_match_plain_within_eval_tolerance_before_ppl_accept": False,
        "decode_update_mode": qkv.get("runtime_update_mode") or "incremental_append_current_token_no_full_context_requantize",
        "persistent_plain_kv_cache_allowed": False,
        "qkv_policy_contract": qkv,
        "k_bits": _juju_int_or_none(qkv.get("k_bits")),
        "v_bits": _juju_int_or_none(qkv.get("v_bits")),
        "key_bits": _juju_int_or_none(qkv.get("k_bits")),
        "value_bits": _juju_int_or_none(qkv.get("v_bits")),
        "normal_bits": _juju_int_or_none(qkv.get("normal_bits")),
        "key_normal_bits": _juju_int_or_none(qkv.get("key_normal_bits")),
        "value_normal_bits": _juju_int_or_none(qkv.get("value_normal_bits")),
        "outlier_channels": _juju_int_or_none(qkv.get("outlier_channels")),
        "outlier_bits": _juju_int_or_none(qkv.get("outlier_bits")),
        "key_outlier_bits": _juju_int_or_none(qkv.get("key_outlier_bits")),
        "value_outlier_bits": _juju_int_or_none(qkv.get("value_outlier_bits")),
        "key_effective_bits": qkv.get("key_effective_bits"),
        "value_effective_bits": qkv.get("value_effective_bits"),
        "group_size": _juju_int_or_none(qkv.get("group_size")),
        "page_size_tokens": _juju_int_or_none(qkv.get("page_size_tokens")),
        "sink_tokens": _juju_int_or_none(qkv.get("sink_tokens")),
        "enable_qjl": enable_qjl,
        "qjl_enabled": enable_qjl,
        "enable_rotation": bool(enable_rotation),
        "rotation_enabled": bool(enable_rotation),
        "rotation_seed": _juju_int_or_none(qkv.get("rotation_seed")),
        "qjl_seed": _juju_int_or_none(qkv.get("qjl_seed")),
        "rotation_backend": qkv.get("rotation_backend"),
        "codebook_distribution": qkv.get("codebook_distribution"),
        "bit_budget_semantics": qkv.get("bit_budget_semantics"),
        "qkv_policy_hash": _juju_int_or_none(qkv.get("qkv_policy_hash")),
        "qkv_policy_hash_hex": qkv.get("qkv_policy_hash_hex"),
        "qkv_state_key_scope": [
            "request_id",
            "layer",
            "kv_head",
            "head_dim",
            "context_epoch",
            "qkv_policy_hash",
        ],
        "qkv_state_reuse_forbidden_across": [
            "different_layer",
            "different_request",
            "different_context_contents",
            "different_head_dim",
            "different_policy",
        ],
        "executor_contract": "do_not_use_head_only_qkv_state_cache; layer_and_epoch_are_part_of_identity",
        "query_policy": dict(qkv.get("query_policy") or {}),
        "key_cache_policy": dict(qkv.get("key") or {}),
        "value_cache_policy": dict(qkv.get("value") or {}),
        "turboquant_policy": dict(qkv.get("turboquant") or {}),
    }


def _juju_qkv_contract_fields(qkv):
    qkv = _juju_effective_qkv_schema({"qkv_policy_contract": dict(qkv or {})}, {})
    enable_qjl = bool(_juju_bool_or_none(qkv.get("enable_qjl")))
    enable_rotation = _juju_bool_or_none(qkv.get("enable_rotation"))
    if enable_rotation is None:
        enable_rotation = True
    return {
        "qkv_policy_contract": qkv,
        "qkv_cache_schema_effective": qkv,
        "qkv_packed_cache_required": True,
        "required_quantized_qkv": True,
        "persistent_plain_kv_cache_allowed": False,
        "plain_kv_persistent_storage": False,
        "plain_fallback_allowed": False,
        "plain_kv_runtime_allowed": False,
        "k_bits": _juju_int_or_none(qkv.get("k_bits")),
        "v_bits": _juju_int_or_none(qkv.get("v_bits")),
        "key_bits": _juju_int_or_none(qkv.get("k_bits")),
        "value_bits": _juju_int_or_none(qkv.get("v_bits")),
        "normal_bits": _juju_int_or_none(qkv.get("normal_bits")),
        "key_normal_bits": _juju_int_or_none(qkv.get("key_normal_bits")),
        "value_normal_bits": _juju_int_or_none(qkv.get("value_normal_bits")),
        "outlier_channels": _juju_int_or_none(qkv.get("outlier_channels")),
        "outlier_bits": _juju_int_or_none(qkv.get("outlier_bits")),
        "key_outlier_bits": _juju_int_or_none(qkv.get("key_outlier_bits")),
        "value_outlier_bits": _juju_int_or_none(qkv.get("value_outlier_bits")),
        "key_effective_bits": qkv.get("key_effective_bits"),
        "value_effective_bits": qkv.get("value_effective_bits"),
        "group_size": _juju_int_or_none(qkv.get("group_size")),
        "page_size_tokens": _juju_int_or_none(qkv.get("page_size_tokens")),
        "sink_tokens": _juju_int_or_none(qkv.get("sink_tokens")),
        "enable_qjl": enable_qjl,
        "qjl_enabled": enable_qjl,
        "enable_rotation": bool(enable_rotation),
        "rotation_enabled": bool(enable_rotation),
        "rotation_seed": _juju_int_or_none(qkv.get("rotation_seed")),
        "qjl_seed": _juju_int_or_none(qkv.get("qjl_seed")),
        "rotation_backend": qkv.get("rotation_backend"),
        "codebook_distribution": qkv.get("codebook_distribution"),
        "bit_budget_semantics": qkv.get("bit_budget_semantics"),
        "qkv_policy_hash": _juju_int_or_none(qkv.get("qkv_policy_hash")),
        "qkv_policy_hash_hex": qkv.get("qkv_policy_hash_hex"),
        "qjl": dict(qkv.get("qjl") or {}),
        "rotation": dict(qkv.get("rotation") or {}),
        "codebook": dict(qkv.get("codebook") or {}),
        "normal": dict(qkv.get("normal") or {}),
        "outlier": dict(qkv.get("outlier") or {}),
        "evaluation_policy": _juju_eval_kv_policy(qkv),
        "supported_cache_bits": list(qkv.get("supported_cache_bits") or JUJU_QKV_SUPPORTED_CACHE_BITS),
        "query_policy": dict(qkv.get("query_policy") or {}),
        "key_cache_policy": dict(qkv.get("key") or {}),
        "value_cache_policy": dict(qkv.get("value") or {}),
        "turboquant_policy": dict(qkv.get("turboquant") or {}),
        "residency": dict(qkv.get("residency") or {}),
        "cache_layout": dict(qkv.get("cache_layout") or {}),
    }


def _juju_layer_attention_kind(layer, runtime_arch):
    runtime_arch = runtime_arch or {}
    layer_types = _juju_list_or_none(runtime_arch.get("layer_types"))
    if layer_types and 0 <= int(layer) < len(layer_types):
        text = str(layer_types[int(layer)]).lower()
        if "global" in text or "full" in text:
            return "global_full_attention"
        if "sliding" in text or "local" in text:
            return "sliding_window_attention"
    interval = _juju_first_int(
        runtime_arch.get("full_attention_interval"),
        runtime_arch.get("global_attention_interval"),
    )
    offset = _juju_first_int(
        runtime_arch.get("full_attention_offset"),
        runtime_arch.get("global_attention_offset"),
        interval - 1 if interval else None,
    )
    if interval and interval > 0:
        return "global_full_attention" if int(layer) % interval == int(offset or 0) else "sliding_window_attention"
    return "standard_attention"


def _juju_layer_rope_contract(layer, runtime_arch):
    runtime_arch = runtime_arch or {}
    kind = _juju_layer_attention_kind(layer, runtime_arch)
    params = _juju_config_dict(runtime_arch.get("rope_parameters"))
    full_params = _juju_config_dict(params.get("full_attention")) or _juju_config_dict(params.get("global_attention"))
    sliding_params = _juju_config_dict(params.get("sliding_attention")) or _juju_config_dict(params.get("local_attention"))
    selected = full_params if kind == "global_full_attention" and full_params else sliding_params
    if not selected:
        selected = {}
    theta = first_present(
        runtime_arch.get("full_rope_theta") if kind == "global_full_attention" else runtime_arch.get("sliding_rope_theta"),
        selected.get("rope_theta"),
        selected.get("theta"),
        runtime_arch.get("rope_theta"),
        runtime_arch.get("theta"),
    )
    head_dim = _juju_first_int(
        runtime_arch.get("global_head_dim") if kind == "global_full_attention" else runtime_arch.get("head_dim"),
        runtime_arch.get("head_dim"),
    )
    partial = _juju_float_or_none(first_present(selected.get("partial_rotary_factor"), runtime_arch.get("partial_rotary_factor")))
    rope_dim = _juju_first_int(
        selected.get("rope_dimension_count"),
        selected.get("qk_rope_head_dim"),
        runtime_arch.get("qk_rope_head_dim"),
        int(head_dim * partial) if head_dim and partial else None,
        head_dim,
    )
    rope_type = first_present(selected.get("rope_type"), selected.get("type"), runtime_arch.get("rope_type"), "default")
    frequency_dim = head_dim if str(rope_type).lower() == "proportional" and head_dim else rope_dim
    return {
        "kind": kind,
        "rope_type": rope_type,
        "theta": theta,
        "head_dim": head_dim,
        "rope_dim": rope_dim,
        "frequency_dim": frequency_dim,
        "partial_rotary_factor": partial,
        "rotate_half": True,
        "proportional_frequency_dim_uses_full_head_dim": str(rope_type).lower() == "proportional",
    }


def _juju_value_norm_mode_is_unweighted(value):
    if value in (None, ""):
        return False
    text = str(value).strip().lower().replace("-", "_")
    return text in {
        "unweighted_rmsnorm_contract",
        "unweighted_rms_norm_contract",
        "unweighted_rmsnorm",
        "parameter_free_rmsnorm",
        "parameter_free_rms_norm",
        "rmsnorm_no_weight",
        "rms_norm_no_weight",
    }


def _juju_layer_selector_matches(value, layer):
    if value in (None, "", False):
        return False
    if value is True:
        return True
    try:
        layer_i = int(layer)
    except Exception:
        return False
    if isinstance(value, dict):
        for key in ("layers", "layer_ids", "indices", "attention_layers", "value_norm_layers"):
            if _juju_layer_selector_matches(value.get(key), layer_i):
                return True
        for key in (layer_i, str(layer_i)):
            if key in value:
                mapped = value.get(key)
                return mapped is True or _juju_value_norm_mode_is_unweighted(mapped)
        return False
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, dict):
                item_layer = _juju_first_int(item.get("layer"), item.get("layer_id"), item.get("index"))
                if item_layer == layer_i and (
                    item.get("unweighted_value_norm_is_contractual_when_declared") is True or
                    _juju_value_norm_mode_is_unweighted(item.get("value_norm_mode")) or
                    _juju_value_norm_mode_is_unweighted(item.get("v_norm_mode")) or
                    _juju_value_norm_mode_is_unweighted(item.get("norm_mode"))
                ):
                    return True
            else:
                try:
                    if int(item) == layer_i:
                        return True
                except Exception:
                    pass
        return False
    if isinstance(value, str):
        for part in re.split(r"[,\\s]+", value.strip()):
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                try:
                    if int(lo) <= layer_i <= int(hi):
                        return True
                except Exception:
                    pass
            else:
                try:
                    if int(part) == layer_i:
                        return True
                except Exception:
                    pass
    return False


def _juju_runtime_declares_unweighted_v_norm(layer, runtime_arch):
    runtime_arch = dict(runtime_arch or {})
    for key in (
        "value_norm_mode",
        "v_norm_mode",
        "attention_value_norm_mode",
        "attention.value_norm_mode",
    ):
        if _juju_value_norm_mode_is_unweighted(runtime_arch.get(key)):
            return True
    for key in (
        "unweighted_value_norm",
        "unweighted_v_norm",
        "attention_unweighted_value_norm",
        "attention_unweighted_v_norm",
    ):
        if _juju_bool_or_none(runtime_arch.get(key)) is True:
            return True
    for key in (
        "unweighted_value_norm_layers",
        "unweighted_v_norm_layers",
        "layers_with_unweighted_v_norm",
        "layers_with_unweighted_v_norm_contract",
        "value_norm_layers",
    ):
        if _juju_layer_selector_matches(runtime_arch.get(key), layer):
            return True
    for table_key in ("attention_layer_contract_table", "attention_layer_table", "layers"):
        rows = runtime_arch.get(table_key)
        if not isinstance(rows, list):
            continue
        if _juju_layer_selector_matches(rows, layer):
            return True
    return False


def _juju_layer_has_implicit_unweighted_v_norm(layer, suffixes, runtime_arch):
    suffixes = {str(x or "").lower() for x in (suffixes or set())}
    if any(x in suffixes for x in JUJU_V_NORM_SUFFIXES):
        return False
    if _juju_runtime_declares_unweighted_v_norm(layer, runtime_arch):
        return True
    attention_k_eq_v = _juju_bool_or_none((runtime_arch or {}).get("attention_k_eq_v")) is True
    has_q_norm = any(x in suffixes for x in JUJU_Q_NORM_SUFFIXES)
    has_k_norm = any(x in suffixes for x in JUJU_K_NORM_SUFFIXES)
    return bool(attention_k_eq_v and has_q_norm and has_k_norm)


def _juju_value_raw_input_name(value_projection_present, attention_k_eq_v):
    if value_projection_present:
        return "v_raw"
    if attention_k_eq_v:
        return "k_raw"
    return "missing_value_projection"


def _juju_contract_source_config(contract):
    arch = dict(contract.get("arch_meta") or {})
    for value in (
        contract.get("source_config"),
        contract.get("hf_config"),
        contract.get("config_json"),
        arch.get("source_config"),
        arch.get("hf_config"),
        arch.get("config_json"),
    ):
        data = _juju_config_dict(value)
        if data:
            return data
    return {}


def _juju_contract_text_config(contract):
    arch = dict(contract.get("arch_meta") or {})
    source_config = _juju_contract_source_config(contract)
    for parent in (source_config, arch, contract):
        for key in ("text_config", "llm_config", "model_config", "language_config", "transformer_config"):
            data = _juju_config_dict(parent.get(key) if isinstance(parent, dict) else None)
            if data:
                return data
    return {}


def _juju_first_config_value(contract, *keys):
    arch = dict(contract.get("arch_meta") or {})
    text_config = _juju_contract_text_config(contract)
    source_config = _juju_contract_source_config(contract)
    for parent in (text_config, source_config, arch):
        if not isinstance(parent, dict):
            continue
        for key in keys:
            if key in parent and parent.get(key) not in (None, ""):
                return parent.get(key)
    return None


def _juju_effective_qkv_schema(contract, runtime_arch=None):
    runtime_arch = dict(runtime_arch or {})
    qkv = dict(contract.get("qkv_cache_schema") or contract.get("qkv_policy_contract") or {})
    synthesized = not bool(qkv)
    explicit_normal_bits = qkv.get("normal_bits") not in (None, "") or contract_value(qkv, "normal.bits", default=None) not in (None, "")
    explicit_key_normal_bits = (
        qkv.get("key_normal_bits") not in (None, "") or
        contract_value(qkv, "key.normal_bits", default=None) not in (None, "") or
        contract_value(qkv, "key.normal.bits", default=None) not in (None, "")
    )
    explicit_value_normal_bits = (
        qkv.get("value_normal_bits") not in (None, "") or
        contract_value(qkv, "value.normal_bits", default=None) not in (None, "") or
        contract_value(qkv, "value.normal.bits", default=None) not in (None, "")
    )
    explicit_enable_qjl = qkv.get("enable_qjl") not in (None, "") or contract_value(qkv, "qjl.enabled", default=None) not in (None, "")

    def fill(key, *values):
        if qkv.get(key) not in (None, ""):
            return
        value = _juju_first_int(*values)
        if value is not None:
            qkv[key] = value

    qkv.setdefault("format", "JUJU_QKV_POLICY_V1")
    qkv.setdefault(
        "source",
        "source_contract_preserved" if not synthesized else "juju_generator_synthesized_runtime_qkv_contract",
    )
    qkv.setdefault("backend", "qkv_quantized_per_layer_head_cache")
    qkv["required_quantized_qkv"] = True
    qkv["persistent_plain_kv_cache_allowed"] = False
    qkv["plain_fallback_allowed"] = False
    qkv.setdefault("k_bits", 3)
    qkv.setdefault("v_bits", 2)
    qkv.setdefault("enable_rotation", True)
    qkv.setdefault("rotation_backend", "gaussian_qr_orthogonal")
    qkv.setdefault("codebook_distribution", "exact_beta")
    qkv.setdefault("bit_budget_semantics", "total_bits_include_qjl_residual; mse_codebook_bits_are_total_bits_minus_one")
    qkv.setdefault("rotation_seed", 1234)
    qkv.setdefault("qjl_seed", 5678)
    qkv.setdefault("outlier_channels", 32)
    qkv.setdefault("outlier_bits", 3)
    qkv.setdefault("sink_tokens", 4)
    qkv.setdefault("group_size", 64)
    qkv.setdefault("page_size_tokens", 16)
    qkv.setdefault("cache_dtype", "qkv_quantized_uint_packed")
    qkv.setdefault("scale_dtype", "float32")
    qkv.setdefault("zero_dtype", "float32")
    qkv.setdefault("residency_policy", "ram_tracked_via_tier_usage_device_vram_when_enabled")
    qkv.setdefault("runtime_update_mode", "incremental_append_current_token_no_full_context_requantize")
    qkv.setdefault("decode_contract", "decode_reads_qkv_state_directly_plain_kv_is_validation_only")
    qkv.setdefault("state_identity", [
        "request_id",
        "layer",
        "kv_head",
        "head_dim",
        "context_epoch",
        "qkv_policy_hash",
    ])
    qkv.setdefault("reuse_forbidden_across", [
        "different_layer",
        "different_request",
        "different_context_contents",
        "different_head_dim",
        "different_policy",
    ])
    qkv.setdefault("validation", {})
    if isinstance(qkv["validation"], dict):
        qkv["validation"]["required_quantized_qkv"] = True
        qkv["validation"]["plain_reference_is_eval_probe_only"] = True
        qkv["validation"]["reject_generation_if_qkv_backend_unavailable"] = True
        qkv["validation"]["qkv_decode_requires_plain_comparison_probe"] = False
        qkv["validation"]["ppl_must_use_qkv_backend"] = True
        qkv["validation"]["ppl_plain_reference_forbidden"] = True
    fill("num_attention_heads", _juju_first_config_value(contract, "num_attention_heads", "n_heads"), runtime_arch.get("num_attention_heads"), runtime_arch.get("n_heads"))
    fill("num_key_value_heads", _juju_first_config_value(contract, "num_key_value_heads", "n_kv_heads"), runtime_arch.get("num_key_value_heads"), runtime_arch.get("n_kv_heads"))
    fill("num_global_key_value_heads", _juju_first_config_value(contract, "num_global_key_value_heads"), runtime_arch.get("num_global_key_value_heads"), qkv.get("num_key_value_heads"))
    fill("head_dim", _juju_first_config_value(contract, "head_dim", "key_length"), runtime_arch.get("head_dim"), runtime_arch.get("key_length"))
    fill("value_head_dim", _juju_first_config_value(contract, "value_head_dim", "v_head_dim", "value_length"), runtime_arch.get("value_head_dim"), runtime_arch.get("v_head_dim"), qkv.get("head_dim"))
    fill("global_head_dim", _juju_first_config_value(contract, "global_head_dim", "global_key_length"), runtime_arch.get("global_head_dim"), qkv.get("head_dim"))
    fill("global_value_head_dim", _juju_first_config_value(contract, "global_value_head_dim", "global_value_length"), runtime_arch.get("global_value_head_dim"), qkv.get("global_head_dim"))
    fill("max_seq_len", _juju_first_config_value(contract, "max_position_embeddings", "max_seq_len", "context_length"), runtime_arch.get("max_position_embeddings"), runtime_arch.get("context_length"))
    k_bits = _juju_int_or_none(qkv.get("k_bits"))
    v_bits = _juju_int_or_none(qkv.get("v_bits"))
    normal_bits = _juju_first_int(qkv.get("normal_bits"), contract_value(qkv, "normal.bits", default=None))
    outlier_channels = _juju_int_or_none(qkv.get("outlier_channels"))
    outlier_bits = _juju_int_or_none(qkv.get("outlier_bits"))
    key_normal_bits = _juju_first_int(
        qkv.get("key_normal_bits"),
        contract_value(qkv, "key.normal_bits", default=None),
        contract_value(qkv, "key.normal.bits", default=None),
    )
    value_normal_bits = _juju_first_int(
        qkv.get("value_normal_bits"),
        contract_value(qkv, "value.normal_bits", default=None),
        contract_value(qkv, "value.normal.bits", default=None),
    )
    key_outlier_bits = _juju_first_int(
        qkv.get("key_outlier_bits"),
        contract_value(qkv, "key.outlier_bits", default=None),
        contract_value(qkv, "key.outlier.bits", default=None),
    )
    value_outlier_bits = _juju_first_int(
        qkv.get("value_outlier_bits"),
        contract_value(qkv, "value.outlier_bits", default=None),
        contract_value(qkv, "value.outlier.bits", default=None),
    )
    group_size = _juju_int_or_none(qkv.get("group_size"))
    page_tokens = _juju_int_or_none(qkv.get("page_size_tokens"))
    sink_tokens = _juju_int_or_none(qkv.get("sink_tokens"))
    if normal_bits is None:
        normal_bits = v_bits if v_bits is not None else (k_bits if k_bits is not None else 2)
        qkv["normal_bits_source"] = "derived_from_value_bits_for_non_outlier_qkv_channels"
    else:
        qkv.setdefault("normal_bits_source", "source_contract" if explicit_normal_bits else "source_contract_default")
    if key_normal_bits is None:
        key_normal_bits = normal_bits
        qkv["key_normal_bits_source"] = "derived_from_legacy_normal_bits" if explicit_normal_bits else "turboquant_2p5_non_outlier_default"
    else:
        qkv.setdefault("key_normal_bits_source", "source_contract" if explicit_key_normal_bits else "source_contract_default")
    if value_normal_bits is None:
        value_normal_bits = normal_bits
        qkv["value_normal_bits_source"] = "derived_from_legacy_normal_bits" if explicit_normal_bits else "turboquant_2p5_non_outlier_default"
    else:
        qkv.setdefault("value_normal_bits_source", "source_contract" if explicit_value_normal_bits else "source_contract_default")
    if key_outlier_bits is None:
        key_outlier_bits = outlier_bits
        qkv["key_outlier_bits_source"] = "derived_from_legacy_outlier_bits"
    else:
        qkv.setdefault("key_outlier_bits_source", "source_contract")
    if value_outlier_bits is None:
        value_outlier_bits = outlier_bits
        qkv["value_outlier_bits_source"] = "derived_from_legacy_outlier_bits"
    else:
        qkv.setdefault("value_outlier_bits_source", "source_contract")
    raw_qkv_bits = any(
        bit in (16, 32)
        for bit in (
            k_bits,
            v_bits,
            normal_bits,
            outlier_bits,
            key_normal_bits,
            value_normal_bits,
            key_outlier_bits,
            value_outlier_bits,
        )
        if bit is not None
    )
    qjl_value = _juju_bool_or_none(qkv.get("enable_qjl"))
    if qjl_value is None:
        qjl_value = _juju_bool_or_none(contract_value(qkv, "qjl.enabled", default=None))
    if qjl_value is None:
        enable_qjl = not raw_qkv_bits
        qkv["enable_qjl_source"] = "derived_from_quantized_qkv_bits"
    else:
        enable_qjl = bool(qjl_value)
        qkv.setdefault("enable_qjl_source", "source_contract" if explicit_enable_qjl else "source_contract_default")
    if enable_qjl and raw_qkv_bits:
        enable_qjl = False
        qkv["enable_qjl_source"] = "disabled_for_plain_or_raw_qkv_bits"
    enable_rotation = _juju_bool_or_none(qkv.get("enable_rotation"))
    if enable_rotation is None:
        enable_rotation = True

    def _split_effective_bits(normal, outlier, channels, dim):
        normal = _juju_int_or_none(normal)
        outlier = _juju_int_or_none(outlier)
        channels = _juju_int_or_none(channels) or 0
        dim = _juju_int_or_none(dim)
        if normal is None:
            return None
        if outlier is None or channels <= 0 or dim is None or dim <= 0:
            return float(normal)
        channels = max(0, min(channels, dim))
        return float(channels * outlier + (dim - channels) * normal) / float(dim)

    def _turboquant_paper_label_bits(normal, outlier, channels, dim):
        normal = _juju_int_or_none(normal)
        outlier = _juju_int_or_none(outlier)
        channels = _juju_int_or_none(channels) or 0
        dim = _juju_int_or_none(dim)
        if dim != 128 or channels != 32:
            return None
        if normal == 2 and outlier == 3:
            return 2.5
        if normal == 3 and outlier == 4:
            return 3.5
        return None

    def _qjl_mse_bits(total_bits):
        total_bits = _juju_int_or_none(total_bits)
        if total_bits is None:
            return None
        if not enable_qjl or total_bits in (16, 32):
            return total_bits
        return total_bits - 1 if total_bits > 1 else None

    head_dim_for_bits = _juju_first_int(qkv.get("head_dim"), qkv.get("global_head_dim"), 128)
    key_effective_bits = _split_effective_bits(key_normal_bits, key_outlier_bits, outlier_channels, head_dim_for_bits)
    value_effective_bits = _split_effective_bits(value_normal_bits, value_outlier_bits, outlier_channels, head_dim_for_bits)
    key_paper_label_bits = _turboquant_paper_label_bits(
        key_normal_bits, key_outlier_bits, outlier_channels, head_dim_for_bits)
    value_paper_label_bits = _turboquant_paper_label_bits(
        value_normal_bits, value_outlier_bits, outlier_channels, head_dim_for_bits)
    qkv["query_policy"] = {
        "target": "query_activation",
        "cached": False,
        "persistent_cache": False,
        "quantized_cache_bits": None,
        "storage": "runtime_fp_activation_not_kv_cache",
        "role": "attention_query_for_inner_product_against_cached_keys",
    }
    key_cache_policy = {
        "target": "key_cache",
        "cached": True,
        "bits": k_bits,
        "normal_bits": key_normal_bits,
        "outlier_bits": key_outlier_bits,
        "outlier_channels": outlier_channels,
        "paper_label_bits": key_paper_label_bits,
        "effective_bits": key_effective_bits,
        "mse_bits": _qjl_mse_bits(key_normal_bits),
        "outlier_mse_bits": _qjl_mse_bits(key_outlier_bits),
        "qjl_residual_bits": 1 if enable_qjl and not raw_qkv_bits else 0,
        "quant_axis": "per_channel_head_dim",
        "scale_axis": "head_dim_channel_group",
        "group_axis": "head_dim",
        "token_axis_independent": False,
        "estimator": "turboquantprod_inner_product",
    }
    value_cache_policy = {
        "target": "value_cache",
        "cached": True,
        "bits": v_bits,
        "normal_bits": value_normal_bits,
        "outlier_bits": value_outlier_bits,
        "outlier_channels": outlier_channels,
        "paper_label_bits": value_paper_label_bits,
        "effective_bits": value_effective_bits,
        "mse_bits": _qjl_mse_bits(value_normal_bits),
        "outlier_mse_bits": _qjl_mse_bits(value_outlier_bits),
        "qjl_residual_bits": 1 if enable_qjl and not raw_qkv_bits else 0,
        "quant_axis": "per_token_value_vector",
        "scale_axis": "token_vector_group",
        "group_axis": "value_head_dim",
        "token_axis_independent": True,
        "estimator": "turboquantprod_vector_reconstruction",
    }
    turboquant_policy = {
        "paper": "TurboQuant_Qprod",
        "kv_only": True,
        "query_cached": False,
        "rotation_backend": qkv.get("rotation_backend"),
        "codebook_distribution": qkv.get("codebook_distribution"),
        "algorithm": "mse_quantizer_with_1bit_qjl_residual_when_codebook_bits_are_used",
        "bit_budget_semantics": qkv.get("bit_budget_semantics"),
        "split_channel_semantics": "outlier_and_non_outlier_channels_use_independent_turboquant_instances",
        "default_2p5_bits": {
            "paper_label_bits": 2.5,
            "head_dim": 128,
            "outlier_channels": 32,
            "outlier_bits": 3,
            "normal_channels": 96,
            "normal_bits": 2,
            "effective_bits": (32 * 3 + 96 * 2) / 128,
            "bit_accounting_note": "paper_label_bits_is_profile_name_not_arithmetic_average",
            "arithmetic_checked": True,
        },
        "default_3p5_bits": {
            "paper_label_bits": 3.5,
            "head_dim": 128,
            "outlier_channels": 32,
            "outlier_bits": 4,
            "normal_channels": 96,
            "normal_bits": 3,
            "effective_bits": (32 * 4 + 96 * 3) / 128,
            "bit_accounting_note": "paper_label_bits_is_profile_name_not_arithmetic_average",
            "arithmetic_checked": True,
        },
        "key_cache": key_cache_policy,
        "value_cache": value_cache_policy,
        "query": qkv["query_policy"],
    }
    qkv.update({
        "k_bits": k_bits,
        "v_bits": v_bits,
        "key_bits": k_bits,
        "value_bits": v_bits,
        "normal_bits": normal_bits,
        "key_normal_bits": key_normal_bits,
        "value_normal_bits": value_normal_bits,
        "outlier_channels": outlier_channels,
        "outlier_bits": outlier_bits,
        "key_outlier_bits": key_outlier_bits,
        "value_outlier_bits": value_outlier_bits,
        "key_paper_label_bits": key_paper_label_bits,
        "value_paper_label_bits": value_paper_label_bits,
        "key_effective_bits": key_effective_bits,
        "value_effective_bits": value_effective_bits,
        "rotation_backend": qkv.get("rotation_backend"),
        "codebook_distribution": qkv.get("codebook_distribution"),
        "bit_budget_semantics": qkv.get("bit_budget_semantics"),
        "group_size": group_size,
        "page_size_tokens": page_tokens,
        "sink_tokens": sink_tokens,
        "enable_qjl": enable_qjl,
        "qjl_enabled": enable_qjl,
        "enable_rotation": bool(enable_rotation),
        "rotation_enabled": bool(enable_rotation),
        "qkv_packed_cache_required": True,
        "required_quantized_qkv": True,
        "persistent_plain_kv_cache_allowed": False,
        "plain_kv_persistent_storage": False,
        "plain_fallback_allowed": False,
        "plain_kv_runtime_allowed": False,
    })
    qkv["bit_layout"] = {
        "key_bits": k_bits,
        "value_bits": v_bits,
        "normal_bits": normal_bits,
        "key_normal_bits": key_normal_bits,
        "value_normal_bits": value_normal_bits,
        "normal_bits_semantics": "non_outlier_channel_quant_bits",
        "outlier_channels": outlier_channels,
        "outlier_bits": outlier_bits,
        "key_outlier_bits": key_outlier_bits,
        "value_outlier_bits": value_outlier_bits,
        "key_paper_label_bits": key_paper_label_bits,
        "value_paper_label_bits": value_paper_label_bits,
        "key_effective_bits": key_effective_bits,
        "value_effective_bits": value_effective_bits,
        "qjl_required_for_codebook_bits": not raw_qkv_bits,
        "plain_or_raw_bits_present": raw_qkv_bits,
        "query_cached": False,
        "key_quant_axis": key_cache_policy["quant_axis"],
        "value_quant_axis": value_cache_policy["quant_axis"],
        "key_value_quant_axes_may_differ": True,
    }
    if isinstance(qkv.get("validation"), dict):
        qkv["validation"]["normal_bits_semantics"] = "non_outlier_channel_quant_bits"
        qkv["validation"]["key_value_split_bits_required"] = True
        qkv["validation"]["query_must_not_be_persistent_kv_cache"] = True
        qkv["validation"]["turboquant_total_bits_include_qjl_residual"] = True
        qkv["validation"]["qjl_enabled_for_codebook_quantized_qkv"] = not raw_qkv_bits
        qkv["validation"]["reject_inconsistent_normal_bits"] = True
        qkv["validation"]["format_generation_requires_qkv_policy_self_check"] = True
    qkv["normal"] = {"bits": normal_bits, "semantics": "non_outlier_channel_quant_bits", "source": qkv.get("normal_bits_source")}
    qkv["outlier"] = {"channels": outlier_channels, "bits": outlier_bits}
    qkv["key"] = key_cache_policy
    qkv["value"] = value_cache_policy
    qkv["turboquant"] = turboquant_policy
    qkv["residency"] = {"sink_tokens": sink_tokens, "policy": qkv.get("residency_policy")}
    qkv["rotation"] = {
        "enabled": bool(enable_rotation),
        "backend": qkv.get("rotation_backend"),
        "seed": _juju_int_or_none(qkv.get("rotation_seed")),
    }
    qkv["codebook"] = {
        "distribution": qkv.get("codebook_distribution"),
        "generator": "lloyd_max_exact_beta_pdf" if qkv.get("codebook_distribution") == "exact_beta" else "lloyd_max_gaussian_approx_pdf",
    }
    qkv["qjl"] = {"enabled": enable_qjl, "seed": _juju_int_or_none(qkv.get("qjl_seed")), "source": qkv.get("enable_qjl_source")}
    qkv["cache_layout"] = {
        "backend": qkv.get("backend"),
        "dtype": qkv.get("cache_dtype"),
        "group_size": group_size,
        "page_size_tokens": page_tokens,
        "sink_tokens": sink_tokens,
        "state_identity": qkv.get("state_identity"),
    }
    qkv["supported_cache_bits"] = list(qkv.get("supported_cache_bits") or JUJU_QKV_SUPPORTED_CACHE_BITS)
    qkv["evaluation_policy"] = _juju_eval_kv_policy(qkv)
    qkv_hash = juju_qkv_policy_hash(qkv)
    qkv["qkv_policy_hash"] = qkv_hash
    qkv["qkv_policy_hash_hex"] = f"{qkv_hash:016x}"
    return qkv


def _juju_execution_correctness_contract(contract, runtime_arch, qkv):
    runtime_arch = dict(runtime_arch or {})
    qkv_fields = _juju_qkv_contract_fields(qkv)
    return {
        "format": "JUJU_EXECUTION_CORRECTNESS_CONTRACT_V1",
        "tokenizer": {
            "source": "repo_root_runtime_assets",
            "required_any_of": ["tokenizer.json", "tokenizer.model"],
            "chat_template_sources": ["tokenizer_config.json.chat_template", "chat_template.jinja"],
            "missing_chat_template_policy": "base_completion_template_only_never_invent_family_template",
            "special_token_ids_from_source_metadata": True,
        },
        "embedding": {
            "scale": runtime_arch.get("embedding_scale"),
            "scale_source": runtime_arch.get("embedding_scale_source"),
            "semantics": runtime_arch.get("embedding_scale_semantics") or "multiply_token_embedding_before_first_layer",
            "executor_required": bool(runtime_arch.get("embedding_scale") not in (None, "", 1, 1.0)),
        },
        "attention": {
            "qkv_cache_required": True,
            "plain_kv_runtime_allowed": False,
            "score_scale_source": "metadata_or_qk_norm_contract",
            "per_layer_rope_required": True,
            "value_norm_contract_required": True,
            "attention_k_eq_v": _juju_bool_or_none(runtime_arch.get("attention_k_eq_v")),
            "qkv_bits": {
                "key": _juju_int_or_none(qkv.get("k_bits")),
                "value": _juju_int_or_none(qkv.get("v_bits")),
                "normal": _juju_int_or_none(qkv.get("normal_bits")),
                "outlier": _juju_int_or_none(qkv.get("outlier_bits")),
                "key_normal": _juju_int_or_none(qkv.get("key_normal_bits")),
                "value_normal": _juju_int_or_none(qkv.get("value_normal_bits")),
                "key_outlier": _juju_int_or_none(qkv.get("key_outlier_bits")),
                "value_outlier": _juju_int_or_none(qkv.get("value_outlier_bits")),
                "query_cached": False,
                "group_size": _juju_int_or_none(qkv.get("group_size")),
                "page_size_tokens": _juju_int_or_none(qkv.get("page_size_tokens")),
                "sink_tokens": _juju_int_or_none(qkv.get("sink_tokens")),
                "enable_qjl": bool(_juju_bool_or_none(qkv.get("enable_qjl"))),
                "enable_rotation": bool(_juju_bool_or_none(qkv.get("enable_rotation"))),
            },
            "query_policy": dict(qkv.get("query_policy") or {}),
            "key_cache_policy": dict(qkv.get("key") or {}),
            "value_cache_policy": dict(qkv.get("value") or {}),
            "qkv_contract": qkv_fields,
        },
        "tensor_layout": {
            "executor_must_use_tensor_index_shape_offset_size_row_stride": True,
            "name_based_transpose_or_padding_guess_forbidden": True,
            "padding_bytes_decode_as_zero": True,
            "row_stride_bytes_are_storage_extent": True,
        },
        "graph_ir": {
            "executor_must_consume_ops_in_declared_order": True,
            "required_ops_fail_closed": True,
            "optional_ops_may_skip_only_when_contract_declares_fallback": True,
            "tensor_bindings_are_authoritative": True,
        },
        "numerics": {
            "finite_hidden_vectors_required": True,
            "finite_logits_required": True,
            "ppl_acceptance_requires_plain_or_validated_qkv_probe": True,
            "bad_first_token_logprob_is_correctness_failure_not_prompt_issue": True,
        },
        "performance": {
            "hot_tensor_roles": ["token_embedding", "lm_head", "norm", "attention", "router", "shared_expert"],
            "prefetch_order_follows_graph_ir_layer_order": True,
            "trace_token_layer_phase_required": ["load", "io_wait", "attention", "mlp", "lm_head", "kv", "ram", "cpu", "gpu", "vram"],
            "ppl_probe_records_kv_backend_and_cache_bytes": True,
        },
    }


def _juju_kv_layout_contract(contract, runtime_arch):
    qkv = _juju_effective_qkv_schema(contract, runtime_arch)
    qkv_runtime_policy = _juju_qkv_runtime_policy(contract, qkv)
    num_heads = _juju_first_int(
        _juju_first_config_value(contract, "num_attention_heads", "n_heads", "head_count"),
        qkv.get("num_attention_heads"),
        runtime_arch.get("num_attention_heads"),
    )
    sliding_kv_heads = _juju_first_int(
        _juju_first_config_value(contract, "num_key_value_heads", "n_kv_heads", "head_count_kv"),
        qkv.get("num_key_value_heads"),
        qkv.get("kv_heads"),
        runtime_arch.get("num_key_value_heads"),
    )
    global_kv_heads = _juju_first_int(
        _juju_first_config_value(contract, "num_global_key_value_heads", "global_head_count_kv"),
        qkv.get("num_global_key_value_heads"),
        runtime_arch.get("num_global_key_value_heads"),
        sliding_kv_heads,
    )
    head_dim = _juju_first_int(
        _juju_first_config_value(contract, "head_dim", "key_length"),
        qkv.get("head_dim"),
        runtime_arch.get("head_dim"),
        runtime_arch.get("key_length"),
    )
    value_head_dim = _juju_first_int(
        _juju_first_config_value(contract, "value_head_dim", "v_head_dim", "value_length"),
        qkv.get("value_head_dim"),
        qkv.get("v_head_dim"),
        runtime_arch.get("value_head_dim"),
        runtime_arch.get("v_head_dim"),
        head_dim,
    )
    global_head_dim = _juju_first_int(
        _juju_first_config_value(contract, "global_head_dim", "global_key_length"),
        qkv.get("global_head_dim"),
        runtime_arch.get("global_head_dim"),
        head_dim,
    )
    global_value_head_dim = _juju_first_int(
        _juju_first_config_value(contract, "global_value_head_dim", "global_value_length"),
        qkv.get("global_value_head_dim"),
        runtime_arch.get("global_value_head_dim"),
        global_head_dim,
    )
    attention_k_eq_v = _juju_bool_or_none(_juju_first_config_value(contract, "attention_k_eq_v"))
    if attention_k_eq_v is True:
        value_head_dim = head_dim
        global_value_head_dim = global_head_dim
    max_seq = _juju_first_int(
        _juju_first_config_value(contract, "max_position_embeddings", "max_seq_len", "context_length"),
        runtime_arch.get("max_position_embeddings"),
        runtime_arch.get("context_length"),
        qkv.get("max_seq_len"),
        qkv.get("max_position_embeddings"),
    )

    def _entry(num_kv, k_dim, v_dim):
        if num_kv is None or k_dim is None:
            return None
        value_dim = v_dim if v_dim is not None else k_dim
        num_kv = _juju_int_or_none(num_kv)
        k_dim = _juju_int_or_none(k_dim)
        value_dim = _juju_int_or_none(value_dim)
        if num_kv is None or k_dim is None or value_dim is None:
            return None
        return num_kv * (k_dim + value_dim)

    sliding_entry = _entry(sliding_kv_heads, head_dim, value_head_dim)
    global_entry = _entry(global_kv_heads, global_head_dim, global_value_head_dim)
    entry_dim = _juju_first_int(
        max(x for x in [sliding_entry or 0, global_entry or 0] if x is not None),
        qkv.get("entry_dim"),
    )
    page_tokens = _juju_first_int(qkv.get("page_size_tokens"), qkv.get("block_size_tokens"), 16) or 16
    entry_dtype = _juju_scalar_value(qkv.get("dtype") or qkv.get("cache_dtype")) or "quantized_uint_packed"
    qkv_fields = _juju_qkv_contract_fields(qkv)
    return {
        "format": "JUJU_KV_LAYOUT_CONTRACT_V1",
        "layout": "qkv_quantized_per_layer_head_page_major",
        "serving_cache_model": {
            "page_table_required": True,
            "sequence_lengths_required": True,
            "max_decode_seq_len_required": True,
            "non_contiguous_physical_pages_allowed": True,
            "logical_to_physical_block_table_required": True,
            "source_contracts": [
                "vllm_paged_attention_block_tables",
                "flashinfer_paged_kv_csr_page_table",
                "tensorrt_llm_kv_block_pool_reuse_eviction",
            ],
        },
        "runtime_fields": {
            "seq_lens_tensor": {"dtype": "int32", "shape": ["batch"], "required_for_decode": True},
            "max_decode_seq_len": {"dtype": "int32", "required_for_decode": True},
            "block_tables": {"dtype": "int32", "shape": ["batch", "max_blocks_per_seq"], "required_for_decode": True},
            "kv_page_indptr": {"dtype": "int32", "shape": ["batch_plus_1"], "required_for_paged_decode": True},
            "kv_page_indices": {"dtype": "int32", "shape": ["num_used_pages"], "required_for_paged_decode": True},
            "kv_last_page_len": {"dtype": "int32", "shape": ["batch"], "range": [1, page_tokens], "required_for_paged_decode": True},
            "request_epoch": {"dtype": "uint64", "required_for_cache_identity": True},
            "cache_salt_hash": {"dtype": "uint64", "required_for_cross_request_reuse": False},
        },
        "kv_cache_tensor_layout_options": {
            "preferred": "HND_for_low_precision_decode",
            "supported": ["HND", "NHD"],
            "HND": {"shape": ["num_pages", "num_kv_heads", "page_size_tokens", "head_dim"]},
            "NHD": {"shape": ["num_pages", "page_size_tokens", "num_kv_heads", "head_dim"]},
            "qkv_storage": "key_value_quantized_pages_with_scale_zero_side_data",
        },
        "head_layout": {
            "num_attention_heads": num_heads,
            "attention_k_eq_v": attention_k_eq_v,
            "sliding": {
                "num_key_value_heads": sliding_kv_heads,
                "key_head_dim": head_dim,
                "value_head_dim": value_head_dim,
                "entry_dim": sliding_entry,
            },
            "global": {
                "num_key_value_heads": global_kv_heads,
                "key_head_dim": global_head_dim,
                "value_head_dim": global_value_head_dim,
                "entry_dim": global_entry,
            },
            "max_entry_dim": entry_dim,
        },
        "entry_dtype": entry_dtype,
        "key_bits": _juju_int_or_none(qkv.get("k_bits")),
        "value_bits": _juju_int_or_none(qkv.get("v_bits")),
        "normal_bits": _juju_int_or_none(qkv.get("normal_bits")),
        "key_normal_bits": _juju_int_or_none(qkv.get("key_normal_bits")),
        "value_normal_bits": _juju_int_or_none(qkv.get("value_normal_bits")),
        "outlier_channels": _juju_int_or_none(qkv.get("outlier_channels")),
        "outlier_bits": _juju_int_or_none(qkv.get("outlier_bits")),
        "key_outlier_bits": _juju_int_or_none(qkv.get("key_outlier_bits")),
        "value_outlier_bits": _juju_int_or_none(qkv.get("value_outlier_bits")),
        "key_effective_bits": qkv.get("key_effective_bits"),
        "value_effective_bits": qkv.get("value_effective_bits"),
        "group_size": _juju_int_or_none(qkv.get("group_size")),
        "sink_tokens": _juju_int_or_none(qkv.get("sink_tokens")),
        "enable_qjl": bool(_juju_bool_or_none(qkv.get("enable_qjl"))),
        "qjl_enabled": bool(_juju_bool_or_none(qkv.get("enable_qjl"))),
        "enable_rotation": bool(_juju_bool_or_none(qkv.get("enable_rotation"))),
        "rotation_enabled": bool(_juju_bool_or_none(qkv.get("enable_rotation"))),
        "rotation_seed": _juju_int_or_none(qkv.get("rotation_seed")),
        "qjl_seed": _juju_int_or_none(qkv.get("qjl_seed")),
        "qjl": qkv_fields["qjl"],
        "rotation": qkv_fields["rotation"],
        "normal": qkv_fields["normal"],
        "outlier": qkv_fields["outlier"],
        "quant_axis_contract": {
            "key": qkv_fields["key_cache_policy"].get("quant_axis"),
            "value": qkv_fields["value_cache_policy"].get("quant_axis"),
            "key_scale_axis": qkv_fields["key_cache_policy"].get("scale_axis"),
            "value_scale_axis": qkv_fields["value_cache_policy"].get("scale_axis"),
            "key_value_quant_axes_may_differ": True,
        },
        "query_policy": qkv_fields["query_policy"],
        "key_cache_policy": qkv_fields["key_cache_policy"],
        "value_cache_policy": qkv_fields["value_cache_policy"],
        "turboquant_policy": qkv_fields["turboquant_policy"],
        "residency": qkv_fields["residency"],
        "cache_layout": qkv_fields["cache_layout"],
        "qkv_policy_contract": qkv_fields["qkv_policy_contract"],
        "scale_dtype": _juju_scalar_value(qkv.get("scale_dtype")) or "float32",
        "zero_dtype": _juju_scalar_value(qkv.get("zero_dtype")) or "float32",
        "entry_dim": entry_dim,
        "max_seq_len": max_seq,
        "page_size_tokens": page_tokens,
        "tokens_per_block": page_tokens,
        "tokens_per_block_power_of_two_required": True,
        "growth_page_tokens": page_tokens,
        "block_pool_contract": {
            "pool_key_fields": ["attention_window_size", "num_key_value_heads", "key_head_dim", "value_head_dim", "qkv_policy_hash"],
            "separate_pool_per_attention_window_and_head_layout": True,
            "allocate_blocks_on_demand": True,
            "reuse_full_blocks": True,
            "partial_reuse": {"enabled": True, "copy_on_partial_reuse": True},
            "prefix_reuse": {
                "enabled": True,
                "block_identity_fields": ["cache_salt_hash", "prefix_hash", "block_token_hash", "qkv_policy_hash"],
                "search_structure": "radix_or_hash_tree",
                "evict_only_reusable_leaf_blocks_when_tree_is_used": True,
            },
            "eviction": {
                "primary": "priority_then_lru",
                "priority_range": [0, 100],
                "host_offload_allowed": True,
                "secondary_offload_min_priority": 35,
                "event_stream_required": True,
            },
        },
        "attention_window_contract": {
            "per_layer_window_required": True,
            "sliding_window_supported": True,
            "global_window_supported": True,
            "window_source_priority": ["gguf_runtime", "source_config", "runtime_arch", "default_full_context"],
            "sliding_window_size": _juju_first_config_value(contract, "sliding_window", "sliding_window_size", "attention.sliding_window"),
            "global_window_size": max_seq,
        },
        "layer_attention_table_ref": "generation_contract.layers.attention_layer_table",
        "residency_policy": _juju_scalar_value(qkv.get("residency_policy")) or "ram_tracked_via_tier_usage_device_vram_when_enabled",
        "quantized": bool(qkv),
        "runtime_cache_policy": qkv_runtime_policy,
        "evaluation_policy": _juju_eval_kv_policy(qkv),
        "cache_identity": {
            "position_major_entry_layout": True,
            "entry_scope": ["request_id", "position", "layer"],
            "qkv_state_scope": qkv_runtime_policy["qkv_state_key_scope"],
            "qkv_head_index_is_not_global_without_layer": True,
            "plain_kv_cache_is_not_runtime_identity": True,
        },
        "accuracy_contract": {
            "ppl_must_use_qkv_backend": True,
            "ppl_plain_reference_forbidden": True,
            "qkv_decode_requires_plain_comparison_probe": False,
            "qkv_fallback_allowed": qkv_runtime_policy["plain_fallback_allowed"],
        },
        "page_layout": "layer_head_position_page_major",
        "executor_contract": "decode_appends_current_token_to_qkv_state_and_reads_qkv_state_directly_no_plain_kv_hot_path",
    }


def build_juju_runtime_access_plan(tensor_records, contract, runtime_arch):
    layers = sorted({
        int(rec.get("execution_layer"))
        for rec in tensor_records or []
        if int(rec.get("execution_layer", -1)) >= 0
    })
    startup_hot = sorted(
        [
            rec for rec in tensor_records or []
            if int(rec.get("hotset_rank", 1000)) <= 256 or rec.get("access_phase") == "startup"
        ],
        key=lambda rec: (
            int(rec.get("hotset_rank", 1000)),
            int(rec.get("juju_offset") or 0),
            str(rec.get("name") or ""),
        ),
    )
    file_groups = {}
    for rec in tensor_records or []:
        group = str(rec.get("file_locality_group") or "zz_misc")
        item = file_groups.setdefault(group, {
            "tensor_count": 0,
            "bytes": 0,
            "min_offset": None,
            "max_end": 0,
            "phases": {},
        })
        begin = int(rec.get("juju_offset") or 0)
        end = begin + int(rec.get("juju_bytes") or rec.get("stream_bytes") or 0)
        item["tensor_count"] += 1
        item["bytes"] += int(rec.get("juju_bytes") or rec.get("stream_bytes") or 0)
        item["min_offset"] = begin if item["min_offset"] is None else min(item["min_offset"], begin)
        item["max_end"] = max(item["max_end"], end)
        phase = str(rec.get("access_phase") or "stream")
        item["phases"][phase] = item["phases"].get(phase, 0) + 1
    file_group_list = []
    for name, item in sorted(file_groups.items(), key=lambda kv: (kv[1]["min_offset"] or 0, kv[0])):
        file_group_list.append({
            "name": name,
            "tensor_count": item["tensor_count"],
            "bytes": item["bytes"],
            "offset_range": [int(item["min_offset"] or 0), int(item["max_end"])],
            "phases": item["phases"],
        })

    per_layer = []
    for layer in layers:
        entries = [rec for rec in tensor_records if int(rec.get("execution_layer", -1)) == layer]
        by_phase = {}
        for rec in entries:
            phase = str(rec.get("access_phase") or "stream")
            phase_item = by_phase.setdefault(phase, {"tensor_count": 0, "bytes": 0, "ops": {}})
            phase_item["tensor_count"] += 1
            phase_item["bytes"] += int(rec.get("juju_bytes") or rec.get("stream_bytes") or 0)
            op = str(rec.get("execution_op") or "weight")
            phase_item["ops"][op] = phase_item["ops"].get(op, 0) + 1
        per_layer.append({
            "layer": layer,
            "phase_summary": by_phase,
            "attention_prefetch": "enqueue_layer_plus_1_attention_during_current_mlp",
            "expert_prefetch": "router_selected_dynamic_plus_coactivation_history",
        })

    layer_prefetch_plan = _juju_build_layer_prefetch_plan(tensor_records, layers)
    kv_layout_contract = _juju_kv_layout_contract(contract, runtime_arch)
    qkv_fields = _juju_qkv_contract_fields(_juju_effective_qkv_schema(contract, runtime_arch))
    expert_tier_entries = _juju_expert_tier_entries(tensor_records, contract)
    expert_offset_table = _juju_expert_offset_table(tensor_records, contract)
    expert_bundle_table = _juju_expert_bundle_table(tensor_records, contract)
    expert_chunk_table = _juju_expert_chunk_table(tensor_records, contract)
    moe_layers, moe_layer_bitmask_words = _juju_moe_layer_bitmask_words(tensor_records)
    executor_tensor_table = [
        _juju_runtime_tensor_ref(rec)
        for rec in sorted(tensor_records or [], key=_juju_execution_sort_key)
    ]
    return {
        "format": "JUJU_RUNTIME_ACCESS_PLAN_V1",
        "version": 2,
        "source": "tensor_index_execution_metadata",
        "sort_key": ["file_locality_group", "execution_layer", "expert_id", "projection_order", "execution_order", "runtime_priority"],
        "executor_contract": {
            "tensor_table_required": True,
            "layer_prefetch_plan_required": True,
            "kv_layout_contract_required": True,
            "executor_must_consume_graph_ir_op_order": True,
            "unknown_required_tensor_or_op_behavior": "fail_closed",
            "tensor_ref_fields": [
                "name",
                "role",
                "layer",
                "op",
                "shape",
                "encoding",
                "row_stride_bytes",
                "offset",
                "bytes",
                "source_offset",
                "source_bytes",
                "expert_source_segments",
            ],
        },
        "executor_tensor_count": len(executor_tensor_table),
        "executor_tensor_table": executor_tensor_table,
        "startup_hotset_count": len(startup_hot),
        "startup_hotset": [
            {
                "name": rec.get("name"),
                "op": rec.get("execution_op"),
                "role": rec.get("graph_role"),
                "priority": rec.get("runtime_priority"),
                "prefetch": rec.get("prefetch_priority"),
                "offset": rec.get("juju_offset"),
                "bytes": rec.get("juju_bytes"),
                "row_stride_bytes": rec.get("row_stride_bytes"),
                "encoding": rec.get("weight_encoding"),
            }
            for rec in startup_hot[:96]
        ],
        "protected_roles": ["token_embedding", "lm_head", "final_norm", "attention", "router", "norm"],
        "file_locality_group_count": len(file_group_list),
        "file_locality_groups": file_group_list,
        "per_layer": per_layer,
        "moe_layers": moe_layers,
        "moe_layer_bitmask_words": moe_layer_bitmask_words,
        "expert_tier_entries": expert_tier_entries,
        "expert_offset_table": expert_offset_table,
        "expert_bundle_table": expert_bundle_table,
        "expert_chunk_table": expert_chunk_table,
        "expert_offset_table_kind": "layer_expert_projection_o1_lookup",
        "expert_chunk_table_kind": "layer_expert_projection_absolute_offset_lookup",
        "router_calibration_manifest": router_calibration_manifest_from_juju_idx({"tensors": tensor_records}),
        "expert_calibration_contract": juju_expert_calibration_contract(),
        "adaptive_runtime_scheduler": juju_adaptive_runtime_scheduler_contract(),
        "layer_prefetch_plan_count": len(layer_prefetch_plan),
        "layer_prefetch_plan": layer_prefetch_plan,
        "prefetch_schedule": {
            "startup": ["token_embedding", "final_norm", "lm_head", "rope", "first_layer_attention", "first_layer_router"],
            "per_layer": [
                "attention_input_norm",
                "qkv_projection",
                "attention_output",
                "post_attention_norm",
                "ffn_norm_and_router",
                "selected_expert_bundle",
                "shared_expert_bundle",
                "post_ffw_norm",
                "layer_output_scale",
            ],
            "lookahead": {
                "attention": "layer_plus_1_during_current_mlp",
                "experts": "router_topk_current_layer_plus_mutable_coactivation_next_use",
                "eviction": "protect_startup_hotset_and_current_next_layer",
            },
        },
        "kv_layout_contract": kv_layout_contract,
        "qkv_policy_contract": qkv_fields["qkv_policy_contract"],
        "qkv_cache_schema_effective": qkv_fields["qkv_cache_schema_effective"],
        "eval_kv_policy": qkv_fields["evaluation_policy"],
        "attention_layer_contract_table": _juju_attention_layer_contract_table(tensor_records, runtime_arch),
        "bottleneck_trace_contract": {
            "contract_required": True,
            "qkv_contract": qkv_fields,
            "kv_layout_ref": "runtime_access_plan.kv_layout_contract",
            "qkv_policy_ref": "runtime_access_plan.qkv_policy_contract",
            "token_level": ["forward_token_begin", "cpu_ram", "kv_cache", "io_pipeline", "forward_layer", "forward_token_end"],
            "stage_level": ["forward_embed", "attn_standard_qkv_norm", "mlp_moe_end", "lm_head_logprob_end"],
            "required_counters": [
                "token_index",
                "load_ms",
                "io_wait_ms",
                "attention_ms",
                "mlp_ms",
                "lm_head_ms",
                "kv_ms",
                "kv_backend",
                "kv_entry_dim",
                "kv_page_size_tokens",
                "qkv_error_vs_plain",
                "ram_used_bytes",
                "vram_used_bytes",
                "gpu_util_pct",
                "db_used_bytes",
                "cpu_pct",
                "rss_bytes",
                "queue_depth",
                "inflight",
                "kv_bytes",
                "qkv_k_bits",
                "qkv_v_bits",
                "qkv_normal_bits",
                "qkv_outlier_bits",
                "qkv_key_normal_bits",
                "qkv_value_normal_bits",
                "qkv_key_outlier_bits",
                "qkv_value_outlier_bits",
                "qkv_query_cached",
                "qkv_group_size",
                "qkv_page_size_tokens",
                "qkv_qjl_enabled",
                "qkv_rotation_enabled",
                "process_rss_bytes",
                "available_ram_bytes",
                "device_total_bytes",
                "device_free_bytes",
                "expert_hit_rate",
                "expert_miss_latency_us",
                "prefetch_waste_ratio",
                "gpu_idle_gap_us",
                "disk_read_bytes",
                "pcie_copy_bytes",
            ],
        },
    }


def _juju_bucket_stats(tensor_records):
    stats = {}
    for rec in tensor_records or []:
        bucket = str(rec.get("bucket") or "unknown")
        item = stats.setdefault(bucket, {
            "tensor_count": 0,
            "bytes": 0,
            "layers": [],
        })
        item["tensor_count"] += 1
        item["bytes"] += int(rec.get("juju_bytes") or rec.get("source_bytes") or 0)
        layer = _juju_layer_id_from_name(rec.get("name"))
        if layer is not None and layer not in item["layers"]:
            item["layers"].append(int(layer))
    for item in stats.values():
        item["layers"] = sorted(item["layers"])
        item["layer_count"] = len(item["layers"])
    return stats


def _juju_expert_projection_name(name):
    lower = str(name or "").lower()
    if "ffn_gate_up_exps.weight" in lower or "gate_up" in lower:
        return "gate_up"
    if "ffn_gate_exps.weight" in lower or "gate_proj" in lower or "ffn_gate" in lower:
        return "gate"
    if "ffn_up_exps.weight" in lower or "up_proj" in lower or "ffn_up" in lower:
        return "up"
    if "ffn_down_exps.weight" in lower or "down_proj" in lower or "ffn_down" in lower:
        return "down"
    return "expert"


def _juju_projection_order_value(name):
    role = juju_bundle_member_role_from_name(name)
    if role in JUJU_EXPERT_BUNDLE_MEMBER_RANK:
        return JUJU_EXPERT_BUNDLE_MEMBER_RANK[role]
    proj = _juju_expert_projection_name(name)
    return {"gate_up": 0, "gate": 1, "up": 3, "down": 6}.get(proj, 99)


def _juju_expert_id_from_name(name):
    lower = str(name or "").lower()
    patterns = (
        r"(?:^|[._])experts?[._](\d+)(?:[._]|$)",
        r"(?:^|[._])exps[._](\d+)(?:[._]|$)",
        r"(?:^|[._])expert_(\d+)(?:[._]|$)",
    )
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            return int(m.group(1))
    return None


def _juju_expert_count_from_shape(shape):
    values = [int(v or 0) for v in (shape or [])]
    if len(values) >= 3 and values[2] > 0:
        return int(values[2])
    return 0


def _juju_expert_activation_prior(contract, layer, expert, expert_count):
    stats = contract.get("expert_activation_stats") or contract.get("activation_stats") or {}
    layer_keys = (str(layer), int(layer))
    expert_keys = (str(expert), int(expert))
    for layer_key in layer_keys:
        layer_value = stats.get(layer_key) if isinstance(stats, dict) else None
        if isinstance(layer_value, dict):
            candidates = (
                layer_value.get("experts"),
                layer_value.get("expert_frequency"),
                layer_value.get("activation_frequency"),
                layer_value,
            )
            for candidate in candidates:
                if isinstance(candidate, dict):
                    for expert_key in expert_keys:
                        value = candidate.get(expert_key)
                        if isinstance(value, dict):
                            value = (
                                value.get("frequency") or
                                value.get("activation_prior") or
                                value.get("probability") or
                                value.get("count")
                            )
                        if value is not None:
                            try:
                                return max(0.0, float(value))
                            except Exception:
                                pass
                elif isinstance(candidate, list) and 0 <= int(expert) < len(candidate):
                    try:
                        return max(0.0, float(candidate[int(expert)]))
                    except Exception:
                        pass
    flat = contract.get("expert_activation_priors")
    if isinstance(flat, list):
        for item in flat:
            if not isinstance(item, dict):
                continue
            if int(item.get("layer", -1)) == int(layer) and int(item.get("expert", -1)) == int(expert):
                try:
                    return max(0.0, float(item.get("activation_prior") or item.get("frequency") or item.get("score") or 0.0))
                except Exception:
                    return 0.0
    return 1.0 / max(1, int(expert_count or 1))


def _juju_activation_prior_source(contract):
    if contract.get("expert_activation_stats") or contract.get("activation_stats") or contract.get("expert_activation_priors"):
        return "calibration_or_runtime_stats"
    return "structural_uniform_no_calibration"


def _juju_static_buddy_ids(expert, expert_count, max_buddies=MOE_JUJU_MAX_BUDDIES if "MOE_JUJU_MAX_BUDDIES" in globals() else 8):
    n = int(expert_count or 0)
    e = int(expert or 0)
    if n <= 1:
        return []
    candidates = []
    for delta in (1, -1, 2, -2, 4, -4, 8, -8):
        buddy = (e + delta) % n
        if buddy != e and buddy not in candidates:
            candidates.append(buddy)
        if len(candidates) >= int(max_buddies):
            break
    return candidates


def _juju_expert_tier_entries(tensor_records, contract):
    layer_to_experts = {}
    for rec in tensor_records or []:
        if not is_routed_expert_tensor_name(rec.get("name")):
            continue
        layout = rec.get("expert_layout") or {}
        layer = layout.get("layer")
        if layer is None:
            layer = _juju_layer_id_from_name(rec.get("name"))
        if layer is None:
            continue
        if rec.get("bundle_native") and rec.get("expert_id") is not None:
            count = int(rec.get("expert_id") or 0) + 1
        else:
            count = _juju_expert_count_from_shape(rec.get("shape"))
        explicit = _juju_expert_id_from_name(rec.get("name"))
        if count <= 0 and explicit is not None:
            count = explicit + 1
        if count <= 0:
            continue
        layer_to_experts[int(layer)] = max(int(count), layer_to_experts.get(int(layer), 0))

    hot_pct = float(contract_value(contract, "expert_tier_hot_percentile", "expert_tier_policy.hot_percentile", default=0.10) or 0.10)
    warm_pct = float(contract_value(contract, "expert_tier_warm_percentile", "expert_tier_policy.warm_percentile", default=0.35) or 0.35)
    hot_pct = min(max(hot_pct, 0.0), 1.0)
    warm_pct = min(max(warm_pct, hot_pct), 1.0)
    source = _juju_activation_prior_source(contract)
    out = []
    for layer in sorted(layer_to_experts):
        expert_count = int(layer_to_experts[layer])
        rows = []
        total = 0.0
        for expert in range(expert_count):
            prior = _juju_expert_activation_prior(contract, layer, expert, expert_count)
            rows.append([expert, prior])
            total += prior
        if total > 0.0:
            rows = [[expert, prior / total] for expert, prior in rows]
        rows.sort(key=lambda item: (-item[1], item[0]))
        hot_n = int(math.ceil(expert_count * hot_pct)) if source != "structural_uniform_no_calibration" else 0
        warm_n = int(math.ceil(expert_count * warm_pct)) if source != "structural_uniform_no_calibration" else 0
        for rank, (expert, prior) in enumerate(rows):
            if rank < hot_n:
                tier = "hot"
                residency = "FAST_MEM_STREAMABLE"
                runtime_priority = 78
                prefetch_priority = 94
            elif rank < warm_n:
                tier = "warm"
                residency = "FAST_MEM_STREAMABLE"
                runtime_priority = 70
                prefetch_priority = 88
            else:
                tier = "cold"
                residency = "SLOW_MEM"
                runtime_priority = 65
                prefetch_priority = 80
            out.append({
                "layer": int(layer),
                "expert": int(expert),
                "rank": int(rank),
                "tier": tier,
                "activation_prior": round(float(prior), 12),
                "activation_prior_source": source,
                "runtime_priority": int(runtime_priority),
                "prefetch_priority": int(prefetch_priority),
                "residency_hint": residency,
                "buddy_expert_ids": _juju_static_buddy_ids(expert, expert_count),
            })
    return out


def _juju_expert_tier_lookup(tensor_records, contract):
    return {
        (int(item["layer"]), int(item["expert"])): item
        for item in _juju_expert_tier_entries(tensor_records, contract)
    }


def _juju_segment_ranges(offset, size, segment_count):
    offset = int(offset or 0)
    size = int(size or 0)
    segment_count = max(1, int(segment_count or 1))
    if size <= 0:
        return []
    target = int(math.ceil(size / float(segment_count)))
    out = []
    cursor = offset
    remaining = size
    for segment_id in range(segment_count):
        if remaining <= 0:
            break
        take = min(remaining, target)
        if segment_id == segment_count - 1:
            take = remaining
        out.append({
            "segment": int(segment_id),
            "offset": int(cursor),
            "bytes": int(take),
            "end": int(cursor + take),
        })
        cursor += take
        remaining -= take
    return out


def _juju_expert_row_column_range(rec, expert, expert_count):
    shape = [int(v or 0) for v in (rec.get("shape") or [])]
    cols = int(shape[0]) if shape else int(rec.get("logical_cols") or 0)
    rows = int(rec.get("logical_rows") or 0)
    if len(shape) >= 3 and int(shape[2] or 0) > 0:
        rows_per_expert = int(shape[1] or 0)
        return {
            "row_range": [int(expert) * rows_per_expert, (int(expert) + 1) * rows_per_expert],
            "column_range": [0, cols],
            "rows_per_expert": rows_per_expert,
            "cols": cols,
        }
    return {
        "row_range": [0, rows],
        "column_range": [0, cols],
        "rows_per_expert": rows,
        "cols": cols,
    }


def _juju_expert_source_segments_for_record(rec):
    if not is_routed_expert_tensor_name(rec.get("name")):
        return []
    layout = rec.get("expert_layout") or {}
    layer = layout.get("layer")
    if layer is None:
        layer = _juju_layer_id_from_name(rec.get("name"))
    expert_count = int(layout.get("expert_count") or _juju_expert_count_from_shape(rec.get("shape")) or 0)
    if layer is None or expert_count <= 0:
        return []
    source_bytes = int(rec.get("source_bytes") or rec.get("bytes") or 0)
    juju_bytes = int(rec.get("juju_bytes") or rec.get("stream_bytes") or 0)
    source_base = int(rec.get("source_offset") or 0)
    juju_base = int(rec.get("juju_offset") or rec.get("offset") or 0)
    source_per_expert = source_bytes // max(1, expert_count)
    juju_per_expert = int(layout.get("per_expert_bytes") or (juju_bytes // max(1, expert_count)))
    segment_count = int(layout.get("segment_count_per_expert") or 1)
    projection = layout.get("projection") or _juju_expert_projection_name(rec.get("name"))
    out = []
    for expert in range(expert_count):
        juju_offset = juju_base + expert * juju_per_expert
        source_offset = source_base + expert * source_per_expert
        item = {
            "layer": int(layer),
            "expert": int(expert),
            "projection": projection,
            "tensor": rec.get("name"),
            "juju_offset": int(juju_offset),
            "juju_bytes": int(juju_per_expert),
            "source_offset": int(source_offset),
            "source_bytes": int(source_per_expert),
            "segment_count": int(segment_count),
            "juju_segments": _juju_segment_ranges(juju_offset, juju_per_expert, segment_count),
            "source_segments": _juju_segment_ranges(source_offset, source_per_expert, segment_count),
            **_juju_expert_row_column_range(rec, expert, expert_count),
        }
        split = rec.get("combined_gate_up_split") or {}
        if split.get("enabled"):
            gate_bytes = int(split.get("gate_bytes") or 0)
            up_rel = int(split.get("up_rel_offset") or gate_bytes)
            up_bytes = int(split.get("up_bytes") or 0)
            item["combined_gate_up_split"] = {
                "gate_offset": int(juju_offset + int(split.get("gate_rel_offset") or 0)),
                "gate_bytes": gate_bytes,
                "up_offset": int(juju_offset + up_rel),
                "up_bytes": up_bytes,
                "source_gate_offset": int(source_offset),
                "source_gate_bytes": int(min(gate_bytes, source_per_expert)),
                "source_up_offset": int(source_offset + up_rel),
                "source_up_bytes": int(min(up_bytes, max(0, source_per_expert - up_rel))),
            }
        out.append(item)
    return out


def _juju_moe_layer_bitmask_words(tensor_records):
    layers = sorted({
        int(_juju_layer_id_from_name(rec.get("name")))
        for rec in tensor_records or []
        if is_routed_expert_tensor_name(rec.get("name")) and _juju_layer_id_from_name(rec.get("name")) is not None
    })
    if not layers:
        return [], []
    max_layer = max(layers)
    words = [0] * ((max_layer // 64) + 1)
    for layer in layers:
        words[layer // 64] |= 1 << (layer % 64)
    return layers, words


def _juju_expert_offset_table(tensor_records, contract):
    lookup = _juju_expert_tier_lookup(tensor_records, contract)
    rows = []
    for rec in tensor_records or []:
        if not is_routed_expert_tensor_name(rec.get("name")):
            continue
        layout = rec.get("expert_layout") or {}
        layer = layout.get("layer")
        if layer is None:
            layer = _juju_layer_id_from_name(rec.get("name"))
        bundle_native = bool(rec.get("bundle_native"))
        explicit_expert = rec.get("expert_id")
        expert_count = int(layout.get("expert_count") or _juju_expert_count_from_shape(rec.get("shape")) or 0)
        if bundle_native and explicit_expert is not None:
            expert_count = 1
        base_offset = int(layout.get("base_offset") or rec.get("juju_offset") or rec.get("offset") or 0)
        source_base_offset = int(rec.get("source_offset") or 0)
        total_bytes = int(rec.get("juju_bytes") or rec.get("bytes") or rec.get("source_bytes") or 0)
        total_source_bytes = int(rec.get("source_bytes") or rec.get("bytes") or 0)
        per_expert = int(layout.get("per_expert_bytes") or (total_bytes // max(1, expert_count) if expert_count > 0 else 0))
        source_per_expert = int(total_source_bytes // max(1, expert_count) if expert_count > 0 else 0)
        projection = rec.get("bundle_member_role") or layout.get("role") or layout.get("projection") or _juju_expert_projection_name(rec.get("name"))
        segment_count = int(layout.get("segment_count_per_expert") or 1)
        if layer is None or expert_count <= 0 or per_expert <= 0:
            continue
        split = rec.get("combined_gate_up_split") or {}
        for expert in range(expert_count):
            expert_id = int(explicit_expert) if bundle_native and explicit_expert is not None else int(expert)
            tier = lookup.get((int(layer), int(expert_id)), {})
            offset = int(base_offset + expert * per_expert)
            source_offset = int(source_base_offset + expert * source_per_expert)
            row = {
                "layer": int(layer),
                "expert": int(expert_id),
                "projection": projection,
                "tensor": rec.get("name"),
                "offset": offset,
                "bytes": int(per_expert),
                "end": int(offset + per_expert),
                "source_offset": source_offset,
                "source_bytes": int(source_per_expert),
                "source_end": int(source_offset + source_per_expert),
                "segment_count": segment_count,
                "segments": _juju_segment_ranges(offset, per_expert, segment_count),
                "source_segments": _juju_segment_ranges(source_offset, source_per_expert, segment_count),
                "tier": tier.get("tier", "cold"),
                "activation_prior": tier.get("activation_prior", 0.0),
                "prefetch_priority": int(tier.get("prefetch_priority", rec.get("prefetch_priority") or 80)),
                "runtime_priority": int(tier.get("runtime_priority", rec.get("runtime_priority") or 65)),
                "residency_hint": tier.get("residency_hint", rec.get("residency_hint") or "SLOW_MEM"),
                "bundle_id": rec.get("bundle_id"),
                "bundle_offset": rec.get("bundle_offset"),
                "bundle_size": rec.get("bundle_size"),
                "bundle_alignment": rec.get("bundle_alignment"),
                "bundle_member_offset": rec.get("bundle_member_offset"),
                "bundle_member_size": rec.get("bundle_member_size"),
                "bundle_member_role": rec.get("bundle_member_role"),
                **_juju_expert_row_column_range(rec, 0 if bundle_native else expert, expert_count),
            }
            if split.get("enabled"):
                row["combined_gate_up_split"] = {
                    "gate_offset": int(row["offset"] + int(split.get("gate_rel_offset") or 0)),
                    "gate_bytes": int(split.get("gate_bytes") or 0),
                    "up_offset": int(row["offset"] + int(split.get("up_rel_offset") or 0)),
                    "up_bytes": int(split.get("up_bytes") or 0),
                    "source_gate_offset": int(row["source_offset"]),
                    "source_gate_bytes": int(min(int(split.get("gate_bytes") or 0), source_per_expert)),
                    "source_up_offset": int(row["source_offset"] + int(split.get("up_rel_offset") or 0)),
                    "source_up_bytes": int(min(int(split.get("up_bytes") or 0), max(0, source_per_expert - int(split.get("up_rel_offset") or 0)))),
                }
            rows.append(row)
    rows.sort(key=lambda item: (item["layer"], item["expert"], _juju_projection_order_value(item["projection"])))
    return rows


def _juju_expert_bundle_table(tensor_records, contract):
    offsets = _juju_expert_offset_table(tensor_records, contract)
    grouped = {}
    for row in offsets:
        key = (int(row.get("layer") or 0), int(row.get("expert") or 0))
        item = grouped.setdefault(key, {
            "layer": key[0],
            "expert": key[1],
            "offset": None,
            "end": 0,
            "bytes": 0,
            "projection_count": 0,
            "projections": [],
            "fetch_ranges": [],
            "segments": [],
            "tier": row.get("tier", "cold"),
            "activation_prior": row.get("activation_prior", 0.0),
            "prefetch_priority": int(row.get("prefetch_priority") or 80),
            "runtime_priority": int(row.get("runtime_priority") or 65),
            "residency_hint": row.get("residency_hint", "SLOW_MEM"),
            "segment_count": 1,
        })
        begin = int(row.get("offset") or 0)
        size = int(row.get("bytes") or 0)
        bundle_offset = row.get("bundle_offset")
        bundle_size = row.get("bundle_size")
        if bundle_offset is not None and bundle_size:
            item["offset"] = int(bundle_offset) if item["offset"] is None else min(int(item["offset"]), int(bundle_offset))
            item["end"] = max(int(item["end"]), int(bundle_offset) + int(bundle_size))
        else:
            item["offset"] = begin if item["offset"] is None else min(int(item["offset"]), begin)
            item["end"] = max(int(item["end"]), begin + size)
        end = begin + size
        item["projection_count"] += 1
        item["prefetch_priority"] = max(int(item["prefetch_priority"]), int(row.get("prefetch_priority") or 80))
        item["runtime_priority"] = max(int(item["runtime_priority"]), int(row.get("runtime_priority") or 65))
        projection = {
            "name": row.get("projection"),
            "tensor": row.get("tensor"),
            "offset": begin,
            "bytes": size,
            "end": end,
            "source_offset": int(row.get("source_offset") or 0),
            "source_bytes": int(row.get("source_bytes") or 0),
            "source_end": int(row.get("source_end") or 0),
            "row_range": row.get("row_range"),
            "column_range": row.get("column_range"),
            "segments": row.get("segments") or [],
            "source_segments": row.get("source_segments") or [],
            "order": _juju_projection_order_value(row.get("projection")),
            "bundle_member_offset": int(row.get("bundle_member_offset") or (begin - int(item["offset"] or 0))),
            "bundle_member_size": int(row.get("bundle_member_size") or size),
            "bundle_member_role": row.get("bundle_member_role") or row.get("projection"),
        }
        if row.get("combined_gate_up_split"):
            projection["combined_gate_up_split"] = row["combined_gate_up_split"]
        item["projections"].append(projection)
        item["segments"].extend(row.get("segments") or [])
    bundles = []
    for item in grouped.values():
        item["offset"] = int(item["offset"] or 0)
        item["bytes"] = int(item["end"]) - int(item["offset"])
        item["bundle_id"] = len(bundles)
        item["bundle_offset"] = int(item["offset"])
        item["bundle_size"] = int(item["bytes"])
        item["bundle_alignment"] = JUJU_BUNDLE_ALIGNMENT_BYTES
        item["single_contiguous_fetch"] = (
            item["bundle_offset"] % JUJU_BUNDLE_ALIGNMENT_BYTES == 0 and
            item["bundle_size"] % JUJU_BUNDLE_ALIGNMENT_BYTES == 0
        )
        item["projections"].sort(key=lambda p: (int(p.get("order") or 0), int(p.get("offset") or 0)))
        item["projection_offsets"] = [int(p["offset"]) for p in item["projections"]]
        item["projection_bytes"] = [int(p["bytes"]) for p in item["projections"]]
        item["projection_names"] = [str(p.get("name") or "") for p in item["projections"]]
        item["member_roles"] = [str(p.get("bundle_member_role") or p.get("name") or "") for p in item["projections"]]
        item["member_offsets"] = [int(p.get("bundle_member_offset") or 0) for p in item["projections"]]
        item["member_bytes"] = [int(p.get("bundle_member_size") or p.get("bytes") or 0) for p in item["projections"]]
        item["single_range_fetch"] = all(
            int(item["projections"][i]["end"]) == int(item["projections"][i + 1]["offset"])
            for i in range(max(0, len(item["projections"]) - 1))
        )
        ranges = []
        for projection in item["projections"]:
            ranges.append({
                "offset": int(projection["offset"]),
                "bytes": int(projection["bytes"]),
                "end": int(projection["end"]),
                "projection": projection.get("name"),
                "tensor": projection.get("tensor"),
            })
        ranges.sort(key=lambda row: (int(row["offset"]), str(row.get("projection") or "")))
        merged = []
        for row in ranges:
            if merged and int(merged[-1]["end"]) == int(row["offset"]):
                merged[-1]["bytes"] = int(merged[-1]["bytes"]) + int(row["bytes"])
                merged[-1]["end"] = int(row["end"])
                merged[-1].setdefault("projections", []).append(row.get("projection"))
                continue
            merged.append({
                "offset": int(row["offset"]),
                "bytes": int(row["bytes"]),
                "end": int(row["end"]),
                "projections": [row.get("projection")],
            })
        item["fetch_ranges"] = merged
        item["fetch_range_count"] = len(merged)
        item["segment_count"] = len(item.get("segments") or [])
        bundles.append(item)
    bundles.sort(key=lambda item: (int(item["layer"]), int(item["expert"])))
    return {
        "format": JUJU_EXPERT_BUNDLE_TABLE_FORMAT,
        "bundle_count": len(bundles),
        "schema_version": JUJU_IDX_SCHEMA_VERSION,
        "lookup_key": ["layer", "expert"],
        "bundle_unit": "layer_expert",
        "bundle_alignment": JUJU_BUNDLE_ALIGNMENT_BYTES,
        "member_order": list(JUJU_EXPERT_BUNDLE_MEMBER_ORDER),
        "range_fields": ["offset", "bytes", "end"],
        "source_range_fields": ["source_offset", "source_bytes", "source_end"],
        "projection_fields": ["projection_names", "projection_offsets", "projection_bytes", "member_roles", "member_offsets", "member_bytes"],
        "fetch_range_semantics": "single_4kb_aligned_contiguous_juju_range_per_layer_expert_bundle",
        "bundles": bundles,
    }


def _juju_expert_chunk_table(tensor_records, contract):
    rows = _juju_expert_offset_table(tensor_records, contract)
    chunks = []
    for row in rows:
        chunks.append({
            "layer": int(row.get("layer") or 0),
            "expert": int(row.get("expert") or 0),
            "projection": row.get("projection"),
            "tensor": row.get("tensor"),
            "offset": int(row.get("offset") or 0),
            "bytes": int(row.get("bytes") or 0),
            "end": int(row.get("end") or 0),
            "source_offset": int(row.get("source_offset") or 0),
            "source_bytes": int(row.get("source_bytes") or 0),
            "source_end": int(row.get("source_end") or 0),
            "row_range": row.get("row_range"),
            "column_range": row.get("column_range"),
            "segment_count": int(row.get("segment_count") or 1),
            "segments": row.get("segments") or [],
            "source_segments": row.get("source_segments") or [],
            "combined_gate_up_split": row.get("combined_gate_up_split"),
            "tier": row.get("tier", "cold"),
            "prefetch_priority": int(row.get("prefetch_priority") or 80),
            "runtime_priority": int(row.get("runtime_priority") or 65),
        })
    chunks.sort(key=lambda item: (
        int(item["layer"]),
        int(item["expert"]),
        _juju_projection_order_value(item.get("projection")),
        int(item["offset"]),
    ))
    return {
        "format": "JUJU_EXPERT_CHUNK_TABLE_V1",
        "lookup_key": ["layer", "expert", "projection"],
        "chunk_count": len(chunks),
        "offset_unit": "absolute_juju_file_byte_offset",
        "source_offset_unit": "absolute_source_gguf_file_byte_offset",
        "chunks": chunks,
    }


def juju_section_priority(bucket):
    table = {
        "shared_weights": 100,
        "hot_experts": 88,
        "warm_experts": 76,
        "cold_experts": 60,
        "vision_projector": 58,
        "vision_encoder": 48,
        "audio_encoder": 46,
        "video_encoder": 44,
        "document_encoder": 42,
    }
    return int(table.get(str(bucket or ""), 50))


def juju_section_io_hints(bucket, size=0, contract=None):
    bucket = str(bucket or "")
    hw = {}
    if isinstance(contract, dict):
        hw = (
            contract.get("juju_hw") or
            contract.get("hardware_profile") or
            contract.get("hardware_probe") or
            {}
        )
    nvme_block = int(contract_value(hw, "nvme.sequential_block_size", "storage.sequential_block_size", default=0) or 0)
    if bucket == "cold_experts":
        seq = nvme_block or 512 * 1024
        rnd = 4096
        mmap = 0
        distance = 3 if int(size or 0) >= 512 * 1024 * 1024 else 2
    elif bucket == "warm_experts":
        seq = nvme_block or 64 * 1024
        rnd = 4096
        mmap = 0
        distance = 2
    elif bucket == "hot_experts":
        seq = 64 * 1024
        rnd = 4096
        mmap = 1
        distance = 1
    elif bucket == "shared_weights":
        seq = 4096
        rnd = 4096
        mmap = 1
        distance = 0
    else:
        seq = 256 * 1024
        rnd = 4096
        mmap = 0
        distance = 1
    return {
        "sequential_block_size": int(seq),
        "random_block_size": int(rnd),
        "mmap_friendly": int(mmap),
        "prefetch_distance": int(distance),
        "section_priority": juju_section_priority(bucket),
    }


def juju_tensor_expert_layout_fields(tensor, tensor_offset, layout, contract):
    name = tensor.get("name")
    if not is_routed_expert_tensor_name(name):
        return {}
    shape = [int(v or 0) for v in (tensor.get("shape") or [])]
    expert_count = _juju_expert_count_from_shape(shape)
    if expert_count <= 0:
        explicit = _juju_expert_id_from_name(name)
        expert_count = explicit + 1 if explicit is not None else 0
    if expert_count <= 0:
        return {"expert_layout": {"kind": "routed_expert_unknown_axis", "requires_tensor_index_lookup": True}}
    total_bytes = int(layout.get("juju_bytes") or tensor.get("bytes") or 0)
    per_expert = total_bytes // max(1, expert_count)
    combined_gate_up = "ffn_gate_up_exps.weight" in str(name or "").lower() or _juju_expert_projection_name(name) == "gate_up"
    split_bytes = per_expert // 2 if combined_gate_up else 0
    segment_target = int(contract_value(
        contract,
        "preferred_segment_bytes",
        "chunk_io_contract.preferred_segment_bytes",
        "expert_segmentation_contract.preferred_segment_bytes",
        default=1024 * 1024,
    ) or (1024 * 1024))
    segment_count = max(1, min(4, int(math.ceil(per_expert / max(1, segment_target)))))
    return {
        "expert_axis": 2 if len(shape) >= 3 else -1,
        "expert_count": int(expert_count),
        "per_expert_bytes": int(per_expert),
        "expert_offset_formula": "juju_offset + expert_id * per_expert_bytes",
        "expert_layout": {
            "kind": "expert_contiguous_inside_tensor",
            "layer": _juju_layer_id_from_name(name),
            "projection": _juju_expert_projection_name(name),
            "expert_axis": 2 if len(shape) >= 3 else -1,
            "expert_count": int(expert_count),
            "base_offset": int(tensor_offset),
            "per_expert_bytes": int(per_expert),
            "segment_count_per_expert": int(segment_count),
            "segment_bytes_target": int(segment_target),
        },
        "combined_gate_up_split": {
            "enabled": bool(combined_gate_up),
            "gate_rel_offset": 0,
            "gate_bytes": int(split_bytes) if combined_gate_up else 0,
            "up_rel_offset": int(split_bytes) if combined_gate_up else 0,
            "up_bytes": int(per_expert - split_bytes) if combined_gate_up else 0,
        },
    }


def juju_tensor_segmentation_fields(name, bucket, contract):
    segment_policy = u32(contract_value(contract, "segment_policy", "expert_segmentation_contract.segment_policy", default=2))
    allow_partial = bool(contract_value(
        contract,
        "allow_partial_expert_segments",
        "expert_segmentation_contract.allow_partial_expert_segments",
        default=False,
    ))
    importance_ordered = bool(contract_value(
        contract,
        "importance_ordered_rows",
        "expert_segmentation_contract.importance_ordered_rows",
        default=False,
    ))
    runtime_partial = bool(contract_value(
        contract,
        "partial_execution_runtime_enabled",
        "expert_segmentation_contract.partial_execution_runtime_enabled",
        default=False,
    ))
    routed = is_routed_expert_tensor_name(name)
    can_partial = bool(routed and allow_partial and importance_ordered and runtime_partial and segment_policy in {3, 4})
    partial_accuracy = float(contract_value(
        contract,
        "partial_accuracy",
        "expert_segmentation_contract.partial_accuracy",
        default=(0.90 if can_partial else 0.0),
    ) or 0.0)
    return {
        "segment_policy": int(segment_policy),
        "can_partial_exec": bool(can_partial),
        "partial_accuracy": float(partial_accuracy if can_partial else 0.0),
        "segment_contract": {
            "policy": int(segment_policy),
            "partial_enabled": bool(can_partial),
            "requires_importance_ordered_rows": True,
            "requires_runtime_partial_kernel": True,
            "exact_ppl_default": True,
        },
    }


def _juju_layer_expert_groups(tensor_records):
    layers = {}
    for rec in tensor_records or []:
        name = str(rec.get("name") or "")
        if not is_routed_expert_tensor_name(name):
            continue
        layer = _juju_layer_id_from_name(name)
        if layer is None:
            continue
        layer_entry = layers.setdefault(int(layer), {})
        proj = _juju_expert_projection_name(name)
        layer_entry.setdefault(proj, []).append({
            "name": name,
            "bucket": rec.get("bucket"),
            "shape": rec.get("shape"),
            "bytes": int(rec.get("juju_bytes") or rec.get("source_bytes") or 0),
            "prefetch_priority": int(rec.get("prefetch_priority") or 0),
            "runtime_priority": int(rec.get("runtime_priority") or 0),
        })
    out = []
    for layer in sorted(layers):
        groups = {
            key: sorted(value, key=lambda item: item["name"])
            for key, value in sorted(layers[layer].items())
        }
        out.append({
            "layer": layer,
            "projection_groups": groups,
            "tensor_count": sum(len(value) for value in groups.values()),
            "bytes": sum(item["bytes"] for value in groups.values() for item in value),
        })
    return out


def build_juju_predictor_section(tensor_records, contract, split_meta):
    expert_layers = _juju_layer_expert_groups(tensor_records)
    tier_entries = _juju_expert_tier_entries(tensor_records, contract)
    by_layer = {}
    for item in tier_entries:
        by_layer.setdefault(int(item["layer"]), []).append(item)
    transitions = []
    for layer in sorted(by_layer):
        next_layer = layer + 1
        if next_layer not in by_layer:
            continue
        src = sorted(by_layer[layer], key=lambda item: item["rank"])
        dst = sorted(by_layer[next_layer], key=lambda item: item["rank"])
        for rank, item in enumerate(src):
            if not dst:
                break
            target = dst[min(rank, len(dst) - 1)]
            transitions.append({
                "from_layer": int(layer),
                "from_expert": int(item["expert"]),
                "to_layer": int(next_layer),
                "to_expert": int(target["expert"]),
                "probability": round(float(item.get("activation_prior") or 0.0), 12),
                "source": item.get("activation_prior_source", "structural_uniform_no_calibration"),
            })
    source = _juju_activation_prior_source(contract)
    return {
        "format": "JUJU_PREDICTOR_BOOTSTRAP_V1",
        "trained_weights_embedded": source != "structural_uniform_no_calibration",
        "predictor_payload_embedded": True,
        "runtime_mutable": True,
        "scope": "layer_expert_prefetch",
        "split": split_meta,
        "inputs": [
            "router_scores",
            "gate_input_snapshots",
            "mutable_coactivation_index",
            "prefetch_miss_feedback",
        ],
        "fallback_policy": {
            "when_no_history": "activation_prior_then_router_score_order",
            "cold_start": "calibration_prior_then_static_transition_then_runtime_hits",
            "avoid_model_specific_constants": True,
        },
        "activation_prior_source": source,
        "expert_activation_priors": tier_entries,
        "cross_layer_transition_prior": transitions,
        "calibration_contract": {
            "mutable_idx_update_only": True,
            "requires_full_weight_rewrite": False,
            "router_tensors_only": True,
            "expert_weights_required": False,
            "updates": [
                "expert_activation_priors",
                "cross_layer_transition_prior",
                "buddy_expert_ids",
                "expert_tier_entries",
            ],
        },
        "expert_layer_count": len(expert_layers),
        "expert_tensor_count": sum(layer["tensor_count"] for layer in expert_layers),
        "metadata_only": False,
    }


def build_juju_buddy_map_section(tensor_records, contract, split_meta):
    expert_layers = _juju_layer_expert_groups(tensor_records)
    tier_lookup = _juju_expert_tier_lookup(tensor_records, contract)
    buddy_units = []
    expert_bundles = []
    for layer in expert_layers:
        projections = layer["projection_groups"]
        max_experts = 0
        per_projection_bytes = {}
        for proj, entries in projections.items():
            proj_bytes = 0
            proj_experts = 0
            for entry in entries:
                shape = entry.get("shape") or []
                experts = int(shape[2]) if len(shape) >= 3 and int(shape[2] or 0) > 0 else 0
                if experts > 0:
                    proj_experts = max(proj_experts, experts)
                    proj_bytes += int(entry.get("bytes") or 0) // max(1, experts)
            if proj_experts > 0:
                max_experts = max(max_experts, proj_experts)
                per_projection_bytes[proj] = proj_bytes
        buddy_units.append({
            "layer": layer["layer"],
            "unit": "routed_expert_projection_bundle",
            "projection_order": [p for p in ("gate_up", "gate", "up", "down") if p in projections],
            "projection_groups": projections,
            "tensor_count": layer["tensor_count"],
            "bytes": layer["bytes"],
            "expert_count": max_experts,
        })
        if max_experts > 0:
            projection_order = [p for p in ("gate_up", "gate", "up", "down") if p in per_projection_bytes]
            bundle_bytes = sum(per_projection_bytes.get(p, 0) for p in projection_order)
            for expert_id in range(max_experts):
                tier = tier_lookup.get((int(layer["layer"]), int(expert_id)), {})
                expert_bundles.append({
                    "layer": layer["layer"],
                    "expert": expert_id,
                    "unit": "routed_expert_projection_bundle",
                    "projection_order": projection_order,
                    "bytes": int(bundle_bytes),
                    "activation_prior": tier.get("activation_prior", 0.0),
                    "prefetch_priority": int(tier.get("prefetch_priority", 80)),
                    "residency_hint": tier.get("residency_hint", "SLOW_MEM"),
                    "buddy_expert_ids": tier.get("buddy_expert_ids") or _juju_static_buddy_ids(expert_id, max_experts),
                    "coactivation_source": tier.get("activation_prior_source", "static_same_layer_ring"),
                })
    return {
        "format": "JUJU_BUDDY_MAP_V1",
        "construction": "generic_layer_projection_grouping",
        "split": split_meta,
        "buddy_units": buddy_units,
        "expert_bundles": expert_bundles,
        "unit_count": len(buddy_units),
        "expert_bundle_count": len(expert_bundles),
        "tensor_count": sum(unit["tensor_count"] for unit in buddy_units),
        "runtime_update_allowed": True,
    }


def build_juju_tier_hint_section(tensor_records, contract, split_meta):
    stats = _juju_bucket_stats(tensor_records)
    routed = [
        rec for rec in tensor_records or []
        if is_routed_expert_tensor_name(rec.get("name"))
    ]
    tier_entries = _juju_expert_tier_entries(tensor_records, contract)
    moe_layers, moe_layer_mask = _juju_moe_layer_bitmask_words(tensor_records)
    expert_offset_table = _juju_expert_offset_table(tensor_records, contract)
    expert_bundle_table = _juju_expert_bundle_table(tensor_records, contract)
    expert_chunk_table = _juju_expert_chunk_table(tensor_records, contract)
    tier_counts = {"hot": 0, "warm": 0, "cold": 0}
    for item in tier_entries:
        tier_counts[item.get("tier", "cold")] = tier_counts.get(item.get("tier", "cold"), 0) + 1
    return {
        "format": "JUJU_TIER_HINT_V1",
        "split": split_meta,
        "bootstrap_policy": {
            "source": _juju_activation_prior_source(contract),
            "hot_layers_env": "JUJU_BOOTSTRAP_HOT_EXPERT_LAYERS",
            "warm_layers_env": "JUJU_BOOTSTRAP_WARM_EXPERT_LAYERS",
            "runtime_stats_override": True,
            "hardware_cache_override": True,
            "name_pattern_is_fallback_only": True,
        },
        "bucket_stats": stats,
        "routed_expert_tensor_count": len(routed),
        "hot_expert_tensor_count": int(stats.get("hot_experts", {}).get("tensor_count") or 0),
        "warm_expert_tensor_count": int(stats.get("warm_experts", {}).get("tensor_count") or 0),
        "cold_expert_tensor_count": int(stats.get("cold_experts", {}).get("tensor_count") or 0),
        "expert_tier_entries": tier_entries,
        "expert_tier_counts": tier_counts,
        "expert_offset_table": expert_offset_table,
        "expert_bundle_table": expert_bundle_table,
        "expert_chunk_table": expert_chunk_table,
        "expert_offset_table_kind": "layer_expert_projection_o1_lookup",
        "expert_chunk_table_kind": "layer_expert_projection_absolute_offset_lookup",
        "moe_layers": moe_layers,
        "moe_layer_bitmask_words": moe_layer_mask,
        "dense_layer_skip_ready": True,
        "combined_gate_up_split_offsets_ready": any("combined_gate_up_split" in row for row in expert_offset_table),
        "qkv_policy_required": True,
    }


def normalize_juju_expert_calibration_stats(stats):
    out = []
    if not stats:
        return out
    if isinstance(stats, dict):
        iterable = stats.items()
    elif isinstance(stats, list):
        iterable = enumerate(stats)
    else:
        return out
    for layer_key, layer_value in iterable:
        try:
            layer = int(str(layer_key).replace("layer_", "").replace("layer.", ""))
        except Exception:
            continue
        if isinstance(layer_value, dict):
            expert_values = (
                layer_value.get("experts") or
                layer_value.get("expert_frequency") or
                layer_value.get("activation_frequency") or
                layer_value.get("activation_counts") or
                layer_value
            )
        else:
            expert_values = layer_value
        if isinstance(expert_values, dict):
            pairs = expert_values.items()
        elif isinstance(expert_values, list):
            pairs = enumerate(expert_values)
        else:
            continue
        rows = []
        total = 0.0
        for expert_key, value in pairs:
            try:
                expert = int(str(expert_key).replace("expert_", "").replace("expert.", ""))
            except Exception:
                continue
            if isinstance(value, dict):
                value = value.get("frequency") or value.get("activation_prior") or value.get("probability") or value.get("count")
            try:
                score = max(0.0, float(value))
            except Exception:
                continue
            rows.append([expert, score])
            total += score
        if total > 0.0:
            for expert, score in rows:
                out.append({
                    "layer": int(layer),
                    "expert": int(expert),
                    "activation_prior": round(float(score / total), 12),
                })
    return out


def router_calibration_manifest_from_juju_idx(idx):
    tensors = idx.get("tensors") or (idx.get("graph_ir") or {}).get("runtime_access_plan", {}).get("executor_tensor_table") or []
    routers = []
    for rec in tensors:
        if str(rec.get("execution_op") or rec.get("op") or "").lower() not in {"moe_router", "moe_router_scale"}:
            continue
        routers.append({
            "name": rec.get("name"),
            "layer": _juju_layer_id_from_name(rec.get("name")),
            "offset": int(rec.get("juju_offset") or rec.get("offset") or 0),
            "bytes": int(rec.get("juju_bytes") or rec.get("bytes") or rec.get("source_bytes") or 0),
            "row_stride_bytes": int(rec.get("row_stride_bytes") or 0),
            "encoding": rec.get("weight_encoding") or rec.get("encoding"),
            "shape": list(rec.get("shape") or []),
        })
    return {
        "format": "JUJU_ROUTER_CALIBRATION_MANIFEST_V1",
        "router_tensor_count": len(routers),
        "requires_full_weight_download": False,
        "requires_expert_weight_download": False,
        "range_read_only": True,
        "router_tensors": routers,
    }


def apply_juju_expert_calibration_to_idx(idx, calibration_stats):
    if not isinstance(idx, dict):
        raise TypeError("idx must be a dict")
    priors = normalize_juju_expert_calibration_stats(calibration_stats)
    if not priors:
        raise ValueError("no calibration expert activation priors found")
    contract = {
        "expert_activation_priors": priors,
        "expert_tier_policy": {
            "hot_percentile": 0.10,
            "warm_percentile": 0.35,
            "hot_threshold": 0.15,
            "warm_threshold": 0.05,
        },
    }
    tensor_records = idx.get("tensors") or []
    tier_entries = _juju_expert_tier_entries(tensor_records, contract)
    offset_table = _juju_expert_offset_table(tensor_records, contract)
    moe_layers, moe_mask = _juju_moe_layer_bitmask_words(tensor_records)
    idx["expert_activation_priors"] = priors
    idx["expert_tier_entries"] = tier_entries
    idx["expert_offset_table"] = offset_table
    idx["moe_layers"] = moe_layers
    idx["moe_layer_bitmask_words"] = moe_mask
    idx["router_calibration_manifest"] = router_calibration_manifest_from_juju_idx(idx)
    idx["expert_calibration_contract"] = juju_expert_calibration_contract()
    if isinstance(calibration_stats, dict):
        for src_key, dst_key in (
            ("expert_access_count", "expert_access_count"),
            ("coactivation", "expert_coactivation_table"),
            ("coactivation_table", "expert_coactivation_table"),
            ("transition_table", "expert_transition_table"),
            ("router_trace_summary", "router_trace_summary"),
        ):
            value = calibration_stats.get(src_key)
            if value not in (None, "", [], {}):
                idx[dst_key] = value
    idx["calibration_update"] = {
        "format": "JUJU_IDX_CALIBRATION_UPDATE_V1",
        "mutable_idx_only": True,
        "requires_juju_weight_rewrite": False,
        "requires_full_weight_download": False,
        "activation_prior_count": len(priors),
        "tier_entry_count": len(tier_entries),
        "expert_offset_entry_count": len(offset_table),
        "required_for_max_offload_claim": True,
        "required_trace_fields": juju_expert_calibration_contract()["trace_required_fields"],
    }
    graph_ir = idx.get("graph_ir")
    if isinstance(graph_ir, dict):
        graph_ir["expert_activation_priors"] = priors
        graph_ir["expert_tier_entries"] = tier_entries
        graph_ir["expert_offset_table"] = offset_table
        graph_ir["moe_layer_bitmask_words"] = moe_mask
        graph_ir["expert_calibration_contract"] = juju_expert_calibration_contract()
        policy = graph_ir.setdefault("moe_offload_policy", {})
        if isinstance(policy, dict):
            policy["expert_calibration_contract"] = juju_expert_calibration_contract()
            prefetch = policy.setdefault("prefetch", {})
            if isinstance(prefetch, dict):
                prefetch["activation_prior_source"] = "calibration_or_runtime_stats"
                prefetch["mutable_idx_calibration_applied"] = True
            if isinstance(calibration_stats, dict):
                for src_key, dst_key in (
                    ("coactivation", "expert_coactivation_table"),
                    ("coactivation_table", "expert_coactivation_table"),
                    ("transition_table", "expert_transition_table"),
                ):
                    value = calibration_stats.get(src_key)
                    if value not in (None, "", [], {}):
                        graph_ir[dst_key] = value
    return idx


def write_calibrated_juju_idx(index_path, calibration_stats, output_path=None):
    index_path = Path(index_path)
    output_path = Path(output_path) if output_path else index_path
    idx = json.loads(index_path.read_text(encoding="utf-8"))
    apply_juju_expert_calibration_to_idx(idx, calibration_stats)
    output_path.write_text(json.dumps(idx, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {
        "ok": True,
        "index_path": str(index_path),
        "output_path": str(output_path),
        "activation_prior_count": len(idx.get("expert_activation_priors") or []),
        "tier_entry_count": len(idx.get("expert_tier_entries") or []),
    }


def juju_runtime_special_tokens_metadata(runtime_arch):
    runtime_arch = dict(runtime_arch or {})
    return {
        "bos_token_id": runtime_arch.get("bos_token_id"),
        "eos_token_id": runtime_arch.get("eos_token_id"),
        "unk_token_id": runtime_arch.get("unk_token_id"),
        "pad_token_id": runtime_arch.get("pad_token_id"),
        "mask_token_id": runtime_arch.get("mask_token_id"),
        "add_bos_token": runtime_arch.get("add_bos_token"),
        "add_eos_token": runtime_arch.get("add_eos_token"),
        "add_space_prefix": runtime_arch.get("add_space_prefix"),
        "source": "source_config_or_gguf_runtime_metadata",
        "source_by_field": runtime_arch.get("special_token_source_by_field") or {},
        "defaulted_fields": runtime_arch.get("special_token_default_sources") or {},
    }


def build_juju_runtime_metadata_sections(tensor_records, contract, split_meta, runtime_arch=None):
    runtime_arch = dict(runtime_arch or contract.get("arch_meta") or {})
    qkv_fields = _juju_qkv_contract_fields(_juju_effective_qkv_schema(contract, runtime_arch))
    runtime_contract = {
        "format": "JUJU_RUNTIME_CONTRACT_SUMMARY_V1",
        "split": split_meta,
        "qkv_contract": qkv_fields,
        "eval_kv_policy": qkv_fields["evaluation_policy"],
        "required_views": [
            "root_metadata",
            "qkv_policy",
            "kv_layout",
            "runtime_policy",
            "graph_ir",
            "execution_correctness",
            "bottleneck_trace",
        ],
        "tokenizer_contract": juju_tokenizer_contract(),
        "special_tokens": juju_runtime_special_tokens_metadata(runtime_arch),
        "validation_contract": juju_validation_contract(),
        "runtime_loop": "tokenizer_contract_then_graph_ir_ops_then_tensor_layout_then_qkv_cache_then_lm_head",
        "runtime_execution_manifest": {
            "format": "JUJU_RUNTIME_EXECUTION_MANIFEST_V1",
            "location": ["graph_ir.runtime_execution_manifest", "idx.runtime_execution_manifest"],
            "engine_read_first": True,
            "model_name_specific_fallback_allowed": False,
        },
        "plain_kv_runtime_allowed": False,
        "graph_ir_required": True,
        "tensor_layout_records_required": True,
        "trace_required": True,
        "exact_mode": juju_exact_mode_policy(),
        "approx_mode": juju_approx_mode_policy(),
        "ppl_acceptance_contract": juju_ppl_acceptance_contract(),
        "performance_acceptance_contract": juju_performance_acceptance_contract(),
        "adaptive_runtime_scheduler": juju_adaptive_runtime_scheduler_contract(),
        "calibration_contract": {
            **juju_expert_calibration_contract(),
            "best_path": "runtime_or_router_activation_stats_update_mutable_idx",
            "requires_juju_weight_rewrite": False,
            "requires_full_weight_download": False,
            "router_range_manifest": router_calibration_manifest_from_juju_idx({"tensors": tensor_records}),
            "idx_update_function": "apply_juju_expert_calibration_to_idx",
        },
    }
    return [
        (JUJU_SECTION_PREDICTOR, "PREDICTOR", build_juju_predictor_section(tensor_records, contract, split_meta)),
        (JUJU_SECTION_BUDDY_MAP, "BUDDY_MAP", build_juju_buddy_map_section(tensor_records, contract, split_meta)),
        (JUJU_SECTION_TIER_HINT, "TIER_HINT", build_juju_tier_hint_section(tensor_records, contract, split_meta)),
        (JUJU_SECTION_RUNTIME_CONTRACT, "RUNTIME_CONTRACT", runtime_contract),
    ]


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


def _juju_model_family_text(contract, runtime=None):
    runtime = runtime or {}
    arch = dict(contract.get("arch_meta") or {})
    source_config = _juju_contract_source_config(contract)
    text_config = _juju_contract_text_config(contract)
    parts = [
        contract.get("architecture"),
        contract.get("model_type"),
        contract.get("model_id"),
        contract.get("model_name"),
        source_config.get("model_type"),
        text_config.get("model_type"),
        arch.get("architecture"),
        arch.get("model_type"),
        arch.get("model_id"),
        runtime.get("declared_architecture"),
        runtime.get("architecture"),
        runtime.get("model_type"),
        runtime.get("model_id"),
        runtime.get("model_name"),
    ]
    return " ".join(str(x).lower() for x in parts if x not in (None, ""))


def _juju_float_or_none(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) and out > 0.0 else None


def _juju_infer_embedding_scale(contract, runtime):
    hidden = _juju_first_int(runtime.get("hidden_size"), runtime.get("hidden_dim"), _juju_first_config_value(contract, "hidden_size", "hidden_dim"))
    explicit = first_present(
        runtime.get("embedding_scale"),
        runtime.get("scale_emb"),
        (contract.get("arch_meta") or {}).get("embedding_scale"),
        (contract.get("arch_meta") or {}).get("scale_emb"),
        _juju_first_config_value(contract, "embedding_scale", "scale_emb"),
    )
    explicit_float = _juju_float_or_none(explicit)
    if explicit_float is not None:
        return explicit_float, "source_config"
    scale_embedding = first_present(
        runtime.get("scale_embedding"),
        (contract.get("arch_meta") or {}).get("scale_embedding"),
        _juju_first_config_value(contract, "scale_embedding"),
    )
    scale_embedding_float = _juju_float_or_none(scale_embedding)
    if scale_embedding_float is not None:
        return scale_embedding_float, "source_config_scale_embedding"
    if hidden and _juju_bool_or_none(scale_embedding) is True:
        return float(math.sqrt(float(hidden))), "source_config_scale_embedding_true"
    family = _juju_model_family_text(contract, runtime)
    for needles, rule, source in JUJU_EMBEDDING_SCALE_FAMILY_RULES:
        if not hidden or not any(needle in family for needle in needles):
            continue
        if rule == "sqrt_hidden_size":
            return float(math.sqrt(float(hidden))), source
    return None, "absent_no_embedding_scale"


def _juju_default_special_token_bool(field, contract, runtime):
    runtime = dict(runtime or {})
    if field == "add_eos_token":
        return False, "default_false_when_source_tokenizer_policy_absent"
    if field == "add_space_prefix":
        explicit = first_present(
            _juju_bool_or_none(runtime.get("add_prefix_space")),
            _juju_bool_or_none(runtime.get("tokenizer_add_prefix_space")),
            _juju_bool_or_none(_juju_first_config_value(contract, "add_prefix_space", "tokenizer_add_prefix_space")),
        )
        if explicit is not None:
            return explicit, "source_add_prefix_space_alias"
        return False, "default_false_when_source_tokenizer_policy_absent"
    if field == "add_bos_token":
        return False, "default_false_when_source_tokenizer_policy_absent"
    return False, "default_false_unknown_special_token_policy"


def _juju_apply_special_token_policy_defaults(contract, fields):
    defaults = {}
    sources = {}
    for field in ("add_bos_token", "add_eos_token", "add_space_prefix"):
        if fields.get(field) is None:
            value, source = _juju_default_special_token_bool(field, contract, fields)
            fields[field] = value
            defaults[field] = source
            sources[field] = source
        else:
            sources[field] = "source_config_or_gguf_runtime_metadata"
    fields["special_token_default_sources"] = defaults
    fields["special_token_source_by_field"] = sources
    return fields


def _juju_runtime_arch_with_special_token_defaults(contract, runtime_arch):
    return _juju_apply_special_token_policy_defaults(contract or {}, dict(runtime_arch or {}))


def juju_runtime_arch_metadata(contract, directory=None):
    arch = dict(contract.get("arch_meta") or {})
    runtime = dict((directory or {}).get("gguf_runtime") or {})
    out = dict(runtime)
    cfg = lambda *keys: _juju_first_config_value(contract, *keys)

    fields = {
        "declared_architecture": first_present(contract.get("architecture"), arch.get("architecture"), runtime.get("declared_architecture"), runtime.get("architecture")),
        "model_id": first_present(contract.get("model_id"), contract.get("model_name"), runtime.get("model_id")),
        "model_name": first_present(contract.get("model_name"), runtime.get("model_name")),
        "tokenizer_model": first_present(runtime.get("tokenizer_model"), runtime.get("tokenizer.ggml.model"), arch.get("tokenizer_model"), cfg("tokenizer_model", "tokenizer_class")),
        "tokenizer_pre": first_present(runtime.get("tokenizer_pre"), runtime.get("tokenizer.ggml.pre"), arch.get("tokenizer_pre"), cfg("tokenizer_pre")),
        "bos_token_id": first_present(runtime.get("bos_token_id"), arch.get("bos_token_id"), cfg("bos_token_id")),
        "eos_token_id": first_present(runtime.get("eos_token_id"), arch.get("eos_token_id"), cfg("eos_token_id")),
        "unk_token_id": first_present(runtime.get("unk_token_id"), runtime.get("unknown_token_id"), arch.get("unk_token_id"), cfg("unk_token_id")),
        "pad_token_id": first_present(runtime.get("pad_token_id"), runtime.get("padding_token_id"), arch.get("pad_token_id"), cfg("pad_token_id")),
        "mask_token_id": first_present(runtime.get("mask_token_id"), arch.get("mask_token_id"), cfg("mask_token_id")),
        "add_bos_token": first_present(_juju_bool_or_none(runtime.get("add_bos_token")), _juju_bool_or_none(runtime.get("tokenizer_add_bos_token")), _juju_bool_or_none(arch.get("add_bos_token")), _juju_bool_or_none(cfg("add_bos_token"))),
        "add_eos_token": first_present(_juju_bool_or_none(runtime.get("add_eos_token")), _juju_bool_or_none(runtime.get("tokenizer_add_eos_token")), _juju_bool_or_none(arch.get("add_eos_token")), _juju_bool_or_none(cfg("add_eos_token"))),
        "add_space_prefix": first_present(_juju_bool_or_none(runtime.get("add_space_prefix")), _juju_bool_or_none(runtime.get("tokenizer_add_space_prefix")), _juju_bool_or_none(arch.get("add_space_prefix")), _juju_bool_or_none(cfg("add_space_prefix"))),
        "num_hidden_layers": first_present(cfg("num_hidden_layers", "n_layers"), arch.get("n_layers"), arch.get("num_hidden_layers"), runtime.get("num_hidden_layers"), runtime.get("n_layers")),
        "hidden_size": first_present(cfg("hidden_size", "hidden_dim"), arch.get("hidden_dim"), arch.get("hidden_size"), runtime.get("hidden_size"), runtime.get("hidden_dim")),
        "vocab_size": first_present(cfg("vocab_size"), arch.get("vocab_size"), runtime.get("vocab_size")),
        "head_dim": first_present(cfg("head_dim", "key_length"), arch.get("head_dim"), runtime.get("head_dim"), runtime.get("key_length")),
        "value_head_dim": first_present(cfg("value_head_dim", "v_head_dim", "value_length"), arch.get("value_head_dim"), arch.get("v_head_dim"), runtime.get("value_head_dim"), runtime.get("v_head_dim")),
        "global_head_dim": first_present(cfg("global_head_dim", "global_key_length"), arch.get("global_head_dim"), runtime.get("global_head_dim")),
        "global_value_head_dim": first_present(cfg("global_value_head_dim", "global_value_length"), arch.get("global_value_head_dim"), runtime.get("global_value_head_dim")),
        "num_attention_heads": first_present(cfg("num_attention_heads", "n_heads"), arch.get("n_heads"), arch.get("num_attention_heads"), runtime.get("num_attention_heads"), runtime.get("n_heads")),
        "num_key_value_heads": first_present(cfg("num_key_value_heads", "n_kv_heads"), arch.get("n_kv_heads"), arch.get("num_key_value_heads"), runtime.get("num_key_value_heads"), runtime.get("n_kv_heads")),
        "num_global_key_value_heads": first_present(cfg("num_global_key_value_heads"), arch.get("num_global_key_value_heads"), runtime.get("num_global_key_value_heads")),
        "attention_k_eq_v": first_present(_juju_bool_or_none(cfg("attention_k_eq_v")), arch.get("attention_k_eq_v"), runtime.get("attention_k_eq_v")),
        "value_norm_mode": first_present(cfg("value_norm_mode", "v_norm_mode", "attention_value_norm_mode", "attention.value_norm_mode"), arch.get("value_norm_mode"), arch.get("v_norm_mode"), arch.get("attention_value_norm_mode"), runtime.get("value_norm_mode"), runtime.get("v_norm_mode"), runtime.get("attention_value_norm_mode")),
        "v_norm_mode": first_present(cfg("v_norm_mode", "value_norm_mode", "attention_value_norm_mode", "attention.value_norm_mode"), arch.get("v_norm_mode"), arch.get("value_norm_mode"), arch.get("attention_value_norm_mode"), runtime.get("v_norm_mode"), runtime.get("value_norm_mode"), runtime.get("attention_value_norm_mode")),
        "unweighted_value_norm": first_present(_juju_bool_or_none(cfg("unweighted_value_norm", "unweighted_v_norm", "attention_unweighted_value_norm", "attention_unweighted_v_norm")), _juju_bool_or_none(arch.get("unweighted_value_norm")), _juju_bool_or_none(arch.get("unweighted_v_norm")), _juju_bool_or_none(runtime.get("unweighted_value_norm")), _juju_bool_or_none(runtime.get("unweighted_v_norm"))),
        "unweighted_value_norm_layers": first_present(cfg("unweighted_value_norm_layers", "unweighted_v_norm_layers", "layers_with_unweighted_v_norm", "layers_with_unweighted_v_norm_contract"), arch.get("unweighted_value_norm_layers"), arch.get("unweighted_v_norm_layers"), arch.get("layers_with_unweighted_v_norm"), arch.get("layers_with_unweighted_v_norm_contract"), runtime.get("unweighted_value_norm_layers"), runtime.get("unweighted_v_norm_layers"), runtime.get("layers_with_unweighted_v_norm"), runtime.get("layers_with_unweighted_v_norm_contract")),
        "kv_lora_rank": first_present(arch.get("kv_lora_rank"), runtime.get("kv_lora_rank")),
        "q_lora_rank": first_present(arch.get("q_lora_rank"), runtime.get("q_lora_rank")),
        "qk_nope_head_dim": first_present(arch.get("qk_nope_head_dim"), runtime.get("qk_nope_head_dim")),
        "qk_rope_head_dim": first_present(arch.get("qk_rope_head_dim"), runtime.get("qk_rope_head_dim")),
        "experts_per_moe_layer": first_present(arch.get("experts_per_moe_layer"), arch.get("n_experts"), runtime.get("experts_per_moe_layer"), runtime.get("n_experts")),
        "routed_experts_per_token": first_present(arch.get("routed_experts_per_token"), arch.get("top_k"), runtime.get("routed_experts_per_token"), runtime.get("top_k")),
        "expert_intermediate_size": first_present(arch.get("expert_intermediate_size"), arch.get("expert_intermediate_dim"), runtime.get("expert_intermediate_size"), runtime.get("expert_intermediate_dim")),
        "rms_norm_eps": first_present(arch.get("rms_norm_eps"), arch.get("norm_eps"), runtime.get("rms_norm_eps"), runtime.get("norm_eps")),
        "norm_eps": first_present(arch.get("norm_eps"), arch.get("rms_norm_eps"), runtime.get("norm_eps"), runtime.get("rms_norm_eps")),
        "rms_norm_unit_offset": first_present(
            _juju_bool_or_none(arch.get("rms_norm_unit_offset")),
            _juju_bool_or_none(arch.get("rmsnorm_unit_offset")),
            _juju_bool_or_none(arch.get("rms_norm_weight_unit_offset")),
            _juju_bool_or_none(runtime.get("rms_norm_unit_offset")),
            _juju_bool_or_none(runtime.get("rmsnorm_unit_offset")),
            _juju_bool_or_none(runtime.get("rms_norm_weight_unit_offset")),
            False,
        ),
        "rmsnorm_unit_offset": first_present(
            _juju_bool_or_none(arch.get("rms_norm_unit_offset")),
            _juju_bool_or_none(arch.get("rmsnorm_unit_offset")),
            _juju_bool_or_none(arch.get("rms_norm_weight_unit_offset")),
            _juju_bool_or_none(runtime.get("rms_norm_unit_offset")),
            _juju_bool_or_none(runtime.get("rmsnorm_unit_offset")),
            _juju_bool_or_none(runtime.get("rms_norm_weight_unit_offset")),
            False,
        ),
        "rope_theta": first_present(arch.get("rope_theta"), runtime.get("rope_theta"), runtime.get("theta")),
        "theta": first_present(arch.get("rope_theta"), runtime.get("theta"), runtime.get("rope_theta")),
        "sliding_window": first_present(arch.get("sliding_window"), runtime.get("sliding_window")),
        "embedding_scale": first_present(arch.get("embedding_scale"), arch.get("scale_emb"), runtime.get("embedding_scale"), runtime.get("scale_emb")),
        "scale_emb": first_present(arch.get("scale_emb"), arch.get("embedding_scale"), runtime.get("scale_emb"), runtime.get("embedding_scale")),
        "final_logit_softcap": first_present(arch.get("final_logit_softcap"), arch.get("final_logit_softcapping"), runtime.get("final_logit_softcap"), runtime.get("final_logit_softcapping"), runtime.get("logit_softcap")),
        "final_logit_softcapping": first_present(arch.get("final_logit_softcapping"), arch.get("final_logit_softcap"), runtime.get("final_logit_softcapping"), runtime.get("final_logit_softcap"), runtime.get("logit_softcap")),
        "attn_logit_softcap": first_present(arch.get("attn_logit_softcap"), arch.get("attn_logit_softcapping"), arch.get("attention_logit_softcap"), arch.get("attention_logit_softcapping"), runtime.get("attn_logit_softcap"), runtime.get("attn_logit_softcapping"), runtime.get("attention_logit_softcap"), runtime.get("attention_logit_softcapping")),
        "attn_logit_softcapping": first_present(arch.get("attn_logit_softcapping"), arch.get("attn_logit_softcap"), arch.get("attention_logit_softcapping"), arch.get("attention_logit_softcap"), runtime.get("attn_logit_softcapping"), runtime.get("attn_logit_softcap"), runtime.get("attention_logit_softcapping"), runtime.get("attention_logit_softcap")),
        "query_pre_attn_scalar": first_present(arch.get("query_pre_attn_scalar"), arch.get("attention_query_pre_attn_scalar"), arch.get("attn_query_pre_attn_scalar"), runtime.get("query_pre_attn_scalar"), runtime.get("attention_query_pre_attn_scalar"), runtime.get("attn_query_pre_attn_scalar")),
        "attention_query_pre_attn_scalar": first_present(arch.get("attention_query_pre_attn_scalar"), arch.get("query_pre_attn_scalar"), arch.get("attn_query_pre_attn_scalar"), runtime.get("attention_query_pre_attn_scalar"), runtime.get("query_pre_attn_scalar"), runtime.get("attn_query_pre_attn_scalar")),
        "attention_scale": first_present(arch.get("attention_scale"), arch.get("attn_scale"), arch.get("attention_score_scale"), arch.get("f_attn_scale"), arch.get("gemma4.attention.scale"), arch.get("qk_scale"), runtime.get("attention_scale"), runtime.get("attn_scale"), runtime.get("attention_score_scale"), runtime.get("f_attn_scale"), runtime.get("gemma4.attention.scale"), runtime.get("qk_scale")),
        "attention_score_scale": first_present(arch.get("attention_score_scale"), arch.get("attention_scale"), arch.get("attn_scale"), arch.get("f_attn_scale"), arch.get("gemma4.attention.scale"), arch.get("qk_scale"), runtime.get("attention_score_scale"), runtime.get("attention_scale"), runtime.get("attn_scale"), runtime.get("f_attn_scale"), runtime.get("gemma4.attention.scale"), runtime.get("qk_scale")),
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
        "layer_types": first_present(cfg("layer_types"), arch.get("layer_types"), runtime.get("layer_types")),
        "rope_parameters": first_present(cfg("rope_parameters"), arch.get("rope_parameters"), runtime.get("rope_parameters")),
        "rope_type": first_present(arch.get("rope_type"), runtime.get("rope_type")),
    }
    if _juju_bool_or_none(fields.get("attention_k_eq_v")) is True:
        if fields.get("head_dim") is not None:
            fields["value_head_dim"] = fields["head_dim"]
        if fields.get("global_head_dim") is not None:
            fields["global_value_head_dim"] = fields["global_head_dim"]
    embedding_scale, embedding_scale_source = _juju_infer_embedding_scale(contract, {**runtime, **fields})
    fields["embedding_scale"] = embedding_scale
    fields["scale_emb"] = embedding_scale
    fields["embedding_scale_source"] = embedding_scale_source
    fields["embedding_scale_semantics"] = "multiply_token_embedding_before_first_layer" if embedding_scale is not None else "none"
    _juju_apply_special_token_policy_defaults(contract, fields)
    for key, value in fields.items():
        if value is not None:
            out[key] = value
    return out


def build_layer_graph_ir(layer, tensors, runtime_arch=None):
    runtime_arch = runtime_arch or {}
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
    dense_weights = bind(
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    attention_norm_weights = bind("attn_norm.weight", "input_layernorm.weight", "pre_attention_norm.weight")
    q_weights = bind(*JUJU_ATTENTION_Q_SUFFIXES)
    k_weights = bind(*JUJU_ATTENTION_K_SUFFIXES)
    v_weights = bind(*JUJU_ATTENTION_V_SUFFIXES)
    q_norm_weights = bind(*JUJU_Q_NORM_SUFFIXES)
    k_norm_weights = bind(*JUJU_K_NORM_SUFFIXES)
    v_norm_weights = bind(*JUJU_V_NORM_SUFFIXES)
    o_weights = bind(*JUJU_ATTENTION_O_SUFFIXES)
    post_attention_norm_weights = bind(
        "post_attention_norm.weight",
        "post_attention_layernorm.weight",
        "post_attention_layer_norm.weight",
        "post_attn_norm.weight",
    )
    ffn_norm_weights = bind("ffn_norm.weight", "ffn_pre_norm.weight", "pre_ffw_norm.weight", "mlp_norm.weight")
    expert_norm_weights = bind("pre_ffw_norm_2.weight", "ffn_pre_norm_2.weight", "moe_norm.weight")
    router_weights = bind("ffn_gate_inp.weight", "router.weight", "mlp.router.weight", "moe.gate.weight")
    router_scale_weights = bind("ffn_gate_inp.scale", "router.scale", "mlp.router.scale", "moe.gate.scale")
    shared_gate_weights = bind(
        "shared_expert_gate.weight",
        "shared_expert.gate.weight",
        "ffn_shared_gate.weight",
        "shared_gate.weight",
    )
    post_ffw_norm1_weights = bind("post_ffw_norm_1.weight", "ffn_post_norm_1.weight")
    post_ffw_norm2_weights = bind("post_ffw_norm_2.weight", "ffn_post_norm_2.weight")
    post_ffw_norm_weights = bind("post_ffw_norm.weight", "ffn_post_norm.weight")
    layer_output_scale_weights = bind("layer_output_scale.weight", "layer_scalar.weight", "layer_scalar")
    expert_down_scale_weights = bind(
        "ffn_down_exps.scale",
        "ffn_gate_inp.per_expert_scale",
        "router.per_expert_scale",
        "mlp.router.per_expert_scale",
        "moe.gate.per_expert_scale",
    )
    attention_k_eq_v = _juju_bool_or_none(runtime_arch.get("attention_k_eq_v")) is True
    value_projection_present = bool(v_weights)
    value_raw_input = _juju_value_raw_input_name(value_projection_present, attention_k_eq_v)
    layer_suffixes = {_juju_layer_suffix(name) for name in names}
    implicit_unweighted_v_norm = _juju_layer_has_implicit_unweighted_v_norm(layer, layer_suffixes, runtime_arch)
    value_norm_mode = (
        "weighted_rmsnorm" if v_norm_weights else
        "unweighted_rmsnorm_contract" if implicit_unweighted_v_norm else
        "identity"
    )
    rope_contract = _juju_layer_rope_contract(layer, runtime_arch)
    attention_cache_contract = {
        "kv_cache": "qkv_quantized_per_layer_head_cache_required",
        "kv_layout_ref": "graph_ir.kv_layout_contract",
        "cache_backend_policy_ref": "graph_ir.kv_layout_contract.runtime_cache_policy",
        "plain_reference_required_for_ppl": False,
        "ppl_kv_backend": "qkv_quantized_per_layer_head_cache",
        "eval_kv_backend": "qkv_quantized_per_layer_head_cache",
        "qkv_state_scope": ["request_id", "layer", "kv_head", "head_dim", "context_epoch", "qkv_policy_hash"],
        "qkv_update_mode": "append_current_token_before_decode_no_full_context_requantize",
        "forbid_plain_kv_runtime_fallback": True,
    }

    ops = [
        {"op": "rms_norm", "name": "attention_input_norm", "inputs": ["hidden"], "weights": attention_norm_weights, "output": "attention_norm", "optional_behavior": "pass_hidden_when_weight_absent", "required": bool(attention_norm_weights)},
        {"op": "linear", "name": "q_projection", "inputs": ["attention_norm"], "weights": q_weights, "output": "q_raw", "required": bool(q_weights)},
        {"op": "linear", "name": "k_projection", "inputs": ["attention_norm"], "weights": k_weights, "output": "k_raw", "required": bool(k_weights)},
        {"op": "linear", "name": "v_projection", "inputs": ["attention_norm"], "weights": v_weights, "output": "v_raw", "fallback_output": "k_raw", "fallback_semantics": "when_no_v_projection_value_uses_raw_k_projection_before_k_norm", "required": False},
        {"op": "rms_norm", "name": "q_norm", "inputs": ["q_raw"], "weights": q_norm_weights, "output": "q", "optional_behavior": "pass_q_raw_when_weight_absent", "required": bool(q_norm_weights)},
        {"op": "rms_norm", "name": "k_norm", "inputs": ["k_raw"], "weights": k_norm_weights, "output": "k", "optional_behavior": "pass_k_raw_when_weight_absent", "required": bool(k_norm_weights)},
        *([{"op": "rms_norm", "name": "v_norm", "inputs": [value_raw_input], "weights": v_norm_weights, "output": "v", "norm_mode": value_norm_mode, "required": False}]
          if v_norm_weights or implicit_unweighted_v_norm else
          [{"op": "identity", "name": "value_passthrough", "inputs": [value_raw_input], "weights": [], "output": "v", "required": False}]),
        {"op": "rope", "name": "rotary_embedding", "inputs": ["q", "k"], "weights": bind("rope_freqs.weight"), "rope_contract": rope_contract, "required": False},
        {"op": "attention", "name": "attention", "inputs": ["q", "k", "v"], **attention_cache_contract, "attention_scale": "metadata_or_qk_norm_contract", "required": bool(q_weights and k_weights and (v_weights or k_weights))},
        {"op": "linear", "name": "attention_output", "inputs": ["attention"], "weights": o_weights, "output": "attention_out", "required": bool(o_weights)},
        {"op": "rms_norm", "name": "post_attention_norm", "inputs": ["attention_out"], "weights": post_attention_norm_weights, "output": "attention_branch", "optional_behavior": "pass_attention_out_when_weight_absent", "required": bool(post_attention_norm_weights)},
        {"op": "residual", "name": "attention_residual", "inputs": ["hidden", "attention_branch"], "output": "hidden", "required": True},
        {"op": "rms_norm", "name": "ffn_norm", "inputs": ["hidden"], "weights": ffn_norm_weights, "output": "shared_ffn_input", "optional_behavior": "pass_hidden_when_weight_absent", "required": bool(ffn_norm_weights)},
        {"op": "rms_norm", "name": "expert_ffn_norm", "inputs": ["hidden"], "weights": expert_norm_weights, "output": "expert_ffn_input", "optional_behavior": "use_shared_ffn_input_when_weight_absent", "required": bool(expert_norm_weights)},
        {"op": "select", "name": "router_input", "inputs": ["hidden", "expert_ffn_input"], "output": "router_input", "rule": "use_hidden_when_router_has_internal_scale_else_expert_ffn_input", "scale": router_scale_weights, "required": False},
        {"op": "hidden_snapshot", "name": "fate_gate_input_snapshot", "inputs": ["router_input"], "target": "engine_state.gate_input_snapshots[layer]", "required": False},
        {"op": "linear", "name": "moe_router", "inputs": ["router_input"], "weights": router_weights, "scale": router_scale_weights, "output": "expert_scores", "required": bool(router_weights)},
        {"op": "topk", "name": "expert_select", "inputs": ["expert_scores"], "config_key": "adaptive_seq_topk_entropy", "required": bool(router_weights)},
        {"op": "shared_expert_mlp", "name": "shared_experts", "inputs": ["shared_ffn_input"], "weights": shared_expert_weights, "gate": shared_gate_weights, "output": "shared_branch_raw", "required": bool(shared_expert_weights)},
        {"op": "rms_norm", "name": "post_ffw_norm_1", "inputs": ["shared_branch_raw"], "weights": post_ffw_norm1_weights, "output": "shared_branch", "optional_behavior": "pass_shared_branch_raw_when_weight_absent", "required": bool(post_ffw_norm1_weights)},
        {"op": "moe_expert_mlp", "name": "moe_experts", "inputs": ["expert_ffn_input", "selected_experts"], "weights": moe_weights, "per_expert_output_scale": expert_down_scale_weights, "output": "expert_sum_raw", "required": bool(moe_weights)},
        {"op": "rms_norm", "name": "post_ffw_norm_2", "inputs": ["expert_sum_raw"], "weights": post_ffw_norm2_weights, "output": "expert_branch", "optional_behavior": "pass_expert_sum_raw_when_weight_absent", "required": bool(post_ffw_norm2_weights)},
        {"op": "dense_mlp", "name": "dense_ffn_fallback", "inputs": ["shared_ffn_input"], "weights": dense_weights, "output": "dense_branch", "required": bool(dense_weights and not moe_weights and not shared_expert_weights)},
        {"op": "add", "name": "ffn_branch_sum", "inputs": ["shared_branch", "expert_branch", "dense_branch"], "output": "ffn_out", "missing_input": "zero", "required": bool(moe_weights or shared_expert_weights or dense_weights)},
        {"op": "rms_norm", "name": "post_ffw_norm", "inputs": ["ffn_out"], "weights": post_ffw_norm_weights, "output": "ffn_branch", "optional_behavior": "pass_ffn_out_when_weight_absent", "required": bool(post_ffw_norm_weights)},
        {"op": "residual", "name": "ffn_residual", "inputs": ["hidden", "ffn_branch"], "output": "hidden", "required": True},
        {"op": "scale", "name": "layer_output_scale", "inputs": ["hidden"], "weights": layer_output_scale_weights, "output": "hidden", "required": bool(layer_output_scale_weights)},
    ]
    return {
        "layer": int(layer),
        "tensor_prefix": prefix,
        "layer_name_parser": "common_gguf_layer_prefixes",
        "available_tensors": sorted(names),
        "tensor_layout_contract": {
            "format": "JUJU_TENSOR_LAYOUT_REF_V1",
            "shape_map_ref": "graph_ir.tensor_bindings.shape_map",
            "tensor_index_ref": "juju.idx.tensor_records",
            "layer_tensor_count": len(names),
            "required_record_fields": ["name", "shape", "encoding", "offset", "size", "row_layout", "row_stride_bytes"],
            "row_layout_contract": "tensor_index_record_is_authoritative_no_name_based_transpose",
        },
        "semantic_contract": {
            "attention_post_norm_is_separate_from_ffn_norm": True,
            "expert_branch_uses_expert_ffn_norm": bool(expert_norm_weights),
            "router_uses_hidden_when_internal_scale_present": bool(router_scale_weights),
            "value_uses_raw_k_projection_when_v_projection_missing": not bool(v_weights),
            "value_projection_present": value_projection_present,
            "value_source": "v_projection" if value_projection_present else ("raw_k_projection_before_k_norm" if attention_k_eq_v else "missing_value_projection"),
            "value_norm_input": value_raw_input,
            "value_norm_mode": value_norm_mode,
            "value_norm_requires_layer_local_weight_tensor": bool(v_norm_weights),
            "unweighted_value_norm_is_contractual_when_declared": bool(implicit_unweighted_v_norm),
            "attention_kind": rope_contract.get("kind"),
            "rope_contract": rope_contract,
            "shared_and_expert_post_norms_apply_before_branch_sum": bool(post_ffw_norm1_weights or post_ffw_norm2_weights),
            "post_ffw_norm_applies_to_combined_ffn_before_residual": bool(post_ffw_norm_weights),
            "layer_output_scale_after_ffn_residual": bool(layer_output_scale_weights),
        },
        "ops": ops,
    }


def _juju_layer_refs_by_suffix(names, suffixes):
    wanted = {str(x or "").lower() for x in suffixes if x}
    return sorted(
        str(name or "")
        for name in names or []
        if _juju_layer_suffix(name) in wanted
    )


def _juju_attention_layer_contract_table(tensor_records, runtime_arch):
    runtime_arch = dict(runtime_arch or {})
    layers = sorted({
        layer for layer in (_juju_layer_id_from_name(t.get("name")) for t in tensor_records or [])
        if layer is not None
    })
    max_seq = _juju_first_int(
        runtime_arch.get("max_position_embeddings"),
        runtime_arch.get("context_length"),
        runtime_arch.get("max_seq_len"),
    )
    sliding_window = _juju_first_int(
        runtime_arch.get("sliding_window"),
        runtime_arch.get("sliding_window_size"),
    )
    table = []
    for layer in layers:
        names = _juju_tensors_by_layer(tensor_records, layer)
        q_refs = _juju_layer_refs_by_suffix(names, JUJU_ATTENTION_Q_SUFFIXES)
        k_refs = _juju_layer_refs_by_suffix(names, JUJU_ATTENTION_K_SUFFIXES)
        v_refs = _juju_layer_refs_by_suffix(names, JUJU_ATTENTION_V_SUFFIXES)
        o_refs = _juju_layer_refs_by_suffix(names, JUJU_ATTENTION_O_SUFFIXES)
        q_norm_refs = _juju_layer_refs_by_suffix(names, JUJU_Q_NORM_SUFFIXES)
        k_norm_refs = _juju_layer_refs_by_suffix(names, JUJU_K_NORM_SUFFIXES)
        v_norm_refs = _juju_layer_refs_by_suffix(names, JUJU_V_NORM_SUFFIXES)
        layer_suffixes = {_juju_layer_suffix(name) for name in names}
        implicit_unweighted_v_norm = _juju_layer_has_implicit_unweighted_v_norm(layer, layer_suffixes, runtime_arch)
        value_norm_mode = (
            "weighted_rmsnorm" if v_norm_refs else
            "unweighted_rmsnorm_contract" if implicit_unweighted_v_norm else
            "identity"
        )
        kind = _juju_layer_attention_kind(layer, runtime_arch)
        is_global = kind == "global_full_attention"
        attention_k_eq_v = _juju_bool_or_none(runtime_arch.get("attention_k_eq_v")) is True
        head_dim = _juju_first_int(
            runtime_arch.get("global_head_dim") if is_global else runtime_arch.get("head_dim"),
            runtime_arch.get("head_dim"),
        )
        value_head_dim = _juju_first_int(
            runtime_arch.get("global_value_head_dim") if is_global else runtime_arch.get("value_head_dim"),
            runtime_arch.get("value_head_dim"),
            head_dim,
        )
        kv_heads = _juju_first_int(
            runtime_arch.get("num_global_key_value_heads") if is_global else runtime_arch.get("num_key_value_heads"),
            runtime_arch.get("num_key_value_heads"),
        )
        window_size = max_seq if is_global else _juju_first_int(sliding_window, max_seq)
        value_source = "v_projection"
        if not v_refs:
            value_source = "raw_k_projection_before_k_norm" if attention_k_eq_v else "missing_value_projection"
        value_norm_input = _juju_value_raw_input_name(bool(v_refs), attention_k_eq_v)
        missing = []
        if not q_refs:
            missing.append("q_projection")
        if not k_refs:
            missing.append("k_projection")
        if not o_refs:
            missing.append("attention_output")
        if not v_refs and not attention_k_eq_v:
            missing.append("v_projection_or_attention_k_eq_v_contract")
        table.append({
            "format": "JUJU_LAYER_ATTENTION_CONTRACT_V1",
            "layer": int(layer),
            "attention_kind": kind,
            "window_size": window_size,
            "num_attention_heads": _juju_first_int(runtime_arch.get("num_attention_heads")),
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "value_head_dim": value_head_dim,
            "attention_k_eq_v": bool(attention_k_eq_v),
            "value_projection_present": bool(v_refs),
            "value_source": value_source,
            "value_norm_input": value_norm_input,
            "q_projection": q_refs,
            "k_projection": k_refs,
            "v_projection": v_refs,
            "o_projection": o_refs,
            "q_norm": q_norm_refs,
            "k_norm": k_norm_refs,
            "v_norm": v_norm_refs,
            "value_norm_mode": value_norm_mode,
            "value_norm_declared": bool(v_norm_refs or implicit_unweighted_v_norm),
            "value_norm_requires_layer_local_weight_tensor": bool(v_norm_refs),
            "unweighted_value_norm_is_contractual_when_declared": bool(implicit_unweighted_v_norm),
            "rope_contract": _juju_layer_rope_contract(layer, runtime_arch),
            "qkv_cache_backend": "qkv_quantized_per_layer_head_cache",
            "ppl_kv_backend": "qkv_quantized_per_layer_head_cache",
            "eval_kv_backend": "qkv_quantized_per_layer_head_cache",
            "required_complete": not missing,
            "missing": missing,
        })
    return table


def _juju_layer_execution_contract_table(tensor_records, runtime_arch):
    runtime_arch = dict(runtime_arch or {})
    layers = sorted({
        layer for layer in (_juju_layer_id_from_name(t.get("name")) for t in tensor_records or [])
        if layer is not None
    })
    out = []
    for layer in layers:
        names = _juju_tensors_by_layer(tensor_records, layer)
        suffixes = {_juju_layer_suffix(name) for name in names}
        refs = lambda candidates: _juju_layer_refs_by_suffix(names, candidates)
        attention_kind = _juju_layer_attention_kind(layer, runtime_arch)
        is_global = attention_kind == "global_full_attention"
        v_refs = refs(JUJU_ATTENTION_V_SUFFIXES)
        v_norm_refs = refs(JUJU_V_NORM_SUFFIXES)
        implicit_unweighted_v_norm = _juju_layer_has_implicit_unweighted_v_norm(layer, suffixes, runtime_arch)
        value_norm_mode = (
            "weighted_rmsnorm" if v_norm_refs else
            "unweighted_rmsnorm_contract" if implicit_unweighted_v_norm else
            "identity"
        )
        router_scale_refs = refs({"ffn_gate_inp.scale", "router.scale", "mlp.router.scale", "moe.gate.scale"})
        post_ffw_norm1_refs = refs({"post_ffw_norm_1.weight", "ffn_post_norm_1.weight"})
        post_ffw_norm2_refs = refs({"post_ffw_norm_2.weight", "ffn_post_norm_2.weight"})
        post_ffw_norm_refs = refs({"post_ffw_norm.weight", "ffn_post_norm.weight"})
        layer_output_scale_refs = refs({"layer_output_scale.weight", "layer_scalar.weight", "layer_scalar"})
        q_norm_refs = refs(JUJU_Q_NORM_SUFFIXES)
        k_norm_refs = refs(JUJU_K_NORM_SUFFIXES)
        attention_k_eq_v = _juju_bool_or_none(runtime_arch.get("attention_k_eq_v")) is True
        value_norm_input = _juju_value_raw_input_name(bool(v_refs), attention_k_eq_v)
        out.append({
            "format": "JUJU_LAYER_EXECUTION_CONTRACT_V1",
            "layer": int(layer),
            "attention_kind": attention_kind,
            "head_dim": _juju_first_int(
                runtime_arch.get("global_head_dim") if is_global else runtime_arch.get("head_dim"),
                runtime_arch.get("head_dim"),
            ),
            "value_head_dim": _juju_first_int(
                runtime_arch.get("global_value_head_dim") if is_global else runtime_arch.get("value_head_dim"),
                runtime_arch.get("value_head_dim"),
                runtime_arch.get("global_head_dim") if is_global else runtime_arch.get("head_dim"),
                runtime_arch.get("head_dim"),
            ),
            "num_attention_heads": _juju_first_int(runtime_arch.get("num_attention_heads")),
            "num_key_value_heads": _juju_first_int(
                runtime_arch.get("num_global_key_value_heads") if is_global else runtime_arch.get("num_key_value_heads"),
                runtime_arch.get("num_key_value_heads"),
            ),
            "attention_k_eq_v": bool(attention_k_eq_v),
            "value_projection_present": bool(v_refs),
            "value_source": "v_projection" if v_refs else ("raw_k_projection_before_k_norm" if attention_k_eq_v else "missing_value_projection"),
            "value_norm_input": value_norm_input,
            "value_norm_mode": value_norm_mode,
            "value_norm_declared": bool(v_norm_refs or implicit_unweighted_v_norm),
            "value_norm_requires_layer_local_weight_tensor": bool(v_norm_refs),
            "unweighted_value_norm_is_contractual_when_declared": bool(implicit_unweighted_v_norm),
            "attention": {
                "q_projection": refs(JUJU_ATTENTION_Q_SUFFIXES),
                "k_projection": refs(JUJU_ATTENTION_K_SUFFIXES),
                "v_projection": v_refs,
                "o_projection": refs(JUJU_ATTENTION_O_SUFFIXES),
                "q_norm": q_norm_refs,
                "k_norm": k_norm_refs,
                "v_norm": v_norm_refs,
                "value_norm_input": value_norm_input,
                "value_norm_mode": value_norm_mode,
                "value_norm_declared": bool(v_norm_refs or implicit_unweighted_v_norm),
                "value_norm_requires_layer_local_weight_tensor": bool(v_norm_refs),
                "unweighted_value_norm_is_contractual_when_declared": bool(implicit_unweighted_v_norm),
                "rope_contract": _juju_layer_rope_contract(layer, runtime_arch),
                "attention_scale": first_present(runtime_arch.get("attention_scale"), runtime_arch.get("attn_scale"), runtime_arch.get("attention_score_scale"), runtime_arch.get("f_attn_scale"), runtime_arch.get("qk_scale")),
                "query_pre_attn_scalar": first_present(runtime_arch.get("query_pre_attn_scalar"), runtime_arch.get("attention_query_pre_attn_scalar"), runtime_arch.get("attn_query_pre_attn_scalar")),
                "attn_logit_softcap": first_present(runtime_arch.get("attn_logit_softcap"), runtime_arch.get("attn_logit_softcapping"), runtime_arch.get("attention_logit_softcap"), runtime_arch.get("attention_logit_softcapping")),
            },
            "norms": {
                "attention_input_norm": refs({"attn_norm.weight", "input_layernorm.weight", "pre_attention_norm.weight"}),
                "post_attention_norm": refs({"post_attention_norm.weight", "post_attention_layernorm.weight", "post_attention_layer_norm.weight", "post_attn_norm.weight"}),
                "ffn_norm": refs({"ffn_norm.weight", "ffn_pre_norm.weight", "pre_ffw_norm.weight", "mlp_norm.weight"}),
                "expert_ffn_norm": refs({"pre_ffw_norm_2.weight", "ffn_pre_norm_2.weight", "moe_norm.weight"}),
                "post_ffw_norm_1": post_ffw_norm1_refs,
                "post_ffw_norm_2": post_ffw_norm2_refs,
                "post_ffw_norm": post_ffw_norm_refs,
                "rms_norm_eps": first_present(runtime_arch.get("norm_eps"), runtime_arch.get("rms_norm_eps")),
                "rms_norm_unit_offset": _juju_bool_or_none(runtime_arch.get("rms_norm_unit_offset")),
            },
            "router": {
                "router": refs({"ffn_gate_inp.weight", "router.weight", "mlp.router.weight", "moe.gate.weight"}),
                "router_scale": router_scale_refs,
                "router_uses_hidden_when_internal_scale_present": bool(router_scale_refs),
                "routed_experts_per_token": runtime_arch.get("routed_experts_per_token"),
                "experts_per_moe_layer": runtime_arch.get("experts_per_moe_layer"),
                "norm_topk_prob": runtime_arch.get("norm_topk_prob"),
                "scoring_func": runtime_arch.get("scoring_func"),
                "routed_scaling_factor": runtime_arch.get("routed_scaling_factor"),
            },
            "mlp": {
                "shared_experts": sorted(name for name in names if is_shared_expert_tensor_name(name)),
                "routed_experts": sorted(name for name in names if is_routed_expert_tensor_name(name)),
                "dense_mlp": refs({"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}),
                "shared_and_expert_post_norms_apply_before_branch_sum": bool(post_ffw_norm1_refs or post_ffw_norm2_refs),
                "post_ffw_norm_applies_to_combined_ffn_before_residual": bool(post_ffw_norm_refs),
            },
            "tail": {
                "layer_output_scale": layer_output_scale_refs,
                "layer_output_scale_after_ffn_residual": bool(layer_output_scale_refs),
            },
        })
    return out


def _juju_value_norm_contract_table(layer_execution_contract_table):
    out = []
    for row in layer_execution_contract_table or []:
        if not isinstance(row, dict):
            continue
        attention = row.get("attention") if isinstance(row.get("attention"), dict) else {}
        mode = row.get("value_norm_mode") or attention.get("value_norm_mode") or "identity"
        declared = bool(row.get("value_norm_declared") or attention.get("value_norm_declared"))
        requires_weight = bool(row.get("value_norm_requires_layer_local_weight_tensor") or attention.get("value_norm_requires_layer_local_weight_tensor"))
        out.append({
            "format": "JUJU_VALUE_NORM_CONTRACT_V1",
            "layer": row.get("layer"),
            "attention_kind": row.get("attention_kind"),
            "value_projection_present": bool(row.get("value_projection_present")),
            "value_source": row.get("value_source"),
            "value_norm_input": row.get("value_norm_input") or attention.get("value_norm_input"),
            "value_norm_mode": mode,
            "value_norm_declared": declared,
            "v_norm": attention.get("v_norm") or [],
            "requires_layer_local_weight_tensor": requires_weight,
            "value_norm_requires_layer_local_weight_tensor": requires_weight,
            "unweighted_value_norm_is_contractual_when_declared": bool(row.get("unweighted_value_norm_is_contractual_when_declared") or attention.get("unweighted_value_norm_is_contractual_when_declared")),
            "executor_must_not_skip_when_declared": bool(declared),
        })
    return out



def build_juju_forward_contract_validation(generation_contract, runtime_arch, *, token_embd, lm_head, output_norm, layers):
    embedding = dict((generation_contract or {}).get("embedding") or {})
    tokenizer = dict((generation_contract or {}).get("tokenizer") or {})
    lm = dict((generation_contract or {}).get("lm_head") or {})
    final_norm = dict((generation_contract or {}).get("final_norm") or {})
    layer_contract = dict((generation_contract or {}).get("layers") or {})
    feature_counts = dict(layer_contract.get("feature_counts") or {})
    layer_count = int(layer_contract.get("count") or len(layers or []))
    moe_layers = int(feature_counts.get("layers_with_moe_experts") or 0)
    dense_layers = int(feature_counts.get("layers_with_dense_ffn") or 0)
    q_layers = int(feature_counts.get("layers_with_q_projection") or 0)
    k_layers = int(feature_counts.get("layers_with_k_projection") or 0)
    o_layers = int(feature_counts.get("layers_with_o_projection") or 0)
    router_layers = int(feature_counts.get("layers_with_router") or 0)
    required = {
        "tokenizer_any_of": bool(tokenizer.get("required_any_of")),
        "token_embedding_tensor": bool(token_embd),
        "embedding_hidden_size": bool(embedding.get("hidden_size")),
        "embedding_vocab_size": bool(embedding.get("vocab_size")),
        "embedding_scale_contract": embedding.get("scale") is not None or embedding.get("scale_semantics") == "none",
        "layer_count": layer_count > 0,
        "tensor_layout_contract": bool(layer_contract.get("tensor_layout_records_complete")),
        "attention_contract": layer_count > 0 and q_layers == layer_count and k_layers == layer_count and o_layers == layer_count,
        "attention_layer_table": bool(layer_contract.get("attention_layer_table_complete")),
        "layer_execution_contract_table": bool(layer_contract.get("layer_execution_contract_table_complete")),
        "value_norm_contract_table": bool(layer_contract.get("value_norm_contract_table_complete")),
        "router_contract": moe_layers == 0 or router_layers == moe_layers,
        "mlp_contract": (moe_layers + dense_layers) > 0,
        "kv_layout_contract": bool(layer_contract.get("kv_layout_contract_available")),
        "final_norm_tensor": bool(output_norm) or bool(final_norm.get("tensor")),
        "lm_head_tensor": bool(lm_head) or bool(lm.get("tensor")) or bool(lm.get("tied_to_token_embedding")),
    }
    missing = sorted(k for k, ok in required.items() if not ok)
    return {
        "format": "JUJU_FORWARD_CONTRACT_VALIDATION_V1",
        "contract_complete": not missing,
        "required_status": required,
        "missing": missing,
        "fail_closed_if_contract_missing": True,
        "source_priority": [
            "source_config_explicit_values",
            "gguf_runtime_kv",
            "tensor_index_shapes_and_names",
            "architecture_forward_contract_rules",
            "documented_absent_contract",
        ],
        "embedding_scale": {
            "value": embedding.get("scale"),
            "source": embedding.get("scale_source"),
            "semantics": embedding.get("scale_semantics"),
        },
        "layer_features": feature_counts,
        "runtime_arch_keys": sorted(str(k) for k, v in (runtime_arch or {}).items() if v is not None),
    }

def build_generation_contract(*, contract, tensor_records, runtime_arch, token_embd, lm_head, output_norm):
    shape_map = _juju_tensor_shape_map(tensor_records)
    hidden_size = runtime_arch.get("hidden_size") or runtime_arch.get("hidden_dim")
    vocab_size = runtime_arch.get("vocab_size")
    embedding_scale = first_present(
        runtime_arch.get("embedding_scale"),
        runtime_arch.get("scale_emb"),
        (contract.get("arch_meta") or {}).get("embedding_scale"),
        (contract.get("arch_meta") or {}).get("scale_emb"),
    )
    embedding_scale_source = runtime_arch.get("embedding_scale_source") or (
        "source_config" if embedding_scale is not None else "absent_no_embedding_scale"
    )
    feature_counts = {
        "layers_with_post_attention_norm": 0,
        "layers_with_expert_ffn_norm": 0,
        "layers_with_post_ffw_norm": 0,
        "layers_with_post_ffw_norm_1": 0,
        "layers_with_post_ffw_norm_2": 0,
        "layers_with_layer_output_scale": 0,
        "layers_without_v_projection": 0,
        "layers_with_q_projection": 0,
        "layers_with_k_projection": 0,
        "layers_with_v_projection": 0,
        "layers_with_o_projection": 0,
        "layers_with_v_norm": 0,
        "layers_with_unweighted_v_norm_contract": 0,
        "layers_with_global_attention": 0,
        "layers_with_sliding_attention": 0,
        "layers_with_router": 0,
        "layers_with_router_scale": 0,
        "layers_with_moe_experts": 0,
        "layers_with_dense_ffn": 0,
    }
    layers = sorted({
        layer for layer in (_juju_layer_id_from_name(t.get("name")) for t in tensor_records)
        if layer is not None
    })
    attention_layer_table = _juju_attention_layer_contract_table(tensor_records, runtime_arch)
    layer_execution_contract_table = _juju_layer_execution_contract_table(tensor_records, runtime_arch)
    value_norm_contract_table = _juju_value_norm_contract_table(layer_execution_contract_table)
    value_norm_layers = sorted(
        int(row.get("layer"))
        for row in value_norm_contract_table
        if row.get("layer") is not None and row.get("value_norm_mode") != "identity"
    )
    weighted_value_norm_layers = sorted(
        int(row.get("layer"))
        for row in value_norm_contract_table
        if row.get("layer") is not None and row.get("value_norm_requires_layer_local_weight_tensor") is True
    )
    unweighted_value_norm_layers = sorted(
        int(row.get("layer"))
        for row in value_norm_contract_table
        if row.get("layer") is not None and row.get("unweighted_value_norm_is_contractual_when_declared") is True
    )
    by_layer = {layer: {_juju_layer_suffix(n) for n in _juju_tensors_by_layer(tensor_records, layer)} for layer in layers}
    for layer, suffixes in by_layer.items():
        if any(x in suffixes for x in {"post_attention_norm.weight", "post_attention_layernorm.weight", "post_attention_layer_norm.weight", "post_attn_norm.weight"}):
            feature_counts["layers_with_post_attention_norm"] += 1
        if any(x in suffixes for x in {"pre_ffw_norm_2.weight", "ffn_pre_norm_2.weight", "moe_norm.weight"}):
            feature_counts["layers_with_expert_ffn_norm"] += 1
        if any(x in suffixes for x in {"post_ffw_norm.weight", "ffn_post_norm.weight"}):
            feature_counts["layers_with_post_ffw_norm"] += 1
        if any(x in suffixes for x in {"post_ffw_norm_1.weight", "ffn_post_norm_1.weight"}):
            feature_counts["layers_with_post_ffw_norm_1"] += 1
        if any(x in suffixes for x in {"post_ffw_norm_2.weight", "ffn_post_norm_2.weight"}):
            feature_counts["layers_with_post_ffw_norm_2"] += 1
        if any(x in suffixes for x in {"layer_output_scale.weight", "layer_scalar.weight", "layer_scalar"}):
            feature_counts["layers_with_layer_output_scale"] += 1
        if not any(x in suffixes for x in JUJU_ATTENTION_V_SUFFIXES):
            feature_counts["layers_without_v_projection"] += 1
        if any(x in suffixes for x in JUJU_ATTENTION_Q_SUFFIXES):
            feature_counts["layers_with_q_projection"] += 1
        if any(x in suffixes for x in JUJU_ATTENTION_K_SUFFIXES):
            feature_counts["layers_with_k_projection"] += 1
        if any(x in suffixes for x in JUJU_ATTENTION_V_SUFFIXES):
            feature_counts["layers_with_v_projection"] += 1
        if any(x in suffixes for x in JUJU_ATTENTION_O_SUFFIXES):
            feature_counts["layers_with_o_projection"] += 1
        explicit_v_norm = any(x in suffixes for x in JUJU_V_NORM_SUFFIXES)
        implicit_v_norm = _juju_layer_has_implicit_unweighted_v_norm(layer, suffixes, runtime_arch)
        if explicit_v_norm or implicit_v_norm:
            feature_counts["layers_with_v_norm"] += 1
        if implicit_v_norm:
            feature_counts["layers_with_unweighted_v_norm_contract"] += 1
        attention_kind = _juju_layer_attention_kind(layer, runtime_arch)
        if attention_kind == "global_full_attention":
            feature_counts["layers_with_global_attention"] += 1
        elif attention_kind == "sliding_window_attention":
            feature_counts["layers_with_sliding_attention"] += 1
        if any(x in suffixes for x in {"ffn_gate_inp.weight", "router.weight", "mlp.router.weight", "moe.gate.weight"}):
            feature_counts["layers_with_router"] += 1
        if any(x in suffixes for x in {"ffn_gate_inp.scale", "router.scale", "mlp.router.scale", "moe.gate.scale"}):
            feature_counts["layers_with_router_scale"] += 1
        if any("exps." in x or "_exps." in x for x in suffixes):
            feature_counts["layers_with_moe_experts"] += 1
        if any(x in suffixes for x in {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}):
            feature_counts["layers_with_dense_ffn"] += 1
    tensor_layout_records_complete = all(
        rec.get("row_layout") is not None and
        rec.get("row_stride_bytes") is not None and
        rec.get("juju_offset") is not None and
        rec.get("juju_bytes") is not None and
        rec.get("shape") is not None and
        rec.get("weight_encoding") is not None
        for rec in (tensor_records or [])
    )
    qkv_fields = _juju_qkv_contract_fields(_juju_effective_qkv_schema(contract, runtime_arch))
    contract_out = {
        "format": "JUJU_GENERATION_CONTRACT_V1",
        "required_runtime_loop": "tokenizer_contract_then_graph_ir_ops_then_tensor_layout_then_lm_head",
        "contract_source": "generated_from_source_tensor_table_and_config_not_model_name",
        "fail_closed": True,
        "tokenizer": juju_tokenizer_contract(),
        "exact_mode": juju_exact_mode_policy(),
        "approx_mode": juju_approx_mode_policy(),
        "ppl_acceptance_contract": juju_ppl_acceptance_contract(),
        "performance_acceptance_contract": juju_performance_acceptance_contract(),
        "expert_calibration_contract": juju_expert_calibration_contract(),
        "embedding": {
            "tensor": token_embd,
            "shape": shape_map.get(token_embd, []),
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "scale": embedding_scale,
            "scale_source": embedding_scale_source,
            "scale_semantics": runtime_arch.get("embedding_scale_semantics") or ("multiply_token_embedding_before_first_layer" if embedding_scale is not None else "none"),
            "row_layout": "token_major_rows_vocab_by_hidden",
        },
        "layers": {
            "count": len(layers),
            "feature_counts": feature_counts,
            "attention_layer_table": attention_layer_table,
            "attention_layer_table_complete": bool(attention_layer_table) and all(
                bool(row.get("required_complete")) for row in attention_layer_table
            ),
            "layer_execution_contract_table": layer_execution_contract_table,
            "layer_execution_contract_table_complete": len(layer_execution_contract_table) == len(layers),
            "value_norm_contract_table": value_norm_contract_table,
            "value_norm_contract_table_complete": len(value_norm_contract_table) == len(layers),
            "value_norm_layers": value_norm_layers,
            "weighted_value_norm_layers": weighted_value_norm_layers,
            "unweighted_value_norm_layers": unweighted_value_norm_layers,
            "op_order_is_authoritative": True,
            "unknown_required_op_behavior": "fail_closed",
            "optional_missing_op_behavior": "documented_fallback_only",
            "tensor_layout_records_complete": bool(tensor_layout_records_complete),
            "kv_layout_contract_available": False,
            "qkv_contract": qkv_fields,
        },
        "lm_head": {
            "tensor": lm_head,
            "shape": shape_map.get(lm_head, []),
            "tied_to_token_embedding": bool(lm_head and token_embd and lm_head == token_embd),
            "row_layout": "token_major_rows_vocab_by_hidden",
            "required": bool(lm_head),
        },
        "final_norm": {
            "tensor": output_norm,
            "shape": shape_map.get(output_norm, []),
            "required": bool(output_norm),
        },
        "performance_contract": {
            "qkv_contract": qkv_fields,
            "hot_startup_tensors": [x for x in [token_embd, lm_head, output_norm, _juju_first_tensor(tensor_records, "rope_freqs.weight")] if x],
            "protect_roles": ["token_embedding", "lm_head", "final_norm", "attention", "router", "norm"],
            "sidecar_upload_format": "structured_json_yaml_toml_only_no_generated_md_pdf",
            "trace_required_keys": [
                "tokenizer",
                "embedding_contract_binding",
                "final_norm_contract_binding",
                "lm_head_contract_binding",
                "runtime_access_plan",
                "forward_layer",
                "attn_standard_qkv_norm",
                "mlp_moe_end",
                "lm_head_logprob_end",
                "cpu_ram",
                "kv_cache",
                "io_pipeline",
            ],
            "bottleneck_counters": [
                "tokenize_ms",
                "embed_ms",
                "attention_ms",
                "mlp_ms",
                "lm_head_ms",
                "kv_bytes",
                "ram_used_bytes",
                "vram_used_bytes",
                "db_used_bytes",
                "io_wait_ms",
                "queue_depth",
                "inflight",
                "qkv_k_bits",
                "qkv_v_bits",
                "qkv_normal_bits",
                "qkv_outlier_bits",
                "qkv_key_normal_bits",
                "qkv_value_normal_bits",
                "qkv_key_outlier_bits",
                "qkv_value_outlier_bits",
                "qkv_query_cached",
                "qkv_group_size",
                "qkv_page_size_tokens",
                "qkv_qjl_enabled",
                "qkv_rotation_enabled",
                "process_rss_bytes",
                "available_ram_bytes",
                "device_total_bytes",
                "device_free_bytes",
                "expert_hit_rate",
                "expert_miss_latency_us",
                "prefetch_waste_ratio",
                "gpu_idle_gap_us",
                "disk_read_bytes",
                "pcie_copy_bytes",
            ],
            "acceptance_order": [
                "exact_mode_acceptance",
                "performance_mode_acceptance",
            ],
            "exact_mode_acceptance": juju_ppl_acceptance_contract(),
            "performance_mode_acceptance": juju_performance_acceptance_contract(),
        },
        "qkv_policy_contract": qkv_fields["qkv_policy_contract"],
        "qkv_cache_schema_effective": qkv_fields["qkv_cache_schema_effective"],
        "eval_kv_policy": qkv_fields["evaluation_policy"],
    }
    contract_out["forward_contract_validation"] = build_juju_forward_contract_validation(
        contract_out,
        runtime_arch,
        token_embd=token_embd,
        lm_head=lm_head,
        output_norm=output_norm,
        layers=layers,
    )
    return contract_out


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


def build_juju_runtime_execution_manifest(*, generation_contract, runtime_access_plan, runtime_arch, qkv_fields):
    generation_contract = dict(generation_contract or {})
    runtime_access_plan = dict(runtime_access_plan or {})
    runtime_arch = _juju_runtime_arch_with_special_token_defaults(generation_contract, runtime_arch)
    qkv_fields = dict(qkv_fields or {})
    perf = dict(generation_contract.get("performance_contract") or {})
    executor = dict(runtime_access_plan.get("executor_contract") or {})
    tensor_table = runtime_access_plan.get("executor_tensor_table") or []
    role_counts = {}
    op_counts = {}
    for rec in tensor_table:
        if not isinstance(rec, dict):
            continue
        role = str(rec.get("role") or rec.get("graph_role") or "unknown")
        op = str(rec.get("op") or rec.get("execution_op") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        op_counts[op] = op_counts.get(op, 0) + 1
    tokenizer = generation_contract.get("tokenizer") or juju_tokenizer_contract()
    kv_layout = runtime_access_plan.get("kv_layout_contract") or {}
    return {
        "format": "JUJU_RUNTIME_EXECUTION_MANIFEST_V1",
        "schema_version": 1,
        "required": True,
        "source": "generated_from_source_config_tensor_table_and_runtime_contracts",
        "fail_closed_if_missing": True,
        "runtime_loop": [
            "tokenizer_contract",
            "embedding_lookup_and_scale",
            "graph_ir_ops_in_declared_order",
            "tensor_layout_contract",
            "attention_router_norm_mlp_ops",
            "qkv_cache_contract",
            "final_norm",
            "lm_head",
            "sampler",
            "bottleneck_trace",
        ],
        "executor_read_order": {
            "first": "runtime_execution_manifest",
            "then": [
                "tokenizer",
                "special_tokens",
                "embedding",
                "graph_ir",
                "tensor_layout",
                "kv_layout_contract",
                "qkv_policy_contract",
                "runtime_access_plan",
                "bottleneck_trace_contract",
            ],
            "model_name_specific_fallback_allowed": False,
            "unknown_required_field_behavior": "fail_closed",
        },
        "exact_mode": juju_exact_mode_policy(),
        "approx_mode": juju_approx_mode_policy(),
        "ppl_acceptance_contract": juju_ppl_acceptance_contract(),
        "performance_acceptance_contract": juju_performance_acceptance_contract(),
        "tokenizer": tokenizer,
        "special_tokens": {
            "bos_token_id": runtime_arch.get("bos_token_id"),
            "eos_token_id": runtime_arch.get("eos_token_id"),
            "unk_token_id": runtime_arch.get("unk_token_id"),
            "pad_token_id": runtime_arch.get("pad_token_id"),
            "add_bos_token": runtime_arch.get("add_bos_token"),
            "add_eos_token": runtime_arch.get("add_eos_token"),
            "add_space_prefix": runtime_arch.get("add_space_prefix"),
            "source": "source_config_or_gguf_runtime_metadata",
            "source_by_field": runtime_arch.get("special_token_source_by_field") or {},
            "defaulted_fields": runtime_arch.get("special_token_default_sources") or {},
        },
        "embedding": generation_contract.get("embedding") or {},
        "attention": {
            "qkv_cache_required": True,
            "plain_kv_runtime_allowed": False,
            "eval_kv_policy": generation_contract.get("eval_kv_policy") or qkv_fields.get("evaluation_policy") or {},
            "kv_layout_ref": "runtime_execution_manifest.kv_layout_contract",
            "qkv_policy_ref": "runtime_execution_manifest.qkv_policy_contract",
            "score_scale_source": "source_config_or_qk_norm_layout",
            "rope_source": "source_config_or_gguf_runtime_metadata",
            "head_layout": (kv_layout.get("head_layout") or {}),
            "window_contract": (kv_layout.get("attention_window_contract") or {}),
            "layer_table": (generation_contract.get("layers") or {}).get("attention_layer_table") or [],
            "value_norm_contract_table": (generation_contract.get("layers") or {}).get("value_norm_contract_table") or [],
            "value_norm_layers": (generation_contract.get("layers") or {}).get("value_norm_layers") or [],
            "unweighted_value_norm_layers": (generation_contract.get("layers") or {}).get("unweighted_value_norm_layers") or [],
        },
        "layer_execution_contract_table": (generation_contract.get("layers") or {}).get("layer_execution_contract_table") or [],
        "router": {
            "execution_op": "moe_router",
            "topk_source_fields": ["routed_experts_per_token", "experts_per_moe_layer", "norm_topk_prob", "scoring_func"],
            "runtime_arch_values": {
                "experts_per_moe_layer": runtime_arch.get("experts_per_moe_layer"),
                "routed_experts_per_token": runtime_arch.get("routed_experts_per_token"),
                "norm_topk_prob": runtime_arch.get("norm_topk_prob"),
                "scoring_func": runtime_arch.get("scoring_func"),
                "routed_scaling_factor": runtime_arch.get("routed_scaling_factor"),
            },
            "calibration_manifest_ref": "runtime_access_plan.router_calibration_manifest",
            "buddy_map_section_required": True,
            "predictor_section_required": True,
        },
        "norms": {
            "eps": runtime_arch.get("norm_eps") or runtime_arch.get("rms_norm_eps"),
            "source": "source_config_or_gguf_runtime_metadata",
            "optional_missing_op_behavior": generation_contract.get("layers", {}).get("optional_missing_op_behavior"),
            "value_norm_contract_table": (generation_contract.get("layers") or {}).get("value_norm_contract_table") or [],
            "value_norm_layers": (generation_contract.get("layers") or {}).get("value_norm_layers") or [],
            "weighted_value_norm_layers": (generation_contract.get("layers") or {}).get("weighted_value_norm_layers") or [],
            "unweighted_value_norm_layers": (generation_contract.get("layers") or {}).get("unweighted_value_norm_layers") or [],
        },
        "lm_head": generation_contract.get("lm_head") or {},
        "final_norm": generation_contract.get("final_norm") or {},
        "graph_ir": {
            "format": "JUJU_GRAPH_IR_V1",
            "ops_ref": "graph_ir.ops",
            "layers_ref": "graph_ir.layers",
            "executor_must_consume_ops_in_declared_order": True,
            "required_ops_fail_closed": True,
        },
        "tensor_layout": {
            "executor_tensor_table_ref": "runtime_access_plan.executor_tensor_table",
            "tensor_ref_fields": list(executor.get("tensor_ref_fields") or []),
            "role_counts": role_counts,
            "op_counts": op_counts,
            "shape_offset_stride_are_authoritative": True,
            "name_based_shape_or_transpose_guess_forbidden": True,
        },
        "runtime_access_plan": {
            "format": runtime_access_plan.get("format"),
            "version": runtime_access_plan.get("version"),
            "file_locality_group_count": runtime_access_plan.get("file_locality_group_count"),
            "layer_prefetch_plan_count": runtime_access_plan.get("layer_prefetch_plan_count"),
            "startup_hotset_count": runtime_access_plan.get("startup_hotset_count"),
            "expert_offset_table_kind": runtime_access_plan.get("expert_offset_table_kind"),
            "moe_layer_bitmask_words": runtime_access_plan.get("moe_layer_bitmask_words") or [],
            "expert_calibration_contract": runtime_access_plan.get("expert_calibration_contract") or juju_expert_calibration_contract(),
            "adaptive_runtime_scheduler": runtime_access_plan.get("adaptive_runtime_scheduler") or juju_adaptive_runtime_scheduler_contract(),
        },
        "kv_layout_contract": kv_layout,
        "qkv_policy_contract": qkv_fields.get("qkv_policy_contract") or runtime_access_plan.get("qkv_policy_contract") or {},
        "qkv_cache_schema_effective": qkv_fields.get("qkv_cache_schema_effective") or runtime_access_plan.get("qkv_cache_schema_effective") or {},
        "eval_kv_policy": qkv_fields.get("evaluation_policy") or generation_contract.get("eval_kv_policy") or {},
        "bottleneck_trace_contract": runtime_access_plan.get("bottleneck_trace_contract") or {
            "required_counters": perf.get("bottleneck_counters") or [],
            "trace_required_keys": perf.get("trace_required_keys") or [],
        },
        "ppl_correctness_gates": {
            "bad_ppl_is_correctness_failure": True,
            "require_finite_hidden_vectors": True,
            "require_finite_logits": True,
            "require_tokenizer_contract_match": True,
            "require_embedding_scale_semantics": True,
            "require_qkv_policy_match": True,
            "require_lm_head_contract_match": True,
            "require_tensor_logical_hash_equivalence": True,
            "require_original_gguf_vs_juju_ppl_delta": True,
            "require_router_topk_exact_match": True,
            "require_expert_id_sequence_exact_match": True,
        },
    }


def juju_format_self_check(idx, sections, qkv_schema):
    idx = idx or {}
    sections = sections or []
    qkv_schema = qkv_schema or {}
    errors = []
    warnings = []
    split_meta = idx.get("split") if isinstance(idx.get("split"), dict) else {}
    source_name = str(idx.get("source_name") or idx.get("source_path") or "")
    split_count = _juju_int_or_none(split_meta.get("split_count")) or 0
    split_enabled = bool(split_meta.get("enabled")) or split_count > 1
    source_shard_match = re.search(r"(?:^|[-_.])\d{5}-of-\d{5}(?:\.gguf)?$", source_name, re.IGNORECASE)
    source_shard_count = 0
    if source_shard_match:
        count_match = re.search(r"of-(\d{5})(?:\.gguf)?$", source_name, re.IGNORECASE)
        source_shard_count = int(count_match.group(1)) if count_match else 0
    partial_source_shard = bool(source_shard_count > 1)
    partial_shard = bool(split_enabled or partial_source_shard)

    def err(code, **fields):
        item = {"code": code}
        item.update(fields)
        errors.append(item)

    def warn(code, **fields):
        item = {"code": code}
        item.update(fields)
        warnings.append(item)

    k_bits = _juju_int_or_none(qkv_schema.get("k_bits"))
    v_bits = _juju_int_or_none(qkv_schema.get("v_bits"))
    normal_bits = _juju_int_or_none(qkv_schema.get("normal_bits"))
    key_normal_bits = _juju_int_or_none(qkv_schema.get("key_normal_bits"))
    value_normal_bits = _juju_int_or_none(qkv_schema.get("value_normal_bits"))
    outlier_bits = _juju_int_or_none(qkv_schema.get("outlier_bits"))
    key_outlier_bits = _juju_int_or_none(qkv_schema.get("key_outlier_bits"))
    value_outlier_bits = _juju_int_or_none(qkv_schema.get("value_outlier_bits"))
    outlier_channels = _juju_int_or_none(qkv_schema.get("outlier_channels")) or 0
    raw_qkv_bits = any(
        bit in (16, 32)
        for bit in (
            k_bits,
            v_bits,
            normal_bits,
            key_normal_bits,
            value_normal_bits,
            outlier_bits,
            key_outlier_bits,
            value_outlier_bits,
        )
        if bit is not None
    )
    enable_qjl = bool(_juju_bool_or_none(qkv_schema.get("enable_qjl")))
    if k_bits is None or v_bits is None or normal_bits is None or key_normal_bits is None or value_normal_bits is None:
        err(
            "qkv_bits_incomplete",
            k_bits=k_bits,
            v_bits=v_bits,
            normal_bits=normal_bits,
            key_normal_bits=key_normal_bits,
            value_normal_bits=value_normal_bits,
        )
    supported_bits = set(int(x) for x in (qkv_schema.get("supported_cache_bits") or JUJU_QKV_SUPPORTED_CACHE_BITS))
    for name, value in (
        ("k_bits", k_bits),
        ("v_bits", v_bits),
        ("normal_bits", normal_bits),
        ("key_normal_bits", key_normal_bits),
        ("value_normal_bits", value_normal_bits),
        ("outlier_bits", outlier_bits),
        ("key_outlier_bits", key_outlier_bits),
        ("value_outlier_bits", value_outlier_bits),
    ):
        if value is not None and int(value) not in supported_bits:
            err("qkv_bit_width_not_declared_supported", field=name, value=value, supported=sorted(supported_bits))
    if outlier_channels > 0 and (outlier_bits is None or key_outlier_bits is None or value_outlier_bits is None):
        err(
            "qkv_outlier_bits_incomplete",
            outlier_bits=outlier_bits,
            key_outlier_bits=key_outlier_bits,
            value_outlier_bits=value_outlier_bits,
        )
    if not raw_qkv_bits and not enable_qjl:
        err("qkv_codebook_bits_without_qjl", k_bits=k_bits, v_bits=v_bits, normal_bits=normal_bits)
    if (
        str(qkv_schema.get("source") or "") == "juju_generator_synthesized_runtime_qkv_contract" and
        not raw_qkv_bits and outlier_channels > 0 and v_bits is not None and normal_bits != v_bits
    ):
        err("synthesized_qkv_normal_bits_not_derived_from_value_bits", v_bits=v_bits, normal_bits=normal_bits)
    normal = qkv_schema.get("normal") if isinstance(qkv_schema.get("normal"), dict) else {}
    if normal.get("semantics") != "non_outlier_channel_quant_bits":
        err("qkv_normal_bits_semantics_missing")
    query_policy = qkv_schema.get("query_policy") if isinstance(qkv_schema.get("query_policy"), dict) else {}
    if query_policy.get("cached") is not False:
        err("qkv_query_policy_must_be_uncached", cached=query_policy.get("cached"))
    key_policy = qkv_schema.get("key") if isinstance(qkv_schema.get("key"), dict) else qkv_schema.get("key_cache_policy")
    value_policy = qkv_schema.get("value") if isinstance(qkv_schema.get("value"), dict) else qkv_schema.get("value_cache_policy")
    if not isinstance(key_policy, dict) or key_policy.get("cached") is not True:
        err("qkv_key_cache_policy_missing")
    if not isinstance(value_policy, dict) or value_policy.get("cached") is not True:
        err("qkv_value_cache_policy_missing")
    turboquant_policy = qkv_schema.get("turboquant") if isinstance(qkv_schema.get("turboquant"), dict) else qkv_schema.get("turboquant_policy")
    if not isinstance(turboquant_policy, dict) or turboquant_policy.get("query_cached") is not False:
        err("qkv_turboquant_policy_missing_or_query_cached")
    eval_policy = qkv_schema.get("evaluation_policy") if isinstance(qkv_schema.get("evaluation_policy"), dict) else {}
    if eval_policy.get("ppl_kv_backend") != "qkv_quantized_per_layer_head_cache":
        err("qkv_eval_ppl_backend_not_qkv", backend=eval_policy.get("ppl_kv_backend"))
    if eval_policy.get("eval_kv_backend") != "qkv_quantized_per_layer_head_cache":
        err("qkv_eval_backend_not_qkv", backend=eval_policy.get("eval_kv_backend"))
    if eval_policy.get("eval_must_bypass_quantized_qkv") is not False:
        err("qkv_eval_bypasses_quantized_qkv")
    if eval_policy.get("force_qkv_for_ppl") is not True or eval_policy.get("force_qkv_for_eval") is not True:
        err("qkv_eval_force_qkv_missing")
    if eval_policy.get("allow_plain_reference") is not False:
        err("qkv_eval_allows_plain_reference")
    if qkv_schema.get("required_quantized_qkv") is not True or qkv_schema.get("qkv_packed_cache_required") is not True:
        err(
            "qkv_required_quantized_cache_missing",
            required_quantized_qkv=qkv_schema.get("required_quantized_qkv"),
            qkv_packed_cache_required=qkv_schema.get("qkv_packed_cache_required"),
        )
    if qkv_schema.get("persistent_plain_kv_cache_allowed") is not False:
        err("qkv_persistent_plain_cache_allowed", value=qkv_schema.get("persistent_plain_kv_cache_allowed"))
    if qkv_schema.get("plain_kv_persistent_storage") is not False:
        err("qkv_plain_kv_persistent_storage_allowed", value=qkv_schema.get("plain_kv_persistent_storage"))
    if qkv_schema.get("plain_fallback_allowed") is not False:
        err("qkv_plain_fallback_allowed", value=qkv_schema.get("plain_fallback_allowed"))
    if qkv_schema.get("plain_kv_runtime_allowed") is not False:
        err("qkv_plain_runtime_allowed", value=qkv_schema.get("plain_kv_runtime_allowed"))
    if qkv_schema.get("rotation_backend") != "gaussian_qr_orthogonal":
        err("qkv_rotation_backend_not_paper_default", value=qkv_schema.get("rotation_backend"))
    if qkv_schema.get("codebook_distribution") != "exact_beta":
        err("qkv_codebook_distribution_not_exact_beta", value=qkv_schema.get("codebook_distribution"))
    if not _juju_int_or_none(qkv_schema.get("qkv_policy_hash")):
        err("qkv_policy_hash_missing")
    else:
        expected_qkv_policy_hash = juju_qkv_policy_hash(qkv_schema)
        if _juju_int_or_none(qkv_schema.get("qkv_policy_hash")) != expected_qkv_policy_hash:
            err(
                "qkv_policy_hash_mismatch",
                expected=expected_qkv_policy_hash,
                actual=_juju_int_or_none(qkv_schema.get("qkv_policy_hash")),
            )

    if idx.get("format") != JUJU_IDX_FORMAT:
        err("idx_format_missing_or_wrong")
    if int(idx.get("schema_version") or 0) < JUJU_IDX_SCHEMA_VERSION:
        err("idx_schema_version_too_old", schema_version=idx.get("schema_version"))
    if idx.get("mutable_runtime_index") is not True:
        err("idx_mutable_runtime_index_not_enabled")

    required_section_names = {
        "MODEL_META",
        "QKV_POLICY",
        "PREDICTOR",
        "BUDDY_MAP",
        "TIER_HINT",
        "RUNTIME_CONTRACT",
    }
    present_names = {str(s.get("name") or "") for s in sections}
    for name in sorted(required_section_names - present_names):
        err("required_section_missing", section=name)
    for s in sections:
        name = str(s.get("name") or "")
        if name == "COLD_EXPERTS":
            if int(s.get("mmap_friendly") or 0) != 0:
                err("cold_experts_mmap_enabled")
            if int(s.get("sequential_block_size") or 0) < 128 * 1024:
                err("cold_experts_sequential_block_too_small", value=int(s.get("sequential_block_size") or 0))
        if name == "WARM_EXPERTS" and int(s.get("sequential_block_size") or 0) < 64 * 1024:
            warn("warm_experts_sequential_block_small", value=int(s.get("sequential_block_size") or 0))
        if name == "SHARED_WEIGHTS" and int(s.get("mmap_friendly") or 0) != 1:
            warn("shared_weights_not_mmap_friendly")

    tensors = idx.get("tensors") or []
    required_tensor_fields = [
        "name",
        "shape",
        "source_shape",
        "math_shape",
        "op_role",
        "codec_id",
        "weight_encoding",
        "gguf_type",
        "gguf_type_name",
        "quant_family",
        "kernel_key",
        "row_bytes",
        "row_stride_bytes",
        "physical_bytes",
        "juju_offset",
        "juju_bytes",
        "graph_role",
        "execution_op",
        "execution_layer",
    ]
    for rec in tensors:
        missing = [key for key in required_tensor_fields if rec.get(key) in (None, "")]
        if missing:
            err("tensor_contract_fields_missing", name=rec.get("name"), missing=missing)
            break
        if rec.get("bundle_native"):
            bundle_missing = [
                key for key in (
                    "bundle_id",
                    "bundle_offset",
                    "bundle_size",
                    "bundle_member_offset",
                    "bundle_member_size",
                    "bundle_member_role",
                    "expert_id",
                    "expert_projection",
                    "split_policy",
                )
                if rec.get(key) in (None, "")
            ]
            if bundle_missing:
                err("bundle_tensor_contract_fields_missing", name=rec.get("name"), missing=bundle_missing)
                break
    graph_ir = idx.get("graph_ir") or {}
    if graph_ir.get("format") != "JUJU_GRAPH_IR_V1":
        err("graph_ir_missing_or_wrong_format")
    if not graph_ir.get("ops"):
        if partial_shard:
            warn("graph_ir_root_ops_missing_for_partial_shard")
        else:
            err("graph_ir_root_ops_missing")
    if not graph_ir.get("layers"):
        if partial_shard:
            warn("graph_ir_layer_ops_missing_for_partial_shard")
        else:
            err("graph_ir_layer_ops_missing")
    generation_contract = graph_ir.get("generation_contract") or {}
    if generation_contract.get("format") != "JUJU_GENERATION_CONTRACT_V1":
        err("generation_contract_missing_or_wrong_format")
    runtime_execution_manifest = graph_ir.get("runtime_execution_manifest") or idx.get("runtime_execution_manifest") or {}
    if runtime_execution_manifest.get("format") != "JUJU_RUNTIME_EXECUTION_MANIFEST_V1":
        err("runtime_execution_manifest_missing_or_wrong_format")
    else:
        for field in (
            "tokenizer",
            "special_tokens",
            "embedding",
            "attention",
            "layer_execution_contract_table",
            "norms",
            "router",
            "lm_head",
            "graph_ir",
            "tensor_layout",
            "runtime_access_plan",
            "kv_layout_contract",
            "qkv_policy_contract",
            "eval_kv_policy",
            "bottleneck_trace_contract",
            "ppl_correctness_gates",
            "exact_mode",
            "approx_mode",
            "ppl_acceptance_contract",
            "performance_acceptance_contract",
        ):
            if runtime_execution_manifest.get(field) in (None, "", [], {}):
                err("runtime_execution_manifest_field_missing", field=field)
        read_order = runtime_execution_manifest.get("executor_read_order") or {}
        if read_order.get("model_name_specific_fallback_allowed") is not False:
            err("runtime_execution_manifest_allows_model_name_fallback")
        if runtime_execution_manifest.get("ppl_correctness_gates", {}).get("bad_ppl_is_correctness_failure") is not True:
            err("runtime_execution_manifest_ppl_gate_missing")
        exact_mode = runtime_execution_manifest.get("exact_mode") or {}
        if exact_mode.get("buddy_fallback_can_replace_expert") is not False:
            err("exact_mode_allows_buddy_replacement")
        if exact_mode.get("partial_execution_allowed") is not False:
            err("exact_mode_allows_partial_execution")
        if exact_mode.get("seqtopk_can_change_router_topk") is not False:
            err("exact_mode_allows_seqtopk_router_change")
        if exact_mode.get("cold_expert_requantize_allowed") is not False:
            err("exact_mode_allows_cold_requantize")
        if exact_mode.get("predictor_role") != "prefetch_hint_only":
            err("exact_mode_predictor_not_hint_only", value=exact_mode.get("predictor_role"))
        approx_mode = runtime_execution_manifest.get("approx_mode") or {}
        if approx_mode.get("quality_gate_required") is not True:
            err("approx_mode_quality_gate_missing")
        if approx_mode.get("ppl_delta_threshold_required") is not True:
            err("approx_mode_ppl_delta_gate_missing")
        ppl_acceptance = runtime_execution_manifest.get("ppl_acceptance_contract") or {}
        if ppl_acceptance.get("required_for_preserve_claim") is not True:
            err("ppl_acceptance_not_required_for_preserve_claim")
        if ppl_acceptance.get("tensor_logical_hash_all_match_required") is not True:
            err("ppl_acceptance_missing_tensor_hash_gate")
        if ppl_acceptance.get("server_text_tokenization_allowed") is not False:
            err("ppl_acceptance_allows_server_text_tokenization")
        if ppl_acceptance.get("qkv_required_in_ppl") is not True:
            err("ppl_acceptance_qkv_not_required")
        required_response_fields = set(ppl_acceptance.get("required_response_fields") or [])
        for field in ("kv_backend", "qkv_forced_by_format", "input_ids_preview"):
            if field not in required_response_fields:
                err("ppl_acceptance_response_field_missing", field=field)
        special_tokens = runtime_execution_manifest.get("special_tokens") or {}
        for field in ("add_bos_token", "add_eos_token", "add_space_prefix"):
            if special_tokens.get(field) is None:
                special_tokens[field] = False
                defaulted_fields = special_tokens.setdefault("defaulted_fields", {})
                if isinstance(defaulted_fields, dict):
                    defaulted_fields.setdefault(field, "self_check_default_false_when_source_tokenizer_policy_absent")
                warn("runtime_execution_manifest_special_token_policy_defaulted", field=field)
        if special_tokens.get("add_bos_token") is True and special_tokens.get("bos_token_id") is None:
            err("runtime_execution_manifest_special_token_missing", field="bos_token_id")
        if special_tokens.get("add_eos_token") is True and special_tokens.get("eos_token_id") is None:
            err("runtime_execution_manifest_special_token_missing", field="eos_token_id")
        attention = runtime_execution_manifest.get("attention") or {}
        if attention.get("plain_kv_runtime_allowed") is not False:
            err("runtime_execution_manifest_allows_plain_kv")
        eval_policy = attention.get("eval_kv_policy") or runtime_execution_manifest.get("eval_kv_policy") or {}
        if eval_policy.get("allow_plain_reference") is not False:
            err("runtime_execution_manifest_eval_allows_plain_reference")
        if eval_policy.get("force_qkv_for_ppl") is not True:
            err("runtime_execution_manifest_eval_does_not_force_qkv")
        layer_table = attention.get("layer_table") or []
        if not layer_table:
            if partial_shard:
                warn("runtime_execution_manifest_attention_layer_table_missing_for_partial_shard")
            else:
                err("runtime_execution_manifest_attention_layer_table_missing")
        else:
            for row in layer_table:
                if row.get("required_complete") is not True:
                    if partial_shard:
                        warn("attention_layer_contract_incomplete_for_partial_shard", layer=row.get("layer"), missing=row.get("missing") or [])
                    else:
                        err("attention_layer_contract_incomplete", layer=row.get("layer"), missing=row.get("missing") or [])
                    break
    tokenizer = generation_contract.get("tokenizer") or {}
    if not tokenizer.get("required_any_of"):
        err("tokenizer_required_any_of_missing")
    if tokenizer.get("missing_chat_template_policy") != "base_completion_template_only_never_invent_family_template":
        err("tokenizer_missing_chat_template_policy_not_fail_closed")
    forward_validation = generation_contract.get("forward_contract_validation") or {}
    if forward_validation.get("format") != "JUJU_FORWARD_CONTRACT_VALIDATION_V1":
        err("forward_contract_validation_missing")
    elif forward_validation.get("contract_complete") is not True:
        if partial_shard:
            warn("forward_contract_incomplete_for_partial_shard", missing=forward_validation.get("missing") or [])
        else:
            err("forward_contract_incomplete", missing=forward_validation.get("missing") or [])
    layer_contract = generation_contract.get("layers") or {}
    if layer_contract.get("attention_layer_table_complete") is not True:
        if partial_shard:
            warn("generation_attention_layer_table_incomplete_for_partial_shard")
        else:
            err("generation_attention_layer_table_incomplete")
    layer_execution_table = layer_contract.get("layer_execution_contract_table") or graph_ir.get("layer_execution_contract_table") or idx.get("layer_execution_contract_table") or []
    if not layer_execution_table:
        if partial_shard:
            warn("layer_execution_contract_table_missing_for_partial_shard")
        else:
            err("layer_execution_contract_table_missing")
    if layer_contract.get("layer_execution_contract_table_complete") is not True:
        if partial_shard:
            warn("layer_execution_contract_table_incomplete_for_partial_shard")
        else:
            err("layer_execution_contract_table_incomplete")
    value_norm_table = (
        layer_contract.get("value_norm_contract_table") or
        (runtime_execution_manifest.get("attention") or {}).get("value_norm_contract_table") or
        (runtime_execution_manifest.get("norms") or {}).get("value_norm_contract_table") or
        graph_ir.get("value_norm_contract_table") or
        idx.get("value_norm_contract_table") or
        []
    )
    if layer_contract.get("value_norm_contract_table_complete") is not True:
        if partial_shard:
            warn("value_norm_contract_table_incomplete_for_partial_shard")
        else:
            err("value_norm_contract_table_incomplete")
    feature_counts = layer_contract.get("feature_counts") or {}
    if int(feature_counts.get("layers_with_unweighted_v_norm_contract") or 0) > 0:
        execution_rows = [row for row in layer_execution_table if isinstance(row, dict)]
        attention_rows = [row for row in (layer_contract.get("attention_layer_table") or []) if isinstance(row, dict)]
        value_norm_rows = [row for row in value_norm_table if isinstance(row, dict)]
        if not any(row.get("unweighted_value_norm_is_contractual_when_declared") is True for row in execution_rows):
            err("unweighted_v_norm_contract_not_lifted_to_layer_execution_table")
        if not any(row.get("unweighted_value_norm_is_contractual_when_declared") is True for row in attention_rows):
            err("unweighted_v_norm_contract_not_lifted_to_attention_layer_table")
        if not any(row.get("unweighted_value_norm_is_contractual_when_declared") is True for row in value_norm_rows):
            err("unweighted_v_norm_contract_not_lifted_to_value_norm_table")
        for row in value_norm_rows:
            if row.get("unweighted_value_norm_is_contractual_when_declared") is True:
                if row.get("value_norm_mode") != "unweighted_rmsnorm_contract":
                    err("unweighted_v_norm_mode_not_explicit", layer=row.get("layer"), value=row.get("value_norm_mode"))
                    break
                if row.get("value_norm_input") not in ("v_raw", "k_raw"):
                    err("unweighted_v_norm_input_missing", layer=row.get("layer"), value=row.get("value_norm_input"))
                    break
                if row.get("value_projection_present") is True and row.get("value_norm_input") != "v_raw":
                    err("unweighted_v_norm_v_projection_input_not_v_raw", layer=row.get("layer"), value=row.get("value_norm_input"))
                    break
    runtime_access_plan = graph_ir.get("runtime_access_plan") or idx.get("runtime_access_plan") or {}
    if runtime_access_plan.get("format") != "JUJU_RUNTIME_ACCESS_PLAN_V1":
        err("runtime_access_plan_missing_or_wrong_format")
    executor_contract = runtime_access_plan.get("executor_contract") or {}
    tensor_ref_fields = set(executor_contract.get("tensor_ref_fields") or [])
    for field in ("name", "role", "layer", "op", "shape", "encoding", "row_stride_bytes", "offset", "bytes"):
        if field not in tensor_ref_fields:
            err("executor_tensor_ref_field_missing", field=field)
    if not runtime_access_plan.get("executor_tensor_table"):
        if partial_shard:
            warn("executor_tensor_table_missing_for_partial_shard")
        else:
            err("executor_tensor_table_missing")
    if not runtime_access_plan.get("file_locality_groups"):
        if partial_shard:
            warn("runtime_file_locality_groups_missing_for_partial_shard")
        else:
            err("runtime_file_locality_groups_missing")
    if not runtime_access_plan.get("layer_prefetch_plan"):
        if partial_shard:
            warn("layer_prefetch_plan_missing_for_partial_shard")
        else:
            err("layer_prefetch_plan_missing")
    kv_layout = runtime_access_plan.get("kv_layout_contract") or graph_ir.get("kv_layout_contract") or idx.get("kv_layout_contract") or {}
    if not kv_layout:
        err("kv_layout_contract_missing")
    else:
        if kv_layout.get("format") != "JUJU_KV_LAYOUT_CONTRACT_V1":
            err("kv_layout_contract_wrong_format")
        for field in (
            "layout",
            "page_size_tokens",
            "key_bits",
            "value_bits",
            "normal_bits",
            "key_normal_bits",
            "value_normal_bits",
            "outlier_bits",
            "key_outlier_bits",
            "value_outlier_bits",
            "group_size",
            "enable_qjl",
            "runtime_cache_policy",
            "query_policy",
            "key_cache_policy",
            "value_cache_policy",
        ):
            if kv_layout.get(field) in (None, ""):
                err("kv_layout_field_missing", field=field)
    if not runtime_access_plan.get("expert_offset_table"):
        if partial_shard:
            warn("expert_offset_table_missing_for_partial_shard")
        else:
            err("expert_offset_table_missing")
    bundle_table = runtime_access_plan.get("expert_bundle_table") or idx.get("expert_bundle_table") or {}
    if not bundle_table:
        if partial_shard:
            warn("expert_bundle_table_missing_for_partial_shard")
        else:
            err("expert_bundle_table_missing")
    elif bundle_table.get("format") != JUJU_EXPERT_BUNDLE_TABLE_FORMAT:
        err("expert_bundle_table_wrong_format")
    else:
        if int(bundle_table.get("bundle_alignment") or 0) != JUJU_BUNDLE_ALIGNMENT_BYTES:
            err("expert_bundle_alignment_wrong", bundle_alignment=bundle_table.get("bundle_alignment"))
        if bundle_table.get("member_order") != list(JUJU_EXPERT_BUNDLE_MEMBER_ORDER):
            err("expert_bundle_member_order_wrong")
        for bundle in bundle_table.get("bundles") or []:
            if int(bundle.get("bundle_offset") or bundle.get("offset") or 0) % JUJU_BUNDLE_ALIGNMENT_BYTES != 0:
                err("expert_bundle_offset_not_4k_aligned", layer=bundle.get("layer"), expert=bundle.get("expert"))
                break
            if int(bundle.get("bundle_size") or bundle.get("bytes") or 0) % JUJU_BUNDLE_ALIGNMENT_BYTES != 0:
                err("expert_bundle_size_not_4k_aligned", layer=bundle.get("layer"), expert=bundle.get("expert"))
                break
            if bundle.get("single_contiguous_fetch") is not True:
                err("expert_bundle_not_single_contiguous_fetch", layer=bundle.get("layer"), expert=bundle.get("expert"))
                break
    if not runtime_access_plan.get("moe_layer_bitmask_words"):
        warn("moe_layer_bitmask_missing_or_empty")
    if not runtime_access_plan.get("router_calibration_manifest"):
        if partial_shard:
            warn("router_calibration_manifest_missing_for_partial_shard")
        else:
            err("router_calibration_manifest_missing")
    priority_tables = graph_ir.get("priority_tables") or idx.get("priority_tables") or {}
    if not priority_tables.get("section_priorities"):
        err("section_priority_table_missing")
    moe_policy = graph_ir.get("moe_offload_policy") or idx.get("moe_offload_policy") or {}
    streaming = moe_policy.get("streaming") or {}
    if streaming.get("split_combined_gate_up") is not True:
        err("combined_gate_up_split_policy_missing")
    policy_exact = moe_policy.get("exact_mode") or {}
    if policy_exact.get("buddy_fallback_can_replace_expert") is not False:
        err("moe_policy_exact_allows_buddy_replacement")
    expert_unit = moe_policy.get("expert_unit_contract") or {}
    if expert_unit.get("partial_segment_fetch_allowed") is not False:
        err("moe_policy_exact_allows_partial_segment_fetch")
    if streaming.get("allow_partial_expert_segments") is not False:
        err("moe_policy_exact_allows_partial_streaming")
    prefetch = moe_policy.get("prefetch") or {}
    if not prefetch.get("trigger") or not prefetch.get("priority_field"):
        err("moe_prefetch_policy_incomplete")
    if not moe_policy.get("adaptive_runtime_scheduler"):
        err("adaptive_runtime_scheduler_contract_missing")
    if not moe_policy.get("expert_calibration_contract"):
        err("expert_calibration_contract_missing")
    perf = generation_contract.get("performance_contract") or {}
    trace_keys = set(perf.get("trace_required_keys") or [])
    for key in ("forward_layer", "attn_standard_qkv_norm", "mlp_moe_end", "lm_head_logprob_end", "cpu_ram", "kv_cache", "io_pipeline"):
        if key not in trace_keys:
            err("trace_required_key_missing", trace_key=key)
    bottleneck_counters = set(perf.get("bottleneck_counters") or [])
    for key in (
        "attention_ms",
        "mlp_ms",
        "lm_head_ms",
        "kv_bytes",
        "ram_used_bytes",
        "vram_used_bytes",
        "io_wait_ms",
        "qkv_k_bits",
        "qkv_v_bits",
        "qkv_normal_bits",
        "qkv_key_normal_bits",
        "qkv_value_normal_bits",
        "qkv_outlier_bits",
        "qkv_key_outlier_bits",
        "qkv_value_outlier_bits",
        "qkv_qjl_enabled",
    ):
        if key not in bottleneck_counters:
            err("bottleneck_counter_missing", counter=key)

    return {
        "format": "JUJU_FORMAT_SELF_CHECK_V1",
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "partial_shard": partial_shard,
        "partial_source_shard": partial_source_shard,
    }


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
    generation_contract = build_generation_contract(
        contract=contract,
        tensor_records=tensor_records,
        runtime_arch=runtime_arch,
        token_embd=token_embd,
        lm_head=lm_head,
        output_norm=output_norm,
    )
    runtime_access_plan = build_juju_runtime_access_plan(tensor_records, contract, runtime_arch)
    qkv_fields = _juju_qkv_contract_fields(_juju_effective_qkv_schema(contract, runtime_arch))
    generation_contract["layers"]["kv_layout_contract_available"] = bool(runtime_access_plan.get("kv_layout_contract"))
    generation_contract["layers"]["kv_runtime_policy"] = dict(
        (runtime_access_plan.get("kv_layout_contract") or {}).get("runtime_cache_policy") or {}
    )
    generation_contract["layers"]["qkv_contract"] = qkv_fields
    generation_contract["qkv_policy_contract"] = qkv_fields["qkv_policy_contract"]
    generation_contract["qkv_cache_schema_effective"] = qkv_fields["qkv_cache_schema_effective"]
    generation_contract["forward_contract_validation"] = build_juju_forward_contract_validation(
        generation_contract,
        runtime_arch,
        token_embd=token_embd,
        lm_head=lm_head,
        output_norm=output_norm,
        layers=layers,
    )
    runtime_execution_manifest = build_juju_runtime_execution_manifest(
        generation_contract=generation_contract,
        runtime_access_plan=runtime_access_plan,
        runtime_arch=runtime_arch,
        qkv_fields=qkv_fields,
    )
    priority_rules = [
        {"match": "token_embd.weight|output.weight|output_norm.weight|rope_freqs.weight", "priority": 100, "residency": "FAST_MEM", "prefetch": "startup_hot"},
        {"match": "attention/norm/router tensors", "priority": 85, "residency": "FAST_MEM", "prefetch": "layer_hot"},
        {"match": "expert tensors", "priority": "65/70/78 by cold/warm/hot bootstrap tier", "residency": "SLOW_MEM_or_FAST_MEM_STREAMABLE", "prefetch": "expert_stream_or_bootstrap"},
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
        "execution_correctness_contract": _juju_execution_correctness_contract(
            contract,
            runtime_arch,
            _juju_effective_qkv_schema(contract, runtime_arch),
        ),
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
        "generation_contract": generation_contract,
        "runtime_execution_manifest": runtime_execution_manifest,
        "runtime_access_plan": runtime_access_plan,
        "attention_layer_contract_table": generation_contract.get("layers", {}).get("attention_layer_table") or [],
        "layer_execution_contract_table": generation_contract.get("layers", {}).get("layer_execution_contract_table") or [],
        "value_norm_contract_table": generation_contract.get("layers", {}).get("value_norm_contract_table") or [],
        "kv_layout_contract": runtime_access_plan["kv_layout_contract"],
        "qkv_policy_contract": qkv_fields["qkv_policy_contract"],
        "qkv_cache_schema_effective": qkv_fields["qkv_cache_schema_effective"],
        "eval_kv_policy": qkv_fields["evaluation_policy"],
        "tokenizer_contract": juju_tokenizer_contract(),
        "quantization": {
            "weight": contract.get("weight_quant_schema", {}),
            "qkv_cache": contract.get("qkv_cache_schema", {}),
            "qkv_effective_policy": _juju_effective_qkv_schema(contract, runtime_arch),
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
            "layout_contract": {
                "token_embedding": generation_contract["embedding"],
                "lm_head": generation_contract["lm_head"],
                "final_norm": generation_contract["final_norm"],
            },
        },
        "tensor_index_contract": {
            "binary_schema_version": JUJU_BINARY_TENSOR_INDEX_SCHEMA_VERSION,
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
            "bundle_native_required": True,
            "bundle_table_format": JUJU_EXPERT_BUNDLE_TABLE_FORMAT,
            "bundle_member_order": list(JUJU_EXPERT_BUNDLE_MEMBER_ORDER),
        },
        "ops": [
            {"op": "input_tokens", "output": "token_ids", "required": True},
            {"op": "embedding_lookup", "weights": [token_embd], "output": "hidden", "required": bool(token_embd)},
            {"op": "for_each_layer", "layers": [int(x) for x in layers], "body_ref": "layers"},
            {"op": "rms_norm", "name": "final_norm", "weights": [output_norm] if output_norm else [], "required": bool(output_norm)},
            {"op": "lm_head", "weights": [lm_head] if lm_head else [], "tied_to_embedding": bool(lm_head == token_embd), "required": bool(lm_head)},
            {"op": "sampler", "inputs": ["logits"], "required": True},
        ],
        "layers": [build_layer_graph_ir(layer, tensor_records, runtime_arch) for layer in layers],
        "runtime_policy": {
            "execution": "graph_ir_executor_required",
            "unknown_op": "fail_closed",
            "unknown_tensor": "fail_closed_for_required_optional_skip",
            "kv_cache": (runtime_access_plan.get("kv_layout_contract") or {}).get("runtime_cache_policy", {}).get("preferred_backend", "runtime_default"),
            "runtime_access_plan": "JUJU_RUNTIME_ACCESS_PLAN_V1_required_for_prefetch_residency_and_trace",
            "kv_layout_contract": "JUJU_KV_LAYOUT_CONTRACT_V1_required_for_generation_cache_accounting",
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
                "vram_double_admission_guard": {"enabled": True, "counter": "vram_inflight_bytes"},
                "macos_available_ram": {"count_inactive_pages": False},
                "metal_unified_memory_budget_percent": 60,
                "router_seq_topk_entropy": {
                    "enabled": False,
                    "enabled_only_in_approx_mode": True,
                    "exact_mode_must_use_source_router_topk": True,
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
            "access_plan_ref": "graph_ir.runtime_access_plan",
            "kv_layout_ref": "graph_ir.kv_layout_contract",
            "offload_units": ["shared_tensor", "expert_tensor", "dense_ffn_tensor", "qkv_page", "vision_tensor", "audio_tensor", "video_tensor", "document_tensor"],
            "io_policy": {
                "read_unit": "tensor_span",
                "alignment": 4096,
                "mmap": False,
                "mmap_scope": "shared_and_hot_sections_only",
                "cold_experts_io": "direct_aligned_async_read",
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
                    "sequential_block_size": s.get("sequential_block_size", 4096),
                    "random_block_size": s.get("random_block_size", 4096),
                    "priority": int(s.get("section_priority") or juju_section_priority(str(s.get("name", "")).lower())),
                }
                for s in sections
            ],
        },
        "moe_offload_policy": {
            "enabled": True,
            "router_first": True,
            "format": "JUJU_MOE_OFFLOAD_POLICY_V1",
            "exact_mode": juju_exact_mode_policy(),
            "approx_mode": juju_approx_mode_policy(),
            "expert_calibration_contract": juju_expert_calibration_contract(),
            "adaptive_runtime_scheduler": juju_adaptive_runtime_scheduler_contract(),
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
            "expert_unit_contract": {
                "unit": "layer_expert_projection_bundle",
                "projection_order": ["gate_up", "gate", "up", "down"],
                "o1_lookup": "runtime_access_plan.expert_offset_table",
                "tier_lookup": "runtime_access_plan.expert_tier_entries",
                "buddy_lookup": "BUDDY_MAP.expert_bundles",
                "predictor_lookup": "PREDICTOR.expert_activation_priors",
                "segment_lookup": "tensor.expert_layout.segment_count_per_expert",
                "combined_gate_up_split_offsets": "tensor.combined_gate_up_split",
                "partial_segment_fetch_allowed": False,
                "partial_segment_fetch_allowed_only_in_approx_mode": True,
                "single_expert_fetch_goal": "contiguous_or_minimal_range_read_per_selected_expert",
            },
            "stage_contract": {
                "prefill": {
                    "lookahead_enabled": False,
                    "phase_source": "engine.generation_phase == PREFILL",
                    "expert_policy": "stream_or_cpu_execute_selected_experts",
                    "protect_roles": ["token_embedding", "attention", "router", "norm", "lm_head"],
                },
                "decode": {
                    "lookahead_enabled": True,
                    "phase_source": "engine.generation_phase == DECODE",
                    "prediction_inputs": ["current_router_logits", "previous_layer_topk", "coactivation_history", "gate_input_snapshot"],
                    "cache_policy": "activation_aware_admission_and_eviction",
                },
            },
            "activation_trace": {
                "enabled": True,
                "record_per_token_topk": True,
                "record_router_entropy": True,
                "record_selected_expert_ids": True,
                "record_selected_expert_weights": True,
                "record_layer_transition": True,
                "mutable_idx_update": "expert_activation_priors_and_coactivation",
                "cold_start_source": "router_calibration_manifest_or_structural_uniform",
            },
            "predictor": {
                "enabled": True,
                "section_ref": "PREDICTOR",
                "inputs": ["router_scores", "gate_input_snapshots", "mutable_coactivation_index", "prefetch_miss_feedback"],
                "outputs": ["next_layer_expert_scores", "expert_next_use_epoch", "prefetch_priority"],
                "fallback_order": ["calibration_prior", "cross_layer_transition_prior", "static_same_layer_buddy", "router_score"],
                "exact_mode_role": "prefetch_hint_only",
                "may_replace_router_decision": False,
                "online_update": True,
            },
            "prefetch": {
                "unit": "layer_expert_bundle",
                "trigger": "router_topk_and_previous_layer_coactivation",
                "lookahead_layers": [1, 2],
                "coactivation_table": "mutable_runtime_index",
                "fallback_when_no_history": "router_scores",
                "bounded_by": ["ram_budget", "vram_budget", "staging_slots", "io_queue_depth"],
                "priority_field": "prefetch_priority",
                "schedule_contract": {
                    "prefetch_next_layer_attention_during_current_mlp": True,
                    "prefetch_selected_expert_bundle_after_router": True,
                    "prefetch_buddy_experts_when_bandwidth_free": True,
                    "avoid_decode_io_wait_bubbles": True,
                    "io_compute_overlap_required": True,
                    "max_adaptive_depth": 1,
                },
                "budget_inputs": {
                    "nvme_bandwidth_bytes_per_s": "hardware_probe.nvme_read_bw",
                    "pcie_bandwidth_bytes_per_s": "hardware_probe.pcie_h2d_bw",
                    "cpu_gflops": "hardware_probe.cpu_gflops",
                    "gpu_free_bytes": "runtime_device_free_bytes",
                    "ram_free_bytes": "runtime_available_ram_bytes",
                },
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
                "state_updates_atomic": True,
                "no_double_admission": True,
                "do_not_evict_inflight_or_current_layer": True,
            },
            "cpu_hot_miss": {
                "enabled": True,
                "condition": "expert_in_ram_and_decode_batch_le_4",
                "decision": "cpu_ms < pcie_transfer_ms",
                "cpu_gflops_default": float(contract_value(
                    contract,
                    "juju_hw.cpu_gflops",
                    "hardware_profile.cpu_gflops",
                    "hardware_probe.cpu_gflops",
                    default=1500.0,
                ) or 1500.0),
                "requires_runtime_cpu_probe_override": True,
            },
            "score_aware_precision": {
                "enabled": False,
                "enabled_only_in_approx_mode": True,
                "low_score_load_bits": 4,
                "fallback_when_nvfp4_unavailable": "int4",
                "requires_decoder": "engine_int4_or_scale4_decode",
            },
            "streaming": {
                "expert_streaming_required": True,
                "direct_io_alignment": 4096,
                "split_combined_gate_up": True,
                "allow_partial_expert_segments": False,
                "allow_partial_expert_segments_only_in_approx_mode": True,
                "io_backend_priority": ["io_uring_or_native_async", "direct_aligned_read", "mmap_shared_hot_only"],
                "chunk_size_source": "section.sequential_block_size",
                "cold_experts_mmap_forbidden": True,
                "gate_up_split_offsets_required": True,
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
            "require_qkv_policy_match": bool(contract.get("qkv_cache_schema") or contract.get("qkv_policy_contract")),
            "require_qkv_layer_head_epoch_scope": bool(contract.get("qkv_cache_schema") or contract.get("qkv_policy_contract")),
            "allow_plain_kv_reference_for_ppl": False,
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
    source_directory=None,
    source_total_bytes=None,
    print_layout_probes=True,
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
            "xxhash128": juju_xxhash128_hex(raw),
            "checksum_algorithm": "xxh3_128_or_blake2b_128",
            "hash_semantics": "json_section_payload_bytes",
            "mmap_friendly": 0,
        }
        sections.append(entry)
        section_sizes[section_type] = section_sizes.get(section_type, 0) + len(raw)
        return pos + len(raw)

    with requests.Session() as session:
        if source_directory is None:
            directory, total_bytes = read_remote_directory(session, source_url, token=token)
        else:
            directory = source_directory
            total_bytes = int(source_total_bytes or 0)
            if total_bytes <= 0:
                raise RuntimeError("source_total_bytes is required when source_directory is reused")
        print_gguf_byte_diagnostics(directory, source_name)
        if print_layout_probes:
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
        assign_bootstrap_expert_tiers(active_tensors, contract)
        split_meta = juju_effective_split_meta(
            source_name,
            artifact_source_name,
            len(active_tensors),
            sum(int(tensor.get("bytes") or 0) for tensor in active_tensors),
            split_info=split_info,
        )
        row_stride_stats = juju_row_stride_stats(active_tensors)
        modality_meta = juju_modality_metadata(contract, active_tensors)
        modality_flags = int(modality_meta["modality_flags"])
        runtime_arch = juju_runtime_arch_metadata(contract, directory)
        pos = JUJU_HEADER_BYTES
        table_offset = pos
        pos += JUJU_SECTION_TABLE_RESERVED_ENTRIES * JUJU_SECTION_ENTRY_BYTES
        pos = align_up(pos, 4096)

        meta = {
            "format": "JUJU_SHARDED_CONTAINER_V2",
            "source_format": "GGUF",
            "source_role": "conversion_source_only",
            "source_repo_id": source_repo_id,
            "source_path": source_path,
            "source_name": source_name,
            "artifact_source_name": artifact_source_name,
            "weight_file": artifact_names["weights"],
            "index_file": artifact_names["index"],
            "tensor_payload_layout": "4kb_aligned_sections_optional_row_stride_padded_rows",
            "row_stride_policy": {
                "enabled": juju_row_stride_padding_enabled(),
                "alignment_bytes": juju_row_stride_alignment_bytes(),
                "min_row_bytes": juju_row_stride_min_row_bytes(),
                "max_overhead_pct": juju_row_stride_max_overhead_pct(),
                "logical_cols_are_math_extent": True,
                "row_stride_bytes_are_storage_extent": True,
            },
            "row_stride_stats": row_stride_stats,
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
            **juju_contract_metadata(contract, source_name, source_repo_id, runtime_arch),
        }
        pos = add_json_section_at(pos, JUJU_SECTION_MODEL_META, "MODEL_META", meta)
        qkv_schema = _juju_effective_qkv_schema(contract, runtime_arch)
        pos = add_json_section_at(pos, JUJU_SECTION_QKV_POLICY, "QKV_POLICY", qkv_schema)
        # BUGFIX 974b: Pre-flight section count validation ★★★
        # Problem: Section count check at line 4325 happens AFTER writing all tensor data.
        # On large multimodal models, this means tens of GB are written before failure.
        # Solution: Estimate section count before writing and fail fast.
        estimated_sections = 2  # MODEL_META + TENSOR_INDEX (always present)
        estimated_sections += 1
        non_empty_buckets = sum(
            1 for bucket in JUJU_TENSOR_BUCKET_ORDER
            if any(t["bucket"] == bucket and t["bytes"] > 0 for t in active_tensors)
        )
        estimated_sections += non_empty_buckets
        # Runtime metadata sections (tier hint, predictor, etc.) — estimate conservatively
        estimated_sections += 9  # runtime metadata plus runtime contract section
        if estimated_sections > JUJU_SECTION_TABLE_RESERVED_ENTRIES:
            raise RuntimeError(
                f"JUJU section count will exceed limit: estimated {estimated_sections} > "
                f"{JUJU_SECTION_TABLE_RESERVED_ENTRIES}. Reduce multimodal encoders or "
                f"increase JUJU_SECTION_TABLE_RESERVED_ENTRIES."
            )

        bundle_id_counter = 0
        for bucket in JUJU_TENSOR_BUCKET_ORDER:
            group = sorted(
                [t for t in active_tensors if t["bucket"] == bucket and t["bytes"] > 0],
                key=lambda tensor, bucket=bucket: juju_tensor_file_order_key(tensor, bucket),
            )
            if not group:
                continue
            pos = align_up(pos, 4096)
            section_offset = pos
            section_source_ranges = []
            bundle_units, passthrough_tensors = juju_split_bucket_for_bundle_native(group, contract)
            for bundle in bundle_units:
                pos = align_up(pos, JUJU_BUNDLE_ALIGNMENT_BYTES)
                bundle_offset = pos
                bundle_id = bundle_id_counter
                bundle_id_counter += 1
                bundle_records = []
                for spec in bundle["members"]:
                    member_offset = pos
                    source_segment = juju_expert_member_source_segment(spec, member_offset)
                    source_segments.append(source_segment)
                    section_source_ranges.append(source_segment)
                    record = juju_bundle_member_tensor_index_record(
                        spec,
                        member_offset,
                        bundle_id,
                        bundle_offset,
                        contract,
                    )
                    tensor_records.append(record)
                    bundle_records.append(record)
                    pos += int(spec["output_size"])
                pos = align_up(pos, JUJU_BUNDLE_ALIGNMENT_BYTES)
                bundle_size = pos - bundle_offset
                for record in bundle_records:
                    record["bundle_size"] = int(bundle_size)
                    if isinstance(record.get("expert_layout"), dict):
                        record["expert_layout"]["bundle_size"] = int(bundle_size)
            for tensor in passthrough_tensors:
                pos = align_up(pos, 4096)
                tensor_offset = pos
                layout = juju_tensor_storage_layout(tensor)
                source_segment = juju_tensor_source_segment(tensor, tensor_offset, layout)
                source_segments.append(source_segment)
                section_source_ranges.append(source_segment)
                tensor_records.append(juju_tensor_index_record(tensor, bucket, tensor_offset, layout, contract))
                pos += int(layout["juju_bytes"])
            size = pos - section_offset
            section_type = section_type_for_bucket(bucket)
            section_sha = JUJU_ZERO_SHA256
            section_xxhash128 = "0" * 32
            checksum_algorithm = "not_precomputed"
            if juju_precompute_stream_section_sha():
                if juju_fast_section_checksum_enabled():
                    section_xxhash128 = checksum16_juju_section_ranges(
                        session,
                        source_url,
                        section_offset,
                        size,
                        section_source_ranges,
                        token=token,
                        chunk_size=chunk_size,
                    )
                    checksum_algorithm = "xxh3_128_or_blake2b_128"
                else:
                    section_sha = sha256_juju_section_ranges(
                        session,
                        source_url,
                        section_offset,
                        size,
                        section_source_ranges,
                        token=token,
                        chunk_size=chunk_size,
                    )
                    checksum_algorithm = "sha256"
                hash_semantics = "juju_section_bytes_including_alignment_padding"
            else:
                hash_semantics = "not_precomputed_for_streamed_upload"
            io_hints = juju_section_io_hints(bucket, size, contract)
            sections.append({
                "type": section_type,
                "name": bucket.upper()[:32],
                "offset": section_offset,
                "size": size,
                "sha256": section_sha,
                "xxhash128": section_xxhash128,
                "checksum_algorithm": checksum_algorithm,
                "hash_semantics": hash_semantics,
                **io_hints,
            })
            section_sizes[section_type] = section_sizes.get(section_type, 0) + size

        runtime_arch = juju_runtime_arch_metadata(contract, directory)

        for section_type, name, payload in build_juju_runtime_metadata_sections(tensor_records, contract, split_meta, runtime_arch):
            pos = add_json_section_at(pos, section_type, name, payload)

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
        artifact_uid_value = juju_artifact_uid(
            source_repo_id=source_repo_id,
            source_path=source_path,
            source_name=source_name,
            artifact_source_name=artifact_source_name,
            weight_file=artifact_names["weights"],
            split_meta=split_meta,
            tensor_count=len(tensor_records),
        )
        idx = {
            "format": JUJU_IDX_FORMAT,
            "schema_version": JUJU_IDX_SCHEMA_VERSION,
            "mutable_runtime_index": True,
            "artifact_uid": artifact_uid_value,
            "weight_file": artifact_names["weights"],
            "source_repo_id": source_repo_id,
            "source_path": source_path,
            "source_name": source_name,
            "artifact_source_name": artifact_source_name,
            "split": split_meta,
            **juju_idx_split_top_level_fields(split_meta),
            "modality_flags": modality_flags,
            "multimodal_contract": modality_meta,
            "exact_mode": juju_exact_mode_policy(),
            "approx_mode": juju_approx_mode_policy(),
            "ppl_acceptance_contract": juju_ppl_acceptance_contract(),
            "performance_acceptance_contract": juju_performance_acceptance_contract(),
            "expert_calibration_contract": juju_expert_calibration_contract(),
            "adaptive_runtime_scheduler": juju_adaptive_runtime_scheduler_contract(),
            "graph_ir_format": graph_ir["format"],
            "graph_ir_required": True,
            "graph_ir": graph_ir,
            "priority_tables": graph_ir["priority_tables"],
            "moe_offload_policy": graph_ir["moe_offload_policy"],
            "runtime_access_plan": graph_ir["runtime_access_plan"],
            "kv_layout_contract": graph_ir["kv_layout_contract"],
            "generation_contract": graph_ir["generation_contract"],
            "runtime_execution_manifest": graph_ir["runtime_execution_manifest"],
            "qkv_policy_contract": graph_ir["runtime_access_plan"].get("qkv_policy_contract", {}),
            "qkv_cache_schema_effective": graph_ir["runtime_access_plan"].get("qkv_cache_schema_effective", {}),
            "eval_kv_policy": graph_ir.get("eval_kv_policy", {}),
            "attention_layer_contract_table": graph_ir.get("attention_layer_contract_table", []),
            "expert_tier_entries": graph_ir["runtime_access_plan"].get("expert_tier_entries", []),
            "expert_offset_table": graph_ir["runtime_access_plan"].get("expert_offset_table", []),
            "expert_bundle_table": graph_ir["runtime_access_plan"].get("expert_bundle_table", {}),
            "expert_chunk_table": graph_ir["runtime_access_plan"].get("expert_chunk_table", {}),
            "moe_layer_bitmask_words": graph_ir["runtime_access_plan"].get("moe_layer_bitmask_words", []),
            "router_calibration_manifest": graph_ir["runtime_access_plan"].get("router_calibration_manifest", {}),
            "tensor_count": len(tensor_records),
            "tensors": tensor_records,
            "sections": list(sections),
            "row_stride_stats": row_stride_stats,
            **runtime_arch,
        }
        idx["format_self_check"] = juju_format_self_check(idx, sections, qkv_schema)
        idx["format_self_check"]["fatal_enforced"] = bool(juju_enforce_format_self_check())
        if not idx["format_self_check"]["ok"]:
            msg = "JUJU format self-check failed: " + json.dumps(
                idx["format_self_check"]["errors"][:16], ensure_ascii=False
            )
            if juju_enforce_format_self_check():
                raise RuntimeError(msg)
            print("WARNING:", msg[:2000])
        pos = add_json_section_at(pos, JUJU_SECTION_LAYER_ORDER_INDEX, "TENSOR_INDEX", idx)
        index_checksum = int(juju_section_checksum16_hex(sections[-1])[:16], 16) if sections else 0
        file_size_value = pos
        if len(sections) > JUJU_SECTION_TABLE_RESERVED_ENTRIES:
            raise RuntimeError(
                f"too many JUJU sections: {len(sections)} > {JUJU_SECTION_TABLE_RESERVED_ENTRIES}. "
                f"Increase JUJU_SECTION_TABLE_RESERVED_ENTRIES or reduce multimodal encoder sections."
            )

        table = b"".join(pack_section(entry) for entry in sections)
        table_capacity = JUJU_SECTION_TABLE_RESERVED_ENTRIES * JUJU_SECTION_ENTRY_BYTES
        table = table + (b"\x00" * (table_capacity - len(table)))
        header = make_header(contract, artifact_source_name, file_size_value, sections, section_sizes, index_checksum=index_checksum, modality_flags=modality_flags)
        add_fixed(0, header)
        add_fixed(table_offset, table)

    idx["sections"] = sections
    return {
        "format": "juju_sharded_container_v2",
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
        "storage_mode": "remote_range_to_streamed_4kb_aligned_juju_sections_optional_row_stride",
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
        # BUGFIX 977: Add retry adapter for network resilience ★★★
        # Problem: Single HTTP failure after 30min of conversion = restart from scratch.
        # Colab network is unstable; HuggingFace CDN has intermittent 502 errors.
        # Solution: Mount HTTPAdapter with Retry(3, backoff_factor=1) on the session.
        from requests.adapters import HTTPAdapter
        try:
            from urllib3.util.retry import Retry
            retry_strategy = Retry(
                total=5,
                backoff_factor=1,  # 1s, 2s, 4s, 8s, 16s
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "HEAD"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
        except ImportError:
            adapter = HTTPAdapter(max_retries=3)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
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
            item = {
                "kind": segment.get("kind", "source"),
                "offset": int(segment["offset"]),
                "size": int(segment["size"]),
                "source_offset": int(segment["source_offset"]),
            }
            if item["kind"] == "row_padded_source":
                item.update({
                    "rows": int(segment["rows"]),
                    "row_bytes": int(segment["row_bytes"]),
                    "row_stride_bytes": int(segment["row_stride_bytes"]),
                    "source_size": int(segment.get("source_size") or 0),
                })
            segments.append(item)
        self._segments = sorted(segments, key=lambda item: item["offset"])
        self._offsets = [item["offset"] for item in self._segments]
        # BUGFIX 978: Running SHA256 digest for streamed upload integrity ★★★
        # Problem: hash_semantics = "not_precomputed_for_streamed_upload" means
        # uploaded JUJU files have no integrity verification. Partial corruption
        # during upload is undetectable.
        # Solution: Running SHA256 over all read() output. After streaming completes,
        # source_sha256 property returns the hex digest for post-upload verification.
        self._upload_stream_sha_enabled = juju_upload_stream_sha_enabled()
        self._running_sha256 = hashlib.sha256() if self._upload_stream_sha_enabled else None
        self._total_bytes_hashed = 0
        self._progress_enabled = juju_progress_enabled()
        self._progress_last_bytes = 0
        self._progress_last_time = time.monotonic()
        self._progress_interval_bytes = juju_progress_interval_bytes()
        self._progress_interval_s = juju_progress_interval_s()
        if self._progress_enabled:
            print(
                f"[JUJU upload stream] start: file={self._plan.get('weight_file', '<unknown>')} "
                f"bytes={juju_format_bytes(self._size)}",
                flush=True,
            )

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
            if pos == 0 and self._pos != 0:
                self._running_sha256 = hashlib.sha256() if self._upload_stream_sha_enabled else None
                self._total_bytes_hashed = 0
                self._progress_last_bytes = 0
                self._progress_last_time = time.monotonic()
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
                    elif segment["kind"] == "row_padded_source":
                        chunks.append(self._read_row_padded_source_segment(segment, rel, take))
                    else:
                        chunks.append(self._read_source_segment(segment, rel, take))
                    self._pos += take
                    continue
                next_offset = self._segments[idx + 1]["offset"] if idx + 1 < len(self._segments) else self._size
                take = min(end - self._pos, next_offset - self._pos)
                chunks.append(b"\x00" * take)
                self._pos += take
            result = b"".join(chunks)
            if result:
                if self._running_sha256 is not None:
                    self._running_sha256.update(result)
                self._total_bytes_hashed += len(result)
                if self._progress_enabled:
                    now = time.monotonic()
                    if (
                        self._total_bytes_hashed - self._progress_last_bytes >= self._progress_interval_bytes
                        or now - self._progress_last_time >= self._progress_interval_s
                        or self._total_bytes_hashed >= self._size
                    ):
                        pct = (100.0 * self._total_bytes_hashed / self._size) if self._size > 0 else 100.0
                        print(
                            f"[JUJU upload stream] {self._plan.get('weight_file', '<unknown>')}: "
                            f"{juju_format_bytes(self._total_bytes_hashed)}/{juju_format_bytes(self._size)} "
                            f"({pct:.1f}%)",
                            flush=True,
                        )
                        self._progress_last_bytes = self._total_bytes_hashed
                        self._progress_last_time = now
            return result

    @property
    def source_sha256(self):
        """Return hex SHA256 of all bytes read so far (complete after full streaming)."""
        return self._running_sha256.hexdigest() if self._running_sha256 is not None else ""

    @property
    def artifact_sha256(self):
        return self.source_sha256

    @property
    def total_bytes_hashed(self):
        return self._total_bytes_hashed

    @property
    def total_bytes_streamed(self):
        return self._total_bytes_hashed

    def _read_source_segment(self, segment, rel, size):
        return self._read_source_abs(
            int(segment["source_offset"]) + int(rel),
            int(size),
            int(segment["source_offset"]) + int(segment["size"]),
        )

    def _read_source_abs(self, source_abs, size, source_limit):
        # BUGFIX 977: Exponential backoff retry for HTTP range requests ★★★
        # Problem: Single 502 error after hours of conversion = total restart.
        # Solution: 3 retries with exponential backoff (1s, 2s, 4s).
        # The session already has urllib3 Retry, but this handles cache-miss fetches
        # where resp.content might be empty due to transient CDN issues.
        out = []
        remaining = int(size)
        while remaining > 0:
            if self._cache_start <= source_abs < self._cache_end:
                cache_rel = source_abs - self._cache_start
                take = min(remaining, self._cache_end - source_abs)
                out.append(self._cache_data[cache_rel:cache_rel + take])
                source_abs += take
                remaining -= take
                continue
            fetch_end = min(int(source_limit), source_abs + max(self._remote_chunk, remaining)) - 1
            data = None
            last_error = None
            for attempt in range(5):
                try:
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
                    if data:
                        break
                except Exception as e:
                    last_error = e
                    if attempt < 4:
                        delay = (2 ** attempt)  # 1, 2, 4, 8, 16 seconds
                        import sys
                        print(
                            f"[JUJU WARNING] HTTP range fetch failed (attempt {attempt + 1}/5), "
                            f"retrying in {delay}s: {e}",
                            file=sys.stderr,
                        )
                        time.sleep(delay)
            if not data:
                if last_error:
                    raise last_error
                raise EOFError("empty source range while streaming JUJU upload after retries")
            self._cache_start = source_abs
            self._cache_end = source_abs + len(data)
            self._cache_data = data
        return b"".join(out)

    def _read_row_padded_source_segment(self, segment, rel, size):
        out = []
        remaining = int(size)
        pos = int(rel)
        rows = int(segment["rows"])
        row_bytes = int(segment["row_bytes"])
        row_stride = int(segment["row_stride_bytes"])
        source_base = int(segment["source_offset"])
        source_limit = source_base + int(segment.get("source_size") or (rows * row_bytes))
        while remaining > 0:
            row = pos // row_stride
            in_row = pos % row_stride
            if row >= rows:
                out.append(b"\x00" * remaining)
                break
            if in_row < row_bytes:
                take = min(remaining, row_bytes - in_row)
                source_abs = source_base + row * row_bytes + in_row
                out.append(self._read_source_abs(source_abs, take, source_limit))
            else:
                take = min(remaining, row_stride - in_row)
                out.append(b"\x00" * take)
            pos += take
            remaining -= take
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
    source_directory=None,
    source_total_bytes=None,
    print_layout_probes=True,
):
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if juju_progress_enabled():
        split_label = ""
        if isinstance(split_info, dict) and split_info.get("enabled"):
            split_label = f" split={int(split_info.get('split_index') or 0):02d}/{int(split_info.get('split_count') or 0):02d}"
        print(f"[JUJU stream] planning {artifact_source_name or source_name}{split_label}", flush=True)
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
        source_directory=source_directory,
        source_total_bytes=source_total_bytes,
        print_layout_probes=print_layout_probes,
    )
    if juju_progress_enabled():
        print(
            f"[JUJU stream] plan ready: file={plan['weight_file']} "
            f"bytes={juju_format_bytes(plan['bytes'])} tensors={plan['tensor_count']} "
            f"sections={plan['section_count']}",
            flush=True,
        )
    index_path.write_text(json.dumps(plan["index_json"], ensure_ascii=False, indent=2), encoding="utf-8")
    verify_path = index_path.parent / plan["verify_file"]
    if juju_progress_enabled():
        print(f"[JUJU stream] logical hash verify start: {plan['weight_file']}", flush=True)
    with requests.Session() as verify_session:
        verify_manifest = build_juju_stream_verify_manifest(
            session=verify_session,
            source_url=source_url,
            source_name=source_name,
            source_path=source_path,
            weight_file=plan["weight_file"],
            index_path=index_path,
            verify_path=verify_path,
            tensor_records=plan["index_json"].get("tensors") or [],
            token=token,
            chunk_size=chunk_size,
        )
    verify_path.write_text(json.dumps(verify_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if juju_progress_enabled():
        print(f"[JUJU stream] logical hash verify done: {plan['weight_file']} -> {verify_path}", flush=True)
    info = {
        "format": plan["format"],
        "path": f"<stream:{plan['weight_file']}>",
        "index_path": str(index_path),
        "verify_path": str(verify_path),
        "source_name": plan["source_name"],
        "artifact_source_name": plan["artifact_source_name"],
        "weight_file": plan["weight_file"],
        "index_file": plan["index_file"],
        "verify_file": plan["verify_file"],
        "split": plan["split"],
        "bytes": plan["bytes"],
        "index_bytes": index_path.stat().st_size,
        "verify_bytes": verify_path.stat().st_size,
        "index_sha256": sha256_file(index_path),
        "verify_sha256": sha256_file(verify_path),
        "source_bytes": plan["source_bytes"],
        "source_sha256": verify_manifest.get("tensor_payload_equivalence", {}).get("source_logical_sha256", ""),
        "juju_logical_sha256": verify_manifest.get("tensor_payload_equivalence", {}).get("juju_logical_sha256", ""),
        "tensor_payload_equivalence_all_match": bool(verify_manifest.get("tensor_payload_equivalence", {}).get("all_match")),
        "ppl_acceptance": verify_manifest.get("ppl_acceptance", {}),
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
            source_directory=directory,
            source_total_bytes=total_bytes,
            print_layout_probes=False,
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
    source_directory=None,
    source_total_bytes=None,
    print_layout_probes=True,
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
            "xxhash128": juju_xxhash128_hex(raw),
            "checksum_algorithm": "xxh3_128_or_blake2b_128",
            "hash_semantics": "json_section_payload_bytes",
            "mmap_friendly": 0,
        }
        sections.append(entry)
        section_sizes[section_type] = section_sizes.get(section_type, 0) + len(raw)

    with requests.Session() as session:
        if source_directory is None:
            directory, total_bytes = read_remote_directory(session, source_url, token=token)
        else:
            directory = source_directory
            total_bytes = int(source_total_bytes or 0)
            if total_bytes <= 0:
                raise RuntimeError("source_total_bytes is required when source_directory is reused")
        print_gguf_byte_diagnostics(directory, source_name)
        if print_layout_probes:
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
        assign_bootstrap_expert_tiers(active_tensors, contract)
        split_meta = juju_effective_split_meta(
            source_name,
            artifact_source_name,
            len(active_tensors),
            sum(int(tensor.get("bytes") or 0) for tensor in active_tensors),
            split_info=split_info,
        )
        row_stride_stats = juju_row_stride_stats(active_tensors)
        modality_meta = juju_modality_metadata(contract, active_tensors)
        modality_flags = int(modality_meta["modality_flags"])
        runtime_arch = juju_runtime_arch_metadata(contract, directory)
        expected_write_bytes = int((split_info or {}).get("estimated_file_bytes") or 0)
        with output_path.open("wb") as raw_out:
            out = JujuProgressWriter(raw_out, output_path.name, expected_bytes=expected_write_bytes)
            out.write(b"\x00" * JUJU_HEADER_BYTES)
            table_offset = out.tell()
            out.write(b"\x00" * (JUJU_SECTION_TABLE_RESERVED_ENTRIES * JUJU_SECTION_ENTRY_BYTES))
            write_padding(out, 4096)

            meta = {
                "format": "JUJU_SHARDED_CONTAINER_V2",
                "source_format": "GGUF",
                "source_role": "conversion_source_only",
                "source_repo_id": source_repo_id,
                "source_path": source_path,
                "source_name": source_name,
                "artifact_source_name": artifact_source_name,
                "weight_file": output_path.name,
                "index_file": index_path.name,
                "tensor_payload_layout": "4kb_aligned_sections_optional_row_stride_padded_rows",
                "row_stride_policy": {
                    "enabled": juju_row_stride_padding_enabled(),
                    "alignment_bytes": juju_row_stride_alignment_bytes(),
                    "min_row_bytes": juju_row_stride_min_row_bytes(),
                    "max_overhead_pct": juju_row_stride_max_overhead_pct(),
                    "logical_cols_are_math_extent": True,
                    "row_stride_bytes_are_storage_extent": True,
                },
                "row_stride_stats": row_stride_stats,
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
                **juju_contract_metadata(contract, source_name, source_repo_id, runtime_arch),
            }
            add_json_section(out, JUJU_SECTION_MODEL_META, "MODEL_META", meta)
            qkv_schema = _juju_effective_qkv_schema(contract, runtime_arch)
            add_json_section(out, JUJU_SECTION_QKV_POLICY, "QKV_POLICY", qkv_schema)

            bundle_id_counter = 0
            for bucket in JUJU_TENSOR_BUCKET_ORDER:
                group = sorted(
                    [t for t in active_tensors if t["bucket"] == bucket and t["bytes"] > 0],
                    key=lambda tensor, bucket=bucket: juju_tensor_file_order_key(tensor, bucket),
                )
                if not group:
                    continue
                write_padding(out, 4096)
                offset = out.tell()
                digest = hashlib.sha256() if juju_section_stream_sha_enabled() else JujuNoopDigest()
                bundle_units, passthrough_tensors = juju_split_bucket_for_bundle_native(group, contract)
                for bundle in bundle_units:
                    write_padding(out, JUJU_BUNDLE_ALIGNMENT_BYTES, digest=digest)
                    bundle_offset = out.tell()
                    bundle_id = bundle_id_counter
                    bundle_id_counter += 1
                    bundle_records = []
                    for spec in bundle["members"]:
                        member_offset = out.tell()
                        stream_juju_expert_member_payload(
                            session,
                            source_url,
                            spec,
                            out,
                            token,
                            digest,
                            chunk_size=chunk_size,
                        )
                        record = juju_bundle_member_tensor_index_record(
                            spec,
                            member_offset,
                            bundle_id,
                            bundle_offset,
                            contract,
                        )
                        tensor_records.append(record)
                        bundle_records.append(record)
                    write_padding(out, JUJU_BUNDLE_ALIGNMENT_BYTES, digest=digest)
                    bundle_size = out.tell() - bundle_offset
                    for record in bundle_records:
                        record["bundle_size"] = int(bundle_size)
                        if isinstance(record.get("expert_layout"), dict):
                            record["expert_layout"]["bundle_size"] = int(bundle_size)
                for tensor in passthrough_tensors:
                    write_padding(out, 4096, digest=digest)
                    tensor_offset = out.tell()
                    layout = stream_juju_tensor_payload(
                        session,
                        source_url,
                        tensor,
                        out,
                        token,
                        digest,
                        chunk_size=chunk_size,
                    )
                    tensor_records.append(juju_tensor_index_record(tensor, bucket, tensor_offset, layout, contract))
                size = out.tell() - offset
                section_type = section_type_for_bucket(bucket)
                io_hints = juju_section_io_hints(bucket, size, contract)
                sections.append({
                    "type": section_type,
                    "name": bucket.upper()[:32],
                    "offset": offset,
                    "size": size,
                    "sha256": digest.hexdigest(),
                    "xxhash128": "0" * 32,
                    "checksum_algorithm": "sha256_streaming_write" if juju_section_stream_sha_enabled() else "not_precomputed_fast_write",
                    "hash_semantics": "juju_section_bytes_including_alignment_padding" if juju_section_stream_sha_enabled() else "payload_hash_skipped_fast_write",
                    **io_hints,
                })
                section_sizes[section_type] = section_sizes.get(section_type, 0) + size

            runtime_arch = juju_runtime_arch_metadata(contract, directory)

            for section_type, name, payload in build_juju_runtime_metadata_sections(tensor_records, contract, split_meta, runtime_arch):
                add_json_section(out, section_type, name, payload)

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
            artifact_uid_value = juju_artifact_uid(
                source_repo_id=source_repo_id,
                source_path=source_path,
                source_name=source_name,
                artifact_source_name=artifact_source_name,
                weight_file=output_path.name,
                split_meta=split_meta,
                tensor_count=len(tensor_records),
            )
            idx = {
                "format": JUJU_IDX_FORMAT,
                "schema_version": JUJU_IDX_SCHEMA_VERSION,
                "mutable_runtime_index": True,
                "artifact_uid": artifact_uid_value,
                "weight_file": output_path.name,
                "source_repo_id": source_repo_id,
                "source_path": source_path,
                "source_name": source_name,
                "artifact_source_name": artifact_source_name,
                "split": split_meta,
                **juju_idx_split_top_level_fields(split_meta),
                "modality_flags": modality_flags,
                "multimodal_contract": modality_meta,
                "exact_mode": juju_exact_mode_policy(),
                "approx_mode": juju_approx_mode_policy(),
                "ppl_acceptance_contract": juju_ppl_acceptance_contract(),
                "performance_acceptance_contract": juju_performance_acceptance_contract(),
                "expert_calibration_contract": juju_expert_calibration_contract(),
                "adaptive_runtime_scheduler": juju_adaptive_runtime_scheduler_contract(),
                "graph_ir_format": graph_ir["format"],
                "graph_ir_required": True,
                "graph_ir": graph_ir,
                "priority_tables": graph_ir["priority_tables"],
                "moe_offload_policy": graph_ir["moe_offload_policy"],
                "runtime_access_plan": graph_ir["runtime_access_plan"],
                "kv_layout_contract": graph_ir["kv_layout_contract"],
                "generation_contract": graph_ir["generation_contract"],
                "runtime_execution_manifest": graph_ir["runtime_execution_manifest"],
                "qkv_policy_contract": graph_ir["runtime_access_plan"].get("qkv_policy_contract", {}),
                "qkv_cache_schema_effective": graph_ir["runtime_access_plan"].get("qkv_cache_schema_effective", {}),
                "eval_kv_policy": graph_ir.get("eval_kv_policy", {}),
                "attention_layer_contract_table": graph_ir.get("attention_layer_contract_table", []),
                "expert_tier_entries": graph_ir["runtime_access_plan"].get("expert_tier_entries", []),
                "expert_offset_table": graph_ir["runtime_access_plan"].get("expert_offset_table", []),
                "expert_bundle_table": graph_ir["runtime_access_plan"].get("expert_bundle_table", {}),
                "expert_chunk_table": graph_ir["runtime_access_plan"].get("expert_chunk_table", {}),
                "moe_layer_bitmask_words": graph_ir["runtime_access_plan"].get("moe_layer_bitmask_words", []),
                "router_calibration_manifest": graph_ir["runtime_access_plan"].get("router_calibration_manifest", {}),
                "tensor_count": len(tensor_records),
                "tensors": tensor_records,
                "sections": sections,
                "row_stride_stats": row_stride_stats,
                **runtime_arch,
            }
            idx["format_self_check"] = juju_format_self_check(idx, sections, qkv_schema)
            idx["format_self_check"]["fatal_enforced"] = bool(juju_enforce_format_self_check())
            if not idx["format_self_check"]["ok"]:
                msg = "JUJU format self-check failed: " + json.dumps(
                    idx["format_self_check"]["errors"][:16], ensure_ascii=False
                )
                if juju_enforce_format_self_check():
                    raise RuntimeError(msg)
                print("WARNING:", msg[:2000])
            add_json_section(out, JUJU_SECTION_LAYER_ORDER_INDEX, "TENSOR_INDEX", idx)
            index_checksum = int(juju_section_checksum16_hex(sections[-1])[:16], 16) if sections else 0
            file_size_value = out.tell()
            if len(sections) > JUJU_SECTION_TABLE_RESERVED_ENTRIES:
                raise RuntimeError(f"too many JUJU sections: {len(sections)}")
            out.seek(table_offset)
            for entry in sections:
                out.write(pack_section(entry))
            out.seek(0)
            out.write(make_header(contract, artifact_source_name, file_size_value, sections, section_sizes, index_checksum=index_checksum, modality_flags=modality_flags))
            out.flush()
            out.close_report()

    construction_self_check = juju_construction_self_check(idx, sections, tensor_records, file_size_value, output_path=output_path)
    construction_self_check["fatal_enforced"] = bool(juju_enforce_construction_self_check())
    idx["construction_self_check"] = construction_self_check
    if not construction_self_check["ok"] and juju_enforce_construction_self_check():
        raise RuntimeError("JUJU construction self-check failed: " + json.dumps(
            construction_self_check["errors"][:16], ensure_ascii=False
        ))
    if not construction_self_check["ok"]:
        print("WARNING: JUJU construction self-check reported non-fatal issues:", json.dumps(
            construction_self_check["errors"][:16], ensure_ascii=False
        )[:2000])
    index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    verify_name = juju_artifact_names(artifact_source_name)["verify"]
    verify_path = output_path.parent / verify_name
    if juju_payload_verify_enabled():
        with requests.Session() as verify_session:
            verify_manifest = build_juju_verify_manifest(
                session=verify_session,
                source_url=source_url,
                source_name=source_name,
                source_path=source_path,
                output_path=output_path,
                index_path=index_path,
                verify_path=verify_path,
                tensor_records=tensor_records,
                token=token,
                chunk_size=chunk_size,
            )
        if not verify_manifest.get("tensor_payload_equivalence", {}).get("all_match"):
            raise RuntimeError("JUJU logical payload equivalence failed: " + json.dumps(
                verify_manifest.get("tensor_payload_equivalence", {}).get("mismatches", [])[:16],
                ensure_ascii=False,
            ))
    else:
        verify_manifest = build_juju_fast_upload_verify_manifest(
            source_name=source_name,
            source_path=source_path,
            weight_file=output_path.name,
            index_path=index_path,
            tensor_records=tensor_records,
            source_bytes=total_bytes,
            output_bytes=output_path.stat().st_size,
            mode="physical_part_upload_only_no_payload_hash",
        )
    verify_manifest["construction_self_check"] = construction_self_check
    verify_path.write_text(json.dumps(verify_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "format": "juju_sharded_container_v2",
        "path": str(output_path),
        "index_path": str(index_path),
        "verify_path": str(verify_path),
        "source_name": source_name,
        "artifact_source_name": artifact_source_name,
        "weight_file": output_path.name,
        "index_file": index_path.name,
        "verify_file": verify_name,
        "split": split_meta,
        "bytes": output_path.stat().st_size,
        "index_bytes": index_path.stat().st_size,
        "verify_bytes": verify_path.stat().st_size,
        "sha256": sha256_file(output_path) if juju_output_file_sha_enabled() else "",
        "index_sha256": sha256_file(index_path),
        "verify_sha256": sha256_file(verify_path),
        "source_bytes": total_bytes,
        "source_sha256": verify_manifest.get("tensor_payload_equivalence", {}).get("source_logical_sha256", ""),
        "juju_logical_sha256": verify_manifest.get("tensor_payload_equivalence", {}).get("juju_logical_sha256", ""),
        "tensor_payload_equivalence_all_match": bool(verify_manifest.get("tensor_payload_equivalence", {}).get("all_match")),
        "ppl_acceptance": verify_manifest.get("ppl_acceptance", {}),
        "tensor_count": len(tensor_records),
        "section_count": len(sections),
        "storage_mode": "remote_range_to_4kb_aligned_juju_sections_optional_row_stride",
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


def plan_juju_shard_physical_part_writes_from_hf_url(
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
    plans = []
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
        plans.append({
            "source_url": source_url,
            "source_name": source_name,
            "source_path": source_path,
            "output_path": child_output_path,
            "index_path": child_index_path,
            "contract": contract,
            "token": token,
            "source_repo_id": source_repo_id,
            "chunk_size": chunk_size,
            "artifact_source_name": artifact_source_name,
            "tensor_name_allowlist": split["tensor_names"],
            "split_info": split_info,
            "source_directory": directory,
            "source_total_bytes": total_bytes,
            "print_layout_probes": False,
        })
    return plans
