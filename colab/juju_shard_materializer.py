import hashlib
import io
import json
import bisect
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


def juju_artifact_names(source_name):
    stem = Path(source_name).stem
    return {
        "weights": f"{stem}.juju",
        "index": f"{stem}.juju.idx",
        "verify": f"{stem}.juju.verify.json",
    }


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
        nxt = tensors[order[pos + 1]]["relative_offset"] if pos + 1 < len(order) else total_bytes - data_start
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
            size = min(size * 2, total)


def tensor_bucket(name):
    lower = str(name).lower()
    if "shared_expert" in lower or "shared.expert" in lower:
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


def hash_remote_tensor_group(session, url, group, token=None, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    for tensor in group:
        resp = fetch_range(
            session,
            url,
            tensor["source_offset"],
            tensor["source_offset"] + tensor["bytes"] - 1,
            token=token,
            stream=True,
        )
        try:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    digest.update(chunk)
        finally:
            resp.close()
    return digest.hexdigest()


def build_juju_shard_plan_from_hf_url(
    *,
    source_url,
    source_name,
    source_path,
    contract,
    token=None,
    source_repo_id="",
    chunk_size=16 * 1024 * 1024,
):
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
            "weight_file": juju_artifact_names(source_name)["weights"],
            "index_file": juju_artifact_names(source_name)["index"],
            "tensor_payload_layout": "4kb_aligned_tensor_sections",
            "artifact_name_policy": "preserve_original_shard_stem_change_extension_only",
            "gguf_directory": {
                "version": directory["version"],
                "tensor_count": directory["tensor_count"],
                "kv_count": directory["kv_count"],
                "alignment": directory["alignment"],
                "data_start": directory["data_start"],
                "source_bytes": total_bytes,
            },
            "contract": contract,
        }
        pos = add_json_section_at(pos, JUJU_SECTION_MODEL_META, "MODEL_META", meta)
        pos = add_json_section_at(pos, JUJU_SECTION_QKV_POLICY, "QKV_POLICY", contract.get("qkv_cache_schema", {}))

        for bucket in ("shared_weights", "hot_experts", "warm_experts", "cold_experts"):
            group = [t for t in directory["tensors"] if t["bucket"] == bucket and t["bytes"] > 0]
            if not group:
                continue
            pos = align_up(pos, 4096)
            section_offset = pos
            section_hash = hash_remote_tensor_group(session, source_url, group, token=token, chunk_size=chunk_size)
            for tensor in group:
                pos = align_up(pos, 4096)
                tensor_offset = pos
                source_segments.append({
                    "offset": tensor_offset,
                    "size": tensor["bytes"],
                    "source_offset": tensor["source_offset"],
                })
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
                })
                pos += tensor["bytes"]
            size = pos - section_offset
            section_type = section_type_for_bucket(bucket)
            sections.append({
                "type": section_type,
                "name": bucket.upper()[:32],
                "offset": section_offset,
                "size": size,
                "sha256": section_hash,
                "prefetch_distance": 2 if bucket != "shared_weights" else 0,
                "mmap_friendly": 1,
            })
            section_sizes[section_type] = section_sizes.get(section_type, 0) + size

        idx = {
            "format": "JUJU_IDX_JSON_V1",
            "mutable_runtime_index": True,
            "weight_file": juju_artifact_names(source_name)["weights"],
            "source_repo_id": source_repo_id,
            "source_path": source_path,
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
        header = make_header(contract, source_name, file_size_value, sections, section_sizes)
        add_fixed(0, header)
        add_fixed(table_offset, table)

    idx["sections"] = sections
    return {
        "format": "juju_sharded_container_v1",
        "source_url": source_url,
        "source_name": source_name,
        "source_path": source_path,
        "source_repo_id": source_repo_id,
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
        self._plan = plan
        self._size = int(plan["bytes"])
        self._pos = 0
        self._session = requests.Session()
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
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
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
        data = self.read(len(buffer))
        n = len(data)
        buffer[:n] = data
        return n

    def read(self, size=-1):
        if self.closed or self._pos >= self._size:
            return b""
        if size is None or size < 0:
            end = self._size
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
                    start = segment["source_offset"] + rel
                    resp = fetch_range(
                        self._session,
                        self._plan["source_url"],
                        start,
                        start + take - 1,
                        token=self._plan.get("token"),
                        stream=False,
                    )
                    try:
                        chunks.append(resp.content)
                    finally:
                        resp.close()
                self._pos += take
                continue
            next_offset = self._segments[idx + 1]["offset"] if idx + 1 < len(self._segments) else self._size
            take = min(end - self._pos, next_offset - self._pos)
            chunks.append(b"\x00" * take)
            self._pos += take
        return b"".join(chunks)

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
    )
    index_path.write_text(json.dumps(plan["index_json"], ensure_ascii=False, indent=2), encoding="utf-8")
    info = {
        "format": plan["format"],
        "path": f"<stream:{juju_artifact_names(source_name)['weights']}>",
        "index_path": str(index_path),
        "bytes": plan["bytes"],
        "index_bytes": index_path.stat().st_size,
        "index_sha256": sha256_file(index_path),
        "source_bytes": plan["source_bytes"],
        "tensor_count": plan["tensor_count"],
        "section_count": plan["section_count"],
        "storage_mode": plan["storage_mode"],
        "artifact_name_policy": plan["artifact_name_policy"],
    }
    return info, JujuVirtualFile(plan)


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
):
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
                "weight_file": output_path.name,
                "index_file": index_path.name,
                "tensor_payload_layout": "4kb_aligned_tensor_sections",
                "artifact_name_policy": "preserve_original_shard_stem_change_extension_only",
                "gguf_directory": {
                    "version": directory["version"],
                    "tensor_count": directory["tensor_count"],
                    "kv_count": directory["kv_count"],
                    "alignment": directory["alignment"],
                    "data_start": directory["data_start"],
                    "source_bytes": total_bytes,
                },
                "contract": contract,
            }
            add_json_section(out, JUJU_SECTION_MODEL_META, "MODEL_META", meta)
            add_json_section(out, JUJU_SECTION_QKV_POLICY, "QKV_POLICY", contract.get("qkv_cache_schema", {}))

            for bucket in ("shared_weights", "hot_experts", "warm_experts", "cold_experts"):
                group = [t for t in directory["tensors"] if t["bucket"] == bucket and t["bytes"] > 0]
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

            idx = {
                "format": "JUJU_IDX_JSON_V1",
                "mutable_runtime_index": True,
                "weight_file": output_path.name,
                "source_repo_id": source_repo_id,
                "source_path": source_path,
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
            out.write(make_header(contract, source_name, file_size_value, sections, section_sizes))

    index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "format": "juju_sharded_container_v1",
        "path": str(output_path),
        "index_path": str(index_path),
        "bytes": output_path.stat().st_size,
        "index_bytes": index_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "index_sha256": sha256_file(index_path),
        "source_bytes": total_bytes,
        "tensor_count": len(tensor_records),
        "section_count": len(sections),
        "storage_mode": "remote_range_to_4kb_aligned_juju_sections",
        "artifact_name_policy": "original_shard_stem_with_juju_extension",
    }
