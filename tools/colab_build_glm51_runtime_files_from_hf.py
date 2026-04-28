#!/usr/bin/env python3
"""
Colab helper: build StorageLLM runtime sidecars from a Hugging Face JUJU repo.

It reads only the JUJU headers, JSON footers, and scale4 blocks through HTTP
Range requests. It does not download the full 465GB model parts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url


DEFAULT_TOKENIZER_REPO = "lukealonso/GLM-5.1-NVFP4"
JUJU_MAGIC = b"G51STOR1"
JUJU_HEADER = struct.Struct("<8sIIQQQQQQ")
SCALE4_MAGIC = b"G5S4IDX1"
SCALE4_HEADER = struct.Struct("<8sIIQQQQII")
SCALE4_ENTRY = struct.Struct("<QIIIIIIffQQQ")
PROJ_ORDER = ("gate_proj", "up_proj", "down_proj")
CSV_HEADER = [
    "part",
    "shard",
    "shard_file",
    "source_file",
    "weight_key",
    "layer",
    "expert",
    "proj",
    "rows",
    "cols",
    "groups",
    "group_size",
    "dtype",
    "scale_mode",
    "storage_mode",
    "weight_byte_offset",
    "weight_byte_length",
    "weight_block",
    "scale_key",
    "scale_dtype",
    "scale_byte_offset",
    "scale_byte_length",
    "scale_block",
    "scale2_key",
    "scale2_byte_offset",
    "scale2_byte_length",
]


def parse_hf_ref(ref: str, revision: str) -> Tuple[str, str, str]:
    ref = ref.strip()
    prefix = ""
    if ref.startswith("https://huggingface.co/"):
        parsed = urlparse(ref)
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"bad Hugging Face URL: {ref}")
        repo_id = f"{parts[0]}/{parts[1]}"
        if len(parts) >= 4 and parts[2] in {"tree", "resolve", "blob"}:
            revision = parts[3]
            prefix = "/".join(parts[4:])
        return repo_id, revision, prefix.strip("/")
    return ref, revision, prefix


def auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def read_range(session: requests.Session, url: str, start: int, length: int, token: str, label: str = "") -> bytes:
    end = start + length - 1
    headers = auth_headers(token)
    headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(5):
        resp = session.get(url, headers=headers, timeout=120)
        ok_status = resp.status_code == 206 or (start == 0 and resp.status_code == 200)
        if ok_status and len(resp.content) >= length:
            return resp.content[:length]
        time.sleep(2 + attempt * 2)
    raise RuntimeError(f"range read failed {label} status={resp.status_code} got={len(resp.content)} want={length}")


def stream_range_to_file(
    session: requests.Session,
    url: str,
    start: int,
    length: int,
    token: str,
    out,
    label: str = "",
) -> None:
    if length <= 0:
        return
    end = start + length - 1
    headers = auth_headers(token)
    headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(5):
        resp = session.get(url, headers=headers, stream=True, timeout=120)
        ok_status = resp.status_code == 206 or (start == 0 and resp.status_code == 200)
        if not ok_status:
            time.sleep(2 + attempt * 2)
            continue
        written = 0
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            if not chunk:
                continue
            remaining = length - written
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
            out.write(chunk)
            written += len(chunk)
        if written == length:
            return
        time.sleep(2 + attempt * 2)
    raise RuntimeError(f"range stream failed {label} got={written} want={length}")


def list_juju_files(repo_id: str, revision: str, prefix: str, token: str) -> List[str]:
    api = HfApi(token=token or None)
    files = api.list_repo_files(repo_id=repo_id, revision=revision)
    out = []
    for name in files:
        base = Path(name).name
        if prefix and not name.startswith(prefix.rstrip("/") + "/"):
            continue
        if base.startswith("glm5.1-storage-part") and base.endswith(".juju"):
            out.append(name)
    out.sort()
    if not out:
        raise FileNotFoundError(f"no glm5.1-storage-part*.juju in {repo_id}@{revision} prefix={prefix!r}")
    return out


def read_remote_footer(session: requests.Session, url: str, token: str, filename: str) -> Dict[str, Any]:
    raw = read_range(session, url, 0, JUJU_HEADER.size, token, filename)
    magic, version, _flags, _header_bytes, _data_offset, json_offset, json_bytes, file_size, _block_count = (
        JUJU_HEADER.unpack(raw)
    )
    if magic != JUJU_MAGIC or version != 1:
        raise ValueError(f"{filename}: not a GLM5.1 Storage JUJU file")
    payload = read_range(session, url, int(json_offset), int(json_bytes), token, filename + ":footer")
    meta = json.loads(payload.decode("utf-8"))
    meta["_filename"] = filename
    meta["_url"] = url
    meta["_file_size"] = int(file_size)
    return meta


def part_id_from_meta(meta: Dict[str, Any]) -> int:
    part = int(meta.get("part", 0))
    if part > 0:
        return part
    name = Path(str(meta["_filename"])).stem
    digits = "".join(ch for ch in name[-2:] if ch.isdigit())
    return int(digits) if digits else 0


def block_by_id(meta: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(item["id"]): item for item in meta.get("blocks", [])}


def block_range(blocks: Dict[int, Dict[str, Any]], block_id: Optional[int]) -> Tuple[int, int]:
    if block_id is None:
        return 0, 0
    item = blocks[int(block_id)]
    return int(item["offset"]), int(item["length"])


def shard_file_for_runtime(filename: str) -> str:
    return "parts/" + Path(filename).name


def row_for_projection(
    part: int,
    shard_file: str,
    bundle: Dict[str, Any],
    proj: str,
    info: Dict[str, Any],
    blocks: Dict[int, Dict[str, Any]],
) -> List[Any]:
    weight_block = int(info["weight_block"])
    weight_offset, weight_length = block_range(blocks, weight_block)
    scale_block = info.get("raw_scale_block")
    scale2_block = info.get("raw_scale2_block")
    scale_offset, scale_length = block_range(blocks, scale_block)
    scale2_offset, scale2_length = block_range(blocks, scale2_block)
    weight_key = str(info.get("weight_key") or blocks[weight_block].get("key") or "")
    storage_mode = str(bundle.get("storage_mode") or info.get("storage_mode") or "fp4")
    source_file = str(info.get("source_file") or blocks[weight_block].get("source_file") or "")
    return [
        part,
        1000 + part,
        shard_file,
        source_file,
        weight_key,
        int(bundle["layer"]),
        int(bundle["expert"]),
        proj,
        int(info.get("rows", 0)),
        int(info.get("cols", 0)),
        int(info.get("groups", 0)),
        int(info.get("group_size", 0)),
        str(info.get("dtype") or blocks[weight_block].get("dtype") or ""),
        str(info.get("scale_mode") or ""),
        storage_mode,
        weight_offset,
        weight_length,
        weight_block,
        f"{weight_key}_scale" if scale_block is not None else "",
        "F8_E4M3" if scale_block is not None else "",
        scale_offset,
        scale_length,
        "" if scale_block is None else int(scale_block),
        f"{weight_key}_scale_2" if scale2_block is not None else "",
        scale2_offset,
        scale2_length,
    ]


def collect_rows_and_scale4(metas: Iterable[Dict[str, Any]]) -> Tuple[List[List[Any]], List[Dict[str, Any]]]:
    rows: List[List[Any]] = []
    scale4: List[Dict[str, Any]] = []
    seen = set()
    for meta in metas:
        part = part_id_from_meta(meta)
        shard_file = shard_file_for_runtime(str(meta["_filename"]))
        blocks = block_by_id(meta)
        bundles = list(meta.get("expert_bundles", [])) + list(meta.get("raw_expert_bundles", []))
        for bundle in bundles:
            projs = bundle.get("projections", {})
            for proj in PROJ_ORDER:
                info = projs.get(proj)
                if not info:
                    continue
                key = (int(bundle["layer"]), int(bundle["expert"]), proj)
                if key in seen:
                    raise ValueError(f"duplicate expert projection in footers: {key}")
                seen.add(key)
                rows.append(row_for_projection(part, shard_file, bundle, proj, info, blocks))
                if info.get("scale_mode") == "scale4":
                    scale4.append(
                        {
                            "url": meta["_url"],
                            "key": info.get("weight_key") or blocks[int(info["weight_block"])].get("key") or "",
                            "rows": int(info.get("rows", 0)),
                            "groups": int(info.get("groups", 0)),
                            "group_size": int(info.get("group_size", 0)),
                            "max_abs_error": float(info.get("scale4_repack_max_abs_error", 0.0)),
                            "codebook": blocks[int(info["scale4_codebook_block"])],
                            "index": blocks[int(info["scale4_index_block"])],
                        }
                    )
    rows.sort(key=lambda r: (int(r[5]), int(r[6]), str(r[7])))
    return rows, scale4


def write_tensors_csv(path: Path, rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def align(value: int, unit: int = 64) -> int:
    return (value + unit - 1) // unit * unit


def write_scale4(path: Path, entries: List[Dict[str, Any]], token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(entries, key=lambda item: str(item["key"]))
    entry_table_offset = SCALE4_HEADER.size
    string_table_offset = entry_table_offset + len(entries) * SCALE4_ENTRY.size
    key_offsets: List[Tuple[int, bytes]] = []
    string_cursor = string_table_offset
    for item in entries:
        key = str(item["key"]).encode("utf-8")
        key_offsets.append((string_cursor, key))
        string_cursor += len(key) + 1
    data_offset = align(string_cursor, 64)
    pending = []
    cursor = data_offset
    for item in entries:
        codebook_len = int(item["codebook"]["length"])
        index_len = int(item["index"]["length"])
        codebook_offset = cursor
        cursor += codebook_len
        index_offset = cursor
        cursor += index_len
        cursor = align(cursor, 16)
        pending.append((item, codebook_offset, index_offset, codebook_len, index_len))
    file_size = cursor
    session = requests.Session()
    with path.open("wb") as f:
        f.write(
            SCALE4_HEADER.pack(
                SCALE4_MAGIC,
                1,
                len(entries),
                entry_table_offset,
                string_table_offset,
                data_offset,
                file_size,
                0,
                0,
            )
        )
        for (item, codebook_offset, index_offset, _codebook_len, index_len), (key_offset, key) in zip(pending, key_offsets):
            f.write(
                SCALE4_ENTRY.pack(
                    key_offset,
                    len(key),
                    int(item["rows"]),
                    int(item["groups"]),
                    int(item["group_size"]),
                    4,
                    16,
                    1.0,
                    float(item["max_abs_error"]),
                    codebook_offset,
                    index_offset,
                    index_len,
                )
            )
        for _off, key in key_offsets:
            f.write(key)
            f.write(b"\x00")
        if f.tell() < data_offset:
            f.write(b"\x00" * (data_offset - f.tell()))
        for i, (item, codebook_offset, index_offset, codebook_len, index_len) in enumerate(pending, 1):
            if i == 1 or i % 250 == 0 or i == len(pending):
                print(f"scale4 {i}/{len(pending)} {item['key']}", flush=True)
            if f.tell() < codebook_offset:
                f.write(b"\x00" * (codebook_offset - f.tell()))
            cb = item["codebook"]
            ix = item["index"]
            stream_range_to_file(session, item["url"], int(cb["offset"]), codebook_len, token, f, str(item["key"]) + ":codebook")
            if f.tell() < index_offset:
                f.write(b"\x00" * (index_offset - f.tell()))
            stream_range_to_file(session, item["url"], int(ix["offset"]), index_len, token, f, str(item["key"]) + ":index")
            pad = align(f.tell(), 16) - f.tell()
            if pad:
                f.write(b"\x00" * pad)
        if f.tell() != file_size:
            raise RuntimeError(f"scale4 size mismatch expected={file_size} actual={f.tell()}")


def download_tokenizer(output_root: Path, tokenizer_ref: str, token: str) -> Optional[Path]:
    if not tokenizer_ref:
        tokenizer_ref = DEFAULT_TOKENIZER_REPO
    repo_id, revision, prefix = parse_hf_ref(tokenizer_ref, "main")
    filename = (prefix.rstrip("/") + "/tokenizer.json").lstrip("/") if prefix else "tokenizer.json"
    try:
        cached = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token or None)
    except Exception:
        if repo_id == DEFAULT_TOKENIZER_REPO:
            raise
        cached = hf_hub_download(repo_id=DEFAULT_TOKENIZER_REPO, filename="tokenizer.json", token=token or None)
    dst = output_root / "tokenizer.json"
    shutil.copyfile(cached, dst)
    return dst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--juju-hf", required=True, help="HF repo id or URL containing glm5.1-storage-part*.juju")
    parser.add_argument("--tokenizer-hf", default=DEFAULT_TOKENIZER_REPO, help="HF repo id or URL containing tokenizer.json")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prefix", default="", help="optional folder prefix inside the JUJU repo")
    parser.add_argument("--output-root", default="/content/GLM51_RUNTIME")
    parser.add_argument("--skip-scale4", action="store_true")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    repo_id, revision, inferred_prefix = parse_hf_ref(args.juju_hf, args.revision)
    prefix = args.prefix.strip("/") or inferred_prefix
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    token = args.token

    print(f"JUJU repo={repo_id} revision={revision} prefix={prefix!r}")
    files = list_juju_files(repo_id, revision, prefix, token)
    print(f"part files={len(files)}")
    session = requests.Session()
    metas = []
    for i, filename in enumerate(files, 1):
        print(f"footer {i}/{len(files)} {filename}", flush=True)
        url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
        metas.append(read_remote_footer(session, url, token, filename))
    rows, scale4_entries = collect_rows_and_scale4(metas)
    print(f"tensor rows={len(rows)} scale4 entries={len(scale4_entries)}")

    write_tensors_csv(output_root / "tensors.csv", rows)
    print(f"wrote {output_root / 'tensors.csv'}")
    if not args.skip_scale4:
        write_scale4(output_root / "glm5_scale4.gsc4", scale4_entries, token)
        print(f"wrote {output_root / 'glm5_scale4.gsc4'}")
    if not args.skip_tokenizer:
        tokenizer_path = download_tokenizer(output_root, args.tokenizer_hf, token)
        print(f"wrote {tokenizer_path}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
