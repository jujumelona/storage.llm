#!/usr/bin/env python3
"""
Build runtime sidecar files for the GLM5.1 StorageLLM server from .juju parts.

Inputs:
  - glm5.1-storage-partNN.juju files

Outputs:
  - tensors.csv
  - glm5_scale4.gsc4
  - tokenizer.json, downloaded or copied

This does not rebuild model weights. It only reads the compact JSON footer in
each .juju part and copies scale4 blocks that were already stored in those
parts.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BASE_URL = "https://huggingface.co/lukealonso/GLM-5.1-NVFP4/resolve/main"
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


def read_juju_footer(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        raw = f.read(JUJU_HEADER.size)
        if len(raw) != JUJU_HEADER.size:
            raise ValueError(f"{path}: too small for JUJU header")
        magic, version, _flags, _header_bytes, _data_offset, json_offset, json_bytes, file_size, _block_count = (
            JUJU_HEADER.unpack(raw)
        )
        if magic != JUJU_MAGIC or version != 1:
            raise ValueError(f"{path}: not a GLM5.1 Storage JUJU file")
        actual_size = path.stat().st_size
        if actual_size != file_size:
            raise ValueError(f"{path}: size mismatch header={file_size} actual={actual_size}")
        f.seek(json_offset)
        payload = f.read(json_bytes)
    meta = json.loads(payload.decode("utf-8"))
    meta["_path"] = path
    return meta


def part_id_from_name(path: Path, meta: Dict[str, Any]) -> int:
    if int(meta.get("part", 0)) > 0:
        return int(meta["part"])
    stem = path.stem
    digits = "".join(ch for ch in stem[-2:] if ch.isdigit())
    return int(digits) if digits else 0


def rel_part_path(path: Path, output_root: Path) -> str:
    try:
        return path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError:
        return f"parts/{path.name}"


def block_by_id(meta: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(item["id"]): item for item in meta.get("blocks", [])}


def block_range(blocks: Dict[int, Dict[str, Any]], block_id: Optional[int]) -> Tuple[int, int]:
    if block_id is None:
        return 0, 0
    block = blocks[int(block_id)]
    return int(block["offset"]), int(block["length"])


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
    scale_key = f"{weight_key}_scale" if scale_block is not None else ""
    scale2_key = f"{weight_key}_scale_2" if scale2_block is not None else ""
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
        scale_key,
        "F8_E4M3" if scale_block is not None else "",
        scale_offset,
        scale_length,
        "" if scale_block is None else int(scale_block),
        scale2_key,
        scale2_offset,
        scale2_length,
    ]


def iter_bundle_rows(parts: Iterable[Dict[str, Any]], output_root: Path) -> Tuple[List[List[Any]], List[Dict[str, Any]]]:
    rows: List[List[Any]] = []
    scale4_entries: List[Dict[str, Any]] = []
    seen = set()
    for meta in parts:
        path = Path(meta["_path"])
        part = part_id_from_name(path, meta)
        shard_file = rel_part_path(path, output_root)
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
                    raise ValueError(f"duplicate expert projection in JUJU footers: {key}")
                seen.add(key)
                rows.append(row_for_projection(part, shard_file, bundle, proj, info, blocks))
                if info.get("scale_mode") == "scale4":
                    scale4_entries.append(
                        {
                            "part_path": path,
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
    return rows, scale4_entries


def write_tensors_csv(path: Path, rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def align(value: int, unit: int = 64) -> int:
    return (value + unit - 1) // unit * unit


def read_block_bytes(part_path: Path, block: Dict[str, Any]) -> bytes:
    with part_path.open("rb") as f:
        f.seek(int(block["offset"]))
        data = f.read(int(block["length"]))
    if len(data) != int(block["length"]):
        raise ValueError(f"{part_path}: short read for block {block.get('id')}")
    return data


def write_scale4(path: Path, entries: List[Dict[str, Any]]) -> None:
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

    pending: List[Tuple[Dict[str, Any], int, int, bytes, bytes]] = []
    data_cursor = data_offset
    for item in entries:
        codebook = read_block_bytes(Path(item["part_path"]), item["codebook"])
        index = read_block_bytes(Path(item["part_path"]), item["index"])
        codebook_offset = data_cursor
        data_cursor += len(codebook)
        index_offset = data_cursor
        data_cursor += len(index)
        data_cursor = align(data_cursor, 16)
        pending.append((item, codebook_offset, index_offset, codebook, index))

    file_size = data_cursor
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
        for (item, codebook_offset, index_offset, _codebook, index), (key_offset, key) in zip(pending, key_offsets):
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
                    len(index),
                )
            )
        for _key_offset, key in key_offsets:
            f.write(key)
            f.write(b"\x00")
        if f.tell() < data_offset:
            f.write(b"\x00" * (data_offset - f.tell()))
        for _item, codebook_offset, _index_offset, codebook, index in pending:
            if f.tell() < codebook_offset:
                f.write(b"\x00" * (codebook_offset - f.tell()))
            f.write(codebook)
            f.write(index)
            pad = align(f.tell(), 16) - f.tell()
            if pad:
                f.write(b"\x00" * pad)
        if f.tell() != file_size:
            raise RuntimeError(f"scale4 writer size mismatch expected={file_size} actual={f.tell()}")


def copy_or_download_tokenizer(output_root: Path, tokenizer: str, base_url: str, skip: bool) -> Optional[Path]:
    if skip:
        return None
    dst = output_root / "tokenizer.json"
    if tokenizer:
        src = Path(tokenizer)
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copyfile(src, dst)
        return dst
    url = base_url.rstrip("/") + "/tokenizer.json"
    urllib.request.urlretrieve(url, dst)
    return dst


def find_parts(parts_dir: Path) -> List[Path]:
    direct = sorted(parts_dir.glob("glm5.1-storage-part*.juju"))
    nested = sorted((parts_dir / "parts").glob("glm5.1-storage-part*.juju")) if (parts_dir / "parts").exists() else []
    parts = direct or nested
    if not parts:
        raise FileNotFoundError(f"no glm5.1-storage-part*.juju under {parts_dir}")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", default="/content/GLM51_STORAGE/parts")
    parser.add_argument("--output-root", default="/content/GLM51_STORAGE/runtime")
    parser.add_argument("--tokenizer", default="", help="local tokenizer.json to copy")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    parts_dir = Path(args.parts_dir)
    output_root = Path(args.output_root)
    part_paths = find_parts(parts_dir)
    metas = [read_juju_footer(path) for path in part_paths]
    rows, scale4_entries = iter_bundle_rows(metas, output_root)
    print(f"parts={len(part_paths)} tensor_rows={len(rows)} scale4_entries={len(scale4_entries)}")
    if args.dry_run:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    write_tensors_csv(output_root / "tensors.csv", rows)
    write_scale4(output_root / "glm5_scale4.gsc4", scale4_entries)
    tok = copy_or_download_tokenizer(output_root, args.tokenizer, args.base_url, args.skip_tokenizer)
    print(f"wrote {output_root / 'tensors.csv'}")
    print(f"wrote {output_root / 'glm5_scale4.gsc4'}")
    if tok:
        print(f"wrote {tok}")
    else:
        print("tokenizer skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
