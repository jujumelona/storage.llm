#!/usr/bin/env python3
import csv
import os
import struct
import sys


MAGIC = b"SLTIDX2\0"
VERSION = 2
PROJ = {
    "gate": 0,
    "gate_proj": 0,
    "up": 1,
    "up_proj": 1,
    "down": 2,
    "down_proj": 2,
}


def u32(row, name):
    value = row.get(name, "")
    return int(value) if value else 0


def u64(row, name):
    value = row.get(name, "")
    return int(value) if value else 0


def main(argv):
    if len(argv) not in (2, 3):
        print("usage: build_tensor_index.py <tensors.csv> [tensor_index.bin]", file=sys.stderr)
        return 2
    csv_path = argv[1]
    out_path = argv[2] if len(argv) == 3 else os.path.join(os.path.dirname(csv_path), "tensor_index.bin")

    paths = []
    path_index = {}
    records = []
    keys = []
    key_index = {}
    required = [
        "part", "shard", "shard_file", "layer", "expert", "proj",
        "rows", "cols", "groups", "group_size",
        "weight_byte_offset", "weight_byte_length",
        "scale_byte_offset", "scale_byte_length",
        "scale2_byte_offset", "scale2_byte_length",
    ]

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit("missing columns: " + ", ".join(missing))
        for row in reader:
            shard_file = row.get("shard_file") or row.get("source_file") or ""
            if not shard_file:
                raise SystemExit("empty shard_file")
            idx = path_index.get(shard_file)
            if idx is None:
                if len(paths) >= 65535:
                    raise SystemExit("too many tensor paths")
                idx = len(paths)
                path_index[shard_file] = idx
                paths.append(shard_file)
            proj = PROJ.get((row.get("proj") or "").strip())
            if proj is None:
                raise SystemExit("unknown proj: " + str(row.get("proj")))
            weight_key = (row.get("weight_key") or "").strip()
            if weight_key:
                kidx = key_index.get(weight_key)
                if kidx is None:
                    kidx = len(keys)
                    key_index[weight_key] = kidx
                    keys.append(weight_key)
                flags = 1
            else:
                kidx = 0xFFFFFFFF
                flags = 0
            # BUGFIX 968: Validate scale fields — disambiguate "no scale" from offset=0 ★★
            # Problem: u64() returns 0 for empty string. offset=0 and length=0 is
            # indistinguishable from "scale at file offset 0". FP4 dequant uses
            # offset 0 → reads garbage → wrong output.
            # Solution: Sentinel 0xFFFFFFFFFFFFFFFF means "no scale". Warn on
            # weight with bytes > 0 but scale_byte_length == 0.
            SENTINEL = 0xFFFFFFFFFFFFFFFF
            w_off = u64(row, "weight_byte_offset")
            w_len = u64(row, "weight_byte_length")
            s_off = u64(row, "scale_byte_offset")
            s_len = u64(row, "scale_byte_length")
            s2_off = u64(row, "scale2_byte_offset")
            s2_len = u64(row, "scale2_byte_length")
            # Apply sentinel for missing scale/scale2
            if s_len == 0:
                if w_len > 0:
                    print(
                        f"WARNING: layer={row.get('layer')} expert={row.get('expert')} "
                        f"proj={row.get('proj')} has weight_byte_length={w_len} but "
                        f"scale_byte_length=0 — using sentinel (no scale)",
                        file=sys.stderr,
                    )
                s_off = SENTINEL
                s_len = SENTINEL
            if s2_len == 0:
                s2_off = SENTINEL
                s2_len = SENTINEL
            records.append((
                w_off,
                w_len,
                s_off,
                s_len,
                s2_off,
                s2_len,
                u32(row, "part"),
                u32(row, "shard"),
                u32(row, "layer"),
                u32(row, "expert"),
                proj,
                u32(row, "rows"),
                u32(row, "cols"),
                u32(row, "groups"),
                u32(row, "group_size"),
                idx,
                flags,
                kidx,
            ))

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<IIII", VERSION, len(paths), len(records), len(keys)))
        for path in paths:
            data = path.encode("utf-8")
            if not data or len(data) > 65535:
                raise SystemExit("bad path length: " + path)
            out.write(struct.pack("<H", len(data)))
            out.write(data)
        for key in keys:
            data = key.encode("utf-8")
            if not data or len(data) > 65535:
                raise SystemExit("bad key length")
            out.write(struct.pack("<H", len(data)))
            out.write(data)
        for rec in records:
            out.write(struct.pack("<QQQQQQIIIIIIIIIHHI", *rec))
    os.replace(tmp_path, out_path)
    print(f"wrote {out_path} tensors={len(records)} paths={len(paths)} scale4_keys={len(keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
