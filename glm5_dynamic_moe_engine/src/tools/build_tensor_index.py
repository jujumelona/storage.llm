#!/usr/bin/env python3
import csv
import os
import struct
import sys


MAGIC = b"SLTIDX1\0"
VERSION = 1
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
            records.append((
                u32(row, "part"),
                u32(row, "shard"),
                u32(row, "layer"),
                u32(row, "expert"),
                proj,
                u32(row, "rows"),
                u32(row, "cols"),
                u32(row, "groups"),
                u32(row, "group_size"),
                u64(row, "weight_byte_offset"),
                u64(row, "weight_byte_length"),
                u64(row, "scale_byte_offset"),
                u64(row, "scale_byte_length"),
                u64(row, "scale2_byte_offset"),
                u64(row, "scale2_byte_length"),
                idx,
                0,
                0,
            ))

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<IIII", VERSION, len(paths), len(records), 0))
        for path in paths:
            data = path.encode("utf-8")
            if not data or len(data) > 65535:
                raise SystemExit("bad path length: " + path)
            out.write(struct.pack("<H", len(data)))
            out.write(data)
        for rec in records:
            out.write(struct.pack("<IIIIIIIIIQQQQQQHHI", *rec))
    os.replace(tmp_path, out_path)
    print(f"wrote {out_path} tensors={len(records)} paths={len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
