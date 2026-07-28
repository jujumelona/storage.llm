import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "repack_juju_mxfp4_8x8.py"
SPEC = importlib.util.spec_from_file_location("repack_juju_mxfp4_8x8", TOOL)
REPACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPACK)


def test_repack_mxfp4_8x8_preserves_size_and_interleaves_rows():
    rows = 8
    cols = 64
    blocks = cols // 32
    source = bytes((i * 37 + 11) & 0xFF for i in range(rows * blocks * 17))
    tiled = REPACK.repack_mxfp4_8x8(source, rows, cols)
    assert len(tiled) == len(source)
    for block in range(blocks):
        tile = tiled[block * 136:(block + 1) * 136]
        for row in range(rows):
            original = source[(row * blocks + block) * 17:(row * blocks + block + 1) * 17]
            assert tile[row] == original[0]
            assert tile[8 + row * 8:8 + (row + 1) * 8] == original[1:9]
            assert tile[72 + row * 8:72 + (row + 1) * 8] == original[9:17]


def test_repack_q5_1_8x8_preserves_size_and_interleaves_blocks():
    rows = 8
    cols = 64
    blocks = cols // 32
    source = bytes((i * 29 + 5) & 0xFF for i in range(rows * blocks * 24))
    tiled = REPACK.repack_q5_1_8x8(source, rows, cols)
    assert len(tiled) == len(source)
    for block in range(blocks):
        tile = tiled[block * 192:(block + 1) * 192]
        for row in range(rows):
            original = source[(row * blocks + block) * 24:(row * blocks + block + 1) * 24]
            assert tile[row * 24:(row + 1) * 24] == original


def test_repack_q8_0_8x8_preserves_size_and_interleaves_blocks():
    rows = 8
    cols = 64
    blocks = cols // 32
    source = bytes((i * 23 + 17) & 0xFF for i in range(rows * blocks * 34))
    tiled = REPACK.repack_q8_0_8x8(source, rows, cols)
    assert len(tiled) == len(source)
    for block in range(blocks):
        tile = tiled[block * 272:(block + 1) * 272]
        for row in range(rows):
            original = source[(row * blocks + block) * 34:(row * blocks + block + 1) * 34]
            assert tile[row * 34:(row + 1) * 34] == original


def test_repack_file_updates_only_expert_mxfp4_records(tmp_path):
    rows = 8
    cols = 32
    source = bytes((i * 13 + 7) & 0xFF for i in range(rows * 17))
    weight = tmp_path / "x.juju"
    index = tmp_path / "x.juju.idx"
    weight.write_bytes(source)
    index.write_text(json.dumps({
        "tensors": [{
            "name": "blk.0.ffn_down_exps.weight",
            "gguf_type": 39,
            "juju_offset": 0,
            "juju_bytes": len(source),
            "logical_rows": rows,
            "logical_cols": cols,
            "storage_layout": "source_gguf_quant_block_layout_preserved",
            "expert_layout": {"projection": "down"},
        }],
    }), encoding="utf-8")
    count, size = REPACK.repack_file(weight, index)
    record = json.loads(index.read_text(encoding="utf-8"))["tensors"][0]
    assert count == 1
    assert size == len(source)
    assert len(weight.read_bytes()) == len(source)
    assert record["storage_layout"] == REPACK.TILED_LAYOUT
    assert record["kernel_contract"]["runtime_repack_forbidden"] is True


def test_repack_file_updates_expert_q5_1_records(tmp_path):
    rows = 8
    cols = 32
    source = bytes((i * 19 + 3) & 0xFF for i in range(rows * 24))
    weight = tmp_path / "x.juju"
    index = tmp_path / "x.juju.idx"
    weight.write_bytes(source)
    index.write_text(json.dumps({
        "tensors": [{
            "name": "blk.0.ffn_down_exps.weight",
            "gguf_type": REPACK.Q5_1_GGUF_TYPE,
            "juju_offset": 0,
            "juju_bytes": len(source),
            "logical_rows": rows,
            "logical_cols": cols,
            "storage_layout": "source_gguf_quant_block_layout_preserved",
            "expert_layout": {"projection": "down"},
        }],
    }), encoding="utf-8")
    count, size = REPACK.repack_file(weight, index)
    record = json.loads(index.read_text(encoding="utf-8"))["tensors"][0]
    assert count == 1
    assert size == len(source)
    assert len(weight.read_bytes()) == len(source)
    assert record["storage_layout"] == REPACK.Q5_1_TILED_LAYOUT
    assert record["kernel_contract"]["runtime_repack_forbidden"] is True


def test_repack_file_updates_expert_q8_0_records(tmp_path):
    rows = 8
    cols = 32
    source = bytes((i * 31 + 9) & 0xFF for i in range(rows * 34))
    weight = tmp_path / "x.juju"
    index = tmp_path / "x.juju.idx"
    weight.write_bytes(source)
    index.write_text(json.dumps({
        "tensors": [{
            "name": "blk.0.ffn_down_exps.weight",
            "gguf_type": REPACK.Q8_0_GGUF_TYPE,
            "juju_offset": 0,
            "juju_bytes": len(source),
            "logical_rows": rows,
            "logical_cols": cols,
            "storage_layout": "source_gguf_quant_block_layout_preserved",
            "expert_layout": {"projection": "down"},
        }],
    }), encoding="utf-8")
    count, size = REPACK.repack_file(weight, index)
    record = json.loads(index.read_text(encoding="utf-8"))["tensors"][0]
    assert count == 1
    assert size == len(source)
    assert len(weight.read_bytes()) == len(source)
    assert record["storage_layout"] == REPACK.Q8_0_TILED_LAYOUT
    assert record["kernel_contract"]["runtime_repack_forbidden"] is True

