"""Known MoE naming adapters for JUJU metadata generation.

The C++ reader is metadata-first; these adapters are only used by the writer or
by tests when a source file has legacy names and no explicit metadata yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MoENameInfo:
    family: str
    layer: Optional[int]
    expert: Optional[int]
    projection: str
    routed: bool


KNOWN_MOE_FAMILIES = (
    "mixtral", "qwen", "deepseek", "glm", "kimi", "gemma", "dbrx",
    "arctic", "grok", "phi", "olmoe", "minimax", "generic",
)

_LAYER_PATTERNS = (
    re.compile(r"(?:^|[.])blk\.(\d+)\."),
    re.compile(r"(?:^|[.])block\.(\d+)\."),
    re.compile(r"(?:^|[.])blocks\.(\d+)\."),
    re.compile(r"(?:^|[.])model\.layers\.(\d+)\."),
    re.compile(r"(?:^|[.])decoder\.layers\.(\d+)\."),
    re.compile(r"(?:^|[.])layers\.(\d+)\."),
    re.compile(r"(?:^|[.])transformer\.h\.(\d+)\."),
    re.compile(r"(?:^|[.])h\.(\d+)\."),
)

_EXPERT_PATTERNS = (
    re.compile(r"(?:^|[._])experts?[._](\d+)(?:[._]|$)"),
    re.compile(r"(?:^|[._])exps[._](\d+)(?:[._]|$)"),
    re.compile(r"(?:^|[._])routed_experts?[._](\d+)(?:[._]|$)"),
    re.compile(r"(?:^|[._])moe_experts?[._](\d+)(?:[._]|$)"),
    re.compile(r"(?:^|[._])expert_(\d+)(?:[._]|$)"),
    re.compile(r"(?:^|[._])e(\d+)(?:[._]|$)"),
)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


def layer_from_name(name: str) -> Optional[int]:
    text = str(name or "")
    for pat in _LAYER_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    return None


def expert_from_name(name: str) -> Optional[int]:
    text = str(name or "").lower()
    for pat in _EXPERT_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    return None


def projection_from_name(name: str) -> str:
    text = str(name or "").lower()
    n = _norm(text)
    if "ffn_gate_up_exps" in text or "gate_up" in n or "w1_w3" in n or "w13" in n:
        return "gate_up"
    if "gate_proj" in text or "ffn_gate" in text or re.search(r"(?:^|_)w1(?:_|$)", n) or re.search(r"(?:^|_)wi_0(?:_|$)", n):
        return "gate"
    if "up_proj" in text or "ffn_up" in text or re.search(r"(?:^|_)w3(?:_|$)", n) or re.search(r"(?:^|_)wi_1(?:_|$)", n):
        return "up"
    if "down_proj" in text or "ffn_down" in text or re.search(r"(?:^|_)w2(?:_|$)", n) or re.search(r"(?:^|_)wo(?:_|$)", n):
        return "down"
    return "expert"


def is_routed_expert_name(name: str) -> bool:
    n = _norm(name)
    if "shared_expert" in n or "shared_experts" in n:
        return False
    markers = (
        "exps", "experts", "block_sparse_moe", "sparse_moe", "moe_experts",
        "routed_experts", "feed_forward_experts", "ffn_experts", "mlp_experts",
    )
    return any(m in n for m in markers)


def infer_family(model_id: str, model_type: str = "") -> str:
    text = f"{model_id} {model_type}".lower()
    for fam in KNOWN_MOE_FAMILIES:
        if fam != "generic" and fam in text:
            return fam
    if "deepseek" in text or "kimi" in text:
        return "deepseek" if "deepseek" in text else "kimi"
    return "generic"


def inspect_name(name: str, model_id: str = "", model_type: str = "") -> MoENameInfo:
    return MoENameInfo(
        family=infer_family(model_id, model_type),
        layer=layer_from_name(name),
        expert=expert_from_name(name),
        projection=projection_from_name(name),
        routed=is_routed_expert_name(name),
    )
