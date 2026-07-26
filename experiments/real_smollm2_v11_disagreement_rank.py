from __future__ import annotations

import csv
import gc
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "conjugacy_v10", HERE / "real_smollm2_v10_conjugacy_certificate.py"
)
v10 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v10)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260726)
np.random.seed(20260726)

ROOT = Path("out/real_smollm2_v11_disagreement_rank")
ROOT.mkdir(parents=True, exist_ok=True)

BASE_ID = v10.BASE_ID
MATH_ID = v10.MATH_ID
SQL_ID = v10.SQL_ID
PARAMETER_RATIO_LIMIT = 1.8
RANK_TOL_RELATIVE = 1e-10


def llama_parameter_count(config, hidden_size: int) -> int:
    """Dense Llama-style count with the original head and MLP ratios.

    Hidden size is restricted to multiples of num_attention_heads. GQA ratio and
    intermediate/hidden ratio are preserved. Biases are absent for this lineage.
    """
    n_layers = int(config.num_hidden_layers)
    n_heads = int(config.num_attention_heads)
    n_kv = int(config.num_key_value_heads)
    vocab = int(config.vocab_size)
    if hidden_size % n_heads != 0:
        raise ValueError("hidden size must be divisible by attention heads")
    head_dim = hidden_size // n_heads
    kv_width = n_kv * head_dim
    intermediate = int(round(float(config.intermediate_size) * hidden_size / config.hidden_size))

    embedding = vocab * hidden_size
    attention = hidden_size * hidden_size + 2 * kv_width * hidden_size + hidden_size * hidden_size
    mlp = 3 * intermediate * hidden_size
    norms = 2 * hidden_size
    final_norm = hidden_size
    return embedding + n_layers * (attention + mlp + norms) + final_norm


def maximum_hidden_under_budget(config) -> dict:
    parent_hidden = int(config.hidden_size)
    parent_count = llama_parameter_count(config, parent_hidden)
    best_hidden = parent_hidden
    best_ratio = 1.0
    for hidden in range(parent_hidden, parent_hidden * 3):
        if hidden % int(config.num_attention_heads) != 0:
            continue
        ratio = llama_parameter_count(config, hidden) / parent_count
        if ratio <= PARAMETER_RATIO_LIMIT:
            best_hidden = hidden
            best_ratio = ratio
        else:
            break
    return {
        "parent_hidden": parent_hidden,
        "maximum_hidden": best_hidden,
        "maximum_disagreement_rank": best_hidden - parent_hidden,
        "parameter_ratio": best_ratio,
        "parent_parameter_formula": parent_count,
        "maximum_parameter_formula": llama_parameter_count(config, best_hidden),
    }


def effective_mlp_rows(layer, config, attention_jacobian: torch.Tensor):
    eps = float(config.rms_norm_eps)
    gamma = layer.post_attention_layernorm.weight.detach().cpu().double()
    pre = (gamma[:, None] * attention_jacobian) / math.sqrt(eps)
    gate = layer.mlp.gate_proj.weight.detach().cpu().double() @ pre
    up = layer.mlp.up_proj.weight.detach().cpu().double() @ pre
    down = layer.mlp.down_proj.weight.detach().cpu().double()
    return gate, up, down


def quadratic_map_on_directions(layer, config, attention_jacobian: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Exact second-order SwiGLU jet evaluated on deterministic directions.

    SiLU(t) = 0.5 t + O(t^2), hence the quadratic term is
      q(x) = 0.5 W_down[(Gx) odot (Ux)].
    Columns of x are independent directions.
    """
    gate, up, down = effective_mlp_rows(layer, config, attention_jacobian)
    features = (gate @ x) * (up @ x)
    return 0.5 * (down @ features)


def numerical_rank(singular_values: torch.Tensor) -> int:
    if singular_values.numel() == 0:
        return 0
    threshold = float(singular_values.max()) * RANK_TOL_RELATIVE
    return int((singular_values > threshold).sum())


def rank_for_energy(singular_values: torch.Tensor, fraction: float) -> int:
    energy = singular_values.square()
    total = float(energy.sum())
    if total <= 0.0:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    return int(torch.searchsorted(cumulative, torch.tensor(fraction, dtype=cumulative.dtype)).item() + 1)


def energy_at_rank(singular_values: torch.Tensor, rank: int) -> float:
    energy = singular_values.square()
    total = float(energy.sum())
    if total <= 0.0:
        return 1.0
    return float(energy[: min(rank, len(energy))].sum() / total)


def deterministic_direction_bank(d: int) -> torch.Tensor:
    identity = torch.eye(d, dtype=torch.float64)
    shifted = (identity + torch.roll(identity, shifts=1, dims=0)) / math.sqrt(2.0)
    alternating = (identity - torch.roll(identity, shifts=1, dims=0)) / math.sqrt(2.0)
    return torch.cat([identity, shifted, alternating], dim=1)


def layer_certificate(math_layer, sql_layer, config, rank_budget: int, layer_index: int):
    d = int(config.hidden_size)
    directions = deterministic_direction_bank(d)

    jm = v10.zero_jacobian(math_layer, config)
    js = v10.zero_jacobian(sql_layer, config)
    delta_j = 0.5 * (jm - js)

    qm = quadratic_map_on_directions(math_layer, config, jm, directions)
    qs = quadratic_map_on_directions(sql_layer, config, js, directions)
    delta_q = 0.5 * (qm - qs)

    linear_s = torch.linalg.svdvals(delta_j).sort(descending=True).values
    quadratic_s = torch.linalg.svdvals(delta_q).sort(descending=True).values
    witness = torch.cat([delta_j, delta_q], dim=1)
    witness_s = torch.linalg.svdvals(witness).sort(descending=True).values

    witness_norm = float(torch.linalg.vector_norm(witness))
    normalized_covariance = witness @ witness.T / max(witness_norm * witness_norm, 1e-300)

    row = {
        "layer": layer_index,
        "linear_disagreement_rank": numerical_rank(linear_s),
        "quadratic_disagreement_rank": numerical_rank(quadratic_s),
        "joint_witness_rank": numerical_rank(witness_s),
        "rank90": rank_for_energy(witness_s, 0.90),
        "rank95": rank_for_energy(witness_s, 0.95),
        "rank99": rank_for_energy(witness_s, 0.99),
        "energy_at_budget": energy_at_rank(witness_s, rank_budget),
        "budget_exact_rank_pass": numerical_rank(witness_s) <= rank_budget,
        "budget_99pct_pass": energy_at_rank(witness_s, rank_budget) >= 0.99,
        "largest_singular_value": float(witness_s.max()),
        "smallest_retained_singular_value": float(witness_s[min(rank_budget, d) - 1]),
        "smallest_singular_value": float(witness_s.min()),
    }
    return row, normalized_covariance


def main() -> None:
    ids = [BASE_ID, MATH_ID, SQL_ID]
    configs = [AutoConfig.from_pretrained(model_id) for model_id in ids]
    signatures = [v10.config_signature(config) for config in configs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError({"status": "CONFIG_MISMATCH", "signatures": signatures})

    tokenizer_audit = v10.tokenizer_audit(ids)
    if not all(tokenizer_audit["vocab_equal"]) or not all(
        all(row) for row in tokenizer_audit["probe_ids_equal"]
    ):
        raise RuntimeError({"status": "TOKENIZER_MISMATCH", "audit": tokenizer_audit})

    print("Loading full public checkpoints")
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    math_model = AutoModelForCausalLM.from_pretrained(MATH_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    sql_model = AutoModelForCausalLM.from_pretrained(SQL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    models = [base, math_model, sql_model]
    if not all(v10.finite_state(model) for model in models):
        raise RuntimeError("nonfinite parent parameters")
    shape_audit = v10.assert_same_shapes(models)

    budget = maximum_hidden_under_budget(configs[0])
    rank_budget = int(budget["maximum_disagreement_rank"])
    d = int(configs[0].hidden_size)

    layer_rows = []
    global_covariance = torch.zeros((d, d), dtype=torch.float64)
    for layer_index, (math_layer, sql_layer) in enumerate(
        zip(math_model.model.layers, sql_model.model.layers)
    ):
        row, covariance = layer_certificate(
            math_layer, sql_layer, configs[0], rank_budget, layer_index
        )
        layer_rows.append(row)
        global_covariance += covariance
        print(json.dumps(row))

    global_eigenvalues = torch.linalg.eigvalsh(global_covariance).clamp_min(0.0).sort(descending=True).values
    global_s = torch.sqrt(global_eigenvalues)
    global_rank = numerical_rank(global_s)
    global_energy_budget = energy_at_rank(global_s, rank_budget)
    global_rank90 = rank_for_energy(global_s, 0.90)
    global_rank95 = rank_for_energy(global_s, 0.95)
    global_rank99 = rank_for_energy(global_s, 0.99)

    exact_rejected = bool(
        global_rank > rank_budget
        or any(not row["budget_exact_rank_pass"] for row in layer_rows)
    )
    approximate_99_rejected = bool(
        global_energy_budget < 0.99
        or any(not row["budget_99pct_pass"] for row in layer_rows)
    )

    exact_hidden_required = int(configs[0].hidden_size) + global_rank
    exact_ratio = llama_parameter_count(configs[0], exact_hidden_required) / llama_parameter_count(
        configs[0], int(configs[0].hidden_size)
    ) if exact_hidden_required % int(configs[0].num_attention_heads) == 0 else None

    result = {
        "status": (
            "EXACT_AND_99PCT_LOW_RANK_DISAGREEMENT_REJECTED"
            if exact_rejected and approximate_99_rejected
            else "LOW_RANK_DISAGREEMENT_NOT_FULLY_REJECTED"
        ),
        "models": {"base": BASE_ID, "math": MATH_ID, "sql": SQL_ID},
        "config_signature": signatures[0],
        "tokenizer_audit": tokenizer_audit,
        "state_shape_audit": shape_audit,
        "parameter_budget": budget,
        "mathematical_definition": {
            "mean_state": "m=(h_math+h_sql)/2",
            "disagreement_state": "d=(h_math-h_sql)/2",
            "linearized_dynamics": "m'=Jbar m + DeltaJ d; d'=DeltaJ m + Jbar d",
            "quadratic_term": "exact second-order SwiGLU jet at h=0 evaluated on deterministic canonical directions",
            "witness_bank": "identity directions plus normalized adjacent sums and differences",
            "exact_scope": "fixed identity coordinates and the standard dense Llama/RMSNorm/SwiGLU graph",
            "not_claimed": "not a theorem over arbitrary nonlinear encoders, learned transports, or non-Llama architectures",
        },
        "global_certificate": {
            "joint_witness_rank": global_rank,
            "rank90": global_rank90,
            "rank95": global_rank95,
            "rank99": global_rank99,
            "energy_at_1p8_rank_budget": global_energy_budget,
            "exact_hidden_required": exact_hidden_required,
            "exact_dense_parameter_ratio_if_head_divisible": exact_ratio,
            "exact_low_rank_rejected": exact_rejected,
            "approximate_99pct_rejected": approximate_99_rejected,
        },
        "layer_summary": {
            "layers": len(layer_rows),
            "minimum_joint_rank": min(row["joint_witness_rank"] for row in layer_rows),
            "median_joint_rank": float(np.median([row["joint_witness_rank"] for row in layer_rows])),
            "maximum_joint_rank": max(row["joint_witness_rank"] for row in layer_rows),
            "minimum_energy_at_budget": min(row["energy_at_budget"] for row in layer_rows),
            "median_energy_at_budget": float(np.median([row["energy_at_budget"] for row in layer_rows])),
            "layers_exactly_exceeding_budget": sum(not row["budget_exact_rank_pass"] for row in layer_rows),
            "layers_below_99pct_at_budget": sum(not row["budget_99pct_pass"] for row in layer_rows),
        },
        "layer_rows": layer_rows,
    }

    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (ROOT / "LAYER_DISAGREEMENT_RANK.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)

    report = [
        "# SmolLM2 Math/SQL Low-Rank Disagreement-State Certificate",
        "",
        f"Status: **{result['status']}**",
        f"1.8x maximum dense hidden size: **{budget['maximum_hidden']}**",
        f"Allowed disagreement rank: **{rank_budget}**",
        f"Global witness rank: **{global_rank}**",
        f"Global 99% energy rank: **{global_rank99}**",
        f"Energy retained at rank {rank_budget}: **{global_energy_budget:.8f}**",
        f"Layers whose exact rank exceeds budget: **{result['layer_summary']['layers_exactly_exceeding_budget']}/{len(layer_rows)}**",
        f"Layers below 99% energy at budget: **{result['layer_summary']['layers_below_99pct_at_budget']}/{len(layer_rows)}**",
        "",
        "The certificate uses checkpoint tensors only. No prompts, activations, logits, gradients, labels, or external data are used.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    del base, math_model, sql_model
    gc.collect()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
