from __future__ import annotations

import csv
import gc
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260726)
np.random.seed(20260726)

ROOT = Path("out/real_smollm2_v10_conjugacy_certificate")
ROOT.mkdir(parents=True, exist_ok=True)

BASE_ID = "HuggingFaceTB/SmolLM2-135M"
MATH_ID = "Ashed00/SmolMath-135M"
SQL_ID = "Ellight/code-smolLM2-135m-text-to-sql"
EXACT_TOL = 1e-7


def finite_state(model: torch.nn.Module) -> bool:
    with torch.inference_mode():
        return all(bool(torch.isfinite(p.detach()).all().item()) for p in model.parameters())


def config_signature(config) -> dict:
    fields = [
        "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "vocab_size", "hidden_act",
        "rope_theta", "rms_norm_eps", "tie_word_embeddings", "attention_bias",
        "mlp_bias",
    ]
    return {field: getattr(config, field, None) for field in fields}


def tokenizer_audit(ids: list[str]) -> dict:
    tokenizers = [AutoTokenizer.from_pretrained(model_id) for model_id in ids]
    prompts = [
        "The capital of France is", "Question: 17 + 29 =", "SELECT name FROM users",
        "def fibonacci(n):", "Water freezes at 0 degrees Celsius.",
    ]
    base_vocab = tokenizers[0].get_vocab()
    result = {
        "vocab_equal": [tok.get_vocab() == base_vocab for tok in tokenizers],
        "probe_ids_equal": [],
    }
    for tok in tokenizers:
        result["probe_ids_equal"].append([
            tok(text, add_special_tokens=False)["input_ids"]
            == tokenizers[0](text, add_special_tokens=False)["input_ids"]
            for text in prompts
        ])
    return result


def assert_same_shapes(models: list[torch.nn.Module]) -> dict:
    states = [model.state_dict() for model in models]
    keys = list(states[0].keys())
    if any(list(state.keys()) != keys for state in states[1:]):
        raise RuntimeError("state key mismatch")
    mismatches = []
    for key in keys:
        shapes = [tuple(state[key].shape) for state in states]
        if any(shape != shapes[0] for shape in shapes[1:]):
            mismatches.append({"key": key, "shapes": shapes})
    if mismatches:
        raise RuntimeError({"shape_mismatches": mismatches[:50]})
    return {"key_count": len(keys), "all_equal": True}


def kv_repeated_v(layer, config) -> torch.Tensor:
    wv = layer.self_attn.v_proj.weight.detach().cpu().double()
    n_heads = int(config.num_attention_heads)
    n_kv = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // n_heads))
    if n_heads % n_kv != 0:
        raise RuntimeError("num_attention_heads must be divisible by num_key_value_heads")
    return (
        wv.reshape(n_kv, head_dim, config.hidden_size)
        .repeat_interleave(n_heads // n_kv, dim=0)
        .reshape(n_heads * head_dim, config.hidden_size)
    )


def zero_jacobian(layer, config) -> torch.Tensor:
    """Exact single-token Jacobian of one decoder layer at h=0.

    With one token, attention probability is exactly one. The SwiGLU branch has
    zero first derivative at the origin, so J = I + W_o repeat(W_v) D_gamma / sqrt(eps).
    """
    d = int(config.hidden_size)
    eps = float(config.rms_norm_eps)
    gamma = layer.input_layernorm.weight.detach().cpu().double()
    wo = layer.self_attn.o_proj.weight.detach().cpu().double()
    wv_rep = kv_repeated_v(layer, config)
    normalized_input = gamma[:, None] * torch.eye(d, dtype=torch.float64)
    attention_linear = wo @ wv_rep @ normalized_input / math.sqrt(eps)
    return torch.eye(d, dtype=torch.float64) + attention_linear


def mlp_atom_invariants(layer, config, j_attention_residual: torch.Tensor) -> torch.Tensor:
    """Per-channel Frobenius norms of the exact quadratic SwiGLU atoms at zero.

    For the full decoder layer, x1 = J_attention_residual x at first order.
    The quadratic MLP coefficient is
      0.5 * d_k \otimes (a_k \otimes b_k + b_k \otimes a_k).
    Its Frobenius norm is invariant under orthogonal input/output state changes,
    reciprocal channel scaling, and channel permutation.
    """
    eps = float(config.rms_norm_eps)
    gamma = layer.post_attention_layernorm.weight.detach().cpu().double()
    pre = (gamma[:, None] * j_attention_residual) / math.sqrt(eps)
    gate = layer.mlp.gate_proj.weight.detach().cpu().double() @ pre
    up = layer.mlp.up_proj.weight.detach().cpu().double() @ pre
    down = layer.mlp.down_proj.weight.detach().cpu().double()

    gate_sq = gate.square().sum(dim=1)
    up_sq = up.square().sum(dim=1)
    gate_up = (gate * up).sum(dim=1)
    down_sq = down.square().sum(dim=0)
    atom_sq = 0.5 * down_sq * (gate_sq * up_sq + gate_up.square())
    return torch.sqrt(atom_sq.clamp_min(0.0)).sort().values


def singular_values(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(matrix).sort(descending=True).values


def relative_spectrum_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-30))


def embedding_spectrum(model) -> torch.Tensor:
    emb = model.model.embed_tokens.weight.detach().cpu().double()
    gram = emb.T @ emb
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    return torch.sqrt(eigenvalues).sort(descending=True).values


def layer_delta_geometry(base, math_model, sql_model, layer_index: int) -> dict:
    base_state = base.state_dict()
    math_state = math_model.state_dict()
    sql_state = sql_model.state_dict()
    prefix = f"model.layers.{layer_index}."
    dot = 0.0
    norm_math = 0.0
    norm_sql = 0.0
    for key in base_state:
        if not key.startswith(prefix):
            continue
        db = base_state[key].detach().cpu().double()
        dm = math_state[key].detach().cpu().double() - db
        ds = sql_state[key].detach().cpu().double() - db
        dot += float((dm * ds).sum())
        norm_math += float(dm.square().sum())
        norm_sql += float(ds.square().sum())
    denom = math.sqrt(max(norm_math * norm_sql, 1e-300))
    cosine = dot / denom
    sql_unique = 1.0 - (dot * dot) / max(norm_math * norm_sql, 1e-300)
    return {
        "layer": layer_index,
        "delta_cosine": cosine,
        "math_delta_norm": math.sqrt(norm_math),
        "sql_delta_norm": math.sqrt(norm_sql),
        "orthogonal_unique_fraction_each": max(0.0, min(1.0, sql_unique)),
    }


def pair_certificate(name_a: str, model_a, name_b: str, model_b, config) -> dict:
    embedding_a = embedding_spectrum(model_a)
    embedding_b = embedding_spectrum(model_b)
    embedding_lb = relative_spectrum_distance(embedding_a, embedding_b)

    rows = []
    jacobian_num_sq = 0.0
    jacobian_den_sq = 0.0
    atom_num_sq = 0.0
    atom_den_sq = 0.0

    for layer_index, (layer_a, layer_b) in enumerate(zip(model_a.model.layers, model_b.model.layers)):
        ja = zero_jacobian(layer_a, config)
        jb = zero_jacobian(layer_b, config)
        sva = singular_values(ja)
        svb = singular_values(jb)
        jac_abs = float(torch.linalg.vector_norm(sva - svb))
        jac_den = float(torch.linalg.vector_norm(svb))
        jac_rel = jac_abs / max(jac_den, 1e-300)
        jacobian_num_sq += jac_abs * jac_abs
        jacobian_den_sq += jac_den * jac_den

        atoms_a = mlp_atom_invariants(layer_a, config, ja)
        atoms_b = mlp_atom_invariants(layer_b, config, jb)
        atom_abs = float(torch.linalg.vector_norm(atoms_a - atoms_b))
        atom_den = float(torch.linalg.vector_norm(atoms_b))
        atom_rel = atom_abs / max(atom_den, 1e-300)
        atom_num_sq += atom_abs * atom_abs
        atom_den_sq += atom_den * atom_den

        rows.append({
            "pair": f"{name_a}__{name_b}",
            "layer": layer_index,
            "jacobian_two_sided_orthogonal_min_relative_error": jac_rel,
            "jacobian_exact_conjugacy_rejected": jac_rel > EXACT_TOL,
            "swiglu_atom_spectrum_relative_mismatch": atom_rel,
            "standard_channel_conjugacy_rejected": atom_rel > EXACT_TOL,
            "jacobian_singular_min_a": float(sva.min()),
            "jacobian_singular_max_a": float(sva.max()),
            "jacobian_singular_min_b": float(svb.min()),
            "jacobian_singular_max_b": float(svb.max()),
            "atom_invariant_min_a": float(atoms_a.min()),
            "atom_invariant_max_a": float(atoms_a.max()),
            "atom_invariant_min_b": float(atoms_b.min()),
            "atom_invariant_max_b": float(atoms_b.max()),
        })
        print(json.dumps(rows[-1]))

    aggregate_jacobian_lb = math.sqrt(jacobian_num_sq / max(jacobian_den_sq, 1e-300))
    aggregate_atom_mismatch = math.sqrt(atom_num_sq / max(atom_den_sq, 1e-300))
    exact_rejected = bool(
        embedding_lb > EXACT_TOL
        or any(row["jacobian_exact_conjugacy_rejected"] for row in rows)
        or any(row["standard_channel_conjugacy_rejected"] for row in rows)
    )
    return {
        "pair": [name_a, name_b],
        "allowed_transport_class": "layer-boundary scaled-orthogonal maps required by RMSNorm Euclidean denominator",
        "embedding_singular_spectrum_relative_lower_bound": embedding_lb,
        "aggregate_relaxed_layerwise_jacobian_lower_bound": aggregate_jacobian_lb,
        "aggregate_swiglu_atom_spectrum_mismatch": aggregate_atom_mismatch,
        "max_layer_jacobian_lower_bound": max(row["jacobian_two_sided_orthogonal_min_relative_error"] for row in rows),
        "median_layer_jacobian_lower_bound": float(np.median([row["jacobian_two_sided_orthogonal_min_relative_error"] for row in rows])),
        "max_layer_atom_mismatch": max(row["swiglu_atom_spectrum_relative_mismatch"] for row in rows),
        "layers_rejecting_exact_jacobian_conjugacy": sum(row["jacobian_exact_conjugacy_rejected"] for row in rows),
        "layers_rejecting_standard_swiglu_channel_conjugacy": sum(row["standard_channel_conjugacy_rejected"] for row in rows),
        "exact_architecture_preserving_conjugacy_rejected": exact_rejected,
        "layer_rows": rows,
    }


def main() -> None:
    ids = [BASE_ID, MATH_ID, SQL_ID]
    configs = [AutoConfig.from_pretrained(model_id) for model_id in ids]
    signatures = [config_signature(config) for config in configs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError({"status": "CONFIG_MISMATCH", "signatures": signatures})

    tokenizers = tokenizer_audit(ids)
    if not all(tokenizers["vocab_equal"]) or not all(all(row) for row in tokenizers["probe_ids_equal"]):
        raise RuntimeError({"status": "TOKENIZER_MISMATCH", "audit": tokenizers})

    print("Loading full public checkpoints")
    models = [
        AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
        for model_id in ids
    ]
    if not all(finite_state(model) for model in models):
        raise RuntimeError("nonfinite parent parameters")
    shape_audit = assert_same_shapes(models)
    base, math_model, sql_model = models

    delta_geometry = [
        layer_delta_geometry(base, math_model, sql_model, layer)
        for layer in range(int(configs[0].num_hidden_layers))
    ]

    certificates = [
        pair_certificate("math", math_model, "sql", sql_model, configs[0]),
        pair_certificate("base", base, "math", math_model, configs[0]),
        pair_certificate("base", base, "sql", sql_model, configs[0]),
    ]

    primary = certificates[0]
    status = (
        "EXACT_CONJUGACY_REJECTED"
        if primary["exact_architecture_preserving_conjugacy_rejected"]
        else "EXACT_CONJUGACY_NOT_REJECTED"
    )
    result = {
        "status": status,
        "models": {"base": BASE_ID, "math": MATH_ID, "sql": SQL_ID},
        "config_signature": signatures[0],
        "tokenizer_audit": tokenizers,
        "state_shape_audit": shape_audit,
        "mathematical_scope": {
            "exact_claim": "The Jacobian singular-spectrum bound is the exact minimum for each layer linearization under independent two-sided orthogonal boundary maps.",
            "chain_claim": "Summing independently minimized layer bounds is optimistic, hence a lower bound for any globally consistent transport chain.",
            "rmsnorm_constraint": "Any linear map preserving the RMSNorm Euclidean denominator for all states is scaled orthogonal.",
            "swiglu_claim": "The sorted quadratic atom norms are invariants for standard neuronwise SwiGLU conjugacy under state rotations, reciprocal channel scaling, and channel permutation.",
            "not_claimed": "This is not a lower bound over arbitrary nonlinear transports or architectures outside the fixed Llama/RMSNorm/SwiGLU graph.",
        },
        "primary_math_sql_certificate": {key: value for key, value in primary.items() if key != "layer_rows"},
        "all_pair_certificates": [{key: value for key, value in cert.items() if key != "layer_rows"} for cert in certificates],
        "delta_geometry_summary": {
            "mean_cosine": float(np.mean([row["delta_cosine"] for row in delta_geometry])),
            "median_cosine": float(np.median([row["delta_cosine"] for row in delta_geometry])),
            "mean_orthogonal_unique_fraction": float(np.mean([row["orthogonal_unique_fraction_each"] for row in delta_geometry])),
        },
        "delta_geometry": delta_geometry,
        "verdict": (
            "No exact architecture-preserving linear state transport exists between the two specialists; sequential block reuse without a learned/nonlinear transport is invalid."
            if status == "EXACT_CONJUGACY_REJECTED"
            else "The tested invariants did not reject exact conjugacy; stronger higher-order tests are required."
        ),
    }
    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (ROOT / "LAYER_CERTIFICATE.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(certificates[0]["layer_rows"][0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for certificate in certificates:
            writer.writerows(certificate["layer_rows"])

    with (ROOT / "DELTA_GEOMETRY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(delta_geometry[0].keys()))
        writer.writeheader()
        writer.writerows(delta_geometry)

    report = [
        "# SmolLM2 Specialist Conjugacy Certificate",
        "",
        f"Status: **{status}**",
        "",
        f"Math/SQL embedding spectrum lower bound: **{primary['embedding_singular_spectrum_relative_lower_bound']:.8g}**",
        f"Aggregate relaxed Jacobian lower bound: **{primary['aggregate_relaxed_layerwise_jacobian_lower_bound']:.8g}**",
        f"Maximum layer Jacobian lower bound: **{primary['max_layer_jacobian_lower_bound']:.8g}**",
        f"Layers rejecting exact Jacobian conjugacy: **{primary['layers_rejecting_exact_jacobian_conjugacy']}/{configs[0].num_hidden_layers}**",
        f"Aggregate SwiGLU atom mismatch: **{primary['aggregate_swiglu_atom_spectrum_mismatch']:.8g}**",
        f"Layers rejecting standard SwiGLU channel conjugacy: **{primary['layers_rejecting_standard_swiglu_channel_conjugacy']}/{configs[0].num_hidden_layers}**",
        f"Mean specialist-delta cosine: **{result['delta_geometry_summary']['mean_cosine']:.8g}**",
        f"Mean orthogonal unique fraction: **{result['delta_geometry_summary']['mean_orthogonal_unique_fraction']:.8g}**",
        "",
        result["verdict"],
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2))

    del models
    gc.collect()


if __name__ == "__main__":
    main()
