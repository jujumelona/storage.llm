from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "operator_v9", HERE / "real_smollm2_v9_operator_splitting.py"
)
v9 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v9)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260726)
np.random.seed(20260726)

ROOT = Path("out/real_smollm2_v12_parallel_residual_union")
ROOT.mkdir(parents=True, exist_ok=True)
BASE_ID = "HuggingFaceTB/SmolLM2-135M"
MATH_ID = "Ashed00/SmolMath-135M"
SQL_ID = "Ellight/code-smolLM2-135m-text-to-sql"
N_GEN = 32
N_MCQ = 48
WIKI_BLOCKS = 12
BOOTSTRAPS = 3000
v9.N_GEN = N_GEN
v9.N_MCQ = N_MCQ
v9.WIKI_BLOCKS = WIKI_BLOCKS
v9.BOOTSTRAPS = BOOTSTRAPS


def build_text2sql() -> list[dict[str, str]]:
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    rows = []
    for row in dataset:
        question = str(row.get("question") or "").strip()
        context = str(row.get("context") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not context or not answer:
            continue
        rows.append({
            "prompt": (
                "Database schema:\n" + context
                + "\n\nQuestion: " + question
                + "\nSQL query:\n"
            ),
            "target": answer,
        })
    return v9.deterministic_sample(rows, N_GEN, 1202)


def attention_linear_residual(layer: nn.Module, config: Any) -> torch.Tensor:
    d = int(config.hidden_size)
    n_heads = int(config.num_attention_heads)
    n_kv = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", d // n_heads))
    repeat = n_heads // n_kv
    eps = float(config.rms_norm_eps)

    v = layer.self_attn.v_proj.weight.detach().cpu().double()
    v = v.reshape(n_kv, head_dim, d).repeat_interleave(repeat, dim=0).reshape(d, d)
    o = layer.self_attn.o_proj.weight.detach().cpu().double()
    gamma = layer.input_layernorm.weight.detach().cpu().double()
    return o @ v @ torch.diag(gamma) / math.sqrt(eps)


def mlp_quadratic_witness(layer: nn.Module, config: Any) -> torch.Tensor:
    d = int(config.hidden_size)
    eps = float(config.rms_norm_eps)
    gamma = layer.post_attention_layernorm.weight.detach().cpu().double()
    gate = layer.mlp.gate_proj.weight.detach().cpu().double() * (gamma / math.sqrt(eps))[None, :]
    up = layer.mlp.up_proj.weight.detach().cpu().double() * (gamma / math.sqrt(eps))[None, :]
    down = layer.mlp.down_proj.weight.detach().cpu().double()

    identity = torch.eye(d, dtype=torch.float64)
    shifted = (identity + torch.roll(identity, 1, dims=0)) / math.sqrt(2.0)
    directions = torch.cat([identity, shifted], dim=1)
    return 0.5 * (down @ ((gate @ directions) * (up @ directions)))


def energy_preserving_coefficients(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, dict[str, float]]:
    ga = float(torch.linalg.vector_norm(a))
    gb = float(torch.linalg.vector_norm(b))
    if ga <= 1e-30 or gb <= 1e-30:
        return 0.5, 0.5, {"gain_a": ga, "gain_b": gb, "cosine": 0.0, "raw_a": 0.5, "raw_b": 0.5}
    cosine = float((a * b).sum() / (ga * gb))
    cosine = max(-0.999999, min(1.0, cosine))
    target = math.sqrt((ga * ga + gb * gb) / (4.0 * (1.0 + cosine)))
    ca = target / ga
    cb = target / gb
    if ca > 1.0 or cb > 1.0:
        scale = 1.0 / max(ca, cb)
        ca *= scale
        cb *= scale
    return ca, cb, {
        "gain_a": ga,
        "gain_b": gb,
        "cosine": cosine,
        "raw_a": target / ga,
        "raw_b": target / gb,
        "coefficient_a": ca,
        "coefficient_b": cb,
    }


class ParallelResidualUnionLayer(nn.Module):
    """One persistent hidden trajectory with two immediately fused residual banks.

    Both attention banks see the same pre-attention state. Their residuals are
    fused before the MLP stage. Both MLP banks then see the same fused state and
    are fused immediately. No parent-specific state persists across a boundary.
    """

    def __init__(
        self,
        layer_a: nn.Module,
        layer_b: nn.Module,
        attn_coeff_a: float,
        attn_coeff_b: float,
        mlp_coeff_a: float,
        mlp_coeff_b: float,
    ) -> None:
        super().__init__()
        self.layer_a = layer_a
        self.layer_b = layer_b
        self.register_buffer("attn_coeff_a", torch.tensor(attn_coeff_a, dtype=torch.float32), persistent=True)
        self.register_buffer("attn_coeff_b", torch.tensor(attn_coeff_b, dtype=torch.float32), persistent=True)
        self.register_buffer("mlp_coeff_a", torch.tensor(mlp_coeff_a, dtype=torch.float32), persistent=True)
        self.register_buffer("mlp_coeff_b", torch.tensor(mlp_coeff_b, dtype=torch.float32), persistent=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        common = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_value": None,
            "output_attentions": False,
            "use_cache": False,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        }

        residual = hidden_states
        norm_a = self.layer_a.input_layernorm(hidden_states)
        norm_b = self.layer_b.input_layernorm(hidden_states)
        attn_a = self.layer_a.self_attn(hidden_states=norm_a, **common)[0]
        attn_b = self.layer_b.self_attn(hidden_states=norm_b, **common)[0]
        hidden_states = (
            residual
            + self.attn_coeff_a.to(hidden_states.dtype) * attn_a
            + self.attn_coeff_b.to(hidden_states.dtype) * attn_b
        )

        residual = hidden_states
        mlp_a = self.layer_a.mlp(self.layer_a.post_attention_layernorm(hidden_states))
        mlp_b = self.layer_b.mlp(self.layer_b.post_attention_layernorm(hidden_states))
        hidden_states = (
            residual
            + self.mlp_coeff_a.to(hidden_states.dtype) * mlp_a
            + self.mlp_coeff_b.to(hidden_states.dtype) * mlp_b
        )
        return (hidden_states,)


def evaluate_all(model: nn.Module, tokenizer: Any, datasets: dict[str, Any]):
    start = time.time()
    records = {
        "wikitext": v9.evaluate_wiki(model, tokenizer, datasets["wikitext"]),
        "gsm8k": v9.evaluate_targets(model, tokenizer, datasets["gsm8k"]),
        "text2sql": v9.evaluate_targets(model, tokenizer, datasets["text2sql"]),
        "openbookqa": v9.evaluate_mcq(model, tokenizer, datasets["openbookqa"]),
        "piqa": v9.evaluate_mcq(model, tokenizer, datasets["piqa"]),
    }
    return records, time.time() - start


def main() -> None:
    configs_raw = [AutoConfig.from_pretrained(model_id) for model_id in [BASE_ID, MATH_ID, SQL_ID]]
    configs = [v9.structural_config(config) for config in configs_raw]
    if any(config != configs[0] for config in configs[1:]):
        raise RuntimeError({"status": "STRUCTURAL_CONFIG_MISMATCH", "configs": configs})

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    math_tokenizer = AutoTokenizer.from_pretrained(MATH_ID)
    sql_tokenizer = AutoTokenizer.from_pretrained(SQL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_checks = {
        "math": v9.tokenizer_audit(tokenizer, math_tokenizer),
        "sql": v9.tokenizer_audit(tokenizer, sql_tokenizer),
    }
    if not all(entry["vocab_equal"] and entry["probe_ids_equal"] for entry in tokenizer_checks.values()):
        raise RuntimeError({"status": "TOKENIZER_MISMATCH", "audit": tokenizer_checks})

    print("Loading full public checkpoints")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    math_model = AutoModelForCausalLM.from_pretrained(MATH_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    sql_model = AutoModelForCausalLM.from_pretrained(SQL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()

    shapes = [{key: tuple(value.shape) for key, value in model.state_dict().items()} for model in [base_model, math_model, sql_model]]
    if shapes[0] != shapes[1] or shapes[0] != shapes[2]:
        raise RuntimeError("STATE_SHAPE_MISMATCH")

    parent_parameter_count = int(sum(p.numel() for p in base_model.parameters()))
    datasets = {
        "wikitext": v9.build_wikitext(),
        "gsm8k": v9.build_gsm8k(),
        "text2sql": build_text2sql(),
        "openbookqa": v9.build_openbookqa(),
        "piqa": v9.build_piqa(),
    }

    records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    timings: dict[str, float] = {}
    for name, model in [("parent_base", base_model), ("parent_math", math_model), ("parent_sql", sql_model)]:
        rec, elapsed = evaluate_all(model, tokenizer, datasets)
        records[name] = rec
        summaries[name] = v9.summarize(rec)
        timings[name] = elapsed
        print(name, json.dumps(summaries[name], indent=2))

    parent_hashes = v9.parent_parameter_hashes({
        "base": base_model,
        "math": math_model,
        "sql": sql_model,
    })

    coefficient_audit = []
    union_layers = []
    for layer_index, (math_layer, sql_layer) in enumerate(zip(math_model.model.layers, sql_model.model.layers)):
        attn_a = attention_linear_residual(math_layer, configs_raw[0])
        attn_b = attention_linear_residual(sql_layer, configs_raw[0])
        ca_attn, cb_attn, attn_audit = energy_preserving_coefficients(attn_a, attn_b)

        mlp_a = mlp_quadratic_witness(math_layer, configs_raw[0])
        mlp_b = mlp_quadratic_witness(sql_layer, configs_raw[0])
        ca_mlp, cb_mlp, mlp_audit = energy_preserving_coefficients(mlp_a, mlp_b)
        coefficient_audit.append({
            "layer": layer_index,
            "attention": attn_audit,
            "mlp": mlp_audit,
        })
        union_layers.append(ParallelResidualUnionLayer(
            math_layer,
            sql_layer,
            ca_attn,
            cb_attn,
            ca_mlp,
            cb_mlp,
        ))

    base_model.model.layers = nn.ModuleList(union_layers)
    union_model = base_model.eval()
    new_parameter_count = int(sum(p.numel() for p in union_model.parameters()))
    parameter_ratio = new_parameter_count / parent_parameter_count

    prompts = [
        "The capital of France is", "Question: 19 + 23 =", "SELECT name FROM users",
        "A metal conducts electricity because", "Goal: open a jar\nBest solution:",
    ]
    logits_before_gauge = v9.prompt_logits(union_model, tokenizer, prompts)
    gauge_audit = v9.permutation_gauge(union_model)
    logits_after_gauge = v9.prompt_logits(union_model, tokenizer, prompts)
    gauge_logit_audit = v9.compare_logits(logits_before_gauge, logits_after_gauge)

    math_model.model.layers = nn.ModuleList([])
    sql_model.model.layers = nn.ModuleList([])
    del math_model, sql_model
    gc.collect()

    union_records, union_elapsed = evaluate_all(union_model, tokenizer, datasets)
    records["parallel_residual_union"] = union_records
    summaries["parallel_residual_union"] = v9.summarize(union_records)
    timings["parallel_residual_union"] = union_elapsed

    finite = v9.finite_audit(union_model)
    copy_audit = v9.exact_copy_audit(union_model, parent_hashes)
    parents = ["parent_base", "parent_math", "parent_sql"]
    comparisons = {
        parent: v9.bootstrap_composite(union_records, records[parent], 12000 + index)
        for index, parent in enumerate(parents)
    }

    domains = list(union_records)
    mcq_domains = ["openbookqa", "piqa"]
    best_parent_loss = {
        domain: min(summaries[parent][domain]["loss"] for parent in parents)
        for domain in domains
    }
    best_parent_accuracy = {
        domain: max(summaries[parent][domain]["accuracy"] for parent in parents)
        for domain in mcq_domains
    }
    loss_within_3pct = all(
        summaries["parallel_residual_union"][domain]["loss"] <= 1.03 * best_parent_loss[domain]
        for domain in domains
    )
    accuracy_within_5pp = all(
        summaries["parallel_residual_union"][domain]["accuracy"] >= best_parent_accuracy[domain] - 0.05
        for domain in mcq_domains
    )
    composite_beats_all = all(entry["ci95"][1] < 0.0 for entry in comparisons.values())
    structural_pass = bool(
        parameter_ratio <= 1.8
        and gauge_logit_audit["relative_rms"] < 1e-5
        and copy_audit["exact_copy_count"] == 0
        and finite["all_finite"]
        and hasattr(union_model, "lm_head")
        and union_model.config.tie_word_embeddings
    )
    promoted = bool(loss_within_3pct and accuracy_within_5pp and composite_beats_all and structural_pass)

    speed_ratio_vs_mean_parent = union_elapsed / float(np.mean([timings[parent] for parent in parents]))
    result = {
        "status": "PARALLEL_RESIDUAL_UNION_PASS" if promoted else "PARALLEL_RESIDUAL_UNION_NOT_PROMOTED",
        "method": {
            "name": "single_trajectory_parallel_residual_union",
            "attention_update": "x <- x + cM*A_M(N_M(x)) + cS*A_S(N_S(x))",
            "mlp_update": "h <- h + cM*M_M(N_M(h)) + cS*M_S(N_S(h))",
            "coefficient_rule": "equalized branch jet contribution with fused isotropic energy equal to mean parent jet energy",
            "training": False,
            "router": False,
            "persistent_parallel_states": False,
            "probability_or_logit_mixture": False,
        },
        "models": {"base": BASE_ID, "math": MATH_ID, "sql": SQL_ID},
        "structure": {
            "parent_parameter_count": parent_parameter_count,
            "new_parameter_count": new_parameter_count,
            "parameter_ratio": parameter_ratio,
            "single_persistent_hidden_state": True,
            "single_embedding": True,
            "single_tied_lm_head": True,
            "two_attention_and_mlp_operator_banks": True,
            "coefficient_audit": coefficient_audit,
            "gauge_audit": gauge_audit,
            "gauge_logit_audit": gauge_logit_audit,
            "exact_copy_audit": copy_audit,
            "finite_audit": finite,
        },
        "evaluation": {
            "wiki_blocks": len(union_records["wikitext"]),
            "gsm8k_examples": len(union_records["gsm8k"]),
            "text2sql_examples": len(union_records["text2sql"]),
            "openbookqa_examples": len(union_records["openbookqa"]),
            "piqa_examples": len(union_records["piqa"]),
            "mcq_metric": "choice-normalized total answer-sequence cross entropy",
            "bootstrap_resamples": BOOTSTRAPS,
        },
        "summaries": summaries,
        "timings_seconds": timings,
        "speed_ratio_vs_mean_parent": speed_ratio_vs_mean_parent,
        "comparisons": comparisons,
        "best_parent_loss": best_parent_loss,
        "best_parent_accuracy": best_parent_accuracy,
        "all_losses_within_3pct": loss_within_3pct,
        "all_mcq_accuracies_within_5pp": accuracy_within_5pp,
        "composite_significantly_beats_all_three_parents": composite_beats_all,
        "structural_pass": structural_pass,
        "promoted": promoted,
    }
    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (ROOT / "METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "model", "domain", "example_id", "loss", "correct", "prediction", "label", "target_tokens"
        ])
        writer.writeheader()
        for model_name, domain_map in records.items():
            for domain, rows in domain_map.items():
                for row in rows:
                    writer.writerow({
                        "model": model_name,
                        "domain": domain,
                        "example_id": row.get("example_id"),
                        "loss": row.get("loss"),
                        "correct": row.get("correct"),
                        "prediction": row.get("prediction"),
                        "label": row.get("label"),
                        "target_tokens": row.get("target_tokens"),
                    })

    report = [
        "# Real SmolLM2 One-Trajectory Parallel Residual Union",
        "",
        f"Status: **{result['status']}**",
        f"Parameter ratio: **{parameter_ratio:.6f}x**",
        f"Gauge relative RMS: **{gauge_logit_audit['relative_rms']:.8g}**",
        f"Exact copied parameters: **{copy_audit['exact_copy_count']}**",
        f"All losses within 3%: **{loss_within_3pct}**",
        f"All MCQ accuracies within 5pp: **{accuracy_within_5pp}**",
        f"Composite significantly beats all parents: **{composite_beats_all}**",
        f"Speed ratio vs mean parent: **{speed_ratio_vs_mean_parent:.4f}x**",
        "",
        "Unlike sequential operator splitting, both branches see the same state and are fused immediately after each sublayer. No branch-specific state crosses a layer boundary.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    if promoted:
        checkpoint = {
            "state_dict": {key: value.detach().cpu().to(torch.bfloat16) for key, value in union_model.state_dict().items()},
            "base_config": configs_raw[0].to_dict(),
            "method": result["method"],
            "coefficient_audit": coefficient_audit,
            "gauge_audit": gauge_audit,
        }
        torch.save(checkpoint, ROOT / "PARALLEL_RESIDUAL_UNION_SINGLE_CHECKPOINT.pt")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
