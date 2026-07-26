from __future__ import annotations

import csv
import gc
import importlib.util
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v1 = load_module("weight_field_v1", "real_smollm2_weight_field.py")
v2 = load_module("weight_field_v2", "real_smollm2_weight_field_v2.py")
v3 = load_module("weight_field_v3", "real_smollm2_weight_field_v3_blind.py")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260726)
np.random.seed(20260726)

ROOT = Path("out/real_smollm2_v5_strict")
ROOT.mkdir(parents=True, exist_ok=True)
BASE_ID = v1.BASE_ID
INSTRUCT_ID = v1.INSTRUCT_ID
N = 192
WIKI_BLOCKS = 24
BLOCK_SIZE = 256
BOOTSTRAPS = 5000
v1.WIKI_BLOCKS = WIKI_BLOCKS
v1.BLOCK_SIZE = BLOCK_SIZE


def chat_prompt(tokenizer, content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_arc(tokenizer, subset: str, offset: int, n: int):
    ds = load_dataset("allenai/ai2_arc", subset, split="validation")
    rows = []
    for row in ds:
        texts = list(row["choices"]["text"])
        labels = list(row["choices"]["label"])
        answer = str(row["answerKey"])
        mapping = {str(label): i for i, label in enumerate(labels)}
        if answer not in mapping or len(texts) < 2:
            continue
        letters = [chr(ord("A") + i) for i in range(len(texts))]
        content = "Answer the multiple-choice question with only the option letter.\n\n"
        content += "Question: " + row["question"] + "\n"
        content += "\n".join(f"{letters[i]}. {text}" for i, text in enumerate(texts))
        content += "\nAnswer:"
        rows.append({"prompt": chat_prompt(tokenizer, content), "choices": letters, "label": mapping[answer]})
    return rows[offset : offset + n]


def build_hella(offset: int, n: int):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    rows = []
    for row in ds:
        label = int(row["label"])
        if label >= 0:
            rows.append({"prompt": row["ctx"].strip(), "choices": list(row["endings"]), "label": label})
    return rows[offset : offset + n]


def build_boolq(tokenizer, offset: int, n: int):
    try:
        ds = load_dataset("google/boolq", split="validation")
    except Exception:
        ds = load_dataset("super_glue", "boolq", split="validation")
    rows = []
    for row in ds:
        content = (
            "Read the passage and answer the question with only yes or no.\n\n"
            f"Passage: {row['passage']}\n\nQuestion: {row['question']}\nAnswer:"
        )
        rows.append({
            "prompt": chat_prompt(tokenizer, content),
            "choices": ["yes", "no"],
            "label": 0 if bool(row["answer"]) else 1,
        })
    return rows[offset : offset + n]


@torch.inference_mode()
def evaluate_wiki_blocks(model, tokenizer, texts: list[str]):
    ids = v1.encode_no_special(tokenizer, "\n\n".join(texts))
    usable = min(len(ids) - 1, WIKI_BLOCKS * BLOCK_SIZE)
    records = []
    for block_id, start in enumerate(range(0, usable, BLOCK_SIZE)):
        chunk = ids[start : start + BLOCK_SIZE + 1]
        if len(chunk) < 32:
            continue
        x = torch.tensor([chunk[:-1]], dtype=torch.long)
        y = torch.tensor([chunk[1:]], dtype=torch.long)
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        records.append({"example_id": block_id, "nll": float(loss)})
    return records


@torch.inference_mode()
def evaluate_mcq_records(model, tokenizer, examples: list[dict]):
    records = []
    for i, row in enumerate(examples):
        nlls = v1.option_nlls(model, tokenizer, row["prompt"], row["choices"])
        pred = int(np.argmin(nlls))
        records.append({
            "example_id": i,
            "correct_nll": float(nlls[row["label"]]),
            "correct": int(pred == row["label"]),
            "prediction": pred,
            "label": int(row["label"]),
        })
    return records


def summarize_records(records_by_domain: dict[str, list[dict]]):
    result = {}
    for domain, rows in records_by_domain.items():
        key = "nll" if domain == "wikitext" else "correct_nll"
        entry = {"n": len(rows), "nll": float(np.mean([row[key] for row in rows]))}
        if domain != "wikitext":
            entry["accuracy"] = float(np.mean([row["correct"] for row in rows]))
        result[domain] = entry
    result["balanced_nll"] = float(np.mean([entry["nll"] for entry in result.values() if isinstance(entry, dict)]))
    result["mean_mcq_accuracy"] = float(np.mean([result[d]["accuracy"] for d in result if d != "wikitext" and isinstance(result[d], dict)]))
    return result


def bootstrap_composite(candidate, parent, rng, n_boot=BOOTSTRAPS):
    domains = list(candidate)
    values = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        rels = []
        for domain in domains:
            ck = "nll" if domain == "wikitext" else "correct_nll"
            c = np.array([row[ck] for row in candidate[domain]], dtype=np.float64)
            p = np.array([row[ck] for row in parent[domain]], dtype=np.float64)
            idx = rng.integers(0, len(c), len(c))
            rels.append(c[idx].mean() / p[idx].mean() - 1.0)
        values[b] = np.mean(rels)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "wins_fraction": float(np.mean(values < 0)),
    }


def bootstrap_domain(candidate_rows, parent_rows, domain: str, rng, n_boot=BOOTSTRAPS):
    key = "nll" if domain == "wikitext" else "correct_nll"
    c = np.array([row[key] for row in candidate_rows], dtype=np.float64)
    p = np.array([row[key] for row in parent_rows], dtype=np.float64)
    n = len(c)
    nll_delta = np.empty(n_boot, dtype=np.float64)
    acc_delta = None if domain == "wikitext" else np.empty(n_boot, dtype=np.float64)
    if domain != "wikitext":
        ca = np.array([row["correct"] for row in candidate_rows], dtype=np.float64)
        pa = np.array([row["correct"] for row in parent_rows], dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        nll_delta[b] = c[idx].mean() - p[idx].mean()
        if acc_delta is not None:
            acc_delta[b] = ca[idx].mean() - pa[idx].mean()
    result = {
        "nll_delta_mean": float(nll_delta.mean()),
        "nll_delta_ci95": [float(np.quantile(nll_delta, 0.025)), float(np.quantile(nll_delta, 0.975))],
    }
    if acc_delta is not None:
        result.update({
            "accuracy_delta_mean": float(acc_delta.mean()),
            "accuracy_delta_ci95": [float(np.quantile(acc_delta, 0.025)), float(np.quantile(acc_delta, 0.975))],
        })
    return result


def fixed_candidate_state(a, b):
    means, qdelta, scales, total_n, _, _ = v2.build_field_fp16(a, b)

    def qfn(key: str):
        if "embed_tokens" in key or key.startswith("lm_head") or key == "model.norm.weight":
            return -1.0
        if "layernorm" in key.lower():
            return -1.0
        return 0.0

    return v2.materialize_fn(means, qdelta, scales, qfn), total_n


def canonical_rotation(state: dict[str, torch.Tensor], config):
    d = int(config.hidden_size)
    gram = torch.zeros((d, d), dtype=torch.float64)

    def add_term(term: torch.Tensor):
        nonlocal gram
        term = term.double()
        scale = torch.trace(term).abs().clamp_min(1e-30)
        gram += term / scale

    embed = state["model.embed_tokens.weight"].float()
    add_term(embed.T @ embed)
    for layer in range(int(config.num_hidden_layers)):
        prefix = f"model.layers.{layer}."
        for name in ["q_proj", "k_proj", "v_proj"]:
            w = state[prefix + f"self_attn.{name}.weight"].float()
            add_term(w.T @ w)
        w = state[prefix + "self_attn.o_proj.weight"].float()
        add_term(w @ w.T)
        for name in ["gate_proj", "up_proj"]:
            w = state[prefix + f"mlp.{name}.weight"].float()
            add_term(w.T @ w)
        w = state[prefix + "mlp.down_proj.weight"].float()
        add_term(w @ w.T)

    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    U = eigvecs[:, order]
    for j in range(d):
        col = U[:, j]
        idx = int(torch.argmax(col.abs()))
        if col[idx] < 0:
            U[:, j] = -col
    R = U.T.float()
    orth_error = float((R @ R.T - torch.eye(d)).abs().max())

    new_state: dict[str, torch.Tensor] = {}
    new_state["model.embed_tokens.weight"] = embed @ R.T
    for layer in range(int(config.num_hidden_layers)):
        prefix = f"model.layers.{layer}."
        gamma_attn = state[prefix + "input_layernorm.weight"].float()
        gamma_mlp = state[prefix + "post_attention_layernorm.weight"].float()
        D_attn_Rt = gamma_attn[:, None] * R.T
        D_mlp_Rt = gamma_mlp[:, None] * R.T
        new_state[prefix + "input_layernorm.weight"] = torch.ones_like(gamma_attn)
        new_state[prefix + "post_attention_layernorm.weight"] = torch.ones_like(gamma_mlp)
        for name in ["q_proj", "k_proj", "v_proj"]:
            key = prefix + f"self_attn.{name}.weight"
            new_state[key] = state[key].float() @ D_attn_Rt
        key = prefix + "self_attn.o_proj.weight"
        new_state[key] = R @ state[key].float()
        for name in ["gate_proj", "up_proj"]:
            key = prefix + f"mlp.{name}.weight"
            new_state[key] = state[key].float() @ D_mlp_Rt
        key = prefix + "mlp.down_proj.weight"
        new_state[key] = R @ state[key].float()

    gamma_final = state["model.norm.weight"].float()
    new_state["model.norm.weight"] = torch.ones_like(gamma_final)
    new_state["lm_head.weight"] = embed @ (gamma_final[:, None] * R.T)

    new_config = deepcopy(config)
    new_config.tie_word_embeddings = False
    new_config.torch_dtype = "float32"
    audit = {
        "orthogonality_max_abs": orth_error,
        "eigenvalue_max": float(eigvals.max()),
        "eigenvalue_min": float(eigvals.min()),
        "eigenvalue_condition": float(eigvals.max() / eigvals.clamp_min(1e-30).min()),
        "max_basis_coordinate_abs": float(R.abs().max()),
        "mean_basis_coordinate_abs": float(R.abs().mean()),
    }
    return new_state, new_config, audit


def compare_model_logits(model_a, model_b, tokenizer, prompts):
    refs = v1.prompt_logits(model_a, tokenizer, prompts)
    cands = v1.prompt_logits(model_b, tokenizer, prompts)
    return v1.compare_logits(refs, cands)


def exact_copy_audit(new_state, parent_states):
    copies = []
    for key, tensor in new_state.items():
        for parent_name, parent in parent_states.items():
            if key in parent and tensor.shape == parent[key].shape:
                if torch.equal(tensor.cpu(), parent[key].float().cpu()):
                    copies.append({"new_key": key, "parent": parent_name, "parent_key": key})
    return {"exact_copy_count": len(copies), "copies": copies[:100]}


def main():
    config = AutoConfig.from_pretrained(BASE_ID)
    config_b = AutoConfig.from_pretrained(INSTRUCT_ID)
    fields = ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size", "hidden_act", "rope_theta", "tie_word_embeddings"]
    config_audit = {field: [getattr(config, field), getattr(config_b, field)] for field in fields}
    if any(pair[0] != pair[1] for pair in config_audit.values()):
        raise RuntimeError({"config_mismatch": config_audit})

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    chat_tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_ID)
    if tokenizer.get_vocab() != chat_tokenizer.get_vocab():
        raise RuntimeError("tokenizer mismatch")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    model_instruct = AutoModelForCausalLM.from_pretrained(INSTRUCT_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    state_base = v1.clone_float_state(model_base)
    state_instruct = v1.clone_float_state(model_instruct)
    candidate_state, parent_params = fixed_candidate_state(state_base, state_instruct)
    model_candidate = v1.load_from_state(config, candidate_state)

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wiki_texts = [row["text"] for row in wiki if row["text"].strip()][1800:3200]
    datasets = {
        "arc_easy": build_arc(chat_tokenizer, "ARC-Easy", 320, N),
        "arc_challenge": build_arc(chat_tokenizer, "ARC-Challenge", 64, N),
        "hellaswag": build_hella(512, N),
        "boolq": build_boolq(chat_tokenizer, 512, N),
    }
    if any(len(rows) != N for rows in datasets.values()):
        raise RuntimeError({name: len(rows) for name, rows in datasets.items()})

    models = {"parent_base": model_base, "parent_instruct": model_instruct, "fixed_candidate": model_candidate}
    all_records = {}
    summaries = {}
    for model_name, model in models.items():
        start = time.time()
        records = {"wikitext": evaluate_wiki_blocks(model, tokenizer, wiki_texts)}
        for domain, rows in datasets.items():
            records[domain] = evaluate_mcq_records(model, tokenizer, rows)
        all_records[model_name] = records
        summaries[model_name] = summarize_records(records)
        summaries[model_name]["seconds"] = time.time() - start
        print(model_name, json.dumps(summaries[model_name], indent=2))
        (ROOT / "PARTIAL.json").write_text(json.dumps({"summaries": summaries}, indent=2), encoding="utf-8")

    rng = np.random.default_rng(20260726)
    comparisons = {}
    for parent_name in ["parent_base", "parent_instruct"]:
        entry = {"composite_relative_nll": bootstrap_composite(all_records["fixed_candidate"], all_records[parent_name], rng)}
        entry["domains"] = {}
        for domain in all_records["fixed_candidate"]:
            entry["domains"][domain] = bootstrap_domain(all_records["fixed_candidate"][domain], all_records[parent_name][domain], domain, rng)
        comparisons[parent_name] = entry

    nll_domains = list(all_records["fixed_candidate"])
    mcq_domains = [d for d in nll_domains if d != "wikitext"]
    best_parent_nll = {d: min(summaries["parent_base"][d]["nll"], summaries["parent_instruct"][d]["nll"]) for d in nll_domains}
    best_parent_acc = {d: max(summaries["parent_base"][d]["accuracy"], summaries["parent_instruct"][d]["accuracy"]) for d in mcq_domains}
    no_nll_regression = all(summaries["fixed_candidate"][d]["nll"] <= 1.03 * best_parent_nll[d] for d in nll_domains)
    no_acc_regression = all(summaries["fixed_candidate"][d]["accuracy"] >= best_parent_acc[d] - 0.05 for d in mcq_domains)
    composite_significant = all(comparisons[p]["composite_relative_nll"]["ci95"][1] < 0 for p in comparisons)

    prompts = [
        "The history of mathematics begins", "Water boils at", "Question: 13 + 29 =", "def binary_search(xs, x):",
        chat_prompt(chat_tokenizer, "Explain photosynthesis in one sentence."),
    ]
    rotated_state, rotated_config, rotation_audit = canonical_rotation(candidate_state, config)
    model_rotated = AutoModelForCausalLM.from_config(rotated_config, torch_dtype=torch.float32)
    missing, unexpected = model_rotated.load_state_dict(rotated_state, strict=False)
    if missing or unexpected:
        raise RuntimeError({"rotation_load_missing": missing, "rotation_load_unexpected": unexpected})
    model_rotated.eval()
    rotation_logit_audit = compare_model_logits(model_candidate, model_rotated, tokenizer, prompts)
    copy_audit = exact_copy_audit(rotated_state, {"base": state_base, "instruct": state_instruct})

    rotated_params = int(sum(t.numel() for t in rotated_state.values()))
    parent_parameter_count = int(sum(p.numel() for p in model_base.parameters()))
    rotated_ratio = rotated_params / parent_parameter_count
    rotation_pass = rotation_logit_audit["relative_rms"] < 1e-5 and copy_audit["exact_copy_count"] == 0 and rotated_ratio <= 1.8
    promoted = bool(no_nll_regression and no_acc_regression and composite_significant and rotation_pass)

    if promoted:
        model_dir = ROOT / "CANONICAL_ROTATED_MODEL"
        model_rotated.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        reloaded = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, local_files_only=True).eval()
        reload_audit = compare_model_logits(model_rotated, reloaded, tokenizer, prompts)
        del reloaded
    else:
        reload_audit = None

    result = {
        "status": "STRICT_INDEPENDENT_PASS" if promoted else "STRICT_INDEPENDENT_NOT_PROMOTED",
        "candidate_definition": {
            "embedding": "base",
            "final_norm": "base",
            "all_layer_norms": "base",
            "attention_and_mlp": "FP16-rounded arithmetic midpoint(base,instruct)",
            "candidate_frozen_before_this_evaluation": True,
        },
        "evaluation": {
            "mcq_n_each": N,
            "wikitext_blocks": len(all_records["fixed_candidate"]["wikitext"]),
            "bootstrap_resamples": BOOTSTRAPS,
            "offsets": {"arc_easy": 320, "arc_challenge": 64, "hellaswag": 512, "boolq": 512, "wikitext_text": 1800},
        },
        "config_audit": config_audit,
        "summaries": summaries,
        "comparisons": comparisons,
        "best_parent_nll": best_parent_nll,
        "best_parent_accuracy": best_parent_acc,
        "no_domain_nll_regression_over_3pct": no_nll_regression,
        "no_mcq_accuracy_regression_over_5pp": no_acc_regression,
        "composite_significantly_beats_both_parents": composite_significant,
        "canonical_rotation": {
            **rotation_audit,
            "logit_audit": rotation_logit_audit,
            "exact_copy_audit": copy_audit,
            "parent_parameter_count": parent_parameter_count,
            "rotated_parameter_count": rotated_params,
            "parameter_ratio": rotated_ratio,
            "single_lm_head": True,
            "tie_word_embeddings": False,
            "reload_audit": reload_audit,
            "pass": rotation_pass,
        },
        "promoted": promoted,
    }
    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (ROOT / "EXAMPLE_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "domain", "example_id", "nll", "correct_nll", "correct", "prediction", "label"])
        writer.writeheader()
        for model_name, domains in all_records.items():
            for domain, rows in domains.items():
                for row in rows:
                    writer.writerow({"model": model_name, "domain": domain, **row})

    report = [
        "# Strict Independent SmolLM2 Validation and Canonical Gauge Compilation",
        "",
        f"Status: **{result['status']}**",
        f"Composite significant vs both parents: **{composite_significant}**",
        f"All NLL domains within 3% of best parent: **{no_nll_regression}**",
        f"All MCQ accuracies within 5pp of best parent: **{no_acc_regression}**",
        f"Canonical rotation relative RMS logit error: **{rotation_logit_audit['relative_rms']:.8g}**",
        f"Exact copied tensors after rotation: **{copy_audit['exact_copy_count']}**",
        f"Rotated parameter ratio: **{rotated_ratio:.6f}x**",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
