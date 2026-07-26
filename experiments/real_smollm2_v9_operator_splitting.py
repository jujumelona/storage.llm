from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260726)
np.random.seed(20260726)

ROOT = Path("out/real_smollm2_v9_operator_splitting")
ROOT.mkdir(parents=True, exist_ok=True)

BASE_ID = "HuggingFaceTB/SmolLM2-135M"
MATH_ID = "Ashed00/SmolMath-135M"
CODE_ID = "lhoestq/finetune_smollm2_python"
CODE_SUBFOLDER = "final_checkpoint"

N_GEN = 24
N_MCQ = 32
WIKI_BLOCKS = 8
BLOCK_SIZE = 256
MAX_LENGTH = 512
BATCH_SIZE = 8
BOOTSTRAPS = 3000


class SplitDecoderLayer(nn.Module):
    """One hidden stream, two fixed specialist residual operators.

    If L_i(h) is a pretrained Llama decoder block, define F_i(h)=L_i(h)-h.
    The layer performs a weight-only second-moment-normalized Lie step:
        h <- h + c_first F_first(h)
        h <- h + c_second F_second(h)
    with order A->B on even layers and B->A on odd layers.
    """

    def __init__(
        self,
        layer_a: nn.Module,
        layer_b: nn.Module,
        coeff_a: float,
        coeff_b: float,
        a_first: bool,
    ) -> None:
        super().__init__()
        self.layer_a = layer_a
        self.layer_b = layer_b
        self.register_buffer("coeff_a", torch.tensor(float(coeff_a), dtype=torch.float32), persistent=True)
        self.register_buffer("coeff_b", torch.tensor(float(coeff_b), dtype=torch.float32), persistent=True)
        self.a_first = bool(a_first)

    def _step(self, layer: nn.Module, hidden_states: torch.Tensor, coeff: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        outputs = layer(hidden_states=hidden_states, **kwargs)
        proposed = outputs[0]
        return hidden_states + coeff.to(hidden_states.dtype) * (proposed - hidden_states)

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
        if self.a_first:
            hidden_states = self._step(self.layer_a, hidden_states, self.coeff_a, **common)
            hidden_states = self._step(self.layer_b, hidden_states, self.coeff_b, **common)
        else:
            hidden_states = self._step(self.layer_b, hidden_states, self.coeff_b, **common)
            hidden_states = self._step(self.layer_a, hidden_states, self.coeff_a, **common)
        return (hidden_states,)


def structural_config(config: Any) -> dict[str, Any]:
    fields = [
        "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
        "hidden_act", "rope_theta", "max_position_embeddings", "rms_norm_eps",
        "attention_bias", "mlp_bias", "tie_word_embeddings",
    ]
    return {field: getattr(config, field, None) for field in fields}


def normalized_frobenius(weight: torch.Tensor) -> float:
    weight = weight.detach().float()
    return float(torch.linalg.vector_norm(weight) / math.sqrt(max(1, weight.shape[1])))


def block_gain(layer: nn.Module, head_dim: int) -> dict[str, float]:
    attn = layer.self_attn
    mlp = layer.mlp
    q = normalized_frobenius(attn.q_proj.weight)
    k = normalized_frobenius(attn.k_proj.weight)
    v = normalized_frobenius(attn.v_proj.weight)
    o = normalized_frobenius(attn.o_proj.weight)
    gate = normalized_frobenius(mlp.gate_proj.weight)
    up = normalized_frobenius(mlp.up_proj.weight)
    down = normalized_frobenius(mlp.down_proj.weight)
    gamma_attn = float(layer.input_layernorm.weight.detach().float().square().mean().sqrt())
    gamma_mlp = float(layer.post_attention_layernorm.weight.detach().float().square().mean().sqrt())

    # Isotropic second-moment proxy. Q/K affect attention concentration; V/O
    # determine transported residual magnitude. This is used only to equalize
    # the two fixed residual steps, not fitted against evaluation data.
    attention_gain = gamma_attn * o * v * (1.0 + (q * k) / math.sqrt(float(head_dim)))
    mlp_gain = gamma_mlp * down * gate * up
    total = max(attention_gain + mlp_gain, 1e-30)
    return {
        "q": q, "k": k, "v": v, "o": o,
        "gate": gate, "up": up, "down": down,
        "attention_gain": attention_gain,
        "mlp_gain": mlp_gain,
        "total_gain": total,
    }


def gain_coefficients(layer_a: nn.Module, layer_b: nn.Module, head_dim: int) -> tuple[float, float, dict[str, Any]]:
    gain_a = block_gain(layer_a, head_dim)
    gain_b = block_gain(layer_b, head_dim)
    ga = gain_a["total_gain"]
    gb = gain_b["total_gain"]
    # c_a g_a = c_b g_b and c_a+c_b=1.
    coeff_a = gb / (ga + gb)
    coeff_b = ga / (ga + gb)
    return coeff_a, coeff_b, {"math": gain_a, "code": gain_b, "coeff_math": coeff_a, "coeff_code": coeff_b}


def tokenizer_audit(base_tokenizer: Any, other_tokenizer: Any) -> dict[str, Any]:
    probes = [
        "Hello world", "Question: 17 + 24 =", "def quicksort(xs):",
        "The chemical symbol for oxygen is", "\n    return value",
    ]
    return {
        "vocab_equal": base_tokenizer.get_vocab() == other_tokenizer.get_vocab(),
        "probe_ids_equal": all(
            base_tokenizer(text, add_special_tokens=False)["input_ids"]
            == other_tokenizer(text, add_special_tokens=False)["input_ids"]
            for text in probes
        ),
        "special_ids": {
            "base": [base_tokenizer.bos_token_id, base_tokenizer.eos_token_id, base_tokenizer.pad_token_id],
            "other": [other_tokenizer.bos_token_id, other_tokenizer.eos_token_id, other_tokenizer.pad_token_id],
        },
    }


def deterministic_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) < n:
        raise RuntimeError({"available": len(rows), "required": n})
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(rows), size=n, replace=False))
    return [rows[int(index)] for index in indices]


def build_wikitext() -> list[str]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [row["text"] for row in dataset if row["text"].strip()]
    return texts[200:900]


def build_gsm8k() -> list[dict[str, str]]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for row in dataset:
        answer = str(row["answer"])
        final = answer.split("####")[-1].strip().replace(",", "")
        rows.append({"prompt": f"Question: {row['question']}\nAnswer:", "target": " " + final})
    return deterministic_sample(rows, N_GEN, 901)


def build_mbpp() -> list[dict[str, str]]:
    try:
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    except Exception:
        dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")
    rows = []
    for row in dataset:
        text = str(row.get("prompt") or row.get("text") or "").strip()
        code = str(row.get("code") or "").strip()
        tests = row.get("test_list") or []
        if not text or not code:
            continue
        prompt = "Task: " + text
        if tests:
            prompt += "\nTests:\n" + "\n".join(map(str, tests[:3]))
        prompt += "\nPython solution:\n"
        rows.append({"prompt": prompt, "target": code})
    return deterministic_sample(rows, N_GEN, 902)


def build_openbookqa() -> list[dict[str, Any]]:
    dataset = load_dataset("allenai/openbookqa", "main", split="validation")
    rows = []
    for row in dataset:
        labels = [str(x) for x in row["choices"]["label"]]
        choices = list(row["choices"]["text"])
        mapping = {label: index for index, label in enumerate(labels)}
        answer = str(row["answerKey"])
        if answer not in mapping:
            continue
        prompt = "Science question: " + row["question_stem"] + "\nAnswer:"
        rows.append({"prompt": prompt, "choices": choices, "label": mapping[answer]})
    return deterministic_sample(rows, N_MCQ, 903)


def build_piqa() -> list[dict[str, Any]]:
    dataset = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    rows = []
    for row in dataset:
        label = int(row["label"])
        if label in (0, 1):
            rows.append({
                "prompt": "Goal: " + row["goal"] + "\nBest solution:",
                "choices": [row["sol1"], row["sol2"]],
                "label": label,
            })
    return deterministic_sample(rows, N_MCQ, 904)


@torch.inference_mode()
def evaluate_wiki(model: nn.Module, tokenizer: Any, texts: list[str]) -> list[dict[str, Any]]:
    ids = tokenizer("\n\n".join(texts), add_special_tokens=False)["input_ids"]
    usable = min(len(ids) - 1, WIKI_BLOCKS * BLOCK_SIZE)
    records = []
    for block_id, start in enumerate(range(0, usable, BLOCK_SIZE)):
        chunk = ids[start : start + BLOCK_SIZE + 1]
        if len(chunk) < 64:
            continue
        x = torch.tensor([chunk[:-1]], dtype=torch.long)
        y = torch.tensor([chunk[1:]], dtype=torch.long)
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        records.append({"example_id": block_id, "loss": float(loss)})
    if not records:
        raise RuntimeError("empty WikiText evaluation")
    return records


def prepare_target_sequence(tokenizer: Any, prompt: str, target: str) -> tuple[list[int], int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not target_ids:
        target_ids = [tokenizer.eos_token_id]
    target_ids = target_ids[: min(256, MAX_LENGTH - 1)]
    max_prompt = MAX_LENGTH - len(target_ids)
    prompt_ids = prompt_ids[-max_prompt:] if max_prompt > 0 else []
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id]
    return prompt_ids + target_ids, len(prompt_ids)


@torch.inference_mode()
def evaluate_targets(model: nn.Module, tokenizer: Any, examples: list[dict[str, str]]) -> list[dict[str, Any]]:
    prepared = [prepare_target_sequence(tokenizer, row["prompt"], row["target"]) for row in examples]
    records = []
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for batch_start in range(0, len(prepared), BATCH_SIZE):
        batch_items = prepared[batch_start : batch_start + BATCH_SIZE]
        max_len = max(len(sequence) for sequence, _ in batch_items)
        input_ids = torch.full((len(batch_items), max_len), pad, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, (sequence, _) in enumerate(batch_items):
            input_ids[i, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[i, : len(sequence)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        logp = logits.log_softmax(-1)
        for i, (sequence, target_start) in enumerate(batch_items):
            positions = torch.arange(target_start, len(sequence), dtype=torch.long)
            targets = input_ids[i, positions]
            token_nll = -logp[i, positions - 1, targets]
            records.append({
                "example_id": batch_start + i,
                "loss": float(token_nll.mean()),
                "target_tokens": int(len(positions)),
            })
    return records


@torch.inference_mode()
def evaluate_mcq(model: nn.Module, tokenizer: Any, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for example_id, row in enumerate(examples):
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            prompt_ids = [tokenizer.eos_token_id]
        for choice_id, choice in enumerate(row["choices"]):
            choice_ids = tokenizer(" " + str(choice), add_special_tokens=False)["input_ids"]
            if not choice_ids:
                choice_ids = [tokenizer.eos_token_id]
            choice_ids = choice_ids[: min(256, MAX_LENGTH - 1)]
            prompt_trim = prompt_ids[-(MAX_LENGTH - len(choice_ids)) :]
            flat.append({
                "example_id": example_id,
                "choice_id": choice_id,
                "sequence": prompt_trim + choice_ids,
                "start": len(prompt_trim),
            })

    scores: dict[int, dict[int, float]] = {i: {} for i in range(len(examples))}
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for batch_start in range(0, len(flat), BATCH_SIZE):
        items = flat[batch_start : batch_start + BATCH_SIZE]
        max_len = max(len(item["sequence"]) for item in items)
        input_ids = torch.full((len(items), max_len), pad, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, item in enumerate(items):
            input_ids[i, : len(item["sequence"])] = torch.tensor(item["sequence"], dtype=torch.long)
            attention_mask[i, : len(item["sequence"])] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        logp = logits.log_softmax(-1)
        for i, item in enumerate(items):
            positions = torch.arange(item["start"], len(item["sequence"]), dtype=torch.long)
            targets = input_ids[i, positions]
            sequence_nll = -logp[i, positions - 1, targets].sum()
            scores[item["example_id"]][item["choice_id"]] = float(sequence_nll)

    records = []
    for example_id, row in enumerate(examples):
        nlls = np.array([scores[example_id][i] for i in range(len(row["choices"]))], dtype=np.float64)
        logits = -nlls
        maximum = float(logits.max())
        log_partition = maximum + math.log(float(np.exp(logits - maximum).sum()))
        label = int(row["label"])
        prediction = int(np.argmin(nlls))
        records.append({
            "example_id": example_id,
            "loss": float(nlls[label] + log_partition),
            "correct": int(prediction == label),
            "prediction": prediction,
            "label": label,
        })
    return records


def summarize(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result = {}
    for domain, rows in records.items():
        entry = {"n": len(rows), "loss": float(np.mean([row["loss"] for row in rows]))}
        if "correct" in rows[0]:
            entry["accuracy"] = float(np.mean([row["correct"] for row in rows]))
        result[domain] = entry
    result["balanced_loss"] = float(np.mean([entry["loss"] for entry in result.values() if isinstance(entry, dict)]))
    return result


def bootstrap_composite(candidate: dict[str, list[dict[str, Any]]], parent: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAPS, dtype=np.float64)
    domains = list(candidate)
    for b in range(BOOTSTRAPS):
        relative = []
        for domain in domains:
            c = np.array([row["loss"] for row in candidate[domain]], dtype=np.float64)
            p = np.array([row["loss"] for row in parent[domain]], dtype=np.float64)
            index = rng.integers(0, len(c), len(c))
            relative.append(c[index].mean() / p[index].mean() - 1.0)
        values[b] = float(np.mean(relative))
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "wins_fraction": float(np.mean(values < 0.0)),
    }


def tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def parent_parameter_hashes(models: dict[str, nn.Module]) -> dict[str, set[tuple[tuple[int, ...], str]]]:
    result = {}
    for name, model in models.items():
        values = set()
        for _, parameter in model.named_parameters():
            values.add((tuple(parameter.shape), tensor_hash(parameter.float())))
        result[name] = values
    return result


def exact_copy_audit(model: nn.Module, parent_hashes: dict[str, set[tuple[tuple[int, ...], str]]]) -> dict[str, Any]:
    copies = []
    for name, parameter in model.named_parameters():
        signature = (tuple(parameter.shape), tensor_hash(parameter.float()))
        for parent_name, hashes in parent_hashes.items():
            if signature in hashes:
                copies.append({"new_parameter": name, "parent": parent_name, "shape": list(parameter.shape)})
    return {"exact_copy_count": len(copies), "copies": copies[:100]}


def permutation_gauge(model: nn.Module) -> dict[str, Any]:
    d = int(model.config.hidden_size)
    energy = torch.zeros(d, dtype=torch.float64)
    signed_anchor = model.model.embed_tokens.weight.detach().double().sum(dim=0)
    energy += model.model.embed_tokens.weight.detach().double().square().sum(dim=0)

    for split_layer in model.model.layers:
        for bank in [split_layer.layer_a, split_layer.layer_b]:
            for projection in [bank.self_attn.q_proj, bank.self_attn.k_proj, bank.self_attn.v_proj, bank.mlp.gate_proj, bank.mlp.up_proj]:
                energy += projection.weight.detach().double().square().sum(dim=0)
            for projection in [bank.self_attn.o_proj, bank.mlp.down_proj]:
                energy += projection.weight.detach().double().square().sum(dim=1)

    permutation = torch.argsort(energy, descending=True)
    signs = torch.where(signed_anchor[permutation] >= 0, 1.0, -1.0).float()
    identity = torch.arange(d)
    if torch.equal(permutation.cpu(), identity):
        permutation = torch.roll(permutation, shifts=1)
        signs = signs[torch.roll(identity, shifts=1)]

    with torch.no_grad():
        embed = model.model.embed_tokens.weight.data.float()
        transformed_embed = embed[:, permutation] * signs
        model.model.embed_tokens.weight = nn.Parameter(transformed_embed.to(embed.dtype))
        # Keep tied embedding/head exactly tied.
        model.lm_head.weight = model.model.embed_tokens.weight

        model.model.norm.weight.data = model.model.norm.weight.data[permutation].clone()
        for split_layer in model.model.layers:
            for bank in [split_layer.layer_a, split_layer.layer_b]:
                bank.input_layernorm.weight.data = bank.input_layernorm.weight.data[permutation].clone()
                bank.post_attention_layernorm.weight.data = bank.post_attention_layernorm.weight.data[permutation].clone()
                for projection in [bank.self_attn.q_proj, bank.self_attn.k_proj, bank.self_attn.v_proj, bank.mlp.gate_proj, bank.mlp.up_proj]:
                    weight = projection.weight.data
                    projection.weight.data = (weight[:, permutation] * signs.to(weight.dtype)).contiguous()
                for projection in [bank.self_attn.o_proj, bank.mlp.down_proj]:
                    weight = projection.weight.data
                    projection.weight.data = (weight[permutation, :] * signs.to(weight.dtype)[:, None]).contiguous()

    return {
        "permutation_is_identity": bool(torch.equal(permutation.cpu(), identity)),
        "moved_coordinates": int((permutation.cpu() != identity).sum()),
        "negative_signs": int((signs < 0).sum()),
        "energy_max": float(energy.max()),
        "energy_min": float(energy.min()),
        "permutation": permutation.cpu().tolist(),
        "signs": signs.cpu().tolist(),
    }


@torch.inference_mode()
def prompt_logits(model: nn.Module, tokenizer: Any, prompts: list[str]) -> list[torch.Tensor]:
    outputs = []
    for prompt in prompts:
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        outputs.append(model(input_ids=ids, use_cache=False).logits.detach().cpu().float())
    return outputs


def compare_logits(a: list[torch.Tensor], b: list[torch.Tensor]) -> dict[str, float]:
    delta = torch.cat([(x - y).reshape(-1) for x, y in zip(a, b)])
    reference = torch.cat([x.reshape(-1) for x in a])
    rms = delta.square().mean().sqrt()
    return {
        "max_abs": float(delta.abs().max()),
        "rms": float(rms),
        "relative_rms": float(rms / (reference.square().mean().sqrt() + 1e-30)),
    }


def finite_audit(model: nn.Module) -> dict[str, Any]:
    failures = []
    with torch.inference_mode():
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter.detach()).all().item()):
                failures.append(name)
    return {"all_finite": not failures, "nonfinite_parameters": failures}


def evaluate_all(model: nn.Module, tokenizer: Any, datasets: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], float]:
    start = time.time()
    records = {
        "wikitext": evaluate_wiki(model, tokenizer, datasets["wikitext"]),
        "gsm8k": evaluate_targets(model, tokenizer, datasets["gsm8k"]),
        "mbpp": evaluate_targets(model, tokenizer, datasets["mbpp"]),
        "openbookqa": evaluate_mcq(model, tokenizer, datasets["openbookqa"]),
        "piqa": evaluate_mcq(model, tokenizer, datasets["piqa"]),
    }
    return records, time.time() - start


def main() -> None:
    base_config = AutoConfig.from_pretrained(BASE_ID)
    math_config = AutoConfig.from_pretrained(MATH_ID)
    code_config = AutoConfig.from_pretrained(CODE_ID, subfolder=CODE_SUBFOLDER)
    configs = {
        "base": structural_config(base_config),
        "math": structural_config(math_config),
        "code": structural_config(code_config),
    }
    if configs["base"] != configs["math"] or configs["base"] != configs["code"]:
        raise RuntimeError({"status": "STRUCTURAL_CONFIG_MISMATCH", "configs": configs})

    base_tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    math_tokenizer = AutoTokenizer.from_pretrained(MATH_ID)
    code_tokenizer = AutoTokenizer.from_pretrained(CODE_ID)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    tokenizers = {
        "math": tokenizer_audit(base_tokenizer, math_tokenizer),
        "code": tokenizer_audit(base_tokenizer, code_tokenizer),
    }
    if not all(entry["vocab_equal"] and entry["probe_ids_equal"] for entry in tokenizers.values()):
        raise RuntimeError({"status": "TOKENIZER_MISMATCH", "audit": tokenizers})

    print("Loading public common-base specialists...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    math_model = AutoModelForCausalLM.from_pretrained(MATH_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    code_model = AutoModelForCausalLM.from_pretrained(CODE_ID, subfolder=CODE_SUBFOLDER, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()

    base_keys = {name: tuple(tensor.shape) for name, tensor in base_model.state_dict().items()}
    math_keys = {name: tuple(tensor.shape) for name, tensor in math_model.state_dict().items()}
    code_keys = {name: tuple(tensor.shape) for name, tensor in code_model.state_dict().items()}
    if base_keys != math_keys or base_keys != code_keys:
        raise RuntimeError({"status": "STATE_SHAPE_MISMATCH"})

    parent_param_count = int(sum(parameter.numel() for parameter in base_model.parameters()))
    datasets = {
        "wikitext": build_wikitext(),
        "gsm8k": build_gsm8k(),
        "mbpp": build_mbpp(),
        "openbookqa": build_openbookqa(),
        "piqa": build_piqa(),
    }

    parent_hashes = parent_parameter_hashes({"base": base_model, "math": math_model, "code": code_model})

    records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    timings: dict[str, float] = {}
    for name, model in [("parent_base", base_model), ("parent_math", math_model), ("parent_code", code_model)]:
        model_records, elapsed = evaluate_all(model, base_tokenizer, datasets)
        records[name] = model_records
        summaries[name] = summarize(model_records)
        timings[name] = elapsed
        print(name, json.dumps(summaries[name], indent=2))

    coefficient_audit = []
    split_layers = []
    head_dim = int(getattr(base_config, "head_dim", base_config.hidden_size // base_config.num_attention_heads))
    for layer_index, (math_layer, code_layer) in enumerate(zip(math_model.model.layers, code_model.model.layers)):
        coeff_math, coeff_code, audit = gain_coefficients(math_layer, code_layer, head_dim)
        audit["layer"] = layer_index
        audit["order"] = "math_then_code" if layer_index % 2 == 0 else "code_then_math"
        coefficient_audit.append(audit)
        split_layers.append(SplitDecoderLayer(
            layer_a=math_layer,
            layer_b=code_layer,
            coeff_a=coeff_math,
            coeff_b=coeff_code,
            a_first=(layer_index % 2 == 0),
        ))

    # Base supplies one tokenizer shell, one embedding, one final norm, and one tied LM head.
    base_model.model.layers = nn.ModuleList(split_layers)
    split_model = base_model.eval()
    split_param_count = int(sum(parameter.numel() for parameter in split_model.parameters()))
    parameter_ratio = split_param_count / parent_param_count

    prompts = [
        "The capital of France is", "Question: 19 + 23 =", "def fibonacci(n):",
        "A metal conducts electricity because", "Goal: open a jar\nBest solution:",
    ]
    logits_before_gauge = prompt_logits(split_model, base_tokenizer, prompts)
    gauge_audit = permutation_gauge(split_model)
    logits_after_gauge = prompt_logits(split_model, base_tokenizer, prompts)
    gauge_logit_audit = compare_logits(logits_before_gauge, logits_after_gauge)

    # Drop wrappers; their layer modules are now owned by split_model.
    math_model.model.layers = nn.ModuleList([])
    code_model.model.layers = nn.ModuleList([])
    del math_model, code_model
    gc.collect()

    split_records, split_elapsed = evaluate_all(split_model, base_tokenizer, datasets)
    records["operator_split"] = split_records
    summaries["operator_split"] = summarize(split_records)
    timings["operator_split"] = split_elapsed

    finite = finite_audit(split_model)
    copy_audit = exact_copy_audit(split_model, parent_hashes)

    parents = ["parent_base", "parent_math", "parent_code"]
    comparisons = {
        parent: bootstrap_composite(records["operator_split"], records[parent], 1000 + index)
        for index, parent in enumerate(parents)
    }
    domains = list(split_records)
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
        summaries["operator_split"][domain]["loss"] <= 1.03 * best_parent_loss[domain]
        for domain in domains
    )
    accuracy_within_5pp = all(
        summaries["operator_split"][domain]["accuracy"] >= best_parent_accuracy[domain] - 0.05
        for domain in mcq_domains
    )
    composite_beats_all = all(entry["ci95"][1] < 0.0 for entry in comparisons.values())
    structural_pass = bool(
        parameter_ratio <= 1.8
        and gauge_logit_audit["relative_rms"] < 1e-5
        and copy_audit["exact_copy_count"] == 0
        and finite["all_finite"]
        and hasattr(split_model, "lm_head")
        and split_model.config.tie_word_embeddings
    )
    promoted = bool(loss_within_3pct and accuracy_within_5pp and composite_beats_all and structural_pass)

    speed_ratio_vs_fastest_parent = split_elapsed / min(timings[parent] for parent in parents)
    speed_ratio_vs_mean_parent = split_elapsed / float(np.mean([timings[parent] for parent in parents]))

    result = {
        "status": "OPERATOR_SPLITTING_PASS" if promoted else "OPERATOR_SPLITTING_NOT_PROMOTED",
        "method": {
            "name": "single_state_alternating_residual_operator_splitting",
            "definition": "F_i(h)=L_i(h)-h; h<-h+c_i F_i(h), alternating math/code order by layer",
            "coefficient_rule": "c_math*g_math=c_code*g_code and c_math+c_code=1 using isotropic weight-only second-moment gain",
            "training": False,
            "router": False,
            "parallel_hidden_streams": False,
            "probability_mixture": False,
        },
        "models": {"base": BASE_ID, "math": MATH_ID, "code": CODE_ID + "/" + CODE_SUBFOLDER},
        "config_audit": configs,
        "tokenizer_audit": tokenizers,
        "state_shapes_equal": True,
        "evaluation": {
            "wiki_blocks": len(records["operator_split"]["wikitext"]),
            "gsm8k_examples": len(records["operator_split"]["gsm8k"]),
            "mbpp_examples": len(records["operator_split"]["mbpp"]),
            "openbookqa_examples": len(records["operator_split"]["openbookqa"]),
            "piqa_examples": len(records["operator_split"]["piqa"]),
            "mcq_metric": "choice-normalized total answer-sequence cross entropy",
            "bootstrap_resamples": BOOTSTRAPS,
        },
        "structure": {
            "parent_parameter_count": parent_param_count,
            "new_parameter_count": split_param_count,
            "parameter_ratio": parameter_ratio,
            "single_embedding": True,
            "single_tied_lm_head": True,
            "single_hidden_stream": True,
            "two_operator_banks": True,
            "coefficient_audit": coefficient_audit,
            "gauge_audit": gauge_audit,
            "gauge_logit_audit": gauge_logit_audit,
            "exact_copy_audit": copy_audit,
            "finite_audit": finite,
        },
        "summaries": summaries,
        "timings_seconds": timings,
        "speed_ratio_vs_fastest_parent": speed_ratio_vs_fastest_parent,
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
        writer = csv.DictWriter(handle, fieldnames=["model", "domain", "example_id", "loss", "correct", "prediction", "label", "target_tokens"])
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
        "# Real SmolLM2 Specialist Operator-Splitting Validation",
        "",
        f"Status: **{result['status']}**",
        f"Parameter ratio: **{parameter_ratio:.6f}x**",
        f"Gauge relative RMS logit error: **{gauge_logit_audit['relative_rms']:.8g}**",
        f"Exact copied parameters after gauge: **{copy_audit['exact_copy_count']}**",
        f"All losses within 3% of the per-domain best parent: **{loss_within_3pct}**",
        f"All MCQ accuracies within 5pp: **{accuracy_within_5pp}**",
        f"Composite significantly beats base, math, and code parents: **{composite_beats_all}**",
        f"Speed ratio vs mean parent: **{speed_ratio_vs_mean_parent:.4f}x**",
        "",
        "The candidate is frozen by the stated weight-only gain rule. No coefficient search, labels, gradients, router, or logit mixture are used.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    if promoted:
        # Save only on a strict pass; otherwise no misleading checkpoint is emitted.
        checkpoint = {
            "state_dict": {key: value.detach().cpu().to(torch.bfloat16) for key, value in split_model.state_dict().items()},
            "base_config": base_config.to_dict(),
            "method": result["method"],
            "coefficient_audit": coefficient_audit,
            "gauge_audit": gauge_audit,
        }
        torch.save(checkpoint, ROOT / "OPERATOR_SPLIT_SINGLE_CHECKPOINT.pt")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
