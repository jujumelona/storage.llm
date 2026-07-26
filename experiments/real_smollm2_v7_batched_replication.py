from __future__ import annotations

import csv
import gc
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v1 = load_module("wf_v1", "real_smollm2_weight_field.py")
v5 = load_module("wf_v5", "real_smollm2_v5_strict.py")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260726)
np.random.seed(20260726)

ROOT = Path("out/real_smollm2_v7_batched_replication")
ROOT.mkdir(parents=True, exist_ok=True)
BASE_ID = v1.BASE_ID
INSTRUCT_ID = v1.INSTRUCT_ID
N = 96
BOOTSTRAPS = 5000
BATCH_SEQUENCES = 16
v1.WIKI_BLOCKS = 16
v1.BLOCK_SIZE = 256


def chat_prompt(tokenizer, content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def deterministic_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    if len(rows) < n:
        raise RuntimeError({"available": len(rows), "required": n})
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(rows), size=n, replace=False))
    return [rows[int(i)] for i in indices]


def openbookqa_rows(tokenizer):
    ds = load_dataset("allenai/openbookqa", "main", split="validation")
    rows = []
    for row in ds:
        labels = [str(x) for x in row["choices"]["label"]]
        texts = list(row["choices"]["text"])
        answer = str(row["answerKey"])
        mapping = {label: i for i, label in enumerate(labels)}
        if answer not in mapping:
            continue
        letters = [chr(ord("A") + i) for i in range(len(texts))]
        content = "Answer the science multiple-choice question with only the option letter.\n\n"
        content += "Question: " + row["question_stem"] + "\n"
        content += "\n".join(f"{letters[i]}. {text}" for i, text in enumerate(texts))
        content += "\nAnswer:"
        rows.append({"prompt": chat_prompt(tokenizer, content), "choices": letters, "label": mapping[answer]})
    return rows


def commonsenseqa_rows(tokenizer):
    ds = load_dataset("tau/commonsense_qa", split="validation")
    rows = []
    for row in ds:
        labels = [str(x) for x in row["choices"]["label"]]
        texts = list(row["choices"]["text"])
        answer = str(row["answerKey"])
        mapping = {label: i for i, label in enumerate(labels)}
        if answer not in mapping:
            continue
        letters = [chr(ord("A") + i) for i in range(len(texts))]
        content = "Answer the commonsense multiple-choice question with only the option letter.\n\n"
        content += "Question: " + row["question"] + "\n"
        content += "\n".join(f"{letters[i]}. {text}" for i, text in enumerate(texts))
        content += "\nAnswer:"
        rows.append({"prompt": chat_prompt(tokenizer, content), "choices": letters, "label": mapping[answer]})
    return rows


def winogrande_rows(tokenizer):
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    rows = []
    for row in ds:
        answer = int(row["answer"]) - 1
        if answer not in (0, 1):
            continue
        sentence = row["sentence"]
        content = "Choose the option that correctly fills the blank. Answer with only A or B.\n\n"
        content += f"Sentence: {sentence}\nA. {row['option1']}\nB. {row['option2']}\nAnswer:"
        rows.append({"prompt": chat_prompt(tokenizer, content), "choices": ["A", "B"], "label": answer})
    return rows


def piqa_rows(tokenizer):
    ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    rows = []
    for row in ds:
        label = int(row["label"])
        if label not in (0, 1):
            continue
        content = "Choose the better physical solution. Answer with only A or B.\n\n"
        content += f"Goal: {row['goal']}\nA. {row['sol1']}\nB. {row['sol2']}\nAnswer:"
        rows.append({"prompt": chat_prompt(tokenizer, content), "choices": ["A", "B"], "label": label})
    return rows


def boolq_rows(tokenizer):
    try:
        ds = load_dataset("google/boolq", split="validation")
    except Exception:
        ds = load_dataset("super_glue", "boolq", split="validation")
    rows = []
    for row in ds:
        content = (
            "Read the passage and answer with only yes or no.\n\n"
            f"Passage: {row['passage']}\n\nQuestion: {row['question']}\nAnswer:"
        )
        rows.append({"prompt": chat_prompt(tokenizer, content), "choices": ["yes", "no"], "label": 0 if bool(row["answer"]) else 1})
    return rows


@torch.inference_mode()
def evaluate_mcq_batched(model, tokenizer, examples: list[dict], batch_sequences: int = BATCH_SEQUENCES):
    flat = []
    for example_id, row in enumerate(examples):
        prompt_ids = v1.encode_no_special(tokenizer, row["prompt"])
        if not prompt_ids:
            prompt_ids = [tokenizer.eos_token_id]
        for choice_id, choice in enumerate(row["choices"]):
            option_ids = v1.encode_no_special(tokenizer, " " + choice)
            if not option_ids:
                option_ids = [tokenizer.eos_token_id]
            flat.append({
                "example_id": example_id,
                "choice_id": choice_id,
                "sequence": prompt_ids + option_ids,
                "start": len(prompt_ids),
            })

    scores = {i: {} for i in range(len(examples))}
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for start_batch in range(0, len(flat), batch_sequences):
        batch_items = flat[start_batch : start_batch + batch_sequences]
        max_len = max(len(item["sequence"]) for item in batch_items)
        input_ids = torch.full((len(batch_items), max_len), pad, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, item in enumerate(batch_items):
            seq = item["sequence"]
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, : len(seq)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        logp = logits.log_softmax(-1)
        for i, item in enumerate(batch_items):
            seq = item["sequence"]
            token_positions = torch.arange(item["start"], len(seq), dtype=torch.long)
            pred_positions = token_positions - 1
            targets = input_ids[i, token_positions]
            nll = -logp[i, pred_positions, targets].mean()
            scores[item["example_id"]][item["choice_id"]] = float(nll)

    records = []
    for example_id, row in enumerate(examples):
        nlls = [scores[example_id][i] for i in range(len(row["choices"]))]
        prediction = int(np.argmin(nlls))
        records.append({
            "example_id": example_id,
            "correct_nll": float(nlls[row["label"]]),
            "correct": int(prediction == row["label"]),
            "prediction": prediction,
            "label": int(row["label"]),
        })
    return records


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
    candidate_state, _ = v5.fixed_candidate_state(state_base, state_instruct)
    model_candidate = v1.load_from_state(config, candidate_state)

    datasets = {
        "openbookqa_chat": deterministic_sample(openbookqa_rows(chat_tokenizer), N, 101),
        "commonsenseqa_chat": deterministic_sample(commonsenseqa_rows(chat_tokenizer), N, 102),
        "winogrande_chat": deterministic_sample(winogrande_rows(chat_tokenizer), N, 103),
        "piqa_chat": deterministic_sample(piqa_rows(chat_tokenizer), N, 104),
        "boolq_chat": deterministic_sample(boolq_rows(chat_tokenizer), N, 105),
    }
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wiki_texts = [row["text"] for row in wiki if row["text"].strip()][4400:5200]

    models = {"parent_base": model_base, "parent_instruct": model_instruct, "fixed_candidate": model_candidate}
    all_records = {}
    summaries = {}
    for name, model in models.items():
        start = time.time()
        records = {"wikitext": v5.evaluate_wiki_blocks(model, tokenizer, wiki_texts)}
        for domain, rows in datasets.items():
            records[domain] = evaluate_mcq_batched(model, tokenizer, rows)
        all_records[name] = records
        summaries[name] = v5.summarize_records(records)
        summaries[name]["seconds"] = time.time() - start
        print(name, json.dumps(summaries[name], indent=2))
        (ROOT / "PARTIAL.json").write_text(json.dumps({"summaries": summaries}, indent=2), encoding="utf-8")

    rng = np.random.default_rng(20260726)
    comparisons = {}
    for parent_name in ["parent_base", "parent_instruct"]:
        entry = {"composite_relative_nll": v5.bootstrap_composite(all_records["fixed_candidate"], all_records[parent_name], rng, BOOTSTRAPS)}
        entry["domains"] = {domain: v5.bootstrap_domain(all_records["fixed_candidate"][domain], all_records[parent_name][domain], domain, rng, BOOTSTRAPS) for domain in all_records["fixed_candidate"]}
        comparisons[parent_name] = entry

    domains = list(all_records["fixed_candidate"])
    mcq_domains = [domain for domain in domains if domain != "wikitext"]
    best_nll = {d: min(summaries["parent_base"][d]["nll"], summaries["parent_instruct"][d]["nll"]) for d in domains}
    best_acc = {d: max(summaries["parent_base"][d]["accuracy"], summaries["parent_instruct"][d]["accuracy"]) for d in mcq_domains}
    no_nll_regression = all(summaries["fixed_candidate"][d]["nll"] <= 1.03 * best_nll[d] for d in domains)
    no_acc_regression = all(summaries["fixed_candidate"][d]["accuracy"] >= best_acc[d] - 0.05 for d in mcq_domains)
    composite_significant = all(comparisons[p]["composite_relative_nll"]["ci95"][1] < 0 for p in comparisons)

    prompts = ["An ecosystem contains", "The square root of 81 is", "Question: 17 + 24 =", "def quicksort(items):", chat_prompt(chat_tokenizer, "Name one property of iron.")]
    rotated_state, rotated_config, rotation_audit = v5.canonical_rotation(candidate_state, config)
    model_rotated = AutoModelForCausalLM.from_config(rotated_config, torch_dtype=torch.float32)
    missing, unexpected = model_rotated.load_state_dict(rotated_state, strict=False)
    if missing or unexpected:
        raise RuntimeError({"missing": missing, "unexpected": unexpected})
    model_rotated.eval()
    rotation_logits = v5.compare_model_logits(model_candidate, model_rotated, tokenizer, prompts)
    copy_audit = v5.exact_copy_audit(rotated_state, {"base": state_base, "instruct": state_instruct})
    parent_params = int(sum(p.numel() for p in model_base.parameters()))
    rotated_params = int(sum(t.numel() for t in rotated_state.values()))
    ratio = rotated_params / parent_params
    rotation_pass = rotation_logits["relative_rms"] < 1e-5 and copy_audit["exact_copy_count"] == 0 and ratio <= 1.8
    promoted = bool(no_nll_regression and no_acc_regression and composite_significant and rotation_pass)

    if promoted:
        model_dir = ROOT / "CANONICAL_ROTATED_MODEL"
        model_rotated.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        reloaded = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, local_files_only=True).eval()
        reload_audit = v5.compare_model_logits(model_rotated, reloaded, tokenizer, prompts)
    else:
        reload_audit = None

    result = {
        "status": "BATCHED_CROSS_BENCHMARK_PASS" if promoted else "BATCHED_CROSS_BENCHMARK_NOT_PROMOTED",
        "candidate_frozen": True,
        "sample_selection_frozen": True,
        "evaluation": {"n_per_mcq_domain": N, "domains": list(datasets), "wikitext_blocks": len(all_records["fixed_candidate"]["wikitext"]), "batch_sequences": BATCH_SEQUENCES, "bootstrap_resamples": BOOTSTRAPS},
        "config_audit": config_audit,
        "summaries": summaries,
        "comparisons": comparisons,
        "best_parent_nll": best_nll,
        "best_parent_accuracy": best_acc,
        "no_domain_nll_regression_over_3pct": no_nll_regression,
        "no_mcq_accuracy_regression_over_5pp": no_acc_regression,
        "composite_significantly_beats_both_parents": composite_significant,
        "canonical_rotation": {**rotation_audit, "logit_audit": rotation_logits, "exact_copy_audit": copy_audit, "parameter_ratio": ratio, "reload_audit": reload_audit, "pass": rotation_pass},
        "promoted": promoted,
    }
    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (ROOT / "EXAMPLE_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "domain", "example_id", "nll", "correct_nll", "correct", "prediction", "label"])
        writer.writeheader()
        for model_name, domain_map in all_records.items():
            for domain, rows in domain_map.items():
                for row in rows:
                    writer.writerow({"model": model_name, "domain": domain, **row})

    (ROOT / "REPORT.md").write_text("\n".join([
        "# Batched Cross-Benchmark Replication",
        "",
        f"Status: **{result['status']}**",
        f"Composite significant vs both parents: **{composite_significant}**",
        f"All NLL domains within 3%: **{no_nll_regression}**",
        f"All accuracies within 5pp: **{no_acc_regression}**",
        f"Rotation relative RMS: **{rotation_logits['relative_rms']:.8g}**",
        f"Exact copied tensors: **{copy_audit['exact_copy_count']}**",
        f"Parameter ratio: **{ratio:.6f}x**",
    ]), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
