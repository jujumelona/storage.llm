from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "dual_prefix_v16", HERE / "real_smollm2_v16_dual_prefix_selector.py"
)
v16 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v16)

ROOT = Path("out/real_smollm2_v17_observable_instance_selector")
ROOT.mkdir(parents=True, exist_ok=True)
v16.ROOT = ROOT


def observable_instance_text(row: dict[str, Any]) -> str:
    text = str(row["feature_text"])
    choices = row.get("choices")
    if choices:
        rendered = "\n".join(
            f"Choice {index + 1}: {choice}" for index, choice in enumerate(choices)
        )
        text += "\n\nObservable answer choices:\n" + rendered
    return text


def tokenized_feature_inputs(tokenizer: Any, datasets: dict[str, list[dict[str, Any]]]):
    rows, keys = v16.ordered_rows(datasets)
    sequences = []
    for row in rows:
        ids = tokenizer(
            observable_instance_text(row),
            add_special_tokens=False,
            truncation=True,
            max_length=v16.v14.MAX_FEATURE_TOKENS,
        )["input_ids"]
        if not ids:
            ids = [tokenizer.eos_token_id]
        sequences.append(ids)
    return sequences, keys


v16.tokenized_feature_inputs = tokenized_feature_inputs


if __name__ == "__main__":
    v16.main()
    results_path = ROOT / "RESULTS.json"
    result = json.loads(results_path.read_text(encoding="utf-8"))
    result["selector_observation"] = {
        "uses_all_inference_observable_text": True,
        "mcq_answer_choices_included": True,
        "target_or_correct_label_included": False,
        "motivation": "Bayes-optimal MCQ route may depend on the candidate set, not only the question prompt",
    }
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = ROOT / "REPORT.md"
    report_path.write_text(
        report_path.read_text(encoding="utf-8")
        + "\n\nMCQ selector features include the complete observable candidate set; correct labels and target answers remain excluded.\n",
        encoding="utf-8",
    )
