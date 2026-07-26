from __future__ import annotations

from pathlib import Path

source_path = Path(__file__).with_name("real_smollm2_v9_operator_splitting.py")
source = source_path.read_text(encoding="utf-8")

source = source.replace(
    'ROOT = Path("out/real_smollm2_v9_operator_splitting")',
    'ROOT = Path("out/real_smollm2_v9_operator_splitting_sql")',
)
source = source.replace(
    'CODE_ID = "lhoestq/finetune_smollm2_python"',
    'CODE_ID = "Ellight/code-smolLM2-135m-text-to-sql"',
)
source = source.replace(
    'CODE_SUBFOLDER = "final_checkpoint"',
    'CODE_SUBFOLDER = ""',
)
source = source.replace("build_mbpp", "build_text2sql")
source = source.replace('"mbpp"', '"text2sql"')
source = source.replace("mbpp_examples", "text2sql_examples")

start = source.index("def build_text2sql()")
end = source.index("\ndef build_openbookqa", start)
replacement = '''def build_text2sql() -> list[dict[str, str]]:
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    rows = []
    for row in dataset:
        question = str(row.get("question") or "").strip()
        context = str(row.get("context") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not context or not answer:
            continue
        prompt = (
            "Database schema:\\n" + context
            + "\\n\\nQuestion: " + question
            + "\\nSQL query:\\n"
        )
        rows.append({"prompt": prompt, "target": answer})
    return deterministic_sample(rows, N_GEN, 902)

'''
source = source[:start] + replacement + source[end + 1 :]

# An empty subfolder is equivalent to repository root for HF APIs.
namespace = {"__name__": "__main__", "__file__": str(source_path)}
exec(compile(source, str(source_path), "exec"), namespace)
