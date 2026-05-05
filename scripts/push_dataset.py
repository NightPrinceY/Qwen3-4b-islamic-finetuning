"""
Push Islamic Arabic Q&A dataset to HuggingFace Hub.

Converts JSONL → Parquet with both chat (messages) and flat (question/answer)
columns so the HF data viewer works properly.

Run:
    python scripts/push_dataset.py
"""
import json
import os
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Sequence, Value

SPLITS_DIR  = Path("data/splits")
HF_REPO     = "NightPrince/islamic-arabic-qa"
HF_TOKEN    = os.environ["HF_TOKEN"]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def flatten(rows: list[dict]) -> dict:
    messages_col, system_col, question_col, answer_col = [], [], [], []
    for row in rows:
        msgs = row["messages"]
        system   = next((m["content"] for m in msgs if m["role"] == "system"),    "")
        question = next((m["content"] for m in msgs if m["role"] == "user"),      "")
        answer   = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        messages_col.append(msgs)
        system_col.append(system)
        question_col.append(question)
        answer_col.append(answer)
    return {
        "messages": messages_col,
        "system":   system_col,
        "question": question_col,
        "answer":   answer_col,
    }


def main():
    print("Loading splits ...")
    splits = {
        "train":      load_jsonl(SPLITS_DIR / "train.jsonl"),
        "validation": load_jsonl(SPLITS_DIR / "val.jsonl"),
        "test":       load_jsonl(SPLITS_DIR / "test.jsonl"),
    }

    print("Converting to HF datasets ...")
    dataset_dict = DatasetDict({
        name: Dataset.from_dict(flatten(rows))
        for name, rows in splits.items()
    })

    print(dataset_dict)

    print(f"\nPushing to {HF_REPO} ...")
    dataset_dict.push_to_hub(
        HF_REPO,
        token=HF_TOKEN,
        private=False,
    )
    print("Dataset pushed.")


if __name__ == "__main__":
    main()
