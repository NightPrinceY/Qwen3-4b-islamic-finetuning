"""
Step 1 — Load raw data, clean, format into chat template, and split.

Input:  data/raw/*.json
Output: data/splits/train.jsonl, val.jsonl, test.jsonl
"""
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a knowledgeable Islamic scholar assistant. "
    "Answer questions accurately based on Quran, Sunnah, and classical Islamic scholarship. "
    "Cite sources where possible. Be concise but thorough."
)

SPLITS = {"train": 0.85, "val": 0.10, "test": 0.05}
SEED = 42


def format_sample(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def load_raw(raw_dir: Path) -> list[dict]:
    samples = []
    for path in raw_dir.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        for item in data:
            samples.append(format_sample(item["question"], item["answer"]))
    return samples


def split_and_save(samples: list[dict], splits_dir: Path):
    random.seed(SEED)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    subsets = {
        "train": samples[:n_train],
        "val": samples[n_train : n_train + n_val],
        "test": samples[n_train + n_val :],
    }

    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in subsets.items():
        path = splits_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for sample in subset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"{name}: {len(subset)} samples → {path}")


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    splits_dir = Path("data/splits")

    samples = load_raw(raw_dir)
    print(f"Loaded {len(samples)} total samples")

    split_and_save(samples, splits_dir)
    print("Done.")
