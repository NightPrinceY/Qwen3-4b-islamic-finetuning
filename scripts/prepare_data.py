"""
Step 1 — Load, clean, merge, and split all datasets.

Training sources:
  - SahmBenchmark/fatwa-training_standardized_new  (HuggingFace, Apache 2.0)
  - majdelhaj/ar-qa-dataset  islamic-fiqh + islamic-misc  (GitHub, TSV)
  - Yasbok/Alpaca_arabic_instruct  (HuggingFace, small general-Arabic mix)

Evaluation sources (saved separately, never in training):
  - SahmBenchmark/fatwa-qa-evaluation       (open QA eval)
  - SahmBenchmark/fatwa-mcq-evaluation_standardized  (MCQ accuracy eval)
  - QCRI/IslamicFaithQA  Arabic split        (broader Islamic eval)

Outputs:
  data/splits/train.jsonl
  data/splits/val.jsonl
  data/splits/test.jsonl
  data/splits/eval_fatwa_qa.jsonl
  data/splits/eval_fatwa_mcq.jsonl
  data/splits/eval_islamic_faith_qa.jsonl
"""

import json
import random
import re
import urllib.request
from pathlib import Path

from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
SPLITS = {"train": 0.85, "val": 0.10, "test": 0.05}

# How many samples to take from each noisy source
AR_QA_FIQH_LIMIT   = 12_000
AR_QA_MISC_LIMIT   =  4_000
ALPACA_LIMIT       =  2_500

# Minimum answer word count for ar-qa-dataset quality filter
MIN_ANSWER_WORDS = 30

SYSTEM_PROMPT = (
    "أنت مساعد عالم إسلامي متخصص. أجب على الأسئلة بدقة استناداً إلى "
    "القرآن الكريم والسنة النبوية والفقه الإسلامي الكلاسيكي. "
    "استشهد بالمصادر حيثما أمكن. كن موجزاً لكن شاملاً."
)

AR_QA_URLS = {
    "fiqh": "https://raw.githubusercontent.com/majdelhaj/ar-qa-dataset/master/islamic-fiqh-ask-fm.txt",
    "misc": "https://raw.githubusercontent.com/majdelhaj/ar-qa-dataset/master/islamic-misc.txt",
}

SPLITS_DIR  = Path("data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def to_chat(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
    }


def save_jsonl(samples: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  saved {len(samples):,} → {path}")


def word_count(text: str) -> int:
    return len(text.split())


def is_clean(q: str, a: str) -> bool:
    """Basic quality filter — skip empty, too-short, or URL-only answers."""
    if not q or not a:
        return False
    if word_count(a) < MIN_ANSWER_WORDS:
        return False
    if re.fullmatch(r"https?://\S+", a.strip()):
        return False
    return True


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_fatwa_training() -> list[dict]:
    """SahmBenchmark/fatwa-training_standardized_new — already in chat format."""
    print("Loading SahmBenchmark/fatwa-training_standardized_new ...")
    ds = load_dataset("SahmBenchmark/fatwa-training_standardized_new", split="train")
    samples = []
    for row in ds:
        convs = row["conversations"]
        # convs is a list of {role, content} dicts — extract user/assistant pair
        user_turn = next((c["content"] for c in convs if c["role"] == "user"), None)
        asst_turn = next((c["content"] for c in convs if c["role"] == "assistant"), None)
        if user_turn and asst_turn:
            samples.append(to_chat(user_turn, asst_turn))
    print(f"  loaded {len(samples):,} fatwa-training samples")
    return samples


def load_ar_qa(category: str, url: str, limit: int) -> list[dict]:
    """ar-qa-dataset TSV from GitHub — filter by quality then cap at limit."""
    print(f"Downloading ar-qa-dataset/{category} ...")
    raw_path = Path(f"data/raw/ar_qa_{category}.txt")
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, raw_path)
        print(f"  downloaded → {raw_path}")

    samples, skipped = [], 0
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            q, a = parts
            if is_clean(q, a):
                samples.append(to_chat(q, a))
            else:
                skipped += 1

    print(f"  {len(samples):,} kept, {skipped:,} filtered (min {MIN_ANSWER_WORDS} words)")

    random.seed(SEED)
    random.shuffle(samples)
    samples = samples[:limit]
    print(f"  capped at {len(samples):,}")
    return samples


def load_alpaca_arabic(limit: int) -> list[dict]:
    """Yasbok/Alpaca_arabic_instruct — general Arabic instruction mix.
    Downloads the parquet file directly to avoid datasets schema issues."""
    print("Loading Yasbok/Alpaca_arabic_instruct ...")
    import pandas as pd

    parquet_path = Path("data/raw/alpaca_arabic.parquet")
    if not parquet_path.exists():
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://huggingface.co/datasets/Yasbok/Alpaca_arabic_instruct"
            "/resolve/main/data/train-00000-of-00001-10520e8228c2c104.parquet"
        )
        urllib.request.urlretrieve(url, parquet_path)
        print(f"  downloaded → {parquet_path}")

    df = pd.read_parquet(parquet_path)
    samples = []
    for _, row in df.iterrows():
        instruction = str(row.get("instruction", "") or "").strip()
        inp         = str(row.get("input", "")        or "").strip()
        output      = str(row.get("output", "")       or "").strip()
        if not instruction or not output:
            continue
        question = f"{instruction}\n{inp}".strip() if inp else instruction
        samples.append(to_chat(question, output))

    random.seed(SEED)
    random.shuffle(samples)
    samples = samples[:limit]
    print(f"  sampled {len(samples):,} alpaca-arabic samples")
    return samples


# ── Eval loaders (saved as-is, no system prompt injection needed) ─────────────

def load_eval_fatwa_qa() -> list[dict]:
    print("Loading SahmBenchmark/fatwa-qa-evaluation ...")
    ds = load_dataset("SahmBenchmark/fatwa-qa-evaluation", split="test")
    samples = []
    for row in ds:
        samples.append({
            "id":       row["id"],
            "category": row["category"],
            "question": row["question"],
            "answer":   row["answer"],
        })
    print(f"  loaded {len(samples):,} fatwa-qa-eval samples")
    return samples


def load_eval_fatwa_mcq() -> list[dict]:
    print("Loading SahmBenchmark/fatwa-mcq-evaluation_standardized ...")
    ds = load_dataset("SahmBenchmark/fatwa-mcq-evaluation_standardized", split="test")
    samples = []
    for row in ds:
        samples.append({
            "id":                row["id"],
            "category":          row["category"],
            "query":             row["query"],
            "choices":           row["choices"],
            "answer":            row["answer"],
            "gold":              row["gold"],
            "original_question": row["original_question"],
            "original_answer":   row["original_answer"],
        })
    print(f"  loaded {len(samples):,} fatwa-mcq-eval samples")
    return samples


def load_eval_islamic_faith_qa() -> list[dict]:
    print("Loading QCRI/IslamicFaithQA (Arabic) ...")
    ds = load_dataset("QCRI/IslamicFaithQA", "arabic", split="test")
    samples = []
    for row in ds:
        samples.append({
            "id":            row["id"],
            "category":      row.get("category", ""),
            "category_type": row.get("category_type", ""),
            "question":      row["question"],
            "gold_answer":   row["gold_answer"],
            "difficulty":    row.get("difficulty", 0),
        })
    print(f"  loaded {len(samples):,} IslamicFaithQA-Arabic samples")
    return samples


# ── Split & save ──────────────────────────────────────────────────────────────

def split_and_save(samples: list[dict]) -> None:
    random.seed(SEED)
    random.shuffle(samples)

    n       = len(samples)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    subsets = {
        "train": samples[:n_train],
        "val":   samples[n_train : n_train + n_val],
        "test":  samples[n_train + n_val :],
    }

    for name, subset in subsets.items():
        save_jsonl(subset, SPLITS_DIR / f"{name}.jsonl")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("TRAINING DATA")
    print("=" * 60)

    training = []
    training += load_fatwa_training()
    training += load_ar_qa("fiqh", AR_QA_URLS["fiqh"], AR_QA_FIQH_LIMIT)
    training += load_ar_qa("misc", AR_QA_URLS["misc"],  AR_QA_MISC_LIMIT)
    training += load_alpaca_arabic(ALPACA_LIMIT)

    # Deduplicate by first user message
    seen, unique = set(), []
    for s in training:
        key = s["messages"][1]["content"][:120]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    removed = len(training) - len(unique)
    print(f"\nDeduplication: removed {removed:,} duplicates")
    print(f"Final training pool: {len(unique):,} samples")

    print("\n" + "=" * 60)
    print("SPLITTING")
    print("=" * 60)
    split_and_save(unique)

    print("\n" + "=" * 60)
    print("EVALUATION DATA")
    print("=" * 60)
    save_jsonl(load_eval_fatwa_qa(),          SPLITS_DIR / "eval_fatwa_qa.jsonl")
    save_jsonl(load_eval_fatwa_mcq(),         SPLITS_DIR / "eval_fatwa_mcq.jsonl")
    save_jsonl(load_eval_islamic_faith_qa(),  SPLITS_DIR / "eval_islamic_faith_qa.jsonl")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for f in sorted(SPLITS_DIR.glob("*.jsonl")):
        count = sum(1 for _ in open(f, encoding="utf-8"))
        print(f"  {f.name:<35} {count:>6,} samples")
    print("Done.")


if __name__ == "__main__":
    main()
