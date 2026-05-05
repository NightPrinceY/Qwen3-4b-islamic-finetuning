"""
Step 5 — Evaluate base vs fine-tuned on all three eval sets.

Requires both models served via vLLM (update model names below).
Run serve.py twice on different ports, or serve each model separately.

Three evaluations:
  1. MCQ accuracy  — fatwa_mcq  (2,000 samples) → concrete accuracy % per model
  2. QA qualitative — fatwa_qa  (first N samples) → side-by-side answers
  3. Broader Islamic — islamic_faith_qa (first N samples) → coverage check

Run:
    python scripts/evaluate.py
"""
import json
import re
from pathlib import Path
from openai import OpenAI

BASE_URL        = "http://localhost:8000/v1"
BASE_MODEL      = "qwen3-4b"
FINETUNED_MODEL = "qwen3-4b-islamic"

SPLITS_DIR = Path("data/splits")

COMMON_KWARGS = dict(
    temperature=0.0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)


def load_jsonl(path: Path, n: int = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if n and len(rows) >= n:
                break
    return rows


def ask_qa(client: OpenAI, model: str, messages: list[dict]) -> str:
    """Ask a chat-format question (excludes ground-truth assistant turn)."""
    response = client.chat.completions.create(
        model=model,
        messages=messages[:-1],
        max_tokens=600,
        **COMMON_KWARGS,
    )
    return response.choices[0].message.content


def ask_mcq(client: OpenAI, model: str, query: str) -> str:
    """Ask an MCQ question, extract answer letter from response."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        max_tokens=10,
        **COMMON_KWARGS,
    )
    text = response.choices[0].message.content.strip().lower()
    match = re.search(r"\b([abcd])\b", text)
    return match.group(1) if match else "?"


# ── Evaluation 1: MCQ Accuracy ────────────────────────────────────────────────

def eval_mcq(client: OpenAI, model: str, samples: list[dict]) -> dict:
    correct, total = 0, 0
    by_category: dict[str, list[bool]] = {}

    for s in samples:
        pred   = ask_mcq(client, model, s["query"])
        gold   = s["answer"]
        is_correct = (pred == gold)
        correct += is_correct
        total   += 1

        cat = s.get("category", "other")
        by_category.setdefault(cat, []).append(is_correct)

    accuracy = correct / total * 100 if total else 0
    cat_acc  = {c: sum(v)/len(v)*100 for c, v in by_category.items()}
    return {"accuracy": accuracy, "correct": correct, "total": total,
            "by_category": cat_acc}


# ── Evaluation 2: QA Qualitative ──────────────────────────────────────────────

def eval_qa_qualitative(client: OpenAI, samples: list[dict], n: int = 20):
    print(f"\n{'='*70}")
    print("  QA QUALITATIVE — base vs fine-tuned")
    print(f"{'='*70}")

    for i, s in enumerate(samples[:n], 1):
        messages     = s["messages"]
        question     = messages[1]["content"]
        ground_truth = messages[2]["content"]
        base_ans     = ask_qa(client, BASE_MODEL, messages)
        ft_ans       = ask_qa(client, FINETUNED_MODEL, messages)

        print(f"\n[{i}/{n}] Q: {question[:100]}...")
        print(f"\n  Ground truth : {ground_truth[:200]}...")
        print(f"\n  Base         : {base_ans[:200]}...")
        print(f"\n  Fine-tuned   : {ft_ans[:200]}...")
        print(f"  {'─'*65}")


# ── Evaluation 3: IslamicFaithQA coverage ────────────────────────────────────

def eval_faith_qa(client: OpenAI, samples: list[dict], n: int = 20):
    print(f"\n{'='*70}")
    print("  ISLAMICFAITHQA — broader coverage check")
    print(f"{'='*70}")

    for i, s in enumerate(samples[:n], 1):
        q    = s["question"]
        gold = s["gold_answer"]
        base_ans = client.chat.completions.create(
            model=BASE_MODEL,
            messages=[{"role": "user", "content": q}],
            max_tokens=400,
            **COMMON_KWARGS,
        ).choices[0].message.content
        ft_ans = client.chat.completions.create(
            model=FINETUNED_MODEL,
            messages=[{"role": "user", "content": q}],
            max_tokens=400,
            **COMMON_KWARGS,
        ).choices[0].message.content

        print(f"\n[{i}/{n}] [{s.get('category_type','')}] {q[:100]}...")
        print(f"  Gold       : {gold[:150]}...")
        print(f"  Base       : {base_ans[:150]}...")
        print(f"  Fine-tuned : {ft_ans[:150]}...")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    client = OpenAI(base_url=BASE_URL, api_key="none")

    # ── 1. MCQ accuracy ───────────────────────────────────────
    print("\n" + "="*70)
    print("  MCQ ACCURACY BENCHMARK")
    print("="*70)
    mcq_samples = load_jsonl(SPLITS_DIR / "eval_fatwa_mcq.jsonl")
    print(f"  Loaded {len(mcq_samples)} MCQ samples")

    print(f"\n  Evaluating base model ({BASE_MODEL}) ...")
    base_mcq = eval_mcq(client, BASE_MODEL, mcq_samples)

    print(f"  Evaluating fine-tuned ({FINETUNED_MODEL}) ...")
    ft_mcq = eval_mcq(client, FINETUNED_MODEL, mcq_samples)

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  Base model   : {base_mcq['accuracy']:5.1f}%  ({base_mcq['correct']}/{base_mcq['total']})   │")
    print(f"  │  Fine-tuned   : {ft_mcq['accuracy']:5.1f}%  ({ft_mcq['correct']}/{ft_mcq['total']})   │")
    delta = ft_mcq['accuracy'] - base_mcq['accuracy']
    print(f"  │  Improvement  : {delta:+.1f}%                     │")
    print(f"  └─────────────────────────────────────────┘")

    print("\n  Per-category accuracy:")
    all_cats = sorted(set(list(base_mcq["by_category"]) + list(ft_mcq["by_category"])))
    for cat in all_cats:
        b = base_mcq["by_category"].get(cat, 0)
        f = ft_mcq["by_category"].get(cat, 0)
        print(f"    {cat:<20}  base: {b:5.1f}%  →  ft: {f:5.1f}%  ({f-b:+.1f}%)")

    # ── 2. QA qualitative ─────────────────────────────────────
    qa_samples = load_jsonl(SPLITS_DIR / "test.jsonl", n=20)
    eval_qa_qualitative(client, qa_samples)

    # ── 3. IslamicFaithQA coverage ───────────────────────────
    faith_samples = load_jsonl(SPLITS_DIR / "eval_islamic_faith_qa.jsonl", n=15)
    eval_faith_qa(client, faith_samples)

    print("\n" + "="*70)
    print("  Evaluation complete.")
    print(f"  MCQ: base {base_mcq['accuracy']:.1f}% → fine-tuned {ft_mcq['accuracy']:.1f}% ({delta:+.1f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
