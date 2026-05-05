"""
Data quality inspection — run before training.
Checks format, language, length distribution, duplicates, and content issues.
Prints flagged samples so you can judge them manually.
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

SPLITS_DIR = Path("data/splits")

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_Q_CHARS   = 10
MAX_Q_CHARS   = 1_500
MIN_A_CHARS   = 60
MAX_A_CHARS   = 4_000
ARABIC_RATIO  = 0.30   # at least 30% of chars should be Arabic script

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
    return arabic / len(text)

def has_html(text: str) -> bool:
    return bool(re.search(r"<[a-zA-Z/][^>]*>", text))

def has_encoding_garbage(text: str) -> bool:
    # Replacement chars, weird control chars
    return bool(re.search(r"[�\x00-\x08\x0B\x0C\x0E-\x1F]", text))

def is_repetitive(text: str, threshold: float = 0.5) -> bool:
    """Flag if >50% of the text is a single repeated word."""
    words = text.split()
    if len(words) < 6:
        return False
    top_word, top_count = Counter(words).most_common(1)[0]
    return (top_count / len(words)) > threshold

def get_turns(sample: dict) -> tuple[str, str, str]:
    msgs = sample.get("messages", [])
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user   = next((m["content"] for m in msgs if m["role"] == "user"),   "")
    asst   = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    return system, user, asst

# ── Per-sample checks ─────────────────────────────────────────────────────────

def check_sample(sample: dict) -> list[str]:
    flags = []
    msgs = sample.get("messages", [])

    # Structure
    if not isinstance(msgs, list) or len(msgs) != 3:
        flags.append(f"BAD_STRUCTURE: {len(msgs)} messages (expected 3)")
        return flags
    roles = [m.get("role") for m in msgs]
    if roles != ["system", "user", "assistant"]:
        flags.append(f"BAD_ROLES: {roles}")

    system, user, asst = get_turns(sample)

    # Empty
    if not user.strip():  flags.append("EMPTY_QUESTION")
    if not asst.strip():  flags.append("EMPTY_ANSWER")

    # Length
    if len(user) < MIN_Q_CHARS:   flags.append(f"SHORT_Q: {len(user)} chars")
    if len(user) > MAX_Q_CHARS:   flags.append(f"LONG_Q: {len(user)} chars")
    if len(asst) < MIN_A_CHARS:   flags.append(f"SHORT_A: {len(asst)} chars")
    if len(asst) > MAX_A_CHARS:   flags.append(f"LONG_A: {len(asst)} chars")

    # Language
    if arabic_ratio(user) < ARABIC_RATIO:
        flags.append(f"LOW_ARABIC_Q: {arabic_ratio(user):.0%}")
    if arabic_ratio(asst) < ARABIC_RATIO:
        flags.append(f"LOW_ARABIC_A: {arabic_ratio(asst):.0%}")

    # Content issues
    if has_html(user) or has_html(asst):     flags.append("HTML_TAGS")
    if has_encoding_garbage(user + asst):    flags.append("ENCODING_GARBAGE")
    if is_repetitive(asst):                  flags.append("REPETITIVE_ANSWER")
    if re.fullmatch(r"https?://\S+", asst.strip()):
        flags.append("URL_ONLY_ANSWER")
    if asst.strip() == user.strip():
        flags.append("ANSWER_EQUALS_QUESTION")

    return flags

# ── File-level analysis ───────────────────────────────────────────────────────

def analyze_file(path: Path, show_flagged: int = 5):
    print(f"\n{'═'*65}")
    print(f"  {path.name}")
    print(f"{'═'*65}")

    samples = load_jsonl(path)
    if not samples:
        print("  !! EMPTY FILE")
        return

    total      = len(samples)
    flagged    = []
    q_lens, a_lens, ar_ratios_q, ar_ratios_a = [], [], [], []

    for i, s in enumerate(samples):
        issues = check_sample(s)
        _, user, asst = get_turns(s)
        q_lens.append(len(user))
        a_lens.append(len(asst))
        ar_ratios_q.append(arabic_ratio(user))
        ar_ratios_a.append(arabic_ratio(asst))
        if issues:
            flagged.append((i, issues, user, asst))

    # Duplicates
    seen_q, dup_count = set(), 0
    for s in samples:
        _, user, _ = get_turns(s)
        key = user.strip()[:100]
        if key in seen_q:
            dup_count += 1
        seen_q.add(key)

    # Stats
    def stats(lst):
        lst = sorted(lst)
        n = len(lst)
        return {
            "min":    lst[0],
            "p10":    lst[int(n*0.10)],
            "median": lst[n//2],
            "p90":    lst[int(n*0.90)],
            "max":    lst[-1],
            "mean":   sum(lst)/n,
        }

    qs = stats(q_lens)
    as_ = stats(a_lens)
    arq = stats(ar_ratios_q)
    ara = stats(ar_ratios_a)

    flag_rate = len(flagged) / total * 100

    print(f"\n  Samples    : {total:,}")
    print(f"  Flagged    : {len(flagged):,}  ({flag_rate:.1f}%)")
    print(f"  Duplicates : {dup_count:,}")

    print(f"\n  Question length (chars)  — min:{qs['min']}  p10:{qs['p10']}  median:{qs['median']}  p90:{qs['p90']}  max:{qs['max']}")
    print(f"  Answer   length (chars)  — min:{as_['min']}  p10:{as_['p10']}  median:{as_['median']}  p90:{as_['p90']}  max:{as_['max']}")
    print(f"  Arabic ratio  (question) — min:{arq['min']:.0%}  median:{arq['median']:.0%}  mean:{arq['mean']:.0%}")
    print(f"  Arabic ratio  (answer)   — min:{ara['min']:.0%}  median:{ara['median']:.0%}  mean:{ara['mean']:.0%}")

    # Flag breakdown
    if flagged:
        all_flags = [f for _, issues, _, _ in flagged for f in issues]
        flag_counts = Counter(f.split(":")[0] for f in all_flags)
        print(f"\n  Flag breakdown:")
        for flag, count in flag_counts.most_common():
            print(f"    {flag:<30} {count:>5}  ({count/total*100:.1f}%)")

    # Show sample flagged entries
    if flagged and show_flagged > 0:
        print(f"\n  ── Sample flagged entries (showing {min(show_flagged, len(flagged))}) ──")
        for idx, issues, q, a in flagged[:show_flagged]:
            print(f"\n  [#{idx}] Flags: {issues}")
            print(f"  Q: {q[:120].strip()!r}")
            print(f"  A: {a[:120].strip()!r}")

    # Show 3 clean random samples
    clean = [(i, s) for i, s in enumerate(samples)
             if not check_sample(s)]
    if clean:
        import random; random.seed(99)
        picks = random.sample(clean, min(3, len(clean)))
        print(f"\n  ── 3 clean random samples ──")
        for idx, s in picks:
            _, q, a = get_turns(s)
            print(f"\n  [#{idx}]")
            print(f"  Q: {q[:150].strip()}")
            print(f"  A: {a[:150].strip()}")

    return {
        "total": total,
        "flagged": len(flagged),
        "flag_rate": flag_rate,
        "duplicates": dup_count,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█"*65)
    print("  DATA QUALITY INSPECTION")
    print("█"*65)

    training_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    eval_files     = ["eval_fatwa_qa.jsonl", "eval_fatwa_mcq.jsonl",
                      "eval_islamic_faith_qa.jsonl"]

    print("\n\n▶ TRAINING SPLITS")
    train_results = {}
    for fname in training_files:
        path = SPLITS_DIR / fname
        if path.exists():
            train_results[fname] = analyze_file(path, show_flagged=3)

    print("\n\n▶ EVALUATION SETS")
    for fname in eval_files:
        path = SPLITS_DIR / fname
        if path.exists():
            # Eval sets have a different schema — just check basic stats
            samples = load_jsonl(path)
            print(f"\n  {fname:<40} {len(samples):,} samples  ✓")

    # Overall verdict
    print("\n\n" + "█"*65)
    print("  OVERALL VERDICT")
    print("█"*65)
    total_samples  = sum(r["total"]   for r in train_results.values())
    total_flagged  = sum(r["flagged"] for r in train_results.values())
    total_dups     = sum(r["duplicates"] for r in train_results.values())
    overall_rate   = total_flagged / total_samples * 100

    print(f"\n  Total training samples : {total_samples:,}")
    print(f"  Total flagged          : {total_flagged:,}  ({overall_rate:.1f}%)")
    print(f"  Remaining duplicates   : {total_dups:,}")

    if overall_rate < 5:
        verdict = "✅ GOOD  — safe to train"
    elif overall_rate < 15:
        verdict = "⚠️  ACCEPTABLE — consider filtering flagged samples"
    else:
        verdict = "❌ RISKY  — clean before training"

    print(f"\n  Verdict: {verdict}\n")


if __name__ == "__main__":
    main()
