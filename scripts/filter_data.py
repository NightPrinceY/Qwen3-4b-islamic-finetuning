"""
Step 1b — Post-filter: remove flagged samples and re-deduplicate.

Reads:  data/splits/{train,val,test}.jsonl
Writes: data/splits/{train,val,test}.jsonl  (in-place, saves originals to data/splits/pre_filter/)
"""

import json
import re
import shutil
from collections import Counter
from pathlib import Path

SPLITS_DIR   = Path("data/splits")
BACKUP_DIR   = Path("data/splits/pre_filter")

# Stricter thresholds than inspect_data.py
MIN_Q_CHARS  = 15
MIN_A_CHARS  = 80
ARABIC_RATIO = 0.40   # at least 40% Arabic script

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_jsonl(samples, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

def arabic_ratio(text):
    if not text: return 0.0
    return sum(1 for c in text if "؀" <= c <= "ۿ") / len(text)

def has_html(text):
    return bool(re.search(r"<[a-zA-Z/][^>]*>", text))

def has_encoding_garbage(text):
    return bool(re.search(r"[\\]x[0-9a-fA-F]{2}", text))

def is_url_only(text):
    return bool(re.fullmatch(r"https?://\S+", text.strip()))

def get_turns(sample):
    msgs = sample.get("messages", [])
    user = next((m["content"] for m in msgs if m["role"] == "user"),      "")
    asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    return user, asst

def should_keep(sample) -> tuple[bool, str]:
    msgs = sample.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) != 3:
        return False, "bad_structure"

    user, asst = get_turns(sample)

    if len(user) < MIN_Q_CHARS:                  return False, "short_q"
    if len(asst) < MIN_A_CHARS:                  return False, "short_a"
    if arabic_ratio(user) < ARABIC_RATIO:        return False, "low_arabic_q"
    if arabic_ratio(asst) < ARABIC_RATIO:        return False, "low_arabic_a"
    if has_html(user) or has_html(asst):         return False, "html"
    if has_encoding_garbage(user + asst):        return False, "encoding"
    if is_url_only(asst):                        return False, "url_only"
    if asst.strip() == user.strip():             return False, "answer_eq_question"

    return True, "ok"

def deduplicate(samples):
    seen, unique = set(), []
    for s in samples:
        user, asst = get_turns(s)
        # Full-text key on first 200 chars of question + first 100 of answer
        key = (user.strip()[:200], asst.strip()[:100])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique

# ── Main ──────────────────────────────────────────────────────────────────────

def filter_file(path: Path) -> dict:
    samples = load_jsonl(path)
    original = len(samples)

    # Filter
    kept, removed_reasons = [], Counter()
    for s in samples:
        keep, reason = should_keep(s)
        if keep:
            kept.append(s)
        else:
            removed_reasons[reason] += 1

    after_filter = len(kept)

    # Deduplicate
    kept = deduplicate(kept)
    dups_removed = after_filter - len(kept)

    print(f"\n  {path.name}")
    print(f"    Before   : {original:,}")
    print(f"    Filtered : {original - after_filter:,}  ({removed_reasons})")
    print(f"    Deduped  : {dups_removed:,}")
    print(f"    After    : {len(kept):,}  (removed {original - len(kept):,} total, {(original-len(kept))/original*100:.1f}%)")

    return kept

def main():
    print("\n" + "="*60)
    print("POST-FILTER PASS")
    print("="*60)

    # Backup originals
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        src = SPLITS_DIR / f"{split}.jsonl"
        if src.exists():
            shutil.copy(src, BACKUP_DIR / f"{split}.jsonl")
    print(f"\nOriginals backed up → {BACKUP_DIR}")

    # Filter and save
    total_before, total_after = 0, 0
    for split in ["train", "val", "test"]:
        path = SPLITS_DIR / f"{split}.jsonl"
        if not path.exists():
            continue
        clean = filter_file(path)
        total_before += sum(1 for _ in open(BACKUP_DIR / f"{split}.jsonl", encoding="utf-8"))
        total_after  += len(clean)
        save_jsonl(clean, path)

    print(f"\n{'='*60}")
    print(f"  Total before : {total_before:,}")
    print(f"  Total after  : {total_after:,}")
    print(f"  Removed      : {total_before - total_after:,}  ({(total_before-total_after)/total_before*100:.1f}%)")
    print(f"\n  ✅ Clean splits saved to {SPLITS_DIR}")

if __name__ == "__main__":
    main()
