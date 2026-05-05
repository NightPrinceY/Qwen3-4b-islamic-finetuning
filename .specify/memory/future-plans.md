# Future Plans

## After v1 (Qwen3-4B) is published

- Re-run the exact same pipeline on **Qwen3-8B** — one model name change, same everything
- Experiment with **DPO alignment** using TRL — create preference pairs from Islamic scholar feedback
- Add **Arabic-only evaluation metrics** (BLEU on Arabic, dialect coverage)
- Connect fine-tuned model to **Muslim Voice AI Agent** as the primary LLM brain

## Dataset Expansion

- Add more categories: Tafsir, Sirah (Prophet's biography), Islamic History
- Add multi-turn conversations, not just single Q&A pairs
- Explore existing Arabic Islamic datasets on HuggingFace to augment

## Serving Improvements

- Benchmark fine-tuned 4B vs base 8B for quality/speed tradeoff
- Explore whether fine-tuned 4B can replace base 8B in the voice agent with better domain accuracy
