---
language:
- ar
license: apache-2.0
task_categories:
- text-generation
- question-answering
tags:
- arabic
- islamic
- fiqh
- fatwa
- aqeedah
- quran
- instruction-tuning
- chat
- conversational
size_categories:
- 10K<n<100K
pretty_name: Islamic Arabic Q&A
dataset_info:
  features:
  - name: messages
    sequence:
      dtype: string
      name: content
  - name: system
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_examples: 17944
  - name: validation
    num_examples: 2101
  - name: test
    num_examples: 1042
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# Islamic Arabic Q&A Dataset

A curated Arabic instruction-tuning dataset focused on Islamic scholarship —
covering **Fiqh**, **Fatwa**, **Aqeedah**, **Quran Sciences**, and **Islamic Finance**.
Built to fine-tune Arabic LLMs for Islamic Q&A tasks.

## Dataset Summary

| Split | Samples |
|-------|---------|
| Train | 17,944 |
| Validation | 2,101 |
| Test | 1,042 |
| **Total** | **21,087** |

## Data Sources

| Source | Samples | License |
|--------|---------|---------|
| [SahmBenchmark/fatwa-training_standardized_new](https://huggingface.co/datasets/SahmBenchmark/fatwa-training_standardized_new) | 9,953 | Apache 2.0 |
| [majdelhaj/ar-qa-dataset](https://huggingface.co/datasets/majdelhaj/ar-qa-dataset) — islamic-fiqh | ~7,000 | — |
| [majdelhaj/ar-qa-dataset](https://huggingface.co/datasets/majdelhaj/ar-qa-dataset) — islamic-misc | ~4,000 | — |
| [Yasbok/Alpaca_arabic_instruct](https://huggingface.co/datasets/Yasbok/Alpaca_arabic_instruct) (subset) | 2,500 | — |

## Quality Filtering

All samples passed the following filters before inclusion:

- **Minimum answer length**: 80 characters
- **Minimum Arabic character ratio**: 40%
- **Near-duplicate removal**: full-text deduplication
- **HTML/encoding garbage removal**: regex-based cleanup

## Format

Each sample contains:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "أنت مساعد عالم إسلامي متخصص. أجب على الأسئلة بدقة استناداً إلى القرآن الكريم والسنة النبوية والفقه الإسلامي الكلاسيكي. استشهد بالمصادر حيثما أمكن. كن موجزاً لكن شاملاً."
    },
    {
      "role": "user",
      "content": "ما حكم زكاة الفطر وما مقدارها؟"
    },
    {
      "role": "assistant",
      "content": "زكاة الفطر واجبة على كل مسلم قادر ..."
    }
  ],
  "system": "أنت مساعد عالم إسلامي متخصص ...",
  "question": "ما حكم زكاة الفطر وما مقدارها؟",
  "answer": "زكاة الفطر واجبة على كل مسلم قادر ..."
}
```

The `messages` column is in standard chat format for direct use with `apply_chat_template`.
The `question` and `answer` columns are flattened for easy browsing and filtering.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("NightPrince/islamic-arabic-qa")

# Access train split
train = dataset["train"]
print(train[0]["question"])
print(train[0]["answer"])
```

### Fine-tuning with TRL SFTTrainer

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("NightPrince/islamic-arabic-qa")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=SFTConfig(
        max_length=1024,
        assistant_only_loss=True,
        dataset_text_field="",
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
trainer.train()
```

## Fine-tuned Model

This dataset was used to fine-tune **[NightPrince/Qwen3-4B-Islamic-Arabic](https://huggingface.co/NightPrince/Qwen3-4B-Islamic-Arabic)** — a QLoRA fine-tune of Qwen3-4B achieving a 6.5% improvement on the Fatwa MCQ benchmark over the base model.

## License

Apache 2.0 — inherited from the primary data source (SahmBenchmark/fatwa-training_standardized_new).

## Citation

```bibtex
@misc{islamic-arabic-qa-2026,
  author       = {Yahya Alnwsany},
  title        = {Islamic Arabic Q\&A Dataset},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/datasets/NightPrince/islamic-arabic-qa},
}
```
