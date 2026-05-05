"""
Generates outputs/MODEL_CARD.md from outputs/training_record.json.
Run after evaluate.py has filled in MCQ accuracy numbers.

Usage:
    python scripts/generate_model_card.py
    python scripts/generate_model_card.py --record outputs/training_record.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_record(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_date(iso: str | None) -> str:
    if not iso:
        return "N/A"
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d")


def generate(record: dict) -> str:
    m   = record["model"]
    lo  = record["lora"]
    tr  = record["training"]
    ds  = record["dataset"]
    hw  = record["hardware"]
    env = record["environment"]
    res = record["results"]
    ev  = record["eval_benchmarks"]

    base_model  = m["base_model"]
    train_date  = fmt_date(record.get("started_at"))
    duration    = f"{record.get('duration_hours', 'N/A')} hours"
    eff_batch   = tr["effective_batch_size"]

    mcq_base = ev.get("fatwa_mcq_accuracy_base")
    mcq_ft   = ev.get("fatwa_mcq_accuracy_finetuned")
    mcq_delta = ev.get("fatwa_mcq_improvement")

    mcq_table = ""
    if mcq_base is not None and mcq_ft is not None:
        mcq_table = f"""
| Model | Fatwa MCQ Accuracy |
|---|---|
| {base_model} (base) | {mcq_base:.1f}% |
| **This model (fine-tuned)** | **{mcq_ft:.1f}%** |
| Improvement | **{mcq_delta:+.1f}%** |
"""
    else:
        mcq_table = "\n> Evaluation results will be added after running `scripts/evaluate.py`.\n"

    trainable_pct = lo.get("trainable_params_pct", "N/A")
    trainable_abs = lo.get("trainable_params", 0)
    total_params  = lo.get("total_params", 0)

    final_loss  = res.get("final_train_loss", "N/A")
    best_eval   = res.get("best_eval_loss", "N/A")
    total_steps = res.get("total_steps", "N/A")

    gpu_list = ", ".join(
        f"{g['name']} ({g['vram_gb']} GB)" for g in hw.get("gpus", [])
    )
    total_vram = hw.get("total_vram_gb", "N/A")

    card = f"""---
language:
- ar
license: apache-2.0
base_model: {base_model}
tags:
- arabic
- islamic
- fiqh
- fatwa
- qlora
- peft
- qwen3
pipeline_tag: text-generation
---

# Qwen3-4B Islamic Arabic

Fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) on a curated
Arabic Islamic Q&A dataset covering Fiqh, Fatwa, Aqeedah, Quran Sciences, and Islamic Finance.

## Model Details

- **Base model**: `{base_model}`
- **Fine-tuning method**: QLoRA (4-bit NF4 quantization + LoRA adapters)
- **Language**: Arabic (ar)
- **Domain**: Islamic scholarship — Fiqh, Fatwa, Aqeedah, Quran Sciences, Islamic Finance
- **Training date**: {train_date}
- **Author**: [Yahya Alnwsany](https://huggingface.co/NightPrince)

## Evaluation{mcq_table}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "NightPrince/Qwen3-4B-Islamic-Arabic"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {{
        "role": "system",
        "content": (
            "أنت مساعد عالم إسلامي متخصص. أجب على الأسئلة بدقة استناداً إلى "
            "القرآن الكريم والسنة النبوية والفقه الإسلامي الكلاسيكي."
        ),
    }},
    {{"role": "user", "content": "ما حكم زكاة الفطر وما مقدارها؟"}},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3, do_sample=True)

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Serve with vLLM

```bash
python -m vllm.entrypoints.openai.api_server \\
    --model NightPrince/Qwen3-4B-Islamic-Arabic \\
    --quantization awq \\
    --dtype float16 \\
    --enforce-eager \\
    --served-model-name qwen3-4b-islamic \\
    --port 8000
```

## Training Details

### Dataset

| Source | Samples | License |
|---|---|---|
| SahmBenchmark/fatwa-training_standardized_new | 9,953 | Apache 2.0 |
| majdelhaj/ar-qa-dataset (islamic-fiqh) | ~10,000 | — |
| majdelhaj/ar-qa-dataset (islamic-misc) | ~4,000 | — |
| Yasbok/Alpaca_arabic_instruct (subset) | 2,500 | — |

**Final splits after quality filtering:**

| Split | Samples |
|---|---|
| Train | {ds['train_samples']:,} |
| Validation | {ds['val_samples']:,} |
| Test | {ds['test_samples']:,} |

**Quality filter applied:** minimum 80 characters answer, minimum 40% Arabic character ratio,
near-duplicate removal.

**System prompt used in all training samples:**
> {ds['system_prompt']}

### Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | {tr['num_train_epochs']} |
| Per-device batch size | {tr['per_device_train_batch_size']} |
| Gradient accumulation | {tr['gradient_accumulation_steps']} |
| Effective batch size | {eff_batch} |
| Learning rate | {tr['learning_rate']} |
| LR scheduler | {tr['lr_scheduler']} |
| Warmup ratio | {tr['warmup_ratio']} |
| Max sequence length | {tr['max_seq_length']} |
| Precision | FP16 (no AMP scaler — QLoRA handles precision internally) |
| Gradient checkpointing | {tr['gradient_checkpointing']} |
| Loss | Assistant turns only |

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | {lo['r']} |
| Alpha | {lo['lora_alpha']} |
| Dropout | {lo['lora_dropout']} |
| Target modules | {', '.join(lo['target_modules'])} |
| Trainable params | {trainable_abs:,} ({trainable_pct}% of total) |

### Training Results

| Metric | Value |
|---|---|
| Final train loss | {final_loss} |
| Best eval loss | {best_eval} |
| Total steps | {total_steps} |
| Training duration | {duration} |

### Hardware

- **GPUs**: {hw['num_gpus']}x — {gpu_list}
- **Total VRAM**: {total_vram} GB
- **CUDA**: {env.get('cuda', 'N/A')}

### Environment

| Package | Version |
|---|---|
| Python | {env.get('python')} |
| PyTorch | {env.get('pytorch')} |
| Transformers | {env.get('transformers')} |
| PEFT | {env.get('peft')} |
| TRL | {env.get('trl')} |
| BitsAndBytes | {env.get('bitsandbytes')} |
| Accelerate | {env.get('accelerate')} |

## Limitations

- Answers reflect classical Islamic scholarship and may not represent all scholarly opinions.
- Not a substitute for a qualified Islamic scholar (مفتٍ).
- Performance on dialectal Arabic may be lower than MSA.
- Model was not trained on Shia fiqh sources — coverage is predominantly Sunni scholarship.

## Citation

```bibtex
@misc{{qwen3-4b-islamic-arabic,
  author       = {{Yahya Alnwsany}},
  title        = {{Qwen3-4B Islamic Arabic}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  url          = {{https://huggingface.co/NightPrince/Qwen3-4B-Islamic-Arabic}},
}}
```

## License

Apache 2.0 — inherited from the base model and primary training dataset.
"""
    return card.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default="outputs/training_record.json")
    parser.add_argument("--out",    default="outputs/MODEL_CARD.md")
    args = parser.parse_args()

    record = load_record(args.record)
    card   = generate(record)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(card, encoding="utf-8")
    print(f"Model card saved → {args.out}")
    print(f"  {len(card.splitlines())} lines, {len(card):,} chars")


if __name__ == "__main__":
    main()
