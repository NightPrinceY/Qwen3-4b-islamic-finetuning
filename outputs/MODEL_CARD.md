---
language:
- ar
license: apache-2.0
base_model: Qwen/Qwen3-4B
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

Fine-tuned version of [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on a curated
Arabic Islamic Q&A dataset covering Fiqh, Fatwa, Aqeedah, Quran Sciences, and Islamic Finance.

## Model Details

- **Base model**: `Qwen/Qwen3-4B`
- **Fine-tuning method**: QLoRA (4-bit NF4 quantization + LoRA adapters)
- **Language**: Arabic (ar)
- **Domain**: Islamic scholarship — Fiqh, Fatwa, Aqeedah, Quran Sciences, Islamic Finance
- **Training date**: 2026-05-05
- **Author**: [Yahya Alnwsany](https://huggingface.co/NightPrince)

## Evaluation
> Evaluation results will be added after running `scripts/evaluate.py`.


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
    {
        "role": "system",
        "content": (
            "أنت مساعد عالم إسلامي متخصص. أجب على الأسئلة بدقة استناداً إلى "
            "القرآن الكريم والسنة النبوية والفقه الإسلامي الكلاسيكي."
        ),
    },
    {"role": "user", "content": "ما حكم زكاة الفطر وما مقدارها؟"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3, do_sample=True)

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Serve with vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
    --model NightPrince/Qwen3-4B-Islamic-Arabic \
    --quantization awq \
    --dtype float16 \
    --enforce-eager \
    --served-model-name qwen3-4b-islamic \
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
| Train | 17,944 |
| Validation | 2,101 |
| Test | 1,042 |

**Quality filter applied:** minimum 80 characters answer, minimum 40% Arabic character ratio,
near-duplicate removal.

**System prompt used in all training samples:**
> أنت مساعد عالم إسلامي متخصص. أجب على الأسئلة بدقة استناداً إلى القرآن الكريم والسنة النبوية والفقه الإسلامي الكلاسيكي. استشهد بالمصادر حيثما أمكن. كن موجزاً لكن شاملاً.

### Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 16 |
| Effective batch size | 64 |
| Learning rate | 0.0002 |
| LR scheduler | cosine |
| Warmup ratio | 0.05 |
| Max sequence length | 1024 |
| Precision | FP16 (no AMP scaler — QLoRA handles precision internally) |
| Gradient checkpointing | True |
| Loss | Assistant turns only |

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 64 |
| Alpha | 128 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | 132,120,576 (5.6512% of total) |

### Training Results

| Metric | Value |
|---|---|
| Final train loss | 1.8918 |
| Best eval loss | 2.4094 |
| Total steps | 843 |
| Training duration | 7.59 hours |

### Hardware

- **GPUs**: 4x — NVIDIA GeForce RTX 2080 Ti (11.0 GB), NVIDIA GeForce RTX 2080 Ti (11.0 GB), NVIDIA GeForce RTX 2080 Ti (11.0 GB), NVIDIA GeForce RTX 2080 Ti (11.0 GB)
- **Total VRAM**: 44.0 GB
- **CUDA**: 13.0

### Environment

| Package | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 4.57.6 |
| PEFT | 0.18.1 |
| TRL | 1.3.0 |
| BitsAndBytes | 0.49.2 |
| Accelerate | 1.13.0 |

## Limitations

- Answers reflect classical Islamic scholarship and may not represent all scholarly opinions.
- Not a substitute for a qualified Islamic scholar (مفتٍ).
- Performance on dialectal Arabic may be lower than MSA.
- Model was not trained on Shia fiqh sources — coverage is predominantly Sunni scholarship.

## Citation

```bibtex
@misc{qwen3-4b-islamic-arabic,
  author       = {Yahya Alnwsany},
  title        = {Qwen3-4B Islamic Arabic},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/NightPrince/Qwen3-4B-Islamic-Arabic},
}
```

## License

Apache 2.0 — inherited from the base model and primary training dataset.