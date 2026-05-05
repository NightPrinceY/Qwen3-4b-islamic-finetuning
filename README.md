# Qwen3-4B Islamic Arabic — Fine-Tuning Pipeline

QLoRA fine-tuning pipeline for [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on a curated Arabic Islamic Q&A dataset.
Covers Fiqh, Fatwa, Aqeedah, Quran Sciences, and Islamic Finance.

**Author:** [Yahya Alnwsany](https://huggingface.co/NightPrince)

---

## HuggingFace Releases

| Artifact | Link |
|----------|------|
| Dataset | [NightPrince/islamic-arabic-qa](https://huggingface.co/datasets/NightPrince/islamic-arabic-qa) |
| Merged FP16 model | [NightPrince/Qwen3-4B-Islamic-Arabic](https://huggingface.co/NightPrince/Qwen3-4B-Islamic-Arabic) |
| LoRA adapter | [NightPrince/Qwen3-4B-Islamic-Arabic-LoRA](https://huggingface.co/NightPrince/Qwen3-4B-Islamic-Arabic-LoRA) |
| INT4 compressed-tensors | [NightPrince/Qwen3-4B-Islamic-Arabic-INT4](https://huggingface.co/NightPrince/Qwen3-4B-Islamic-Arabic-INT4) |
| GGUF (Q4_K_M / Q8_0 / F16) | [NightPrince/Qwen3-4B-Islamic-Arabic-GGUF](https://huggingface.co/NightPrince/Qwen3-4B-Islamic-Arabic-GGUF) |

---

## Results

| Metric | Value |
|--------|-------|
| Final train loss | 1.8918 |
| Best eval loss | 2.4094 |
| Token accuracy (start → end) | ~50% → ~60% |
| Total steps | 843 (3 epochs) |
| Training duration | 7.59 hours |

---

## Pipeline

```
Step 1 — Prepare data     scripts/prepare_data.py
Step 2 — Filter & inspect scripts/filter_data.py + inspect_data.py
Step 3 — Smoke test       scripts/smoke_test.py
Step 4 — Train            bash scripts/launch_training.sh
Step 5 — Merge LoRA       scripts/merge_lora.py
Step 6 — Quantize GGUF    scripts/quantize.py
Step 7 — Evaluate         scripts/evaluate.py
Step 8 — Model card       scripts/generate_model_card.py
Step 9 — Push dataset     scripts/push_dataset.py
```

---

## Project Structure

```
├── configs/
│   ├── accelerate_multigpu.yaml   # 4-GPU DDP config
│   ├── training_config.yaml       # all hyperparameters
│   ├── lora_config.yaml           # LoRA r=64, alpha=128
│   └── deepspeed_zero2.json       # (unused — replaced by Accelerate DDP)
├── scripts/
│   ├── prepare_data.py            # download + format 4 data sources
│   ├── filter_data.py             # quality filter (80 char, 40% Arabic)
│   ├── inspect_data.py            # dataset quality report
│   ├── smoke_test.py              # 10-step sanity check
│   ├── train.py                   # QLoRA SFTTrainer + Accelerate DDP
│   ├── launch_training.sh         # 4-GPU launch wrapper
│   ├── merge_lora.py              # merge adapter → FP16
│   ├── quantize.py                # GGUF (F16 + Q4_K_M + Q8_0) via llama.cpp
│   ├── evaluate.py                # MCQ + QA evaluation via vLLM
│   ├── push_dataset.py            # push to HuggingFace Hub
│   └── generate_model_card.py     # generate MODEL_CARD.md from training_record.json
├── data/
│   └── README.md                  # dataset documentation
├── outputs/
│   ├── training_record.json       # full training metadata + results
│   └── MODEL_CARD.md              # generated HuggingFace model card
├── specs/                         # feature spec (Spec Kit)
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/NightPrinceY/Qwen3-4b-islamic-finetuning
cd Qwen3-4b-islamic-finetuning

# Install into your environment
pip install -r requirements.txt

# Copy and fill in secrets
cp .env.example .env
```

---

## Training

### Requirements

- 4× GPU with ≥11 GB VRAM (tested on RTX 2080 Ti)
- CUDA 12+
- Python 3.11

### Run

```bash
# 1. Prepare dataset
python scripts/prepare_data.py
python scripts/filter_data.py

# 2. Smoke test (single GPU, 10 steps)
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_test.py

# 3. Full training (4 GPUs)
bash scripts/launch_training.sh

# 4. Merge + quantize
python scripts/merge_lora.py
python scripts/quantize.py

# 5. Evaluate (requires vLLM serving)
python scripts/evaluate.py
```

---

## Key Technical Decisions

| Decision | Why |
|----------|-----|
| `fp16=False, bf16=False` | Qwen3's internal autocast emits BF16 — crashes FP16 grad scaler on CC 7.5. QLoRA handles precision internally. |
| `device_map={"": local_rank}` | Required for Accelerate DDP + QLoRA — `"auto"` breaks DDP. |
| `gradient_checkpointing=True` | Safe once fp16=False removes the AMP scaler. Needed to fit batch in 11 GB. |
| Accelerate DDP over DeepSpeed | DeepSpeed ZeRO + QLoRA has known instabilities. DDP is simpler and reliable. |
| llama.cpp for GGUF quantization | AutoAWQ and llm-compressor both incompatible with transformers 4.57. llama.cpp is self-contained. |

---

## Hardware

| Component | Value |
|-----------|-------|
| GPUs | 4× NVIDIA GeForce RTX 2080 Ti (11 GB) |
| Total VRAM | 44 GB |
| CUDA | 13.0 |
| OS | Ubuntu 22.04 (WSL2) |

---

## License

Apache 2.0
