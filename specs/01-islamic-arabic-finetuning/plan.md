# Implementation Plan: Islamic Arabic Fine-Tuning Pipeline

**Branch**: `01-islamic-arabic-finetuning` | **Date**: 2026-05-04
**Spec**: `specs/01-islamic-arabic-finetuning/spec.md`

## Summary

Fine-tune Qwen3-4B on a curated Arabic Islamic Q&A dataset using QLoRA + TRL SFTTrainer + DeepSpeed ZeRO-2 across 4x RTX 2080 Ti GPUs. Merge, quantize to AWQ, serve via vLLM, evaluate, and publish everything publicly.

## Technical Context

**Language/Version**: Python 3.11 (CollegeX venv)
**Primary Dependencies**: PyTorch 2.11, Transformers 4.57, PEFT 0.18, TRL 1.3, DeepSpeed 0.18, AutoAWQ, vLLM 0.20
**Storage**: `data/` for dataset, `outputs/` for artifacts — ~20 GB total
**Testing**: Manual eval via `scripts/evaluate.py` + vLLM smoke test
**Target Platform**: WSL2, 4x RTX 2080 Ti (CC 7.5), CUDA 12.9
**Project Type**: ML training pipeline
**Performance Goals**: >10 tok/s inference on quantized model, <24h total training time
**Constraints**: FP16 only, no bf16, no Flash Attention 2, no awq_marlin, GPU0 always skipped
**Scale/Scope**: 500–3000 training samples, 3 epochs

## Constitution Check

- [x] FP16 enforced in `configs/training_config.yaml` (fp16: true, bf16: false)
- [x] DeepSpeed ZeRO-2 configured in `configs/deepspeed_zero2.json`
- [x] `CUDA_VISIBLE_DEVICES=1,2,3,4` set in launch command
- [x] All hyperparameters in `configs/` — no hardcoding in scripts
- [x] Pipeline order enforced — each step reads from previous step's output dir

## Project Structure

```text
Qwen3-4b-finetuning/
├── configs/
│   ├── lora_config.yaml          # LoRA rank, target modules
│   ├── training_config.yaml      # TrainingArguments
│   └── deepspeed_zero2.json      # ZeRO-2 + FP16
├── data/
│   ├── raw/                      # source JSON files (gitignored)
│   ├── processed/                # intermediate cleaned data
│   └── splits/                   # train.jsonl / val.jsonl / test.jsonl
├── scripts/
│   ├── prepare_data.py           # Step 1
│   ├── train.py                  # Step 2
│   ├── merge_lora.py             # Step 3
│   ├── quantize.py               # Step 4
│   └── evaluate.py               # Step 5
├── outputs/
│   ├── checkpoints/              # LoRA adapters (gitignored)
│   ├── merged/                   # full model (gitignored)
│   ├── quantized/                # AWQ model (gitignored)
│   └── logs/                     # training logs
├── specs/
│   └── 01-islamic-arabic-finetuning/
│       ├── spec.md               # this project's spec
│       ├── plan.md               # this file
│       └── tasks.md              # step-by-step tasks
└── .specify/                     # speckit scaffold
    ├── memory/
    │   ├── constitution.md
    │   └── future-plans.md
    └── templates/
```

## Phase Breakdown

### Phase 0 — Dataset (Task #1)
- Collect Islamic Q&A pairs from verified sources
- Format into chat template via `prepare_data.py`
- Verify ≥500 samples, all JSONL valid, splits created

### Phase 1 — Training (Task #2)
- Launch: `CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --config_file configs/deepspeed_zero2.json --num_processes 4 scripts/train.py`
- Monitor W&B for loss curve
- Save best checkpoint by eval loss

### Phase 2 — Merge + Quantize (Tasks #3, #4)
- `python scripts/merge_lora.py`
- `python scripts/quantize.py`
- Smoke test quantized model with vLLM

### Phase 3 — Evaluate + Publish (Tasks #5–#9)
- Run evaluate.py, document results
- Push to HuggingFace (model + dataset)
- Push to GitHub (training code + README)

### Phase 4 — Integration (Task #10)
- Wire into Muslim Voice AI Agent
