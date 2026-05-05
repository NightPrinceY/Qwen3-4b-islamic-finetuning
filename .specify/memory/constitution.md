# Qwen3-4B Islamic Fine-Tuning Constitution

## Core Principles

### I. Data Quality First
Every training sample must be accurate Islamic scholarship — sourced from Quran, authenticated Hadith, or verified classical scholarship. No hallucinated references, no unverified fatawa. Quality over quantity at every stage.

### II. Reproducibility
Every experiment must be reproducible. All hyperparameters live in `configs/`. No magic numbers in scripts. Anyone cloning the repo can reproduce the exact training run.

### III. Incremental Validation
Validate at every stage before moving to the next: check dataset format before training, check loss curves before merging, check merged model before quantizing. Never skip gates.

### IV. Hardware-Aware Design
All code must account for RTX 2080 Ti (CC 7.5) constraints: FP16 only, no bf16, no Flash Attention 2, no awq_marlin. Use DeepSpeed ZeRO-2. Always set `CUDA_VISIBLE_DEVICES=1,2,3,4` — GPU0 is often occupied.

### V. Publish Everything
Model, dataset, and training code all go public. Write proper HuggingFace model cards and dataset cards. Goal is community impact, not just a private experiment.

## Pipeline Order (NON-NEGOTIABLE)

1. `scripts/prepare_data.py` → `data/splits/`
2. `scripts/train.py` → `outputs/checkpoints/`
3. `scripts/merge_lora.py` → `outputs/merged/`
4. `scripts/quantize.py` → `outputs/quantized/`
5. `scripts/evaluate.py` → documented results
6. Publish dataset → `NightPrince/islamic-arabic-qa`
7. Publish model → `NightPrince/Qwen3-4B-Islamic-Arabic`
8. Publish code → `NightPrinceY/Qwen3-4b-islamic-finetuning`

## Quality Gates

- **Before training**: ≥500 samples, all valid JSONL, consistent system prompt
- **During training**: loss decreasing by epoch 1 — if not, fix data first
- **Before merge**: pick best checkpoint by eval loss, not last
- **Before quantize**: merged model must generate coherent Arabic text
- **Before publish**: evaluate.py shows measurable improvement on ≥10 test samples

## Technical Constraints

| Constraint | Value | Reason |
|---|---|---|
| GPU | 4x RTX 2080 Ti, CUDA_VISIBLE_DEVICES=1,2,3,4 | GPU0 occupied |
| Precision | FP16 | bf16 emulated on CC 7.5 |
| Quantization | AWQ plain | awq_marlin needs CC 8.0+ |
| Attention | FlashInfer | Flash Attention 2 needs CC 8.0+ |
| DeepSpeed | ZeRO-2 | optimal for this hardware |
| Base model | Qwen/Qwen3-4B | fast iteration before 8B |
| LoRA | r=64, alpha=128, all projection layers | |

## Governance

This constitution supersedes any shortcut. If a gate fails, fix the upstream step — never skip forward.

**Version**: 1.0.0 | **Ratified**: 2026-05-04 | **Author**: Yahya Alnwsany
