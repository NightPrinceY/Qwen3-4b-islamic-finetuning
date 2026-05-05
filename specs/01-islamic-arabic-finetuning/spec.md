# Feature Specification: Islamic Arabic Fine-Tuning Pipeline

**Feature Branch**: `01-islamic-arabic-finetuning`
**Created**: 2026-05-04
**Status**: In Progress
**Author**: Yahya Alnwsany

---

## User Scenarios & Testing

### User Story 1 — Fine-Tuned Model Answers Islamic Questions Better Than Base (Priority: P1)

A researcher or Muslim user asks a domain-specific Islamic question in Arabic or English.
The fine-tuned model should give a more accurate, sourced, and domain-appropriate response
than the base Qwen3-4B model.

**Why this priority**: This is the entire goal of the project. Everything else serves this.

**Independent Test**: Query both base and fine-tuned with `scripts/evaluate.py` on 10+ Islamic questions. Fine-tuned responses must be noticeably more accurate and citation-aware.

**Acceptance Scenarios**:

1. **Given** a Fiqh question in English, **When** asked to both models, **Then** fine-tuned cites a specific madhab ruling while base gives a generic answer.
2. **Given** an Arabic question (e.g., أركان الإسلام), **When** sent to fine-tuned model, **Then** response is in Arabic, accurate, and structured.
3. **Given** a Hadith classification question, **When** fine-tuned responds, **Then** it references Sahih/Hasan/Da'if terminology correctly.

---

### User Story 2 — Dataset is Public and Reusable (Priority: P2)

Any Arabic NLP researcher can download the dataset from HuggingFace and use it for their own training.

**Why this priority**: Community impact. A public high-quality Islamic Arabic dataset is rare.

**Independent Test**: Dataset card on HuggingFace has clear format docs. `load_dataset("NightPrince/islamic-arabic-qa")` works without errors.

**Acceptance Scenarios**:

1. **Given** the dataset is published, **When** loaded via HuggingFace datasets, **Then** all splits (train/val/test) load correctly with `messages` field in chat format.
2. **Given** a new user reads the dataset card, **When** they follow the example, **Then** they can fine-tune their own model on it within 30 minutes.

---

### User Story 3 — Training Pipeline is Reproducible (Priority: P3)

Another developer clones the GitHub repo and can reproduce the training run exactly.

**Why this priority**: Professional credibility. Reproducible research is the standard.

**Independent Test**: Clone repo on a fresh machine, follow README, run `scripts/train.py` — same loss curve, same checkpoint quality.

**Acceptance Scenarios**:

1. **Given** a fresh clone, **When** configs are unchanged and dataset is the same, **Then** training reproduces the same eval loss within ±2%.
2. **Given** a developer reads `configs/training_config.yaml`, **When** they adjust `learning_rate`, **Then** the change propagates to training without editing any Python file.

---

### Edge Cases

- What if a training sample contains a weak (Da'if) Hadith presented as authentic? → Manual review gate before adding to dataset.
- What if GPU0 is occupied and training crashes on CUDA init? → `CUDA_VISIBLE_DEVICES=1,2,3,4` hardcoded in training launch command.
- What if loss diverges (NaN/Inf) during training? → FP16 loss scaling in DeepSpeed config handles this; checkpoint from last stable step.
- What if the merged model produces garbage text? → Do not quantize; roll back to best checkpoint and re-merge.

---

## Requirements

### Functional Requirements

- **FR-001**: Dataset MUST contain ≥500 high-quality Islamic Q&A pairs across ≥5 categories
- **FR-002**: All samples MUST follow the chat format: `[system, user, assistant]` messages
- **FR-003**: Training MUST run on 4x RTX 2080 Ti using FP16 + DeepSpeed ZeRO-2
- **FR-004**: LoRA adapter MUST be mergeable into base model without errors
- **FR-005**: Quantized model MUST load and serve via vLLM with `--quantization awq`
- **FR-006**: Evaluation MUST produce side-by-side comparison of base vs fine-tuned
- **FR-007**: Model MUST be published to HuggingFace with a complete model card
- **FR-008**: Dataset MUST be published to HuggingFace with a complete dataset card
- **FR-009**: Training code MUST be published to GitHub with a README covering full pipeline

### Key Entities

- **Sample**: `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`
- **Checkpoint**: LoRA adapter weights saved at eval intervals; best selected by lowest eval loss
- **Merged Model**: Full FP16 model after `merge_and_unload()` — ~8 GB on disk
- **Quantized Model**: AWQ INT4 model — ~4 GB on disk, compatible with vLLM

---

## Success Criteria

- **SC-001**: Fine-tuned model shows qualitative improvement on ≥80% of test samples vs base
- **SC-002**: Dataset has ≥500 samples across ≥5 Islamic knowledge categories
- **SC-003**: Training completes without OOM or NaN errors on 4x 2080 Ti
- **SC-004**: HuggingFace model card documents benchmark comparison with concrete examples
- **SC-005**: GitHub repo README allows a new developer to reproduce training in one session
