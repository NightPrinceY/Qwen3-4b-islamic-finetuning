# Tasks: Islamic Arabic Fine-Tuning Pipeline

**Spec**: `spec.md` | **Plan**: `plan.md` | **Date**: 2026-05-04

---

## Stage 1 — Dataset

- [x] **T-01** Collect Fiqh Q&A pairs — sourced from SahmBenchmark/fatwa-training + ar-qa-dataset/islamic-fiqh
- [x] **T-02** Collect Aqeedah Q&A pairs — covered via ar-qa-dataset/islamic-misc
- [x] **T-03** Collect Quran Sciences Q&A pairs — covered via ar-qa-dataset/islamic-misc + QCRI/IslamicFaithQA
- [x] **T-04** Collect Hadith Sciences Q&A pairs — covered via ar-qa-dataset categories
- [x] **T-05** Collect Islamic Finance Q&A pairs — SahmBenchmark/fatwa-training (Zakat, Riba, Murabaha, Waqf, etc.)
- [x] **T-06** Collect Arabic-only Q&A pairs — all sources are Arabic
- [x] **T-07** Run `scripts/prepare_data.py` — 21,087 clean samples across train/val/test
- [x] **T-08** Quality inspection + filter pass — 0 flagged after filter_data.py (7.1% removed)

**Dataset summary (2026-05-04):**
- train.jsonl: 17,944 | val.jsonl: 2,101 | test.jsonl: 1,042
- eval_fatwa_mcq.jsonl: 2,000 | eval_fatwa_qa.jsonl: 2,000 | eval_islamic_faith_qa.jsonl: 3,810
- Sources: SahmBenchmark (9,953) + ar-qa-dataset/fiqh (10,003) + ar-qa-dataset/misc (4,000) + Alpaca-Arabic (2,500)
- After dedup + filter: 21,087 total, 0 flagged, ✅ GOOD

## Stage 2 — Training

- [x] **T-09** Verified: trl 1.3.0, peft 0.18.1, deepspeed 0.18.9, accelerate 1.13.0, bitsandbytes 0.49.2 ✓
- [x] **T-10** Smoke test PASSED — loss 3.2659, no OOM, no NaN on GPU2 single-GPU run

**Smoke test findings (applied to train.py):**
- `fp16=False / bf16=False` — Qwen3 BF16 autocast breaks FP16 grad scaler on CC 7.5
- `gradient_checkpointing=False` — adds BF16 hooks, same crash
- `dtype=torch.float16` + cast LoRA params to FP16 after PEFT attach
- `device_map={"": local_rank}` for Accelerate DDP (not `"auto"`)
- TRL 1.3 API: `processing_class`, `max_length`, `assistant_only_loss=True`
- `LD_LIBRARY_PATH` must include CUDA 13 libs from CollegeX venv
- Launch via: `bash scripts/launch_training.sh`

- [ ] **T-11** Launch full 4-GPU training: `bash scripts/launch_training.sh`
- [ ] **T-12** Monitor loss — confirm decreasing by step 100
- [ ] **T-13** Identify best checkpoint by lowest eval loss

## Stage 3 — Merge & Quantize

- [ ] **T-14** Run `python scripts/merge_lora.py` — verify `outputs/merged/` contains full model files
- [ ] **T-15** Smoke test merged model: load and generate 3 responses
- [ ] **T-16** Run `python scripts/quantize.py` — verify `outputs/quantized/` created
- [ ] **T-17** Smoke test quantized model via vLLM: serve and query once

## Stage 4 — Evaluate

- [ ] **T-18** Update `scripts/evaluate.py` with actual model names
- [ ] **T-19** Run evaluate.py on full test split — capture outputs
- [ ] **T-20** Document 5 side-by-side examples (base vs fine-tuned) for model card

## Stage 5 — Publish

- [ ] **T-21** Push dataset to HuggingFace: `NightPrince/islamic-arabic-qa`
- [ ] **T-22** Write dataset card (domain, sources, format, splits, example)
- [ ] **T-23** Push model to HuggingFace: `NightPrince/Qwen3-4B-Islamic-Arabic`
- [ ] **T-24** Write model card (base model, training details, eval results, vLLM usage)
- [ ] **T-25** Initialize git, create GitHub repo, push training code + README
- [ ] **T-26** Wire fine-tuned model into Muslim Voice AI Agent and test live

---

**Total tasks**: 26 | **Completed**: 8 / 26 | **Last updated**: 2026-05-04
