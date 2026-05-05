"""
T-10 — Single-GPU smoke test: 10 training steps.
Confirms no OOM, no NaN loss, data loads correctly, model trains.

Run:
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yahya/CollegeX/lib/python3.11/site-packages/nvidia/cu13/lib
    CUDA_VISIBLE_DEVICES=2 python scripts/smoke_test.py
"""
import os
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_NAME   = "Qwen/Qwen3-4B"
TRAIN_FILE   = "data/splits/train.jsonl"
MAX_STEPS    = 10
MAX_SEQ_LEN  = 1024
BATCH_SIZE   = 1
GRAD_ACCUM   = 2
OUTPUT_DIR   = "outputs/smoke"

def main():
    print("=" * 55)
    print("SMOKE TEST — Qwen3-4B QLoRA single GPU")
    print("=" * 55)

    # ── Tokenizer ──────────────────────────────────────────
    print("\n[1/5] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "right"
    print(f"  vocab size: {tokenizer.vocab_size:,}")

    # ── Model (4-bit QLoRA) ────────────────────────────────
    print("\n[2/5] Loading model in 4-bit ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map={"": 0},          # single GPU smoke test — CUDA_VISIBLE_DEVICES picks which one
        dtype=torch.float16,         # force fp16 — bf16 unsupported on CC 7.5
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"  device map: {model.hf_device_map}")

    # ── LoRA ───────────────────────────────────────────────
    print("\n[3/5] Attaching LoRA adapter ...")
    lora_cfg = LoraConfig(
        r=16,                     # smaller r for smoke test speed
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Cast all trainable (LoRA) params to FP16 — Qwen3 defaults to BF16
    # which is unsupported on CC 7.5 (RTX 2080 Ti)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float16)

    # ── Dataset ────────────────────────────────────────────
    print("\n[4/5] Loading dataset (first 100 samples) ...")
    ds = load_dataset("json", data_files={"train": TRAIN_FILE}, split="train[:100]")
    print(f"  loaded {len(ds)} samples")

    # Verify format
    sample = ds[0]
    assert "messages" in sample, "missing 'messages' key"
    assert len(sample["messages"]) == 3, "expected 3 messages"
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["system", "user", "assistant"], f"bad roles: {roles}"
    print(f"  format OK — sample Q: {sample['messages'][1]['content'][:60]}...")

    # ── Train ──────────────────────────────────────────────
    print("\n[5/5] Running 10 training steps ...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        fp16=False,   # no AMP scaler — QLoRA handles precision via bnb internally
        bf16=False,
        logging_steps=1,
        save_steps=MAX_STEPS,
        max_length=MAX_SEQ_LEN,
        report_to="none",
        dataloader_num_workers=0,
        assistant_only_loss=True,
        dataset_text_field="",
        gradient_checkpointing=False,   # disable for smoke test — avoids BF16 hook issue
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=ds,
    )

    result = trainer.train()
    final_loss = result.training_loss
    print(f"\n  Final loss : {final_loss:.4f}")

    # ── Verdict ────────────────────────────────────────────
    print("\n" + "=" * 55)
    if torch.isnan(torch.tensor(final_loss)):
        print("  ❌ FAIL — NaN loss detected")
    elif final_loss > 10:
        print(f"  ⚠️  WARNING — loss {final_loss:.2f} seems high (check data)")
    else:
        print(f"  ✅ PASS — loss {final_loss:.4f}, no OOM, no NaN")
        print("  Ready for full multi-GPU training.")
    print("=" * 55)

if __name__ == "__main__":
    main()
