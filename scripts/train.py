"""
Step 2 — QLoRA fine-tuning with TRL SFTTrainer + Accelerate multi-GPU DDP.

Smoke-test findings applied:
  - fp16=False / bf16=False: Qwen3 internal autocast emits BF16 activations
    that break the FP16 grad scaler. QLoRA via bitsandbytes handles precision
    internally — no external AMP scaler needed.
  - dtype=torch.float16: forces non-quantized buffers (LayerNorm, embed) to FP16
  - Cast LoRA params to FP16 after PEFT attach (inherit BF16 by default)
  - gradient_checkpointing=False: adds BF16 hooks that trigger same scaler bug
  - device_map={"": local_rank}: required for Accelerate DDP with QLoRA
  - TRL 1.3 API: processing_class, max_length, assistant_only_loss, dataset_text_field

Saves outputs/training_record.json with full config + results for model card.

Run:
    bash scripts/launch_training.sh
"""

import json
import platform
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def gpu_info() -> list[dict]:
    info = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        info.append({
            "index": i,
            "name": p.name,
            "vram_gb": round(p.total_memory / 1024**3, 1),
            "compute_capability": f"{p.major}.{p.minor}",
        })
    return info


def package_versions() -> dict:
    import transformers, peft, trl, accelerate, bitsandbytes, datasets
    return {
        "python":       platform.python_version(),
        "pytorch":      torch.__version__,
        "cuda":         torch.version.cuda,
        "transformers": transformers.__version__,
        "peft":         peft.__version__,
        "trl":          trl.__version__,
        "accelerate":   accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "datasets":     datasets.__version__,
    }


def save_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=str)


def main():
    accelerator = Accelerator()
    local_rank  = accelerator.local_process_index
    is_main     = accelerator.is_main_process

    lora_cfg  = load_yaml("configs/lora_config.yaml")
    train_cfg = load_yaml("configs/training_config.yaml")

    record_path = Path("outputs/training_record.json")

    # ── Save full training record at start (main process only) ────────────────
    if is_main:
        print(f"\n{'='*55}")
        print(f"  Qwen3-4B QLoRA Fine-Tuning")
        print(f"  GPUs: {accelerator.num_processes} | Local rank: {local_rank}")
        print(f"{'='*55}\n")

        record = {
            "run_name":    train_cfg["run_name"],
            "started_at":  datetime.utcnow().isoformat() + "Z",
            "finished_at": None,
            "duration_hours": None,

            "model": {
                "base_model":      lora_cfg["model_name"],
                "base_model_type": "causal_lm",
                "quantization":    "QLoRA (4-bit NF4, double quant)",
                "compute_dtype":   "float16",
                "architecture":    "Qwen3",
            },

            "lora": {
                "r":              lora_cfg["lora"]["r"],
                "lora_alpha":     lora_cfg["lora"]["lora_alpha"],
                "lora_dropout":   lora_cfg["lora"]["lora_dropout"],
                "bias":           lora_cfg["lora"]["bias"],
                "target_modules": lora_cfg["lora"]["target_modules"],
                "task_type":      lora_cfg["task_type"],
                "trainable_params_pct": None,   # filled after model load
            },

            "training": {
                "num_train_epochs":            train_cfg["num_train_epochs"],
                "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
                "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
                "effective_batch_size":        (
                    train_cfg["per_device_train_batch_size"]
                    * accelerator.num_processes
                    * train_cfg["gradient_accumulation_steps"]
                ),
                "learning_rate":    train_cfg["learning_rate"],
                "lr_scheduler":     train_cfg["lr_scheduler_type"],
                "warmup_ratio":     train_cfg["warmup_ratio"],
                "max_seq_length":   train_cfg["max_length"],
                "fp16":             False,
                "bf16":             False,
                "gradient_checkpointing": True,
                "optimizer":        "adamw_torch_fused",
                "assistant_only_loss": True,
            },

            "dataset": {
                "sources": [
                    "SahmBenchmark/fatwa-training_standardized_new (Apache 2.0)",
                    "majdelhaj/ar-qa-dataset — islamic-fiqh + islamic-misc (GitHub)",
                    "Yasbok/Alpaca_arabic_instruct — 2,500 samples general Arabic",
                ],
                "language":     "Arabic (ar)",
                "domain":       "Islamic scholarship — Fiqh, Fatwa, Aqeedah, Quran Sciences",
                "system_prompt": (
                    "أنت مساعد عالم إسلامي متخصص. أجب على الأسئلة بدقة استناداً إلى "
                    "القرآن الكريم والسنة النبوية والفقه الإسلامي الكلاسيكي. "
                    "استشهد بالمصادر حيثما أمكن. كن موجزاً لكن شاملاً."
                ),
                "train_samples": count_lines("data/splits/train.jsonl"),
                "val_samples":   count_lines("data/splits/val.jsonl"),
                "test_samples":  count_lines("data/splits/test.jsonl"),
                "quality_filter": "min 80 chars answer, min 40% Arabic ratio, deduped",
            },

            "hardware": {
                "num_gpus":    accelerator.num_processes,
                "cuda_visible": "1,2,3,4",
                "gpus":        gpu_info(),
                "total_vram_gb": sum(g["vram_gb"] for g in gpu_info()),
            },

            "environment": package_versions(),

            "eval_benchmarks": {
                "fatwa_mcq_accuracy_base":       None,
                "fatwa_mcq_accuracy_finetuned":  None,
                "fatwa_mcq_improvement":         None,
            },

            "results": {
                "final_train_loss":  None,
                "best_eval_loss":    None,
                "total_steps":       None,
                "samples_per_second": None,
            },
        }
        save_record(record_path, record)
        print(f"Training record initialised → {record_path}\n")

    accelerator.wait_for_everyone()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        lora_cfg["model_name"], trust_remote_code=True
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model (4-bit QLoRA) ───────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        lora_cfg["model_name"],
        quantization_config=bnb_config,
        device_map={"": local_rank},
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_cfg["lora"]["r"],
        lora_alpha=lora_cfg["lora"]["lora_alpha"],
        lora_dropout=lora_cfg["lora"]["lora_dropout"],
        bias=lora_cfg["lora"]["bias"],
        target_modules=lora_cfg["lora"]["target_modules"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, lora_config)

    # Cast all trainable (LoRA) params to FP16
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float16)

    if is_main:
        model.print_trainable_parameters()
        # Save trainable % to record
        total  = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        record = json.loads(record_path.read_text())
        record["lora"]["trainable_params"]     = trainable
        record["lora"]["total_params"]         = total
        record["lora"]["trainable_params_pct"] = round(trainable / total * 100, 4)
        save_record(record_path, record)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset(
        "json",
        data_files={
            "train":      "data/splits/train.jsonl",
            "validation": "data/splits/val.jsonl",
        },
    )

    # ── SFTConfig ─────────────────────────────────────────────────────────────
    sft_args = SFTConfig(
        output_dir                  = train_cfg["output_dir"],
        num_train_epochs            = train_cfg["num_train_epochs"],
        per_device_train_batch_size = train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size  = train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps = train_cfg["gradient_accumulation_steps"],
        eval_strategy               = train_cfg["eval_strategy"],
        eval_steps                  = train_cfg["eval_steps"],
        save_strategy               = train_cfg["save_strategy"],
        save_steps                  = train_cfg["save_steps"],
        save_total_limit            = train_cfg["save_total_limit"],
        learning_rate               = train_cfg["learning_rate"],
        lr_scheduler_type           = train_cfg["lr_scheduler_type"],
        warmup_ratio                = train_cfg["warmup_ratio"],
        logging_steps               = train_cfg["logging_steps"],
        load_best_model_at_end      = train_cfg["load_best_model_at_end"],
        report_to                   = train_cfg["report_to"],
        run_name                    = train_cfg["run_name"],
        fp16                        = False,
        bf16                        = False,
        gradient_checkpointing      = True,   # safe: fp16=False removes AMP scaler, no BF16 hook crash
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        max_length                  = train_cfg["max_length"],
        assistant_only_loss         = True,
        dataset_text_field          = "",
        dataloader_num_workers      = 4,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = sft_args,
        train_dataset    = dataset["train"],
        eval_dataset     = dataset["validation"],
    )

    t_start = time.time()
    train_result = trainer.train()
    duration_hours = (time.time() - t_start) / 3600

    # ── Save model + update record ────────────────────────────────────────────
    if is_main:
        trainer.save_model()

        # Best eval loss from trainer state
        best_eval_loss = None
        if trainer.state.best_metric is not None:
            best_eval_loss = round(float(trainer.state.best_metric), 6)
        elif trainer.state.log_history:
            eval_losses = [
                e["eval_loss"] for e in trainer.state.log_history
                if "eval_loss" in e
            ]
            if eval_losses:
                best_eval_loss = round(min(eval_losses), 6)

        record = json.loads(record_path.read_text())
        record["finished_at"]    = datetime.utcnow().isoformat() + "Z"
        record["duration_hours"] = round(duration_hours, 2)
        record["results"] = {
            "final_train_loss":   round(train_result.training_loss, 6),
            "best_eval_loss":     best_eval_loss,
            "total_steps":        train_result.global_step,
            "samples_per_second": round(train_result.metrics.get("train_samples_per_second", 0), 2),
            "runtime_seconds":    round(train_result.metrics.get("train_runtime", 0), 1),
        }
        save_record(record_path, record)

        print(f"\n{'='*55}")
        print(f"  Training complete in {duration_hours:.1f}h")
        print(f"  Final train loss : {train_result.training_loss:.4f}")
        print(f"  Best eval loss   : {best_eval_loss}")
        print(f"  Total steps      : {train_result.global_step}")
        print(f"  Record saved     : {record_path}")
        print(f"  Checkpoint       : {train_cfg['output_dir']}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
