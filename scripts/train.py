"""
Step 2 — QLoRA fine-tuning with TRL SFTTrainer + DeepSpeed ZeRO-2.

Run with:
    accelerate launch --config_file configs/deepspeed_zero2.json \
        --num_processes 4 scripts/train.py
"""
import os
import yaml
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    lora_cfg = load_config("configs/lora_config.yaml")
    train_cfg = load_config("configs/training_config.yaml")

    model_name = lora_cfg["model_name"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_cfg["lora"]["r"],
        lora_alpha=lora_cfg["lora"]["lora_alpha"],
        lora_dropout=lora_cfg["lora"]["lora_dropout"],
        bias=lora_cfg["lora"]["bias"],
        target_modules=lora_cfg["lora"]["target_modules"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/splits/train.jsonl",
            "validation": "data/splits/val.jsonl",
        },
    )

    training_args = TrainingArguments(
        **{k: v for k, v in train_cfg.items()},
        deepspeed="configs/deepspeed_zero2.json",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()
    trainer.save_model()
    print("Training complete. Checkpoint saved to:", train_cfg["output_dir"])


if __name__ == "__main__":
    main()
