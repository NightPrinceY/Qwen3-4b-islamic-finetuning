"""
Step 3 — Merge LoRA adapter weights into the base model.

Input:  outputs/checkpoints/
Output: outputs/merged/
"""
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

CHECKPOINT_DIR = "outputs/checkpoints"
MERGED_DIR = "outputs/merged"


def main():
    print(f"Loading adapter from {CHECKPOINT_DIR} ...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR} ...")
    model.save_pretrained(MERGED_DIR)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
