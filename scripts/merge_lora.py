"""
Step 3 — Merge LoRA adapter weights into the base model.

Input:  outputs/checkpoints/  (best checkpoint by eval loss)
Output: outputs/merged/

Run:
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yahya/CollegeX/lib/python3.11/site-packages/nvidia/cu13/lib
    CUDA_VISIBLE_DEVICES=2 python scripts/merge_lora.py
"""
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

CHECKPOINT_DIR = "outputs/checkpoints/checkpoint-800"
MERGED_DIR     = "outputs/merged"


def main():
    print(f"Loading adapter from {CHECKPOINT_DIR} ...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR,
        dtype=torch.float16,         # dtype= not torch_dtype= (deprecated)
        device_map={"": 0},          # single GPU — CUDA_VISIBLE_DEVICES picks which
        trust_remote_code=True,
    )

    print("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR} ...")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"Done. Merged model saved to {MERGED_DIR}")
    print("Verify with: python -c \"from transformers import AutoModelForCausalLM; "
          f"m = AutoModelForCausalLM.from_pretrained('{MERGED_DIR}'); print('OK')\"")


if __name__ == "__main__":
    main()
