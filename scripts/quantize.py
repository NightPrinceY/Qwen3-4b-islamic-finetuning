"""
Step 4 — Quantize merged model to AWQ (INT4) for fast vLLM serving.

Input:  outputs/merged/
Output: outputs/quantized/

Requires: pip install autoawq
"""
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MERGED_DIR = "outputs/merged"
QUANTIZED_DIR = "outputs/quantized"

QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}


def main():
    print(f"Loading merged model from {MERGED_DIR} ...")
    model = AutoAWQForCausalLM.from_pretrained(MERGED_DIR, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR, trust_remote_code=True)

    print("Quantizing to AWQ INT4 ...")
    model.quantize(tokenizer, quant_config=QUANT_CONFIG)

    print(f"Saving quantized model to {QUANTIZED_DIR} ...")
    model.save_quantized(QUANTIZED_DIR)
    tokenizer.save_pretrained(QUANTIZED_DIR)

    print("Done. Load in vLLM with --quantization awq")


if __name__ == "__main__":
    main()
