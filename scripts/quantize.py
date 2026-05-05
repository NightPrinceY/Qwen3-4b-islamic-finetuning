"""
Step 4 — Quantize merged model to GGUF + INT4 using llama.cpp.

AutoAWQ and llm-compressor are both broken on transformers>=4.52.
llama.cpp is self-contained and reliable for all quantization needs.

Outputs:
  outputs/gguf/qwen3-4b-islamic-f16.gguf      — full precision GGUF
  outputs/gguf/qwen3-4b-islamic-q4_k_m.gguf   — 4-bit (recommended for Ollama/LM Studio)
  outputs/gguf/qwen3-4b-islamic-q8_0.gguf     — 8-bit near-lossless

Prerequisites:
  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp --depth=1
  pip install -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

Run:
    python scripts/quantize.py
"""
import subprocess
import sys
from pathlib import Path

MERGED_DIR   = "outputs/merged"
GGUF_DIR     = "outputs/gguf"
LLAMA_CPP    = str(Path.home() / "llama.cpp")
PYTHON       = sys.executable

QUANT_TYPES  = ["Q4_K_M", "Q8_0"]


def run(cmd: list, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    Path(GGUF_DIR).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Convert HF model → F16 GGUF ─────────────────────────────────
    f16_path = f"{GGUF_DIR}/qwen3-4b-islamic-f16.gguf"
    print(f"\nConverting {MERGED_DIR} → {f16_path} ...")
    run([
        PYTHON, f"{LLAMA_CPP}/convert_hf_to_gguf.py",
        MERGED_DIR,
        "--outtype", "f16",
        "--outfile", f16_path,
    ])

    # ── Step 2: Quantize F16 → Q4_K_M and Q8_0 ──────────────────────────────
    quantize_bin = f"{LLAMA_CPP}/build/bin/llama-quantize"
    if not Path(quantize_bin).exists():
        # Try pre-built path
        quantize_bin = f"{LLAMA_CPP}/llama-quantize"

    if Path(quantize_bin).exists():
        for quant in QUANT_TYPES:
            out = f"{GGUF_DIR}/qwen3-4b-islamic-{quant.lower()}.gguf"
            print(f"\nQuantizing → {out} ...")
            run([quantize_bin, f16_path, out, quant])
    else:
        print("\nllama-quantize binary not found — skipping Q4/Q8 quantization.")
        print("Build it with:")
        print(f"  cd {LLAMA_CPP} && cmake -B build && cmake --build build --config Release -j$(nproc)")
        print(f"Then re-run this script to quantize the F16 GGUF.")

    print(f"\nDone. Files in {GGUF_DIR}/:")
    for f in sorted(Path(GGUF_DIR).glob("*.gguf")):
        size_gb = f.stat().st_size / 1024**3
        print(f"  {f.name}  ({size_gb:.1f} GB)")

    print("\nRun with Ollama:")
    print(f"  ollama run {GGUF_DIR}/qwen3-4b-islamic-q4_k_m.gguf")


if __name__ == "__main__":
    main()
