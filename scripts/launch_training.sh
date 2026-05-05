#!/bin/bash
# Full 4-GPU QLoRA training launch script
# Usage: bash scripts/launch_training.sh

set -e

# ── CUDA 13 libs required by bitsandbytes ─────────────────────────────────────
CUDA13_LIBS="/home/yahya/CollegeX/lib/python3.11/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA13_LIBS"

# ── GPUs: skip GPU0 (often occupied), use GPUs 1-4 ───────────────────────────
export CUDA_VISIBLE_DEVICES=1,2,3,4

# ── Load HuggingFace token ─────────────────────────────────────────────────────
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# ── NCCL settings for RTX 2080 Ti ─────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0

# ── Python executable ─────────────────────────────────────────────────────────
PYTHON=/home/yahya/CollegeX/bin/python

echo "============================================================"
echo "  Qwen3-4B Islamic Arabic Fine-Tuning"
echo "  GPUs      : $CUDA_VISIBLE_DEVICES"
echo "  Python    : $PYTHON"
echo "  Started   : $(date)"
echo "============================================================"

$PYTHON -m accelerate.commands.launch \
    --config_file configs/accelerate_multigpu.yaml \
    scripts/train.py \
    2>&1 | tee outputs/logs/training_$(date +%Y%m%d_%H%M%S).log

echo "============================================================"
echo "  Training complete: $(date)"
echo "============================================================"
