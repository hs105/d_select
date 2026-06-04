#!/bin/bash
# ============================================================
# Launch long-context evaluation for Thin Keys paper
# Addresses reviewer criticism #1: no long-context evaluation
# ============================================================
#
# Usage:
#   # Full evaluation (baseline + r512 + r256) on GPU 0
#   bash launch_longcontext.sh
#
#   # Quick test
#   bash launch_longcontext.sh --quick
#
#   # Custom GPU
#   CUDA_DEVICE=cuda:1 bash launch_longcontext.sh
#
#   # PPL only (fastest)
#   bash launch_longcontext.sh --eval ppl
#
#   # With fine-tuned checkpoint
#   CHECKPOINT_DIR=/sg-pretrain/focus/checkpoints_7b bash launch_longcontext.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/sg-pretrain/models/mistral-7b}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/logs/longcontext_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

# Ranks to evaluate: baseline (1024) + compressed
RANKS="${RANKS:-1024,512,256}"

echo "============================================================"
echo "Long-Context Evaluation for Thin Keys"
echo "============================================================"
echo "  Model:      ${MODEL_PATH}"
echo "  Ranks:      ${RANKS}"
echo "  Device:     ${CUDA_DEVICE}"
echo "  Save dir:   ${SAVE_DIR}"
echo "  Extra args: $@"
echo "============================================================"

mkdir -p "${SAVE_DIR}"

# Build command
CMD="python ${SCRIPT_DIR}/eval_longcontext.py \
    --model_path ${MODEL_PATH} \
    --ranks ${RANKS} \
    --device ${CUDA_DEVICE} \
    --save_dir ${SAVE_DIR}"

if [ -n "${CHECKPOINT_DIR}" ]; then
    CMD="${CMD} --checkpoint_dir ${CHECKPOINT_DIR}"
fi

# Pass through any extra args (--quick, --eval ppl, etc.)
CMD="${CMD} $@"

echo ""
echo "Running: ${CMD}"
echo ""

# Log to file and stdout
${CMD} 2>&1 | tee "${SAVE_DIR}/run.log"

echo ""
echo "Results saved to ${SAVE_DIR}"
echo "============================================================"
