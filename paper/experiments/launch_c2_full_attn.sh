#!/bin/bash
# Wait for full_attn seed137 to finish, then launch C2 full_attn on GPUs 0-3
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Waiting for experiment_c full_attn seed137 to finish..."
while true; do
    N=$(ps aux | grep 'experiment_c.py.*full_attn' | grep -v grep | grep -v experiment_c2 | wc -l)
    if [ "$N" -eq 0 ]; then
        log "full_attn seed137 done. GPUs 0-3 free."
        break
    fi
    log "  Still running ($N processes). Checking in 60s..."
    sleep 60
done

sleep 10
log "Launching C2 full_attn on GPUs 0-3..."
cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
    experiment_c2.py \
    --mode full_attn \
    > logs/expC2_7b_full_attn.log 2>&1

log "C2 full_attn finished (exit code: $?)"
