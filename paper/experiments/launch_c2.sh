#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Automated C2 launcher: wait for C seed137 → smoke test → full run
# ══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Step 1: Wait for experiment_c seed137 runs to finish ──
log "═══ WAITING FOR EXPERIMENT C SEED137 TO FINISH ═══"

while true; do
    N=$(ps aux | grep 'experiment_c.py' | grep -v grep | grep -v experiment_c2 | wc -l)
    if [ "$N" -eq 0 ]; then
        log "All experiment_c processes have exited."
        break
    fi
    log "  Still running ($N processes). Checking again in 60s..."
    sleep 60
done

# Brief pause to let GPU memory fully release
sleep 10

# Verify GPUs are free
log "Verifying GPUs are free..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END{print sum}')
log "  Total GPU memory in use: ${GPU_MEM} MiB"
if [ "$GPU_MEM" -gt 5000 ]; then
    log "WARNING: GPUs still have ${GPU_MEM} MiB in use. Waiting 30s..."
    sleep 30
fi

# ── Step 2: Smoke test ──
log "═══ RUNNING SMOKE TEST (100 steps, 2 GPUs each) ═══"
cd "$SCRIPT_DIR"
bash run_experiment_c2.sh smoke
SMOKE_EXIT=$?

if [ "$SMOKE_EXIT" -ne 0 ]; then
    log "ERROR: Smoke test failed with exit code $SMOKE_EXIT"
    log "Check logs:"
    log "  ${LOG_DIR}/expC2_7b_full_attn_smoke.log"
    log "  ${LOG_DIR}/expC2_7b_thin1024_smoke.log"
    exit 1
fi

log "Smoke test passed! Verifying checkpoints..."
ls -lh /sg-pretrain/checkpoints/expC2_7b/ 2>/dev/null || true

# ── Step 3: Clean up smoke checkpoints ──
log "Cleaning up smoke checkpoints..."
rm -rf /sg-pretrain/checkpoints/expC2_7b/
log "  Done."

# ── Step 4: Launch full C2 run ──
log "═══ LAUNCHING FULL EXPERIMENT C2 (20B tokens, ~10.6 days) ═══"
cd "$SCRIPT_DIR"
bash run_experiment_c2.sh
FULL_EXIT=$?

log "═══ EXPERIMENT C2 FINISHED (exit code: $FULL_EXIT) ═══"
