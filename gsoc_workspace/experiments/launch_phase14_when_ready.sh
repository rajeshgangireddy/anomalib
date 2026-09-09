#!/usr/bin/env bash
# Wait for AnoStyler generation to finish, then launch the phase14 sweep.
# Idempotent: safe to re-run; the sweep itself is resumable (skips existing JSONs).
set -u

TOTAL=2115
ROOT="/home/devuser/workspace/code/Anomalib/anomalib"
GEN_ROOT="$ROOT/datasets/SynthetciGenMVAD2"
LOG="/tmp/phase14_auto.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_files() {
  find "$GEN_ROOT" -type f -path "*anostyler_*" -name "*.png" 2>/dev/null | wc -l
}

gen_alive() {
  pgrep -f "generate_mvtec2_anostyler.py" >/dev/null 2>&1 && echo 1 || echo 0
}

log "watcher started (expect $((TOTAL * 2)) files = $TOTAL images + $TOTAL masks)"

while true; do
  count=$(count_files)
  alive=$(gen_alive)
  log "files=$count / $((TOTAL * 2)), generation_alive=$alive"

  if [ "$count" -ge "$((TOTAL * 2))" ]; then
    log "all files present -> launching phase14 sweep"
    break
  fi
  if [ "$alive" -eq 0 ]; then
    log "ERROR: generation processes exited with only $count files. Aborting (fix and re-run)."
    exit 1
  fi
  sleep 300
done

sleep 30
log "generation drained; proceeding"

cd "$ROOT" || exit 1
log "running: python -m gsoc_workspace.experiments.sweep --phase phase14_anostyler --gpus 0 1 --procs-per-gpu 1"
python -m gsoc_workspace.experiments.sweep \
  --phase phase14_anostyler \
  --gpus 0 1 \
  --procs-per-gpu 1 >> "$LOG" 2>&1
log "phase14 sweep finished (exit=$?)"
