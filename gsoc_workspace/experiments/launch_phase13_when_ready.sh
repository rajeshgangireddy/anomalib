#!/usr/bin/env bash
# Wait for AnomalyAny generation to finish, then launch the phase13 sweep.
# Idempotent: safe to re-run; the sweep itself is resumable (skips existing JSONs).
set -u

TOTAL=705
ROOT="/home/devuser/workspace/code/Anomalib/anomalib"
GEN_ROOT="$ROOT/datasets/SynthetciGenMVAD2"
LOG="/tmp/phase13_auto.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_images() {
  find "$GEN_ROOT" -type d -name "anomalyany_*" -prune -o -type f -path "*anomalyany_*" -name "*.png" -print 2>/dev/null | wc -l
}

gen_alive() {
  pgrep -f "generate_mvtec2.py" >/dev/null 2>&1 && echo 1 || echo 0
}

log "watcher started (TOTAL=$TOTAL)"

while true; do
  count=$(count_images)
  alive=$(gen_alive)
  log "images=$count / $TOTAL, generation_alive=$alive"

  if [ "$count" -ge "$TOTAL" ]; then
    log "all $TOTAL images present -> launching phase13 sweep"
    break
  fi
  if [ "$alive" -eq 0 ]; then
    log "ERROR: generation processes exited with only $count/$TOTAL images. Aborting (fix and re-run)."
    exit 1
  fi
  sleep 600
done

# Let the generation process fully drain (final gc/empty_cache/exit) before the sweep
# opens the files, avoiding any partial-write race on the last image.
sleep 60
log "generation drained; proceeding"

cd "$ROOT" || exit 1
log "running: python -m gsoc_workspace.experiments.sweep --phase phase13_anomalyany --gpus 0 1 --procs-per-gpu 1"
python -m gsoc_workspace.experiments.sweep \
  --phase phase13_anomalyany \
  --gpus 0 1 \
  --procs-per-gpu 1 >> "$LOG" 2>&1
log "phase13 sweep finished (exit=$?)"
