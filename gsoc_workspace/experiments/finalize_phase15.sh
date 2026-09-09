#!/usr/bin/env bash
# Finalize phase 15 (Dinomaly): wait for the remote sweep, re-run any stragglers,
# collect the 24 phase15 JSONs to local, and aggregate results.csv.
set -u

ROOT="/home/devuser/workspace/code/Anomalib/anomalib"
REMOTE_HOST="Cluster2"
REMOTE_ROOT="/srv/research/experiments/anomalib_synthetic_anomaly/code"
LOG="/tmp/phase15_finalize.log"
EXPECT=24

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "waiting for remote phase15 sweep"
while true; do
  alive=$(ssh -o BatchMode=yes "$REMOTE_HOST" "pgrep -f '[s]weep --phase phase15' >/dev/null && echo 1 || echo 0" 2>/dev/null)
  done=$(ssh -o BatchMode=yes "$REMOTE_HOST" "ls $REMOTE_ROOT/gsoc_workspace/experiments/results/raw/phase15_*.json 2>/dev/null | wc -l" 2>/dev/null)
  done=${done:-0}
  if [ "$alive" = "0" ]; then break; fi
  sleep 180
done
log "remote sweep finished: $done/$EXPECT JSONs"

# safety net: re-run missing jobs (resumable, skips existing)
if [ "$done" -lt "$EXPECT" ]; then
  log "re-running remote sweep for $((EXPECT - done)) missing jobs"
  ssh -o BatchMode=yes "$REMOTE_HOST" "cd $REMOTE_ROOT && .venv/bin/python -m gsoc_workspace.experiments.sweep --phase phase15_dinomaly --gpus 0 1 2 3 4 5 --procs-per-gpu 1" >> "$LOG" 2>&1
fi

log "collecting remote phase15 JSONs"
scp -o BatchMode=yes "$REMOTE_HOST:$REMOTE_ROOT/gsoc_workspace/experiments/results/raw/phase15_*.json" \
  "$ROOT/gsoc_workspace/experiments/results/raw/" >> "$LOG" 2>&1
total=$(ls "$ROOT"/gsoc_workspace/experiments/results/raw/phase15_*.json 2>/dev/null | wc -l)
log "total phase15 JSONs local: $total (expect $EXPECT)"

cd "$ROOT" || exit 1
python -m gsoc_workspace.experiments.sweep --aggregate-only >> "$LOG" 2>&1
log "phase15 COMPLETE"
