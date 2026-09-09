#!/usr/bin/env bash
# Finalize phase 14: wait for the local + remote sweeps (now at --procs-per-gpu 1,
# memory-safe), re-run any straggler/missing jobs, collect remote JSONs, aggregate.
set -u

ROOT="/home/devuser/workspace/code/Anomalib/anomalib"
REMOTE_HOST="Cluster2"
REMOTE_ROOT="/srv/research/experiments/anomalib_synthetic_anomaly/code"
LOG="/tmp/phase14_finalize.log"
LOCAL_JOBS=24
REMOTE_JOBS=72

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- 1. wait for local sweep ----
log "waiting for local sweep"
while pgrep -f "experiments.sweep --phase phase14_anostyler" >/dev/null 2>&1; do sleep 60; done
local_done=$(ls "$ROOT"/gsoc_workspace/experiments/results/raw/phase14_*.json 2>/dev/null | wc -l)
log "local sweep done: $local_done/$LOCAL_JOBS JSONs"

# ---- 2. wait for remote sweep ----
log "waiting for remote sweep"
while true; do
  remote_alive=$(ssh -o BatchMode=yes "$REMOTE_HOST" "pgrep -f '[p]hase14_anostyler' >/dev/null && echo 1 || echo 0" 2>/dev/null)
  remote_done=$(ssh -o BatchMode=yes "$REMOTE_HOST" "ls $REMOTE_ROOT/gsoc_workspace/experiments/results/raw/phase14_*.json 2>/dev/null | wc -l" 2>/dev/null)
  remote_done=${remote_done:-0}
  if [ "$remote_alive" = "0" ]; then break; fi
  sleep 180
done
log "remote sweep done: $remote_done/$REMOTE_JOBS JSONs"

# ---- 3. re-run missing jobs (safety net; sweep is resumable, skips existing) ----
if [ "$local_done" -lt "$LOCAL_JOBS" ]; then
  log "re-running LOCAL sweep for $((LOCAL_JOBS - local_done)) missing jobs"
  cd "$ROOT" || exit 1
  python -m gsoc_workspace.experiments.sweep --phase phase14_anostyler --gpus 0 1 --procs-per-gpu 1 --categories can vial >> "$LOG" 2>&1
fi
if [ "$remote_done" -lt "$REMOTE_JOBS" ]; then
  log "re-running REMOTE sweep for $((REMOTE_JOBS - remote_done)) missing jobs"
  ssh -o BatchMode=yes "$REMOTE_HOST" "cd $REMOTE_ROOT && .venv/bin/python -m gsoc_workspace.experiments.sweep --phase phase14_anostyler --gpus 0 1 2 3 4 5 --procs-per-gpu 1 --categories fabric fruit_jelly rice sheet_metal wallplugs walnuts" >> "$LOG" 2>&1
fi

# ---- 4. collect remote + aggregate ----
log "collecting remote JSONs"
scp -o BatchMode=yes "$REMOTE_HOST:$REMOTE_ROOT/gsoc_workspace/experiments/results/raw/phase14_*.json" \
  "$ROOT/gsoc_workspace/experiments/results/raw/" >> "$LOG" 2>&1
total=$(ls "$ROOT"/gsoc_workspace/experiments/results/raw/phase14_*.json 2>/dev/null | wc -l)
log "total phase14 JSONs local: $total (expect $((LOCAL_JOBS + REMOTE_JOBS)))"

cd "$ROOT" || exit 1
python -m gsoc_workspace.experiments.sweep --aggregate-only >> "$LOG" 2>&1
log "phase14 COMPLETE"
