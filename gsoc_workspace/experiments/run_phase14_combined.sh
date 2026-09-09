#!/usr/bin/env bash
# Phase 14 combined-machine orchestration: wait for AnoStyler generation, sync images
# to the remote cluster, then run the sweep on local (2 GPUs) + remote (6 GPUs) in
# parallel, collect the remote results, and aggregate everything into results.csv.
#
# Partition (mirrors phase 12): local = {can, vial} (24 jobs), remote = the other 6
# categories (72 jobs). Both write to their own results/raw/; this script merges them.
set -u

TOTAL=2115                       # images (masks add the same count again)
ROOT="/home/devuser/workspace/code/Anomalib/anomalib"
GEN_ROOT="$ROOT/datasets/SynthetciGenMVAD2"
REMOTE_HOST="Cluster2"
REMOTE_ROOT="/srv/research/experiments/anomalib_synthetic_anomaly/code"
REMOTE_DS="$REMOTE_ROOT/datasets/SynthetciGenMVAD2"
LOG="/tmp/phase14_combined.log"

LOCAL_CATS="can vial"
REMOTE_CATS="fabric fruit_jelly rice sheet_metal wallplugs walnuts"
LOCAL_JOBS=24
REMOTE_JOBS=72

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_files() { find "$GEN_ROOT" -type f -path "*anostyler_*" -name "*.png" 2>/dev/null | wc -l; }
gen_alive() { pgrep -f "generate_mvtec2_anostyler.py" >/dev/null 2>&1 && echo 1 || echo 0; }

# ---- 1. wait for generation ----
log "waiting for $((TOTAL * 2)) files (images + masks)"
while true; do
  c=$(count_files); a=$(gen_alive)
  if [ "$c" -ge "$((TOTAL * 2))" ]; then log "generation complete ($c files)"; break; fi
  if [ "$a" -eq 0 ]; then log "ERROR: generation died at $c files; aborting"; exit 1; fi
  sleep 300
done
sleep 30
log "generation drained"

# ---- 2. sync images to remote ----
log "scp anostyler images -> remote"
scp -r -o BatchMode=yes "$GEN_ROOT"/MVTec_AD_2_anostyler_* "$REMOTE_HOST:$REMOTE_DS/" >> "$LOG" 2>&1
remote_count=$(ssh -o BatchMode=yes "$REMOTE_HOST" "find $REMOTE_DS -path '*anostyler_*' -name '*.png' | wc -l" 2>/dev/null)
log "remote anostyler files after sync: $remote_count (expect $((TOTAL * 2)))"

# ---- 3. launch local + remote sweeps in parallel ----
log "launching LOCAL sweep: $LOCAL_CATS (gpus 0 1)"
cd "$ROOT" || exit 1
nohup python -m gsoc_workspace.experiments.sweep \
  --phase phase14_anostyler --gpus 0 1 --procs-per-gpu 2 \
  --categories $LOCAL_CATS > /tmp/phase14_local.log 2>&1 &
LOCAL_PID=$!
log "local sweep pid=$LOCAL_PID"

log "launching REMOTE sweep: $REMOTE_CATS (gpus 0-5)"
# Detach the remote sweep so it survives this ssh session (verified pattern: ssh must
# redirect its own stdin/stdout/stderr, and the remote cmd is wrapped in `sh -c` with
# its own streams redirected -- otherwise ssh blocks until the remote job finishes).
ssh -o BatchMode=yes "$REMOTE_HOST" \
  'nohup sh -c "cd /srv/research/experiments/anomalib_synthetic_anomaly/code && .venv/bin/python -m gsoc_workspace.experiments.sweep --phase phase14_anostyler --gpus 0 1 2 3 4 5 --procs-per-gpu 2 --categories fabric fruit_jelly rice sheet_metal wallplugs walnuts > /tmp/phase14_remote.log 2>&1" </dev/null >/dev/null 2>&1 &' \
  </dev/null >/dev/null 2>&1
log "remote sweep launched"

# ---- 4. wait for local sweep ----
log "waiting for local sweep (pid=$LOCAL_PID)"
while kill -0 "$LOCAL_PID" 2>/dev/null; do sleep 60; done
local_done=$(ls "$ROOT"/gsoc_workspace/experiments/results/raw/phase14_*.json 2>/dev/null | wc -l)
log "local sweep finished: $local_done phase14 JSONs (expect $LOCAL_JOBS)"

# ---- 5. wait for remote sweep ----
# `pgrep -f '[p]hase14_anostyler'` uses the bracket trick so the poll's own ssh-spawned
# shell (whose cmdline contains the literal "[p]hase14_anostyler") does not self-match.
log "waiting for remote sweep"
while true; do
  remote_alive=$(ssh -o BatchMode=yes "$REMOTE_HOST" "pgrep -f '[p]hase14_anostyler' >/dev/null && echo 1 || echo 0" 2>/dev/null)
  remote_done=$(ssh -o BatchMode=yes "$REMOTE_HOST" "ls $REMOTE_ROOT/gsoc_workspace/experiments/results/raw/phase14_*.json 2>/dev/null | wc -l" 2>/dev/null)
  remote_done=${remote_done:-0}
  log "remote: alive=$remote_alive json=$remote_done/$REMOTE_JOBS"
  if [ "$remote_alive" = "0" ] && [ "$remote_done" -ge "$REMOTE_JOBS" ]; then
    log "remote sweep finished ($remote_done JSONs)"
    break
  fi
  if [ "$remote_alive" = "0" ] && [ "$remote_done" -lt "$REMOTE_JOBS" ]; then
    log "ERROR: remote sweep stopped with only $remote_done JSONs; check /tmp/phase14_remote.log"
    exit 1
  fi
  sleep 180
done

# ---- 6. collect remote results + aggregate ----
log "scp remote phase14 results -> local"
scp -o BatchMode=yes "$REMOTE_HOST:$REMOTE_ROOT/gsoc_workspace/experiments/results/raw/phase14_*.json" \
  "$ROOT/gsoc_workspace/experiments/results/raw/" >> "$LOG" 2>&1
total_done=$(ls "$ROOT"/gsoc_workspace/experiments/results/raw/phase14_*.json 2>/dev/null | wc -l)
log "total phase14 JSONs local: $total_done (expect $((LOCAL_JOBS + REMOTE_JOBS)) = 96)"

log "aggregating results.csv"
cd "$ROOT" || exit 1
python -m gsoc_workspace.experiments.sweep --aggregate-only >> "$LOG" 2>&1
log "phase14 combined run COMPLETE"
