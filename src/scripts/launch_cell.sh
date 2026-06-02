#!/usr/bin/env bash
# launch_cell.sh — start one experiment cell as a detached nohup background job.
#
# Positional args (all required, in order):
#   1. RUN_ROOT          full path under ${PROJECT_ROOT}/runs/...
#   2. MANIFEST          full path to manifest CSV
#   3. IMG_DIR           full path to flat image directory
#   4. BACKBONE          ViT-L-14 / ViT-L-14-laion2b / EVA02-L-14 / ViT-B-32
#   5. EPS               integer (4, 8, 16, ...)
#   6. ALPHA             integer (default 4)
#   7. IMAGE_SIZE        integer (224 or 800)
#   8. ITERS             integer (200 default)
#   9. GPU               integer GPU index (0..6)
#  10. METHODS           space-separated, e.g. "cogeo pgd_bare advclip coattack"
#
# The cell is launched detached. Its log is at $RUN_ROOT/cell.log and its PID
# is written to $RUN_ROOT/cell.pid. Health markers are heartbeats inside log.
set -euo pipefail
[[ $# -ge 10 ]] || { echo "usage: launch_cell.sh RUN_ROOT MANIFEST IMG_DIR BACKBONE EPS ALPHA IMAGE_SIZE ITERS GPU METHODS"; exit 64; }
RUN_ROOT="$1"; MANIFEST="$2"; IMG_DIR="$3"; BACKBONE="$4"; EPS="$5"; ALPHA="$6"; IMAGE_SIZE="$7"; ITERS="$8"; GPU="$9"; METHODS="${10}"

mkdir -p "$RUN_ROOT"
LOG="$RUN_ROOT/cell.log"
PIDFILE="$RUN_ROOT/cell.pid"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# detach via setsid+nohup so closing the parent shell doesn't kill us
nohup setsid env \
  RUN_ROOT="$RUN_ROOT" \
  MANIFEST="$MANIFEST" \
  IMG_DIR="$IMG_DIR" \
  BACKBONE="$BACKBONE" \
  EPS="$EPS" \
  ALPHA="$ALPHA" \
  IMAGE_SIZE="$IMAGE_SIZE" \
  ITERS="$ITERS" \
  GPU="$GPU" \
  METHODS="$METHODS" \
  CODE_DIR="$SCRIPT_DIR" \
  bash "$SCRIPT_DIR/run_4way.sh" >>"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "[launch_cell] launched PID=$PID RUN_ROOT=$RUN_ROOT BACKBONE=$BACKBONE EPS=$EPS IMAGE_SIZE=$IMAGE_SIZE GPU=$GPU METHODS=\"$METHODS\""
echo "[launch_cell] log: $LOG"
disown -h "$PID" 2>/dev/null || true
exit 0
