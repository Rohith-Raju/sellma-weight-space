#!/usr/bin/env bash
# The whole overnight run as one chain. Run on the GPU machine:
#
#   scripts/overnight.sh                   start inside tmux session `bench` (window `overnight`), then return
#   tmux attach -t bench                   watch (Ctrl-b d to leave it running)
#   tail -f logs/overnight.log             or follow the log
#
# Steps, in order:
#
#   1. Phase 1 baselines (scripts/phase1_baselines.sh). If that script is already running in the
#      session, this one waits for it; otherwise it runs the two chains itself.
#   2. The edit experiment for both models, side by side:
#        uv run bench experiment all --model qwen-9b-full --combined    -> logs/experiment_qwen-9b-full.log
#        uv run bench experiment all --model gemma-4b-full --combined   -> logs/experiment_gemma-4b-full.log
#
# When everything has finished, logs/overnight.done is written; if a step failed,
# logs/overnight.failed is written instead and the log says which step.
#
# Safe to start again after an interruption: finished baseline runs are skipped, and
# `bench experiment` skips every step whose files are complete.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/overnight.log"
SESSION=bench
WINDOW=overnight
PHASE1="$ROOT/scripts/phase1_baselines.sh"
EXPERIMENT_MODELS="qwen-9b-full gemma-4b-full"
cd "$ROOT"
mkdir -p "$LOG_DIR"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

fail() {
  say "FAILED: $*"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$LOG_DIR/overnight.failed"
  exit 1
}

phase1_windows_running() {
  tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -qE '^(qwen|gemma|finish)$'
}

# Runs both baseline chains here, side by side, then lets the phase 1 script check the results.
run_phase1_here() {
  rm -f "$LOG_DIR/phase1.done" "$LOG_DIR/phase1.failed"
  "$PHASE1" chain qwen &
  "$PHASE1" chain gemma &
  wait
  "$PHASE1" finish
}

phase1() {
  if [ -e "$LOG_DIR/phase1.done" ]; then
    say "phase 1: already complete"
    return
  fi
  if phase1_windows_running; then
    say "phase 1: running in other windows of this session, waiting for it"
    while phase1_windows_running && [ ! -e "$LOG_DIR/phase1.done" ] && [ ! -e "$LOG_DIR/phase1.failed" ]; do
      sleep 30
    done
  fi
  if [ ! -e "$LOG_DIR/phase1.done" ]; then
    say "phase 1: running the baseline chains (finished runs are skipped)"
    run_phase1_here
  fi
  [ -e "$LOG_DIR/phase1.done" ] || fail "phase 1 baselines are incomplete, see logs/phase1.log"
  say "phase 1: complete"
}

run_experiment() {
  local model="$1" log="$LOG_DIR/experiment_$1.log"
  say "experiment $model: start (log: $log)"
  uv run bench experiment all --model "$model" --combined 2>&1 | grep --line-buffered -v "Loading weights" | tee -a "$log"
  local status="${PIPESTATUS[0]}"
  say "experiment $model: finished with exit code $status"
  return "$status"
}

experiments() {
  local model pids="" failed=0
  for model in $EXPERIMENT_MODELS; do
    run_experiment "$model" &
    pids="$pids $!"
  done
  for pid in $pids; do
    wait "$pid" || failed=1
  done
  [ "$failed" -eq 0 ] || fail "an experiment did not finish, see logs/experiment_<model>.log; start this script again to resume"
}

run_all() {
  rm -f "$LOG_DIR/overnight.done" "$LOG_DIR/overnight.failed"
  say "overnight chain started"
  phase1
  experiments
  date -u +%Y-%m-%dT%H:%M:%SZ > "$LOG_DIR/overnight.done"
  say "overnight chain complete"
}

start_in_tmux() {
  if tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -qx "$WINDOW"; then
    echo "window '$WINDOW' already exists in tmux session '$SESSION'; attach with: tmux attach -t $SESSION" >&2
    exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -d -t "$SESSION" -n "$WINDOW" "$0 run"
  else
    tmux new-session -d -s "$SESSION" -n "$WINDOW" "$0 run"
  fi
  echo "started; watch with: tmux attach -t $SESSION   or: tail -f $LOG"
}

case "${1:-start}" in
  start) start_in_tmux ;;
  run)   run_all ;;
  *)     echo "usage: $0 [start | run]" >&2; exit 2 ;;
esac
