#!/usr/bin/env bash
# Phase 1: the baseline runs of the unedited models. Run on the GPU machine:
#
#   scripts/phase1_baselines.sh            start everything inside tmux session `bench`, then return
#   tmux attach -t bench                   watch (Ctrl-b d to leave it running)
#   tail -f logs/phase1.log                or follow the log
#
# Two chains run side by side, one tmux window each, because the GPU has room for both:
#
#   qwen   qwen-9b-full, qwen-9b-q4      on dev and test
#   gemma  gemma-4b-full, gemma-4b-q4    on dev, test and gemma_extra
#
# Every run includes the damage checks. Within a chain the runs go one after another. A run
# that already has a finished result (a summary.json for that model and split, made without
# --limit and with damage checks) is skipped, so the script can simply be started again after
# an interruption. When both chains have finished and all ten results exist, the file
# logs/phase1.done is written; if a run failed, logs/phase1.failed is written instead.
#
# Each llama.cpp run starts its own server on a free port chosen by the operating system, so
# the two chains cannot collide.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/phase1.log"
SESSION=bench
cd "$ROOT"
mkdir -p "$LOG_DIR"

models_of_chain() {
  case "$1" in
    qwen)  echo "qwen-9b-full qwen-9b-q4" ;;
    gemma) echo "gemma-4b-full gemma-4b-q4" ;;
    *)     echo "unknown chain: $1" >&2; exit 2 ;;
  esac
}

splits_of_chain() {
  case "$1" in
    qwen)  echo "dev test" ;;
    gemma) echo "dev test gemma_extra" ;;
  esac
}

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# Prints the folder of a finished full run of model $1 on split $2, or nothing.
finished_run() {
  uv run --quiet python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path

model, split = sys.argv[1:3]
for run_dir in sorted(Path("runs").glob(f"*_{model}_{split}*")):
    try:
        info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continue
    full = info.get("model") == model and info.get("split") == split \
        and info.get("limit") is None and not info.get("skip_damage")
    if full and (run_dir / "summary.json").exists() and (run_dir / "damage.json").exists():
        print(run_dir)
        break
PY
}

run_chain() {
  local chain="$1" model split done_dir
  for model in $(models_of_chain "$chain"); do
    for split in $(splits_of_chain "$chain"); do
      done_dir="$(finished_run "$model" "$split")"
      if [ -n "$done_dir" ]; then
        say "$chain: skip $model $split, already in $done_dir"
        continue
      fi
      say "$chain: start $model $split"
      if uv run bench run "$model" --split "$split" 2>&1 | grep --line-buffered -v "Loading weights" \
           | sed -u "s/^/[$model $split] /" | tee -a "$LOG"; [ "${PIPESTATUS[0]}" -eq 0 ]; then
        say "$chain: done $model $split"
      else
        say "$chain: FAILED $model $split"
      fi
    done
  done
  say "$chain: chain finished"
}

# Waits for both chain windows to end, then writes the marker file.
finish() {
  local chain model split missing=0
  while tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -qE '^(qwen|gemma)$'; do
    sleep 30
  done
  for chain in qwen gemma; do
    for model in $(models_of_chain "$chain"); do
      for split in $(splits_of_chain "$chain"); do
        if [ -z "$(finished_run "$model" "$split")" ]; then
          say "missing result: $model $split"
          missing=1
        fi
      done
    done
  done
  if [ "$missing" -eq 0 ]; then
    rm -f "$LOG_DIR/phase1.failed"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$LOG_DIR/phase1.done"
    say "phase 1 complete: all ten baseline runs have results"
  else
    date -u +%Y-%m-%dT%H:%M:%SZ > "$LOG_DIR/phase1.failed"
    say "phase 1 NOT complete: see the missing results above; start the script again to retry them"
  fi
}

start_in_tmux() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2
    exit 1
  fi
  rm -f "$LOG_DIR/phase1.done" "$LOG_DIR/phase1.failed"
  say "phase 1 started"
  # The chain windows close when their chain ends; `finish` watches for that.
  tmux new-session -d -s "$SESSION" -n qwen "$0 chain qwen"
  tmux new-window -d -t "$SESSION" -n gemma "$0 chain gemma"
  tmux new-window -d -t "$SESSION" -n finish "$0 finish"
  echo "started; watch with: tmux attach -t $SESSION   or: tail -f $LOG"
}

case "${1:-start}" in
  start)  start_in_tmux ;;
  chain)  run_chain "$2" ;;
  finish) finish ;;
  *)      echo "usage: $0 [start | chain qwen|gemma | finish]" >&2; exit 2 ;;
esac
