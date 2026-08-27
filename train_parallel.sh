#!/bin/bash
# Run N independent self-play workers to fill the opening book (book.db).
#
# Each search is single-threaded on purpose (see CLAUDE.md): throughput comes from
# many concurrent games, not from parallelising one search. Every worker is its own
# process with its own in-memory book; they share book.db, which is in WAL
# mode and only ever receives the entries a worker newly computed.
#
#   ./train_parallel.sh                  # 20 workers at depth 10
#   ./train_parallel.sh 8 12             # 8 workers at depth 12
#   WORKERS=20 DEPTH=10 ./train_parallel.sh

set -u

cd "$(dirname "$0")" || exit 1

WORKERS="${1:-${WORKERS:-20}}"
DEPTH="${2:-${DEPTH:-10}}"
EXPLORATION="${EXPLORATION:-0.25}"
RANDOM_PLIES="${RANDOM_PLIES:-2}"
LOG_DIR="${LOG_DIR:-logs/selfplay}"
PID_FILE="training_workers.pid"

PY="./venv/bin/python"
[ -x "$PY" ] || { echo "No $PY — the minichess_engine extension lives in the project venv."; exit 1; }

if [ -s "$PID_FILE" ] && xargs -r -a "$PID_FILE" ps -o pid= -p >/dev/null 2>&1; then
    echo "Workers already running (see $PID_FILE). Run ./train_stop.sh first."
    exit 1
fi

CORES=$(nproc)
if [ "$WORKERS" -gt "$CORES" ]; then
    echo "Warning: $WORKERS workers on $CORES cores — they will contend for CPU."
fi

mkdir -p "$LOG_DIR"
: > "$PID_FILE"

echo "=================================="
echo "Self-play: $WORKERS workers, depth $DEPTH"
echo "  exploration:   $EXPLORATION"
echo "  random plies:  $RANDOM_PLIES"
echo "  logs:          $LOG_DIR/worker-N.log"
echo "  database:      book.db"
echo "=================================="

for i in $(seq 1 "$WORKERS"); do
    # stderr carries the engine's per-depth chatter; keep it out of the logs.
    nohup "$PY" -u src/self_play.py \
        --depth "$DEPTH" \
        --exploration "$EXPLORATION" \
        --random-plies "$RANDOM_PLIES" \
        --quiet \
        > "$LOG_DIR/worker-$i.log" 2>/dev/null &
    echo $! >> "$PID_FILE"
done

echo "Started $WORKERS workers. PIDs in $PID_FILE."
echo
echo "  tail -f $LOG_DIR/worker-1.log            # watch one worker"
echo "  ./train_status.sh                        # cache growth + worker count"
echo "  ./train_stop.sh                          # graceful stop (SIGTERM)"
