#!/bin/bash
# Stop the self-play workers started by train_parallel.sh.
# SIGTERM triggers the graceful shutdown path, which flushes the cache to the DB.
#
# The grace period matters more than it looks. `shutdown_requested` in
# src/self_play.py is only checked at the top of the move loop -- the search itself
# is a blocking Rust call that never sees the flag -- so a worker cannot react to
# SIGTERM until its current move finishes. At depth 10 that has been observed to
# take over 2000s in a full-hand midgame, so size the timeout to the search depth,
# not to how long you feel like waiting.
#
#   ./train_stop.sh              # wait up to 600s per worker
#   ./train_stop.sh 2000         # wait up to 2000s
#   TIMEOUT=2000 ./train_stop.sh

set -u
cd "$(dirname "$0")" || exit 1

TIMEOUT="${1:-${TIMEOUT:-600}}"
case "$TIMEOUT" in
    ''|*[!0-9]*) echo "Timeout must be a whole number of seconds, got: $TIMEOUT" >&2; exit 1 ;;
esac
[ "$TIMEOUT" -gt 0 ] || { echo "Timeout must be greater than 0." >&2; exit 1; }

PID_FILE="training_workers.pid"
[ -s "$PID_FILE" ] || { echo "No $PID_FILE — nothing to stop."; exit 0; }

mapfile -t PIDS < "$PID_FILE"
ALIVE=()
for pid in "${PIDS[@]}"; do
    if kill -TERM "$pid" 2>/dev/null; then ALIVE+=("$pid"); fi
done
echo "Sent SIGTERM to ${#ALIVE[@]} worker(s); waiting up to ${TIMEOUT}s for them to finish the current move and flush the cache..."

# Report progress periodically: a long timeout otherwise looks like a hang.
PROGRESS_EVERY=30
for ((elapsed = 0; elapsed < TIMEOUT; elapsed++)); do
    REMAIN=()
    for pid in "${ALIVE[@]}"; do
        kill -0 "$pid" 2>/dev/null && REMAIN+=("$pid")
    done
    [ ${#REMAIN[@]} -eq 0 ] && break
    ALIVE=("${REMAIN[@]}")
    if [ "$elapsed" -gt 0 ] && [ $((elapsed % PROGRESS_EVERY)) -eq 0 ]; then
        echo "  ${elapsed}s elapsed, ${#ALIVE[@]} worker(s) still finishing their current move..."
    fi
    sleep 1
done

for pid in "${ALIVE[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "Worker $pid still running after ${TIMEOUT}s; sending SIGKILL."
        kill -KILL "$pid" 2>/dev/null
    fi
done

rm -f "$PID_FILE"
echo "Stopped."
