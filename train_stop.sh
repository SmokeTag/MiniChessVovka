#!/bin/bash
# Stop the self-play workers started by train_parallel.sh.
# SIGTERM triggers the graceful shutdown path, which flushes the cache to the DB.

set -u
cd "$(dirname "$0")" || exit 1

PID_FILE="training_workers.pid"
[ -s "$PID_FILE" ] || { echo "No $PID_FILE — nothing to stop."; exit 0; }

mapfile -t PIDS < "$PID_FILE"
ALIVE=()
for pid in "${PIDS[@]}"; do
    if kill -TERM "$pid" 2>/dev/null; then ALIVE+=("$pid"); fi
done
echo "Sent SIGTERM to ${#ALIVE[@]} worker(s); waiting for them to flush the cache..."

for _ in $(seq 1 60); do
    REMAIN=()
    for pid in "${ALIVE[@]}"; do
        kill -0 "$pid" 2>/dev/null && REMAIN+=("$pid")
    done
    [ ${#REMAIN[@]} -eq 0 ] && break
    ALIVE=("${REMAIN[@]}")
    sleep 1
done

for pid in "${ALIVE[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "Worker $pid still running after 60s; sending SIGKILL."
        kill -KILL "$pid" 2>/dev/null
    fi
done

rm -f "$PID_FILE"
echo "Stopped."
