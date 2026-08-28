#!/bin/bash
# Stop a repertoire build started by build_book_parallel.sh.
#
# SIGTERM makes each worker finish the search it is in, flush its dirty rows to book.db
# and exit, so nothing computed is lost. The search is a blocking Rust call that never
# sees the flag, which is why the grace period is sized to the search depth rather than
# to patience: depth 10 has been seen to take over 30s in a full-hand middlegame, and
# more at higher depths.
#
#   ./build_stop.sh            # wait up to 300s per worker
#   ./build_stop.sh 1200       # wait up to 1200s
#
# Stopping is safe at any point: the build is resumable. Re-running the same command
# picks up where it left off, because everything already searched is a book hit.

set -u
cd "$(dirname "$0")" || exit 1

TIMEOUT="${1:-${TIMEOUT:-300}}"
case "$TIMEOUT" in ''|*[!0-9]*) echo "Timeout must be a whole number of seconds, got: $TIMEOUT" >&2; exit 1 ;; esac
[ "$TIMEOUT" -gt 0 ] || { echo "Timeout must be greater than 0." >&2; exit 1; }

PID_FILE="book_workers.pid"
[ -s "$PID_FILE" ] || { echo "No $PID_FILE — nothing to stop."; exit 0; }

# The driver runs the stages in sequence, so killing only the workers would let it start
# the next one. Take it down first, by the pid it recorded -- never by `pkill -f` on the
# script name, which also matches the shell you typed the command into.
DRIVER_PID_FILE="book_build.pid"
if [ -s "$DRIVER_PID_FILE" ]; then
    read -r driver < "$DRIVER_PID_FILE"
    case "$driver" in
        ''|*[!0-9]*) ;;
        *) kill -TERM "$driver" 2>/dev/null && echo "Signalled the build driver (pid $driver)." ;;
    esac
fi

mapfile -t PIDS < "$PID_FILE"
ALIVE=()
for pid in "${PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null && ALIVE+=("$pid")
done
echo "Sent SIGTERM to ${#ALIVE[@]} worker(s); waiting up to ${TIMEOUT}s for the search in flight to finish and flush..."

for ((elapsed = 0; elapsed < TIMEOUT; elapsed++)); do
    REMAIN=()
    for pid in "${ALIVE[@]}"; do kill -0 "$pid" 2>/dev/null && REMAIN+=("$pid"); done
    [ ${#REMAIN[@]} -eq 0 ] && break
    ALIVE=("${REMAIN[@]}")
    if [ "$elapsed" -gt 0 ] && [ $((elapsed % 30)) -eq 0 ]; then
        echo "  ${elapsed}s elapsed, ${#ALIVE[@]} worker(s) still finishing their current search..."
    fi
    sleep 1
done

for pid in "${ALIVE[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "Worker $pid still running after ${TIMEOUT}s; sending SIGKILL (its unflushed searches are lost)."
        kill -KILL "$pid" 2>/dev/null
    fi
done

rm -f "$PID_FILE" "$DRIVER_PID_FILE"
echo "Stopped. Re-run build_book_parallel.sh to resume."
