#!/bin/bash
# Build the opening repertoire in book.db across N workers.
#
# The tree is walked one our-turn tier at a time, and each tier is a stage:
#
#   stage 2   one process, plies 0-2          ~31 searches, seconds
#   stage 4   N shards, --split-ply 2         each takes disjoint subtrees
#   stage 6   N shards, --split-ply 4         ...and so on, two plies per stage
#
# A stage is a barrier, and that is the whole coordination mechanism. Everything a
# stage needs from the tier above it is already in book.db, so when the next stage's
# workers re-walk the prefix, every node up there is a `probe_book` hit that returns in
# ~0s -- which is how a worker learns the move another worker searched without any
# message passing. Below --split-ply each worker owns whole subtrees, so no two workers
# search the same node except where two subtrees transpose (~1% of the tier).
#
#   ./build_book_parallel.sh                 # 20 workers, ply 6, depth 10
#   ./build_book_parallel.sh 12 8 10         # 12 workers, ply 8, depth 10
#   RESIGN=1500 BREADTH=0-8:all,9-16:3 ./build_book_parallel.sh 20 12
#
# Runs in the foreground so you can watch the stages. To detach:
#   nohup ./build_book_parallel.sh 20 6 > logs/book/build.log 2>&1 &

set -u
cd "$(dirname "$0")" || exit 1

WORKERS="${1:-${WORKERS:-20}}"
MAX_PLY="${2:-${MAX_PLY:-6}}"
DEPTH="${3:-${DEPTH:-10}}"
RESIGN="${RESIGN:-1200}"
BREADTH="${BREADTH:-all}"
SCAN_DEPTH="${SCAN_DEPTH:-4}"
LOG_DIR="${LOG_DIR:-logs/book}"
PID_FILE="book_workers.pid"
DRIVER_PID_FILE="book_build.pid"

PY="./venv/bin/python"
[ -x "$PY" ] || { echo "No $PY — the minichess_engine extension lives in the project venv."; exit 1; }

case "$WORKERS$MAX_PLY$DEPTH" in *[!0-9]*) echo "WORKERS, MAX_PLY and DEPTH must be whole numbers." >&2; exit 1 ;; esac
[ "$WORKERS" -ge 1 ] || { echo "Need at least one worker." >&2; exit 1; }

if [ -s "$PID_FILE" ] && xargs -r -a "$PID_FILE" ps -o pid= -p >/dev/null 2>&1; then
    echo "A build is already running (see $PID_FILE). Run ./build_stop.sh first."
    exit 1
fi

CORES=$(nproc)
[ "$WORKERS" -gt "$CORES" ] && echo "Warning: $WORKERS workers on $CORES cores — they will contend for CPU."

mkdir -p "$LOG_DIR"
: > "$PID_FILE"
# build_stop.sh signals this exact pid. Matching on the script *name* instead would be
# a loaded gun: `pkill -f build_book_parallel.sh` also matches the shell you typed the
# command into, and anything else that happens to mention the name.
echo $$ > "$DRIVER_PID_FILE"

stop_workers() {
    # SIGTERM lets each worker finish its current search and flush; the search itself is
    # a blocking Rust call that cannot see the flag, so this is not instant.
    [ -s "$PID_FILE" ] || return
    echo; echo "Stopping workers (they finish the search in flight, then flush)..."
    xargs -r -a "$PID_FILE" kill -TERM 2>/dev/null
    wait
    : > "$PID_FILE"
    rm -f "$DRIVER_PID_FILE"
    echo "Stopped. The build is resumable: re-run the same command to pick up where it left off."
    exit 130
}
trap stop_workers INT TERM

echo "=================================="
echo "Repertoire build: $WORKERS workers, up to ply $MAX_PLY, depth $DEPTH"
echo "  resign cutoff:  $RESIGN"
echo "  opponent breadth: $BREADTH (scan depth $SCAN_DEPTH)"
echo "  logs:           $LOG_DIR/stage-P-worker-N.log"
echo "  database:       book.db"
echo "=================================="

STARTED=$(date +%s)

echo
echo "--- stage 2 (serial seed, plies 0-2) ---"
"$PY" -u build_book.py --max-ply 2 --depth "$DEPTH" --resign "$RESIGN" \
    --opponent-breadth "$BREADTH" --scan-depth "$SCAN_DEPTH" \
    2> "$LOG_DIR/stage-2-seed.err" | grep --line-buffered -Ev '^\s*$'

for (( ply = 4; ply <= MAX_PLY; ply += 2 )); do
    echo
    echo "--- stage $ply ($WORKERS shards, --split-ply $((ply - 2))) ---"
    : > "$PID_FILE"
    for (( i = 0; i < WORKERS; i++ )); do
        # stderr is the engine's per-depth chatter: one line per iterative-deepening
        # iteration per search, which is most of the volume. Kept out of the main log.
        "$PY" -u build_book.py \
            --max-ply "$ply" --split-ply "$((ply - 2))" --depth "$DEPTH" \
            --resign "$RESIGN" --opponent-breadth "$BREADTH" --scan-depth "$SCAN_DEPTH" \
            --shard "$i/$WORKERS" \
            > "$LOG_DIR/stage-$ply-worker-$i.log" 2>/dev/null &
        echo $! >> "$PID_FILE"
    done
    wait
    NEW=$(grep -h 'new entries:' "$LOG_DIR"/stage-"$ply"-worker-*.log 2>/dev/null \
          | awk '{s += $NF} END {print s+0}')
    echo "  stage $ply done: $NEW new entries across $WORKERS shards, $(( $(date +%s) - STARTED ))s total"
    ./build_status.sh 2>/dev/null | sed -n '/book.db:/,$p' | sed 's/^/  /'
done

: > "$PID_FILE"
rm -f "$DRIVER_PID_FILE"
echo
echo "=================================="
echo "Build finished in $(( $(date +%s) - STARTED ))s."
./build_status.sh
