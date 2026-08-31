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
#   EXPORT=0 ./build_book_parallel.sh        # skip the book.tsv refresh at the end
#
# On the way out -- whether it finished or you stopped it -- the build refreshes
# book.tsv, the git-tracked copy of book.db (which is gitignored). It never commits;
# it prints the command if there is anything to keep.
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

# Refresh book.tsv, the versioned copy of the book.
#
# book.db is gitignored and book.tsv is what git tracks, so the two drift the moment a
# build adds rows and nothing says so out loud. Running it here is what keeps "the book"
# and "the book in git" the same claim.
#
# The export is byte-stable -- no timestamp in the header, rows sorted by (fen, rank) --
# so "unchanged" below means this build genuinely added nothing, not that the step
# silently failed.
#
# It exports and never commits. What goes into a commit is not a build script's call,
# and a build that committed on its own would eventually commit a half-finished tier.
export_snapshot() {
    if [ "${EXPORT:-1}" = "0" ]; then
        return 0
    fi
    echo
    echo "--- refreshing book.tsv ---"
    if ! "$PY" export_book.py; then
        echo "  Export failed. book.db is untouched and still authoritative;" >&2
        echo "  re-run ./venv/bin/python export_book.py once you know why." >&2
        return 1
    fi
    git rev-parse --git-dir >/dev/null 2>&1 || return 0
    if [ -z "$(git status --porcelain -- book.tsv 2>/dev/null)" ]; then
        echo "  book.tsv is unchanged: this build added no new rows."
        return 0
    fi
    STAT=$(git diff --numstat -- book.tsv 2>/dev/null | awk '{printf "+%s/-%s lines", $1, $2}')
    echo "  book.tsv changed${STAT:+ ($STAT)}. To keep it:"
    echo "      git add book.tsv && git commit -m 'Extend the book'"
}

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
    # Whatever the workers flushed before dying is real, curated work. Export it here or
    # it sits in a gitignored file with nothing recording that it arrived.
    export_snapshot
    exit 130
}
trap stop_workers INT TERM

# The deepest ply that --opponent-breadth answers with "all", which is as deep as
# --split-ply may go. Above --split-ply every shard walks the same nodes, and that is only
# free while expanding them costs nothing: with "all" the builder just enumerates the
# replies, but past that ply it has to *search* the opponent node to rank them. Split below
# that line and all N workers compute the same scan tier -- measured on stage 12 at
# --split-ply 10, 3,549 unique scans turned into ~67,000 searches and ate four of the
# stage's first four hours. Asking build_book.py's own parser keeps the two readings of
# the spec from drifting; importing it is safe because it chdirs in main(), not at import.
SPLIT_CAP=$("$PY" - "$BREADTH" <<'PY'
import sys
from build_book import HARD_PLY_CAP, breadth_at, parse_breadth

rules = parse_breadth(sys.argv[1])
ply = 0
while ply <= HARD_PLY_CAP and breadth_at(rules, ply) is None:
    ply += 1
print(ply - 1)
PY
) || { echo "Could not read --opponent-breadth $BREADTH" >&2; exit 1; }

echo "=================================="
echo "Repertoire build: $WORKERS workers, up to ply $MAX_PLY, depth $DEPTH"
echo "  resign cutoff:  $RESIGN"
echo "  opponent breadth: $BREADTH (scan depth $SCAN_DEPTH)"
echo "  logs:           $LOG_DIR/stage-P-worker-N.log  (progress)"
echo "                  $LOG_DIR/stage-P-worker-N.err  (engine chatter + any traceback)"
echo "  database:       book.db"
echo "=================================="

STARTED=$(date +%s)

echo
echo "--- stage 2 (serial seed, plies 0-2) ---"
"$PY" -u build_book.py --max-ply 2 --depth "$DEPTH" --resign "$RESIGN" \
    --opponent-breadth "$BREADTH" --scan-depth "$SCAN_DEPTH" \
    2> "$LOG_DIR/stage-2-seed.err" | grep --line-buffered -Ev '^\s*$'

for (( ply = 4; ply <= MAX_PLY; ply += 2 )); do
    # One tier per stage, so the split follows the tier -- but never past the point where
    # the prefix stops being free (SPLIT_CAP), and never above the root, which would hand
    # every subtree to shard 0.
    SPLIT=$(( ply - 2 ))
    CAPPED=""
    if [ "$SPLIT" -gt "$SPLIT_CAP" ]; then
        SPLIT="$SPLIT_CAP"
        CAPPED="  [capped: breadth is a number past ply $SPLIT_CAP, so a deeper split "
        CAPPED+="would make every shard re-scan it]"
    fi
    [ "$SPLIT" -lt 2 ] && SPLIT=2
    echo
    echo "--- stage $ply ($WORKERS shards, --split-ply $SPLIT) ---$CAPPED"
    : > "$PID_FILE"
    STAGE_PIDS=()
    for (( i = 0; i < WORKERS; i++ )); do
        # stderr is the engine's per-depth chatter: one line per iterative-deepening
        # iteration per search, which is most of the volume. It is kept out of the main
        # log but deliberately NOT discarded -- a worker that dies unexpectedly does so
        # by printing a traceback here and nowhere else. Sending it to /dev/null is what
        # let 17 of 22 shards vanish mid-queue on the ply-12 build leaving no record of
        # why, which no amount of after-the-fact forensics could recover.
        "$PY" -u build_book.py \
            --max-ply "$ply" --split-ply "$SPLIT" --depth "$DEPTH" \
            --resign "$RESIGN" --opponent-breadth "$BREADTH" --scan-depth "$SCAN_DEPTH" \
            --shard "$i/$WORKERS" \
            > "$LOG_DIR/stage-$ply-worker-$i.log" \
            2> "$LOG_DIR/stage-$ply-worker-$i.err" &
        STAGE_PIDS+=($!)
        echo $! >> "$PID_FILE"
    done
    # Wait per pid rather than with a bare `wait`, so a shard that dies is named at the
    # end of its stage instead of being folded silently into the entry total. 130 is
    # build_book.py's own "stopped by SIGTERM" exit and is not a failure; anything else
    # non-zero is, and the reason is in the .err file the message points at.
    FAILED=0
    for (( i = 0; i < WORKERS; i++ )); do
        wait "${STAGE_PIDS[$i]}"; rc=$?
        if [ "$rc" -eq 0 ] || [ "$rc" -eq 130 ]; then
            continue
        fi
        FAILED=$(( FAILED + 1 ))
        echo "  !! shard $i/$WORKERS exited $rc — see $LOG_DIR/stage-$ply-worker-$i.err"
        tail -n 3 "$LOG_DIR/stage-$ply-worker-$i.err" 2>/dev/null | sed 's/^/       | /'
    done
    NEW=$(grep -h 'new entries:' "$LOG_DIR"/stage-"$ply"-worker-*.log 2>/dev/null \
          | awk '{s += $NF} END {print s+0}')
    echo "  stage $ply done: $NEW new entries across $WORKERS shards, $(( $(date +%s) - STARTED ))s total"
    if [ "$FAILED" -gt 0 ]; then
        # A tier with a dead shard is not finished, and the next stage will build on the
        # hole. Re-running the same command re-walks and fills it, so say so here rather
        # than letting the entry count read as success.
        echo "  !! $FAILED of $WORKERS shards did not finish: this tier is INCOMPLETE."
        echo "     Re-run the same command once you know why — the walk is idempotent."
    fi
    ./build_status.sh 2>/dev/null | sed -n '/book.db:/,$p' | sed 's/^/  /'
done

: > "$PID_FILE"
rm -f "$DRIVER_PID_FILE"
echo
echo "=================================="
echo "Build finished in $(( $(date +%s) - STARTED ))s."
./build_status.sh
export_snapshot
