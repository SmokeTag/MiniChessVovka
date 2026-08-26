#!/bin/bash
# Snapshot of the self-play workers and the move cache.

set -u
cd "$(dirname "$0")" || exit 1

PID_FILE="training_workers.pid"
RUNNING=0
if [ -s "$PID_FILE" ]; then
    while read -r pid; do
        kill -0 "$pid" 2>/dev/null && RUNNING=$((RUNNING+1))
    done < "$PID_FILE"
fi
echo "Workers running: $RUNNING"
echo "Load average:   $(cut -d' ' -f1-3 /proc/loadavg)"
echo

./venv/bin/python - <<'PY'
import sqlite3
c = sqlite3.connect("file:move_cache.db?mode=ro", uri=True)
total = c.execute("select count(*) from move_cache").fetchone()[0]
print(f"move_cache.db: {total} positions")
for depth, n in c.execute("select depth, count(*) from move_cache group by depth order by depth"):
    print(f"  depth {depth:>2}: {n}")
PY
