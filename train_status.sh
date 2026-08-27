#!/bin/bash
# Snapshot of the self-play workers and the opening book.

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
import os, sqlite3
if not os.path.exists("book.db"):
    print("book.db: not created yet")
    raise SystemExit
c = sqlite3.connect("file:book.db?mode=ro", uri=True)
positions = c.execute("select count(*) from position").fetchone()[0]
rows = c.execute("select count(*) from book_move").fetchone()[0]
print(f"book.db: {positions} positions, {rows} ranked moves")
for depth, n in c.execute(
    "select depth, count(*) from book_move where rank = 1 group by depth order by depth"
):
    print(f"  depth {depth:>2}: {n}")
multi = c.execute("select count(*) from book_move where rank > 1").fetchone()[0]
print(f"  alternatives (rank > 1): {multi}")
PY
