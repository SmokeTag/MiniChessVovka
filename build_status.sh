#!/bin/bash
# What the repertoire build has produced so far, by ply.

set -u
cd "$(dirname "$0")" || exit 1

PID_FILE="book_workers.pid"
RUNNING=0
if [ -s "$PID_FILE" ]; then
    while read -r pid; do
        kill -0 "$pid" 2>/dev/null && RUNNING=$((RUNNING+1))
    done < "$PID_FILE"
fi
echo "Build workers running: $RUNNING"
echo "Load average:          $(cut -d' ' -f1-3 /proc/loadavg)"
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
# By ply and side to move: a repertoire only ever answers for the side on move, so
# even plies are the White repertoire and odd plies the Black one.
print("  ply  entries   (repertoire)")
for ply, n in c.execute(
    "select ply, count(*) from position group by ply order by ply"
):
    if ply is None:
        print(f"   ?    {n:>7}   (no ply recorded)")
        continue
    print(f"  {ply:>3}    {n:>7}   {'White' if ply % 2 == 0 else 'Black'}")
print("  depth of rank-1 rows:")
for depth, n in c.execute(
    "select depth, count(*) from book_move where rank = 1 group by depth order by depth"
):
    print(f"    depth {depth:>2}: {n}")
multi = c.execute("select count(*) from book_move where rank > 1").fetchone()[0]
if multi:
    print(f"  alternatives (rank > 1): {multi}   [opponent-reply scans; never probed]")
PY
