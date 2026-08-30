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

# How far through the current stage the workers are. There is no shared queue to ask, so
# this is read back off the worker logs: each shard prints what it has visited and what is
# still on its frontier, and the stage ends when the *last* shard drains -- which is why the
# slowest one is worth showing beside the total.
#
# The prefix above --split-ply must not count as progress: every shard re-walks all of it,
# so counting it reports a stage as nearly done while it has searched almost none of its own
# subtrees. The prefix is sized from the book (positions below --split-ply) rather than
# guessed from the logs -- guessing it as "the leading run of checkpoints that filed no new
# entry" breaks the moment the re-walk itself files rows, which is exactly what an opponent
# scan tier sitting inside the prefix does (stage 12, --split-ply 10, breadth 9-20:6).
./venv/bin/python - <<'PY'
import glob, os, re, sqlite3

logs = glob.glob("logs/book/stage-*-worker-*.log")
if logs:
    stage = re.search(r"stage-(\d+)-worker", max(logs, key=os.path.getmtime)).group(1)
    save_re = re.compile(r"saved — (\d+) visited \(\+\d+ new\), (\d+) queued")
    visited_re = re.compile(r"visited:\s+(\d+)")
    left_re = re.compile(r"left in queue:\s+(\d+)")
    split_re = re.compile(r"split_ply=(\d+)")

    split_ply = None
    shards = []  # (visited, queued, finished)
    for path in sorted(glob.glob("logs/book/stage-%s-worker-*.log" % stage)):
        with open(path, errors="replace") as fh:
            text = fh.read()
        if split_ply is None:
            m = split_re.search(text)
            split_ply = int(m.group(1)) if m else None
        summary = left_re.search(text)
        saves = save_re.findall(text)
        if summary:  # the shard printed its summary, so it is done walking
            total = visited_re.search(text)
            shards.append((int(total.group(1)) if total else 0, int(summary.group(1)), True))
        elif saves:
            shards.append((int(saves[-1][0]), int(saves[-1][1]), False))

    prefix = 0
    if split_ply and os.path.exists("book.db"):
        db = sqlite3.connect("file:book.db?mode=ro", uri=True)
        prefix = db.execute(
            "select count(*) from position where ply < ?", (split_ply,)).fetchone()[0]

    if shards:
        done = [max(0, visited - prefix) for visited, _q, _f in shards]
        left = [queued for _v, queued, _f in shards]
        pcts = [d / (d + q) if d + q else 1.0 for d, q in zip(done, left)]
        finished = sum(1 for *_r, f in shards if f)
        frontier = sum(done) + sum(left)
        print("Stage %s:               %.1f%% of known frontier   (%d searched, %d queued, "
              "slowest shard %.1f%%%s)"
              % (stage, 100.0 * sum(done) / frontier if frontier else 100.0,
                 sum(done), sum(left), 100.0 * min(pcts),
                 ", %d/%d shards done" % (finished, len(shards)) if finished else ""))
        walked = min(visited for visited, _q, _f in shards)
        if walked < prefix:
            print("  prefix re-walk:       %d/%d nodes   (every shard walks it; a shard's own "
                  "subtrees start after)" % (walked, prefix))
PY
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
# Mate rows are excluded here: the mate break exits iterative deepening early, so their
# `depth` records the iteration that found the mate rather than how far the search got.
# probe_book accepts them at any depth, so counting them as shallow rows makes the book
# look far less deep than it is.  CHECKMATE_SCORE is 1_000_000; search::is_mate_score
# uses 90% of it as the threshold.
MATE_CUTOFF = 900_000
mates = c.execute(
    "select count(*) from book_move where rank = 1 and abs(score) >= ?", (MATE_CUTOFF,)
).fetchone()[0]
print("  depth of rank-1 rows (mates excluded):")
for depth, n in c.execute(
    "select depth, count(*) from book_move where rank = 1 and abs(score) < ?"
    " group by depth order by depth",
    (MATE_CUTOFF,),
):
    print(f"    depth {depth:>2}: {n}")
if mates:
    print(f"    mates:    {mates}   [any depth; accepted by probe_book at any depth]")
multi = c.execute("select count(*) from book_move where rank > 1").fetchone()[0]
if multi:
    print(f"  alternatives (rank > 1): {multi}   [opponent-reply scans; never probed]")
PY
