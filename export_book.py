#!/usr/bin/env python3
"""Dump the opening book to a sorted text file, keyed by FEN.

`book.db` is deliberately gitignored. It is a SQLite file that changes wholesale on
every write -- a VACUUM or a schema migration rewrites every page with identical logical
content -- so tracking it means a fresh ~2MB blob in git history per commit, forever, and
a binary merge conflict with no resolution the first time two branches both add entries.
The book is also designed to grow: measured, ply 6 held 1,997 rows and ply 8 holds
10,000, so "it is small" expires.

This file is the versionable form of the same data. It is ~20x smaller than the DB once
git packs it, it diffs and merges line by line, and it is reviewable -- a repertoire
change reads as a repertoire change.

**The important property is that it is keyed by FEN, not by hash.** A Zobrist hash is
one-way: an artifact keyed on one dies the moment the hash scheme changes, which is
exactly why nothing was salvaged from `move_cache`. A FEN can be re-hashed by any future
build, so this export survives changes the DB does not. It is the same reason the
`position` table exists at all.

    ./venv/bin/python export_book.py                 # -> book.tsv
    ./venv/bin/python export_book.py --out /tmp/x.tsv
    ./venv/bin/python export_book.py --store analysis

Read `import_book.py` for the way back.

## Format

Tab-separated, one ranked move per line, with a `#` comment header:

    fen <TAB> ply <TAB> rank <TAB> move <TAB> score <TAB> depth <TAB> eval_version

`move` is copied verbatim from `book_move.move` -- it is already the canonical stored
form, so round-tripping it is exactly as stable as the DB is.

Two decisions keep the diffs small, and both matter more than they look:

- **Rows are sorted by (fen, rank), and by nothing else.** Not by ply: ply is
  path-dependent (the minimum ever seen wins), so a later build that reaches a position
  by a shorter route would rewrite that row's sort position and shuffle the file for no
  semantic reason. Ply is carried as a column instead.
- **The header holds no timestamp.** Re-exporting an unchanged book must produce a
  byte-identical file, or every export is a diff and the format is worthless.

## What is not exported

A `position` row with no `book_move` beside it -- a position visited but never stored,
which is legal (see the foreign key note in `cache.rs`). It carries no searched work and
the builder re-derives it for free, so a round trip is not expected to reproduce it.

The analysis cache is a separate store and is *not* exported by default. It is whatever a
session happened to look at rather than anything anyone curated, so versioning it is
noise; `--store analysis` is there for when you want to move one around by hand.
"""

import argparse
import os
import sqlite3
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

DB_PATH = "book.db"
DEFAULT_OUT = "book.tsv"

FORMAT_VERSION = 1

STORES = {"book": ("book_move", "position"), "analysis": ("analysis_move", "analysis_position")}

COLUMNS = ("fen", "ply", "rank", "move", "score", "depth", "eval_version")

def export(db_path, store, out_path):
    moves, positions = STORES[store]
    conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    try:
        schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
        rows = conn.execute(
            "SELECT p.fen, p.ply, m.rank, m.move, m.score, m.depth, m.eval_version"
            " FROM %s m JOIN %s p ON p.hash = m.hash"
            " ORDER BY p.fen, m.rank" % (moves, positions)
        ).fetchall()
        childless = conn.execute(
            "SELECT count(*) FROM %s p LEFT JOIN %s m ON m.hash = p.hash"
            " WHERE m.hash IS NULL" % (positions, moves)
        ).fetchone()[0]
        orphans = conn.execute(
            "SELECT count(*) FROM %s m LEFT JOIN %s p ON p.hash = m.hash"
            " WHERE p.hash IS NULL" % (moves, positions)
        ).fetchone()[0]
    finally:
        conn.close()

    if orphans:
        print("WARNING: %d %s row(s) have no %s row and cannot be exported -- their hash "
              "cannot be turned back into a position. Run migrate_book.py."
              % (orphans, moves, positions), file=sys.stderr)

    evs = sorted({r[6] for r in rows})
    lines = [
        "# minichess opening book export",
        "# format_version\t%d" % FORMAT_VERSION,
        "# store\t%s" % store,
        "# db_schema_version\t%d" % schema_version,
        "# eval_version\t%s" % ",".join(str(e) for e in evs),
        "# rows\t%d" % len(rows),
        "# sorted by (fen, rank); no timestamp, so an unchanged book re-exports identically",
        "# " + "\t".join(COLUMNS),
    ]
    for fen, ply, rank, move, score, depth, ev in rows:
        lines.append("%s\t%s\t%d\t%s\t%d\t%d\t%d"
                     % (fen, "" if ply is None else ply, rank, move, score, depth, ev))

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return {"rows": len(rows), "bytes": len(text.encode()), "eval_versions": evs,
            "childless": childless, "orphans": orphans}

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT, help="output path (default: %s)" % DEFAULT_OUT)
    ap.add_argument("--store", choices=sorted(STORES), default="book",
                    help="which pair of tables to export (default: book)")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    if not os.path.exists(DB_PATH):
        print("No %s here; nothing to export." % DB_PATH)
        return 1

    info = export(DB_PATH, args.store, args.out)
    print("Exported %d rows to %s (%.2f MB)."
          % (info["rows"], args.out, info["bytes"] / 1e6))
    print("eval_version(s) present: %s"
          % ", ".join(str(e) for e in info["eval_versions"]))
    if info["childless"]:
        print("%d position row(s) had no moves and were not exported -- they carry no "
              "searched work." % info["childless"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
