#!/usr/bin/env python3
"""Rebuild `book.db` from a `export_book.py` text dump.

The way back from the versionable form. The export is keyed by FEN because a Zobrist
hash is one-way; this is where the hashes come back, by re-deriving each one through the
engine (`from_fen` then `get_position_hash`). That is why this is an engine run and not a
plain SQL load -- and it is the property that lets an export survive a change to the
hashing that the DB itself could not.

    ./venv/bin/python import_book.py                    # book.tsv -> book.db, refuses if non-empty
    ./venv/bin/python import_book.py --merge            # merge into an existing book
    ./venv/bin/python import_book.py --in /tmp/x.tsv --store analysis
    ./venv/bin/python import_book.py --check            # verify the file; write nothing

## Merge rule

`--merge` keeps the **deeper** row when both sides hold the same (position, rank), which
is the rule `search::book_store` already applies: never trade a deeper answer for a
shallower one. On equal depth the imported row wins, so re-importing is how you overwrite
in place. A row whose `eval_version` differs from the engine's current one is stale
whatever its depth -- it is imported, because discarding data on someone's behalf is not
this script's call, but it is counted and reported so a stale file is visible rather than
silently mixed in.

## What it checks before writing anything

- Every FEN parses, and `to_fen(from_fen(fen))` returns it unchanged. A FEN that does not
  round-trip is a corrupted line, not a position, and would file a row under a hash for
  something else.
- No two distinct FENs in the file hash to the same value. That is a Zobrist collision,
  and importing it would silently corrupt both entries.
- `eval_version` values are compared against the engine's, and a mismatch is reported.

## Ordering

`position` rows are written before `book_move` rows, because the version-2 foreign key
makes the parent a precondition. Reverse it and every insert fails.

Run it from the repo root, and **not while a build is running**: twenty workers hold the
write lock, and a build that finishes after the import would flush its in-memory book
over the top of it.
"""

import argparse
import collections
import os
import sqlite3
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

DB_PATH = "book.db"
DEFAULT_IN = "book.tsv"

STORES = {"book": ("book_move", "position"), "analysis": ("analysis_move", "analysis_position")}

Row = collections.namedtuple("Row", "fen ply rank move score depth eval_version")


def parse(path):
    """Read the dump. Returns (rows, header dict). Raises ValueError on a bad line."""
    rows, header = [], {}
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].strip().split("\t")
                if len(parts) == 2:
                    header[parts[0].strip()] = parts[1].strip()
                continue
            f = line.split("\t")
            if len(f) != 7:
                raise ValueError("%s:%d: expected 7 tab-separated fields, got %d"
                                 % (path, lineno, len(f)))
            try:
                rows.append(Row(f[0], None if f[1] == "" else int(f[1]), int(f[2]),
                                f[3], int(f[4]), int(f[5]), int(f[6])))
            except ValueError as e:
                raise ValueError("%s:%d: %s" % (path, lineno, e))
    return rows, header


def resolve(rows, engine):
    """FEN -> hash for every distinct FEN, verifying each one. Returns (map, problems)."""
    by_fen = {}
    problems = []
    seen_hash = {}
    for r in rows:
        if r.fen in by_fen:
            continue
        try:
            gs = engine.from_fen(r.fen)
        except Exception as e:
            problems.append("unparseable FEN %r: %s" % (r.fen, e))
            continue
        back = engine.to_fen(gs)
        if back != r.fen:
            # fen.rs carries exactly what the hash reads, so a FEN that does not
            # round-trip would hash as something other than what the line claims.
            problems.append("FEN does not round-trip: %r -> %r" % (r.fen, back))
            continue
        h = engine.get_position_hash(gs)
        if h in seen_hash and seen_hash[h] != r.fen:
            problems.append("ZOBRIST COLLISION: %r and %r both hash to %s"
                            % (seen_hash[h], r.fen, h))
            continue
        seen_hash[h] = r.fen
        by_fen[r.fen] = h
    return by_fen, problems


def min_ply(rows):
    """Lowest ply seen per FEN -- how early the position can actually appear."""
    out = {}
    for r in rows:
        if r.ply is None:
            out.setdefault(r.fen, None)
            continue
        cur = out.get(r.fen)
        out[r.fen] = r.ply if cur is None else min(cur, r.ply)
    return out


def write(conn, rows, by_fen, plies, store, merge):
    moves_t, pos_t = STORES[store]
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("BEGIN IMMEDIATE")
    inserted = skipped_deeper = 0
    try:
        if not merge:
            # Child first: the foreign key forbids clearing parents out from under it.
            conn.execute("DELETE FROM %s" % moves_t)
            conn.execute("DELETE FROM %s" % pos_t)

        existing = {}
        if merge:
            for h, rk, d in conn.execute("SELECT hash, rank, depth FROM %s" % moves_t):
                existing[(h, rk)] = d

        # Parents before children -- the foreign key makes this a precondition, not a
        # preference. `ply` reconciles to the minimum, matching cache::write_position.
        for fen, h in by_fen.items():
            ply = plies.get(fen)
            cur = conn.execute("SELECT ply FROM %s WHERE hash = ?" % pos_t, (h,)).fetchone()
            if cur is None:
                conn.execute("INSERT INTO %s (hash, fen, ply) VALUES (?, ?, ?)"
                             % pos_t, (h, fen, ply))
            elif ply is not None and (cur[0] is None or ply < cur[0]):
                conn.execute("UPDATE %s SET ply = ? WHERE hash = ?" % pos_t, (ply, h))

        for r in rows:
            h = by_fen.get(r.fen)
            if h is None:
                continue
            if merge and existing.get((h, r.rank), -1) > r.depth:
                skipped_deeper += 1
                continue
            conn.execute(
                "INSERT OR REPLACE INTO %s (hash, rank, move, score, depth, eval_version)"
                " VALUES (?, ?, ?, ?, ?, ?)" % moves_t,
                (h, r.rank, r.move, r.score, r.depth, r.eval_version))
            inserted += 1

        bad = conn.execute("PRAGMA foreign_key_check").fetchall()
        if bad:
            raise RuntimeError("foreign_key_check found %d violation(s): %r"
                               % (len(bad), bad[:5]))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return inserted, skipped_deeper


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", default=DEFAULT_IN,
                    help="input path (default: %s)" % DEFAULT_IN)
    ap.add_argument("--store", choices=sorted(STORES), default="book")
    ap.add_argument("--merge", action="store_true",
                    help="merge into an existing book, keeping the deeper row")
    ap.add_argument("--check", action="store_true",
                    help="verify the file and report; write nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    if not os.path.exists(args.src):
        print("No such file: %s" % args.src)
        return 1

    import ai
    import minichess_engine as engine

    try:
        rows, header = parse(args.src)
    except (ValueError, UnicodeDecodeError) as e:
        print("Malformed export file -- nothing was read:\n  %s" % e)
        return 1
    print("%s: %d rows, format_version %s, store %s"
          % (args.src, len(rows), header.get("format_version", "?"),
             header.get("store", "?")))
    if header.get("store") and header["store"] != args.store:
        print("WARNING: the file says store=%s but --store=%s was given."
              % (header["store"], args.store))

    by_fen, problems = resolve(rows, engine)
    print("%d distinct positions; %d hashed cleanly." % (
        len({r.fen for r in rows}), len(by_fen)))
    if problems:
        print("\nRefusing to import -- %d problem(s):" % len(problems))
        for p in problems[:20]:
            print("  %s" % p)
        if len(problems) > 20:
            print("  ... and %d more" % (len(problems) - 20))
        return 1

    engine_ev = engine.EVAL_VERSION if hasattr(engine, "EVAL_VERSION") else None
    file_evs = sorted({r.eval_version for r in rows})
    print("eval_version in file: %s%s"
          % (", ".join(str(e) for e in file_evs),
             "" if engine_ev is None else "   (engine: %d)" % engine_ev))
    if engine_ev is not None and any(e != engine_ev for e in file_evs):
        stale = sum(1 for r in rows if r.eval_version != engine_ev)
        print("WARNING: %d row(s) were scored by a different evaluation. They import, but "
              "the probe will reject them until re-searched." % stale)

    if args.check:
        print("\n--check: the file is sound. Nothing was written.")
        return 0

    ai.setup_db()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        moves_t, _ = STORES[args.store]
        have = conn.execute("SELECT count(*) FROM %s" % moves_t).fetchone()[0]
        if have and not args.merge:
            print("\n%s already holds %d rows. Re-run with --merge to combine them, or "
                  "clear it with rebuild_book.py first." % (moves_t, have))
            return 1

        print("\n%s %d rows into %s."
              % ("Merging" if args.merge else "Writing", len(rows), moves_t))
        if not args.merge and have:
            print("This REPLACES the %d rows already there." % have)
        if not args.yes:
            try:
                answer = input("Type 'import' to go ahead: ").strip()
            except EOFError:
                answer = ""
            if answer != "import":
                print("Left alone.")
                return 1

        inserted, skipped = write(conn, rows, by_fen, min_ply(rows), args.store, args.merge)
        print("Imported %d row(s)." % inserted)
        if skipped:
            print("Kept %d existing row(s) that were searched deeper." % skipped)
        print("%s now holds %d rows across %d positions."
              % (moves_t, conn.execute("SELECT count(*) FROM %s" % moves_t).fetchone()[0],
                 conn.execute("SELECT count(*) FROM %s"
                              % STORES[args.store][1]).fetchone()[0]))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
