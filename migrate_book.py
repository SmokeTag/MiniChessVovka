#!/usr/bin/env python3
"""Carry book.db forward to the current schema version, keeping every row.

The counterpart to `rebuild_book.py`. Both exist because `setup_db()` refuses a version
mismatch rather than fixing it, which leaves a human two choices: throw the rows away
(rebuild) or copy them into the new shape (this). Rebuilding costs the search time again
-- CPU-days, for a filled repertoire -- so it should be the second thing anyone reaches
for, not the first.

    ./venv/bin/python migrate_book.py             # show the plan, then ask
    ./venv/bin/python migrate_book.py --yes       # no prompt (for scripts)
    ./venv/bin/python migrate_book.py --check     # report only; change nothing

## What version 2 changes

Each move table gains a foreign key on `hash` referencing the position table beside it:

    FOREIGN KEY (hash) REFERENCES position(hash) ON DELETE CASCADE

That makes it impossible to file a ranked move under a hash with no FEN to re-open it.
A Zobrist hash is one-way, so such a row is unreadable forever -- it is what made the old
`move_cache` unmigratable, and the reason nothing was salvaged from it.

The constraint changes what the schema *permits*, not what any row *means*. No score,
depth or eval_version moves, so nothing is re-searched and `EVAL_VERSION` is untouched.

## Why a table rebuild

SQLite has no `ALTER TABLE ... ADD CONSTRAINT`, so adding a foreign key means the
documented 12-step procedure: create the new table under a temporary name, copy the rows,
drop the old, rename. Three details are load-bearing and easy to get wrong:

- **Foreign keys must be OFF while the tables are being swapped.** With enforcement on,
  dropping a parent mid-rebuild cascades into children that are still being copied.
  Enforcement is turned back on for `PRAGMA foreign_key_check`, which is what actually
  proves the result is sound, before anything is committed.
- **`legacy_alter_table` must be ON for the RENAME.** Without it SQLite helpfully
  rewrites references to the renamed table inside other tables' schemas, which is the
  opposite of what a 12-step rebuild wants.
- **One transaction.** A crash halfway leaves the file exactly as it was, not half
  migrated.

## Orphans

A `book_move` row whose hash has no `position` row cannot be copied -- it is precisely
what the new constraint forbids. This script refuses rather than dropping rows silently;
`--drop-orphans` says to discard them, and prints each one first. The reverse case (a
`position` with no `book_move`) is legal, is not an orphan, and is carried over as-is:
SQL cannot require a parent to have children.

Run it from the repo root: DB_PATH in engine_rs/src/cache.rs is the relative string
"book.db", resolved against the process CWD.
"""

import argparse
import datetime
import os
import shutil
import sqlite3
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

DB_PATH = "book.db"
BACKUP_DIR = "backups"

TARGET_VERSION = 2

PAIRS = (("book_move", "position"), ("analysis_move", "analysis_position"))

MOVES_DDL = """CREATE TABLE %s (
            hash         TEXT NOT NULL,
            rank         INTEGER NOT NULL,
            move         TEXT NOT NULL,
            score        INTEGER NOT NULL,
            depth        INTEGER NOT NULL,
            eval_version INTEGER NOT NULL,
            PRIMARY KEY (hash, rank),
            FOREIGN KEY (hash) REFERENCES %s(hash) ON DELETE CASCADE
        )"""

def table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None

def survey(conn):
    """Row counts and orphans per pair, for the plan and for --check."""
    out = []
    for moves, positions in PAIRS:
        if not table_exists(conn, moves):
            out.append({"moves": moves, "positions": positions, "present": False})
            continue
        n_moves = conn.execute("SELECT count(*) FROM %s" % moves).fetchone()[0]
        n_pos = (conn.execute("SELECT count(*) FROM %s" % positions).fetchone()[0]
                 if table_exists(conn, positions) else 0)
        orphans = conn.execute(
            "SELECT m.hash, count(*) FROM %s m"
            " LEFT JOIN %s p ON p.hash = m.hash"
            " WHERE p.hash IS NULL GROUP BY m.hash" % (moves, positions)
        ).fetchall() if table_exists(conn, positions) else []
        childless = conn.execute(
            "SELECT count(*) FROM %s p LEFT JOIN %s m ON m.hash = p.hash"
            " WHERE m.hash IS NULL" % (positions, moves)
        ).fetchone()[0] if table_exists(conn, positions) else 0
        out.append({"moves": moves, "positions": positions, "present": True,
                    "n_moves": n_moves, "n_pos": n_pos,
                    "orphans": orphans, "childless": childless})
    return out

def already_migrated(conn):
    """True when every present move table already carries its foreign key."""
    for moves, _ in PAIRS:
        if not table_exists(conn, moves):
            continue
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (moves,)
        ).fetchone()[0] or ""
        if "FOREIGN KEY" not in sql.upper():
            return False
    return True

def back_up():
    """A verified snapshot beside the file, before anything is touched."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(BACKUP_DIR, "book-premigrate-%s.db" % stamp)
    src = sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True)
    dst = sqlite3.connect(dest)
    try:
        with dst:
            src.backup(dst)
        if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("backup failed its integrity check; refusing to migrate")
    finally:
        dst.close()
        src.close()
    return dest

def migrate(conn, drop_orphans):
    """The 12-step rebuild, in one transaction, for every pair present."""
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("PRAGMA legacy_alter_table = ON")
    conn.execute("BEGIN IMMEDIATE")

    dropped = 0
    try:
        for moves, positions in PAIRS:
            if not table_exists(conn, moves):
                continue
            if not table_exists(conn, positions):
                raise RuntimeError(
                    "%s exists but %s does not; the pair is incomplete and the foreign "
                    "key has nothing to reference." % (moves, positions))

            if drop_orphans:
                cur = conn.execute(
                    "DELETE FROM %s WHERE hash NOT IN (SELECT hash FROM %s)"
                    % (moves, positions))
                dropped += cur.rowcount

            tmp = "%s_migrating" % moves
            conn.execute("DROP TABLE IF EXISTS %s" % tmp)
            conn.execute(MOVES_DDL % (tmp, positions))
            conn.execute(
                "INSERT INTO %s (hash, rank, move, score, depth, eval_version)"
                " SELECT hash, rank, move, score, depth, eval_version FROM %s"
                % (tmp, moves))
            conn.execute("DROP TABLE %s" % moves)
            conn.execute("ALTER TABLE %s RENAME TO %s" % (tmp, moves))

        conn.execute("PRAGMA user_version = %d" % TARGET_VERSION)

        conn.execute("PRAGMA foreign_keys = ON")
        bad = conn.execute("PRAGMA foreign_key_check").fetchall()
        if bad:
            raise RuntimeError("foreign_key_check found %d violation(s): %r"
                               % (len(bad), bad[:5]))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA legacy_alter_table = OFF")
    return dropped

def print_survey(rows):
    for r in rows:
        if not r["present"]:
            print("  %-16s absent" % r["moves"])
            continue
        print("  %-16s %6d moves   %-18s %6d positions"
              % (r["moves"], r["n_moves"], r["positions"], r["n_pos"]))
        if r["childless"]:
            print("      %d position row(s) with no move row -- legal, carried over as-is"
                  % r["childless"])
        if r["orphans"]:
            total = sum(n for _, n in r["orphans"])
            print("      %d move row(s) across %d hash(es) with NO position row"
                  % (total, len(r["orphans"])))
            for h, n in r["orphans"][:10]:
                print("        %s  (%d row(s))" % (h, n))
            if len(r["orphans"]) > 10:
                print("        ... and %d more" % (len(r["orphans"]) - 10))

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--check", action="store_true",
                    help="report what would happen and exit; changes nothing")
    ap.add_argument("--drop-orphans", action="store_true",
                    help="discard move rows whose hash has no position row")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)

    if not os.path.exists(DB_PATH):
        print("No %s here; nothing to migrate." % DB_PATH)
        return 0

    conn = sqlite3.connect(DB_PATH)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        rows = survey(conn)

        print("%s is at schema version %d; this build expects %d."
              % (DB_PATH, version, TARGET_VERSION))
        print()
        print_survey(rows)
        print()

        if version == TARGET_VERSION and already_migrated(conn):
            print("Already at version %d with the foreign key in place. Nothing to do."
                  % TARGET_VERSION)
            return 0
        if version > TARGET_VERSION:
            print("The file is NEWER than this build expects. Refusing to touch it: "
                  "check out a build that matches, or rebuild.")
            return 1

        orphaned = sum(sum(n for _, n in r["orphans"])
                       for r in rows if r["present"] and r["orphans"])
        if orphaned and not args.drop_orphans:
            print("Refusing to migrate: %d move row(s) have no position row, and the "
                  "version-2 foreign key forbids them." % orphaned)
            print("They cannot be repaired -- a Zobrist hash cannot be turned back into "
                  "a position without a FEN beside it.")
            print("Re-run with --drop-orphans to discard exactly those rows.")
            return 1

        print("Migrating to version %d: adding the hash foreign key to %s."
              % (TARGET_VERSION, " and ".join(m for m, _ in PAIRS)))
        print("Every row is copied. No score, depth or eval_version changes, so nothing "
              "is re-searched.")
        if args.drop_orphans and orphaned:
            print("DROPPING %d orphaned move row(s), listed above." % orphaned)
        print("A verified snapshot is written to %s/ first." % BACKUP_DIR)
        print()

        if args.check:
            print("--check: nothing was changed.")
            return 0

        if not args.yes:
            try:
                answer = input("Type 'migrate' to go ahead: ").strip()
            except EOFError:
                answer = ""
            if answer != "migrate":
                print("Left alone.")
                return 1

        snapshot = back_up()
        print("Snapshot: %s" % snapshot)

        dropped = migrate(conn, args.drop_orphans)
        if dropped:
            print("Dropped %d orphaned move row(s)." % dropped)

        after = survey(conn)
        print()
        print("Migrated. %s is now at version %d:"
              % (DB_PATH, conn.execute("PRAGMA user_version").fetchone()[0]))
        print_survey(after)
        print()
        print("If anything looks wrong, the snapshot above is a complete copy of the "
              "file as it was.")
        return 0
    finally:
        conn.close()

if __name__ == "__main__":
    sys.exit(main())
