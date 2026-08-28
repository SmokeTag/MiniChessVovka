#!/usr/bin/env python3
"""Drop the opening book and recreate it at the current schema version.

This is the only thing in the project that deletes book rows. `setup_db()` deliberately
refuses to: "recreate the schema" runs on paths nobody thinks of as destructive -- the
first save of a self-play worker, a stray call from a test, a script that opens the book
just to look at it -- and a SCHEMA_VERSION bump would turn any of them into a silent wipe
of however many CPU-days are in there. So the destructive path is a separate front door
that asks first.

Everything in the book is a pure function of the engine and can be recomputed by
re-searching, so rebuilding costs time, not information.

    ./venv/bin/python rebuild_book.py            # show what is there, then ask
    ./venv/bin/python rebuild_book.py --yes      # no prompt (for scripts)
    ./venv/bin/python rebuild_book.py --analysis # clear the analysis cache instead

The analysis cache (`analysis_move` / `analysis_position`) lives in the same file and is
**left alone** by a book rebuild. Discarding a session's exploration and discarding a
curated repertoire are not the same decision, so neither is a side effect of the other.

Run it from the repo root: DB_PATH in engine_rs/src/cache.rs is the relative string
"book.db", so it is resolved against the process CWD.
"""

import argparse
import os
import sqlite3
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

DB_PATH = "book.db"


def describe():
    """What the file on disk currently holds. Returns None if there is nothing to lose."""
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        try:
            positions = conn.execute("SELECT count(*) FROM position").fetchone()[0]
            moves = conn.execute("SELECT count(*) FROM book_move").fetchone()[0]
            depths = conn.execute(
                "SELECT depth, count(*) FROM book_move WHERE rank = 1"
                " GROUP BY depth ORDER BY depth"
            ).fetchall()
            try:
                cached = conn.execute("SELECT count(*) FROM analysis_move").fetchone()[0]
            except sqlite3.OperationalError:
                cached = 0      # book.db predates the analysis pair
        except sqlite3.OperationalError:
            # A foreign schema is exactly the case this script exists for; it need not
            # have tables we can count.
            return {"version": version, "positions": None, "moves": None, "depths": [],
                    "cached": 0}
        return {"version": version, "positions": positions, "moves": moves,
                "depths": depths, "cached": cached}
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt")
    parser.add_argument("--analysis", action="store_true",
                        help="clear the analysis cache instead, leaving the book alone")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    import ai
    import minichess_engine as rs

    current = describe()
    if current is None:
        print("No %s here -- nothing to rebuild. It is created on the first save."
              % DB_PATH)
        return 0

    print("%s currently holds:" % DB_PATH)
    print("  schema version: %s (this build expects %s)"
          % (current["version"], rs.SCHEMA_VERSION))
    if current["positions"] is None:
        print("  contents:       unreadable under this build's schema")
    else:
        print("  positions:      %d" % current["positions"])
        print("  ranked moves:   %d" % current["moves"])
        for depth, n in current["depths"]:
            print("      depth %2d: %d" % (depth, n))
        print("  analysis cache: %d ranked moves" % current["cached"])
    print()

    if args.analysis:
        print("Clearing DROPS analysis_move and analysis_position.")
        print("book_move and position are NOT touched.")
        word = "clear"
    else:
        print("Rebuilding DROPS book_move and position and stamps version %s."
              % rs.SCHEMA_VERSION)
        print("The analysis cache is NOT touched (use --analysis for that).")
        print("Every row is recomputable by re-searching, but that costs the search "
              "time again.")
        word = "rebuild"

    if not args.yes:
        try:
            answer = input("Type '%s' to go ahead: " % word).strip()
        except EOFError:
            answer = ""
        if answer != word:
            print("Left alone.")
            return 1

    if args.analysis:
        ai.rebuild_analysis()
        print("Cleared: the analysis cache is empty; the book is untouched.")
    else:
        ai.rebuild_book()
        print("Rebuilt: %s is empty at schema version %s." % (DB_PATH, rs.SCHEMA_VERSION))
    return 0


if __name__ == "__main__":
    sys.exit(main())
