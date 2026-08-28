#!/usr/bin/env python3
"""Round-trip the opening book through its text export.

`export_book.py` exists so the book can live in git without the binary: `book.db`
rewrites every page on a VACUUM or a migration, so tracking it means a fresh multi-MB
blob per commit and an unresolvable conflict the first time two branches both add rows.

What makes the export worth trusting is that it is keyed by **FEN**, not by hash. A
Zobrist hash is one-way -- an artifact keyed on one is dead the moment the hash scheme
changes, which is why nothing was salvaged from `move_cache`. So the property under test
is not "the file parses" but "the hashes come back": every FEN must re-derive to the same
hash the export was taken from, or every imported row is filed against the wrong
position and the probe silently answers with someone else's move.

These tests run under `isolated_cache_db()`. They call `ai.setup_db()` and write book
rows, so without it they would build a schema in the repo root -- see the module docstring
in `cache_isolation.py`.
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cache_isolation import IsolatedCacheDB

import ai
import export_book
import import_book
import minichess_engine as engine
from gamestate import GameState


def _seed_book(n_positions=6, depth=4):
    """Search a handful of real positions and file them, so there is a book to export."""
    ai.setup_db()
    gs = GameState()
    gs.setup_initial_board()
    for i in range(n_positions):
        ai.find_best_move_with_score(gs, depth=depth)
        moves = gs.get_all_legal_moves()
        if not moves:
            break
        gs.make_move(moves[i % len(moves)])
    ai.save_move_cache_to_db()


def _rows(db_path):
    conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    try:
        return conn.execute(
            "SELECT p.fen, p.ply, m.rank, m.move, m.score, m.depth, m.eval_version"
            " FROM book_move m JOIN position p ON p.hash = m.hash"
            " ORDER BY p.fen, m.rank").fetchall()
    finally:
        conn.close()


def _hashes(db_path):
    conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    try:
        return dict(conn.execute(
            "SELECT p.fen, p.hash FROM position p"
            " JOIN book_move m ON m.hash = p.hash GROUP BY p.hash"))
    finally:
        conn.close()


class TestBookExportRoundTrip(IsolatedCacheDB):

    def test_export_import_preserves_every_row_and_hash(self):
        _seed_book()
        before, before_hashes = _rows("book.db"), _hashes("book.db")
        self.assertTrue(before, "nothing was seeded, so the round trip proves nothing")

        export_book.export("book.db", "book", "book.tsv")

        # Wipe and rebuild from the text alone.
        ai.rebuild_book()
        self.assertEqual(_rows("book.db"), [], "rebuild_book left rows behind")

        rows, _ = import_book.parse("book.tsv")
        by_fen, problems = import_book.resolve(rows, engine)
        self.assertEqual(problems, [], "the export did not verify")
        conn = sqlite3.connect("book.db", timeout=30)
        try:
            import_book.write(conn, rows, by_fen, import_book.min_ply(rows), "book", False)
        finally:
            conn.close()

        self.assertEqual(_rows("book.db"), before, "a row changed across the round trip")
        # The point of keying on FEN: the hashes must be re-derivable, not merely stored.
        self.assertEqual(_hashes("book.db"), before_hashes,
                         "a position came back under a different hash")

    def test_export_is_byte_stable(self):
        """An unchanged book must re-export identically, or every export is a diff."""
        _seed_book(n_positions=4)
        export_book.export("book.db", "book", "a.tsv")
        export_book.export("book.db", "book", "b.tsv")
        with open("a.tsv", "rb") as fh_a, open("b.tsv", "rb") as fh_b:
            self.assertEqual(fh_a.read(), fh_b.read())

    def test_every_exported_fen_round_trips_through_the_engine(self):
        """A FEN that does not survive from_fen/to_fen would hash as something else."""
        _seed_book()
        export_book.export("book.db", "book", "book.tsv")
        rows, _ = import_book.parse("book.tsv")
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(engine.to_fen(engine.from_fen(r.fen)), r.fen)

    def test_import_refuses_a_fen_that_does_not_parse(self):
        """A corrupted line must stop the import, not file a row under a wrong hash."""
        _seed_book(n_positions=3)
        export_book.export("book.db", "book", "book.tsv")
        with open("book.tsv", "a", encoding="utf-8") as fh:
            fh.write("not-a-fen\t3\t1\t((0, 0), (1, 1), None)\t5\t10\t1\n")
        rows, _ = import_book.parse("book.tsv")
        _, problems = import_book.resolve(rows, engine)
        self.assertEqual(len(problems), 1)
        self.assertIn("unparseable FEN", problems[0])

    def test_import_rejects_a_malformed_line(self):
        with open("broken.tsv", "w", encoding="utf-8") as fh:
            fh.write("# store\tbook\nonly\ttwo\tfields\n")
        with self.assertRaises(ValueError):
            import_book.parse("broken.tsv")

    def test_merge_keeps_the_deeper_row(self):
        """The rule search::book_store already applies: never trade deeper for shallower."""
        _seed_book(n_positions=3)
        export_book.export("book.db", "book", "book.tsv")
        rows, _ = import_book.parse("book.tsv")
        self.assertTrue(rows)

        deep = {r.fen: r.depth for r in rows}
        shallow = [r._replace(depth=1, score=r.score + 999) for r in rows]
        by_fen, problems = import_book.resolve(shallow, engine)
        self.assertEqual(problems, [])

        conn = sqlite3.connect("book.db", timeout=30)
        try:
            inserted, skipped = import_book.write(
                conn, shallow, by_fen, import_book.min_ply(shallow), "book", True)
        finally:
            conn.close()

        self.assertEqual(inserted, 0, "a shallower row was allowed to overwrite")
        self.assertEqual(skipped, len(shallow))
        for fen, _ply, _rank, _move, _score, depth, _ev in _rows("book.db"):
            self.assertEqual(depth, deep[fen])

    def test_childless_positions_are_not_exported(self):
        """A position with no moves carries no searched work; the builder re-derives it."""
        _seed_book(n_positions=3)
        conn = sqlite3.connect("book.db", timeout=30)
        try:
            conn.execute("INSERT INTO position (hash, fen, ply) VALUES (?, ?, ?)",
                         ("f" * 16, "2bnrk/5p/6/6/P5/KRNB2[] w", 0))
            conn.commit()
        finally:
            conn.close()
        info = export_book.export("book.db", "book", "book.tsv")
        self.assertGreaterEqual(info["childless"], 1)
        rows, _ = import_book.parse("book.tsv")
        self.assertNotIn("f" * 16, [r.fen for r in rows])
