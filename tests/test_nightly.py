#!/usr/bin/env python3
"""
Nightly tests for MiniChess - includes database operations and cache verification.
Covers the opening book schema in engine_rs/src/cache.rs.
"""

import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai
import minichess_engine as rs
from tests.cache_isolation import IsolatedCacheDB

DB_PATH = "book.db"

def columns(table):
    conn = sqlite3.connect(DB_PATH)
    try:
        return [col[1] for col in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()

class TestDatabaseMigration(IsolatedCacheDB):
    """Schema creation and the version-driven rebuild."""

    def test_setup_db_creates_book_schema(self):
        ai.setup_db()

        self.assertEqual(
            columns("book_move"),
            ["hash", "rank", "move", "score", "depth", "eval_version"],
        )
        self.assertEqual(columns("position"), ["hash", "fen", "ply"])

    def test_schema_version_is_stamped(self):
        ai.setup_db()

        conn = sqlite3.connect(DB_PATH)
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(version, rs.SCHEMA_VERSION)

    def write_foreign_schema(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("CREATE TABLE book_move (hash TEXT, whatever TEXT)")
        conn.execute("INSERT INTO book_move VALUES ('stale', 'row')")
        conn.execute("PRAGMA user_version = 9999")
        conn.commit()
        conn.close()

    def test_foreign_schema_is_refused_not_rebuilt(self):
        """setup_db must never drop a table on its own.

        "Recreate the schema" runs on paths nobody thinks of as destructive -- a
        worker's first save, a stray call from a test -- so a SCHEMA_VERSION bump that
        dropped on sight would turn any of them into a silent wipe of the training data.
        The refusal has to name both versions and the way out, because it is the only
        thing the operator will see.
        """
        self.write_foreign_schema()

        with self.assertRaises(RuntimeError) as caught:
            ai.setup_db()

        message = str(caught.exception)
        self.assertIn("9999", message, "the refusal must name the on-disk version")
        self.assertIn(str(rs.SCHEMA_VERSION), message,
                      "the refusal must name the expected version")
        self.assertIn("rebuild_book.py", message,
                      "the refusal must say how to rebuild")

        self.assertEqual(columns("book_move"), ["hash", "whatever"])
        conn = sqlite3.connect(DB_PATH)
        try:
            self.assertEqual(
                conn.execute("SELECT count(*) FROM book_move").fetchone()[0], 1
            )
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 9999)
        finally:
            conn.close()

    def test_saving_into_a_foreign_schema_raises_and_keeps_the_rows(self):
        """A worker must fail loudly, not write this build's rows into other tables."""
        ai.setup_db()
        ai.load_move_cache_from_db()
        gs = rs.GameState()
        gs.setup_initial_board()
        rs.find_best_move(gs, 4, 1, None, False)

        conn = sqlite3.connect(DB_PATH)
        conn.execute("DROP TABLE book_move")
        conn.execute("CREATE TABLE book_move (hash TEXT, whatever TEXT)")
        conn.execute("PRAGMA user_version = 9999")
        conn.commit()
        conn.close()

        with self.assertRaises(RuntimeError):
            ai.save_move_cache_to_db()

        ai.rebuild_book()
        ai.save_move_cache_to_db()
        conn = sqlite3.connect(DB_PATH)
        try:
            self.assertEqual(
                conn.execute("SELECT count(*) FROM book_move").fetchone()[0], 1
            )
        finally:
            conn.close()

    def test_rebuild_is_the_explicit_opt_in(self):
        """The destructive path exists -- it just has to be asked for by name."""
        self.write_foreign_schema()

        ai.rebuild_book()

        self.assertEqual(
            columns("book_move"),
            ["hash", "rank", "move", "score", "depth", "eval_version"],
        )
        self.assertEqual(columns("position"), ["hash", "fen", "ply"])
        conn = sqlite3.connect(DB_PATH)
        try:
            self.assertEqual(
                conn.execute("SELECT count(*) FROM book_move").fetchone()[0], 0
            )
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0],
                             rs.SCHEMA_VERSION)
        finally:
            conn.close()

    def test_matching_version_keeps_rows(self):
        """The rebuild must not fire on every open, or the book would never persist."""
        ai.setup_db()
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO book_move (hash, rank, move, score, depth, eval_version)"
            " VALUES ('keepme', 1, '((0, 0), (1, 0), None)', 12, 8, ?)",
            (rs.EVAL_VERSION,),
        )
        conn.commit()
        conn.close()

        ai.setup_db()

        conn = sqlite3.connect(DB_PATH)
        try:
            rows = conn.execute(
                "SELECT count(*) FROM book_move WHERE hash = 'keepme'"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(rows, 1)

class TestDatabaseOperations(IsolatedCacheDB):
    """Book rows survive a load/save cycle through the Rust engine."""

    def test_save_and_load_book(self):
        ai.setup_db()

        conn = sqlite3.connect(DB_PATH)
        for rank, move, score in (
            (1, "((0, 0), (1, 0), None)", 40),
            (2, "((0, 0), (2, 0), None)", 25),
        ):
            conn.execute(
                "INSERT INTO book_move (hash, rank, move, score, depth, eval_version)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                ("test_hash_nightly_1", rank, move, score, 6, rs.EVAL_VERSION),
            )
        conn.execute(
            "INSERT INTO position (hash, fen, ply) VALUES (?, ?, ?)",
            ("test_hash_nightly_1", "2bnrk/5p/6/6/P5/KRNB2[] w", 0),
        )
        conn.commit()
        conn.close()

        ai.load_move_cache_from_db()
        self.assertEqual(ai.book_size(), 1)
        ai.save_move_cache_to_db()

        conn = sqlite3.connect(DB_PATH)
        try:
            rows = conn.execute(
                "SELECT rank, move, score, depth FROM book_move"
                " WHERE hash = 'test_hash_nightly_1' ORDER BY rank"
            ).fetchall()
        finally:
            conn.close()

        self.assertEqual(
            rows,
            [
                (1, "((0, 0), (1, 0), None)", 40, 6),
                (2, "((0, 0), (2, 0), None)", 25, 6),
            ],
        )

if __name__ == '__main__':
    unittest.main()
