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

# Relative on purpose. `DB_PATH` in engine_rs/src/cache.rs is the hardcoded relative
# string "book.db", so the Rust side always opens the DB in the process CWD. Keeping
# this relative too means both halves follow the chdir in IsolatedCacheDB and land on
# the same throwaway file.
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

    def test_foreign_schema_is_rebuilt(self):
        """A DB written by another schema version is dropped and recreated.

        This is what replaced the old "no depth column -> DROP TABLE" sniffing, which
        could not tell a schema it had never seen from one it wrote itself.
        """
        conn = sqlite3.connect(DB_PATH)
        conn.execute("CREATE TABLE book_move (hash TEXT, whatever TEXT)")
        conn.execute("INSERT INTO book_move VALUES ('stale', 'row')")
        conn.execute("PRAGMA user_version = 9999")
        conn.commit()
        conn.close()

        ai.setup_db()

        self.assertEqual(
            columns("book_move"),
            ["hash", "rank", "move", "score", "depth", "eval_version"],
        )
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
        # Nothing was searched, so nothing is dirty and the save is a no-op that must
        # not disturb the rows already there.
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
