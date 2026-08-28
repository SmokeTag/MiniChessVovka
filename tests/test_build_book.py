#!/usr/bin/env python3
"""Tests for the opening-repertoire builder.

Every test runs at a shallow depth inside `isolated_cache_db()`. The builder writes to
whatever `book.db` the CWD resolves to, so without the isolation these would fill the
live book with depth-3 rows.
"""

import argparse
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_book
from tests.cache_isolation import IsolatedCacheDB

import minichess_engine as engine


def build_args(**overrides):
    """The argparse namespace `build()` expects, with test-shaped defaults."""
    args = argparse.Namespace(
        max_ply=4, depth=3, color=("white", "black"), resign=1200,
        opponent_breadth="all", scan_depth=2, split_ply=2, shard=(0, 1),
        save_every=1000,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def read_book():
    """(book_move rows, position rows) keyed so two builds can be compared directly."""
    conn = sqlite3.connect("file:book.db?mode=ro", uri=True)
    moves = {(h, r): (m, s, d) for h, r, m, s, d in
             conn.execute("select hash, rank, move, score, depth from book_move")}
    positions = {h: (f, p) for h, f, p in
                 conn.execute("select hash, fen, ply from position")}
    conn.close()
    return moves, positions


def plies():
    conn = sqlite3.connect("file:book.db?mode=ro", uri=True)
    out = dict(conn.execute("select ply, count(*) from position group by ply"))
    conn.close()
    return out


class TestRepertoireShape(IsolatedCacheDB):
    def test_only_stores_positions_the_engine_is_to_move_in(self):
        """Even plies are White's repertoire, odd plies are Black's, and nothing else.

        This is the property the whole design rests on: the book is probed only at the
        root of a search, so a position where the opponent is to move can never be read
        and must never be written.
        """
        build_book.build(build_args(max_ply=3))
        conn = sqlite3.connect("file:book.db?mode=ro", uri=True)
        for _hash, fen, ply in conn.execute("select hash, fen, ply from position"):
            side_to_move = fen.split()[-1]
            expected = "w" if ply % 2 == 0 else "b"
            self.assertEqual(side_to_move, expected,
                             "ply %d position stored with %s to move: %s" % (ply, side_to_move, fen))
        conn.close()

    def test_one_child_per_our_move_and_all_opponent_replies(self):
        """White's ply-2 count is the number of Black replies to White's *one* chosen
        first move -- not 15 first moves x 15 replies."""
        build_book.build(build_args(max_ply=2, resign=10 ** 9))
        counts = plies()
        self.assertEqual(counts[0], 1)          # the initial position
        self.assertEqual(counts[1], 15)         # Black's repertoire: all 15 White openings
        self.assertLessEqual(counts[2], 16)     # White's: replies to one move, not 15x15

    def test_ply_recorded_is_the_ply_reached(self):
        build_book.build(build_args(max_ply=4))
        self.assertEqual(set(plies()) - {0, 1, 2, 3, 4}, set())


class TestBounds(IsolatedCacheDB):
    def test_resign_cutoff_stops_expansion_but_still_stores_the_node(self):
        """A decided line keeps its own entry -- we may still have to play it -- and just
        does not get a subtree."""
        build_book.build(build_args(max_ply=4, resign=1))
        counts = plies()
        self.assertEqual(counts.get(0), 1)
        self.assertEqual(counts.get(1), 15)
        # Nothing survives a cutoff of 1, so no tier below the seeds exists.
        self.assertNotIn(2, counts)
        self.assertNotIn(3, counts)

    def test_max_ply_is_a_hard_bound(self):
        build_book.build(build_args(max_ply=2))
        self.assertLessEqual(max(plies()), 2)

    def test_opponent_breadth_narrows_the_tier_below_it(self):
        with_all = build_args(max_ply=4, opponent_breadth="all")
        build_book.build(with_all)
        wide = plies().get(4, 0)

        # Same build, but only the best 2 replies at every opponent node.
        os.remove("book.db")
        for sidecar in ("book.db-wal", "book.db-shm"):
            if os.path.exists(sidecar):
                os.remove(sidecar)
        engine.load_move_cache_from_db()
        build_book.build(build_args(max_ply=4, opponent_breadth="0-20:2"))
        narrow = plies().get(4, 0)

        self.assertGreater(wide, 0)
        self.assertLess(narrow, wide)


class TestSharding(IsolatedCacheDB):
    def test_shards_reproduce_the_serial_build_exactly(self):
        """The shards share no state, so this is the only thing that says the split is
        correct rather than merely plausible."""
        build_book.build(build_args(max_ply=4))
        serial_moves, serial_positions = read_book()

        os.remove("book.db")
        for sidecar in ("book.db-wal", "book.db-shm"):
            if os.path.exists(sidecar):
                os.remove(sidecar)
        engine.load_move_cache_from_db()

        build_book.build(build_args(max_ply=2))           # the serial seed stage
        for shard in range(3):
            build_book.build(build_args(max_ply=4, split_ply=2, shard=(shard, 3)))
        sharded_moves, sharded_positions = read_book()

        self.assertEqual(serial_positions, sharded_positions)
        self.assertEqual(serial_moves, sharded_moves)


class TestBreadthSpec(unittest.TestCase):
    """Pure parsing -- no book, no isolation needed."""

    def test_forms(self):
        self.assertEqual(build_book.breadth_at(build_book.parse_breadth("all"), 7), None)
        self.assertEqual(build_book.breadth_at(build_book.parse_breadth("3"), 7), 3)
        rules = build_book.parse_breadth("0-8:all,9-16:3")
        self.assertIsNone(build_book.breadth_at(rules, 8))
        self.assertEqual(build_book.breadth_at(rules, 9), 3)
        self.assertEqual(build_book.breadth_at(rules, 16), 3)
        self.assertIsNone(build_book.breadth_at(rules, 17))  # uncovered -> full breadth

    def test_rejects_nonsense(self):
        for spec in ("0-8:zero", "0-8:0", "nope"):
            with self.assertRaises(argparse.ArgumentTypeError):
                build_book.parse_breadth(spec)


class TestDepthGuard(IsolatedCacheDB):
    def test_a_shallower_search_never_replaces_a_deeper_entry(self):
        """`book_store`'s guard. A narrowing scan can transpose onto a position some other
        line already searched deep; an entry is replaced wholesale, so without the guard
        the deep answer would be silently lost."""
        import ai
        ai.setup_db()
        # The Rust book is process-global, and cache_isolation reloads the *live* book on
        # the way out of every test -- so this one starts with the real book in memory and
        # a search from the initial position is a book hit that stores nothing. Point the
        # in-memory book at this test's (absent, therefore empty) book.db instead.
        engine.load_move_cache_from_db()
        gs = engine.GameState()
        gs.setup_initial_board()

        deep_move, deep_score = engine.find_best_move_with_score(gs.copy(), 6, None, False)
        engine.find_best_move(gs.copy(), 2, 3, None, False)   # shallow MultiPV, same position
        ai.save_move_cache_to_db()

        rows = sorted(sqlite3.connect("file:book.db?mode=ro", uri=True)
                      .execute("select rank, depth, score from book_move"))
        self.assertEqual(rows, [(1, 6, deep_score)],
                         "the depth-2 scan overwrote the depth-6 entry")
        self.assertIsNotNone(deep_move)


if __name__ == "__main__":
    unittest.main()
