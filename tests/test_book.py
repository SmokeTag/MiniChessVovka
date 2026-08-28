#!/usr/bin/env python3
"""What the opening book is allowed to contain, and what it must return.

The old `move_cache` table stored a row per *(hash, depth)*: iterative deepening filed
one for every depth it passed through and the transposition table dumped interior nodes
on top of that, so ~97% of a 100k-row DB was unreachable by the depth-10 probe that
training runs. These tests pin the replacement down from both ends -- exactly what one
search writes, and exactly what a probe will hand back.

Every test runs in a throwaway CWD (`IsolatedCacheDB`): `DB_PATH` in
engine_rs/src/cache.rs is the relative string "book.db", so the CWD is the only seam
between a test and the live book.
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

# White mates in one: Qb1-b5 is covered by the king on c4, so Black has no reply. Used to
# check that the mate break -- which stops iterative deepening early -- files the depth it
# reached rather than the depth it was asked for.
MATE_IN_ONE = "k5/6/2K3/6/6/1Q4[] w"

# Black is in check from the rook on d6 with every flight square but a5 covered, so there
# is exactly one legal move.
ONLY_ONE_LEGAL_MOVE = "k2R2/6/1R4/6/6/5K[] b"


def fresh_book():
    """Start a test from an empty in-memory book.

    The Rust book is process-global and `IsolatedCacheDB` only restores it between tests,
    so without this a developer whose repo-root book.db is full of real training data
    would run these against a warm book and get book hits where a search is expected.
    """
    ai.setup_db()
    ai.load_move_cache_from_db()


def initial_state():
    gs = rs.GameState()
    gs.setup_initial_board()
    return gs


def rows(query, *params):
    conn = sqlite3.connect(DB_PATH)
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


class TestWhatOneSearchWrites(IsolatedCacheDB):
    def test_one_row_per_rank_and_one_position_and_nothing_else(self):
        """A single depth-10 search leaves exactly `top_n` + 1 rows in the DB."""
        fresh_book()
        gs = initial_state()

        ranked = rs.find_best_move(gs, 10, 3, None, False)
        ai.save_move_cache_to_db()

        self.assertEqual(len(ranked), 3)
        book = rows("SELECT hash, rank, depth, eval_version FROM book_move ORDER BY rank")
        self.assertEqual(len(book), 3, "one row per rank, and no per-depth or TT rows")
        self.assertEqual([r[1] for r in book], [1, 2, 3])
        self.assertEqual({r[2] for r in book}, {10}, "depth must be the one completed")
        self.assertEqual({r[3] for r in book}, {rs.EVAL_VERSION})

        positions = rows("SELECT hash, fen, ply FROM position")
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0][0], book[0][0], "same hash in both tables")
        self.assertEqual(positions[0][2], 0, "the initial position is at ply 0")

    def test_forced_moves_are_not_stored(self):
        """A position with one legal move is answered, not filed.

        Nothing searched, so there is no completed depth behind it and no evaluation
        behind its score -- and every row in the book is supposed to be a real result.
        """
        fresh_book()
        gs = rs.from_fen(ONLY_ONE_LEGAL_MOVE)

        ranked = rs.find_best_move(gs, 8, 3, None, False)
        ai.save_move_cache_to_db()

        self.assertEqual(len(ranked), 1)
        self.assertEqual(rows("SELECT count(*) FROM book_move")[0][0], 0)
        self.assertEqual(rows("SELECT count(*) FROM position")[0][0], 0)


class TestStoredDepthIsTheCompletedDepth(IsolatedCacheDB):
    def test_mate_break_records_the_depth_it_reached(self):
        fresh_book()
        gs = rs.from_fen(MATE_IN_ONE)

        rs.find_best_move(gs, 8, 1, None, False)
        ai.save_move_cache_to_db()

        depth = rows("SELECT depth FROM book_move")[0][0]
        self.assertLess(depth, 8, "the mate break exits early; the row must say so")
        self.assertGreaterEqual(depth, 1)

    def test_time_limited_search_never_claims_the_requested_depth(self):
        fresh_book()
        gs = initial_state()

        rs.find_best_move(gs, 14, 1, 0.5, False)
        ai.save_move_cache_to_db()

        stored = rows("SELECT depth FROM book_move")
        if stored:
            self.assertLess(
                stored[0][0], 14,
                "a search cut short by its time limit must not file its answer as deep",
            )


class TestProbe(IsolatedCacheDB):
    def test_shallower_entries_are_rejected_and_deeper_ones_accepted(self):
        fresh_book()
        gs = initial_state()

        rs.find_best_move(gs, 4, 1, None, False)
        ai.save_move_cache_to_db()
        self.assertEqual(rows("SELECT depth FROM book_move")[0][0], 4)

        # Asking for more than the book holds has to re-search, which overwrites the row.
        rs.find_best_move(initial_state(), 8, 1, None, False)
        ai.save_move_cache_to_db()
        self.assertEqual(rows("SELECT depth FROM book_move")[0][0], 8)

        # Asking for less is answered from the deeper entry: same evidence, and more of
        # it. The row is left alone.
        rs.find_best_move(initial_state(), 3, 1, None, False)
        ai.save_move_cache_to_db()
        self.assertEqual(rows("SELECT depth FROM book_move")[0][0], 8)

    def test_rows_from_another_eval_version_are_ignored(self):
        fresh_book()
        gs = initial_state()
        pos_hash = rs.get_position_hash(gs)

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO book_move (hash, rank, move, score, depth, eval_version)"
            " VALUES (?, 1, ?, 999, 20, ?)",
            (pos_hash, "((5, 1), (4, 1), None)", rs.EVAL_VERSION + 1),
        )
        conn.commit()
        conn.close()
        ai.load_move_cache_from_db()

        rs.find_best_move(gs, 4, 1, None, False)
        ai.save_move_cache_to_db()

        score, depth, version = rows(
            "SELECT score, depth, eval_version FROM book_move WHERE rank = 1"
        )[0]
        self.assertEqual(version, rs.EVAL_VERSION)
        self.assertEqual(depth, 4, "the stale deep row must not have answered the probe")
        self.assertNotEqual(score, 999)

    def test_probe_does_not_need_the_position_table(self):
        """The hot path reads `book_move` alone -- dropping `position` cannot break it."""
        fresh_book()
        rs.find_best_move(initial_state(), 6, 2, None, False)
        ai.save_move_cache_to_db()

        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM position")
        conn.commit()
        conn.close()
        ai.load_move_cache_from_db()

        ranked = rs.find_best_move(initial_state(), 6, 2, None, False)
        self.assertEqual(len(ranked), 2)


class TestMultiPV(IsolatedCacheDB):
    def assert_ranked_for_white(self, ranked, n):
        self.assertEqual(len(ranked), n)
        moves = [m for m, _ in ranked]
        self.assertEqual(len(set(map(str, moves))), n, "ranks must be distinct moves")
        scores = [s for _, s in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True),
                         "scores are white-relative, so White's ranks run non-increasing")

    def test_three_ranks_on_a_fresh_search_and_on_a_book_hit(self):
        fresh_book()

        fresh = rs.find_best_move(initial_state(), 8, 3, None, False)
        self.assert_ranked_for_white(fresh, 3)

        ai.save_move_cache_to_db()
        ai.load_move_cache_from_db()

        hit = rs.find_best_move(initial_state(), 8, 3, None, False)
        self.assert_ranked_for_white(hit, 3)
        self.assertEqual(hit, fresh, "a book hit returns the stored moves and scores")

    def test_book_hit_returns_the_stored_score_not_zero(self):
        fresh_book()
        rs.find_best_move(initial_state(), 6, 1, None, False)
        ai.save_move_cache_to_db()
        stored_score = rows("SELECT score FROM book_move WHERE rank = 1")[0][0]

        ai.load_move_cache_from_db()
        move, score = rs.find_best_move_with_score(initial_state(), 6, None, False)

        self.assertIsNotNone(move)
        self.assertEqual(score, stored_score)

    def test_black_ranks_are_ordered_best_first_for_black(self):
        """Scores stay white-relative, so Black's ranks run the other way."""
        fresh_book()
        gs = initial_state()
        gs.make_move(rs.find_best_move(gs, 4, 1, None, False))

        ranked = rs.find_best_move(gs, 6, 3, None, False)

        self.assertEqual(len(ranked), 3)
        scores = [s for _, s in ranked]
        self.assertEqual(scores, sorted(scores),
                         "rank 1 is Black's best, which is White's worst")

    def test_exploration_gets_a_real_second_move(self):
        """A caller asking for two ranks used to get one, silently."""
        fresh_book()
        gs = initial_state()

        ranked = rs.find_best_move(gs, 6, 2, None, False)

        self.assertEqual(len(ranked), 2)
        self.assertNotEqual(str(ranked[0][0]), str(ranked[1][0]))


class TestPositionTable(IsolatedCacheDB):
    def test_every_stored_fen_round_trips_and_hashes_back(self):
        fresh_book()
        gs = initial_state()
        for _ in range(4):
            move = rs.find_best_move(gs, 4, 1, None, False)
            if move is None:
                break
            gs.make_move(move)
        ai.save_move_cache_to_db()

        stored = rows("SELECT hash, fen, ply FROM position")
        self.assertGreaterEqual(len(stored), 4)
        for pos_hash, fen, ply in stored:
            parsed = rs.from_fen(fen)
            self.assertEqual(rs.get_position_hash(parsed), pos_hash,
                             "FEN %r does not hash back to its own row" % fen)
            self.assertEqual(rs.to_fen(parsed), fen, "FEN %r is not canonical" % fen)
            self.assertIsNotNone(ply)

    def test_ply_keeps_the_minimum_seen(self):
        """The same position can be reached at different plies; the book wants the first."""
        fresh_book()
        gs = initial_state()
        rs.find_best_move(gs, 4, 1, None, False)
        ai.save_move_cache_to_db()
        pos_hash = rs.get_position_hash(gs)
        self.assertEqual(rows("SELECT ply FROM position")[0][0], 0)

        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE position SET ply = 12 WHERE hash = ?", (pos_hash,))
        conn.commit()
        conn.close()

        # Re-search the same position: it arrives claiming ply 0 again, which is earlier.
        ai.load_move_cache_from_db()
        rs.find_best_move(initial_state(), 6, 1, None, False)
        ai.save_move_cache_to_db()
        self.assertEqual(rows("SELECT ply FROM position")[0][0], 0)

        # And a later sighting of the same position must not push the minimum back up.
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE position SET ply = 0 WHERE hash = ?", (pos_hash,))
        conn.commit()
        conn.close()
        later = initial_state()
        later.ply = 30
        ai.load_move_cache_from_db()
        rs.find_best_move(later, 8, 1, None, False)
        ai.save_move_cache_to_db()
        self.assertEqual(rows("SELECT ply FROM position")[0][0], 0)


if __name__ == '__main__':
    unittest.main()
