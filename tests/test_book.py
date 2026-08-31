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

MATE_IN_ONE = "k5/6/2K3/6/6/1Q4[] w"

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

        rs.find_best_move(initial_state(), 8, 1, None, False)
        ai.save_move_cache_to_db()
        self.assertEqual(rows("SELECT depth FROM book_move")[0][0], 8)

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

        ai.load_move_cache_from_db()
        rs.find_best_move(initial_state(), 6, 1, None, False)
        ai.save_move_cache_to_db()
        self.assertEqual(rows("SELECT ply FROM position")[0][0], 0)

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

class TestAnalysisMergeKeepsTheBetterEntry(IsolatedCacheDB):
    """What `load_analysis_from_db` does when both stores hold the same position.

    The book builder files rank 1 only -- MultiPV costs 2.4-4.4x -- so a curated row is
    routinely *narrower* than the row a GUI session produced for the same position. The
    merge used to be `if !book.contains_key(hash)`, all or nothing, so the 1-rank book
    row shadowed a 4-rank cached row and `probe_book` then rejected it for holding fewer
    ranks than asked. A hint at 4 lines re-searched the initial position on every single
    launch with the answer already on disk.

    The exception is deliberately narrow: same rank-1 move, same eval_version, and
    strictly more evidence -- deeper, or more ranks, and never less of either. What the
    engine plays cannot change.

    The analysis rows are written by hand rather than by a second search on purpose. A
    single-PV root and a MultiPV root can name different moves at the same depth (the
    drift documented on `multipv_root`), so a test that searched twice would be asserting
    that they happened to agree rather than what the merge rule does.
    """

    def book_row(self, depth=6):
        """File a 1-rank book entry for the initial position and return its row."""
        rs.find_best_move(initial_state(), depth, 1, None, False)
        ai.save_move_cache_to_db()
        ai.load_move_cache_from_db()
        return rows("SELECT hash, move, score, depth, eval_version "
                    "FROM book_move WHERE rank = 1")[0]

    def write_analysis(self, pos_hash, moves):
        """`moves` is [(rank, move_repr, score, depth, eval_version)]."""
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("INSERT OR REPLACE INTO analysis_position(hash, fen, ply) "
                         "SELECT hash, fen, ply FROM position WHERE hash = ?", (pos_hash,))
            for row in moves:
                conn.execute(
                    "INSERT OR REPLACE INTO analysis_move"
                    "(hash, rank, move, score, depth, eval_version) VALUES (?,?,?,?,?,?)",
                    (pos_hash,) + row)
            conn.commit()
        finally:
            conn.close()

    def other_legal_moves(self, exclude, count):
        legal = [str(m) for m in initial_state().get_all_legal_moves()]
        return [m for m in legal if m != exclude][:count]

    def reload(self):
        ai.load_move_cache_from_db()
        return ai.load_analysis_from_db()

    def test_more_ranks_for_the_same_move_replaces_the_narrower_book_row(self):
        fresh_book()
        pos_hash, move, score, depth, ev = self.book_row()
        extra = self.other_legal_moves(move, 2)
        self.write_analysis(pos_hash, [
            (1, move, score, depth, ev),
            (2, extra[0], score - 10, depth, ev),
            (3, extra[1], score - 20, depth, ev),
        ])

        self.assertEqual(self.reload(), 1, "the wider entry must be taken")

        hit = rs.find_best_move(initial_state(), depth, 3, None, False)
        self.assertEqual(len(hit), 3, "a 3-line probe is now answerable from cache")
        self.assertEqual(str(hit[0][0]), move, "and rank 1 is still the book's move")

    def test_the_narrower_book_row_alone_cannot_answer_a_wider_probe(self):
        """The other half of the bug: without the merge there is nothing to hit."""
        fresh_book()
        pos_hash, move, score, depth, ev = self.book_row()

        self.assertEqual(ai.pending_book_writes(), 0, "the load leaves nothing pending")

        hit = rs.find_best_move(initial_state(), depth, 3, None, False)
        self.assertEqual(len(hit), 3)
        self.assertEqual(ai.pending_book_writes(), 1,
                         "a 1-rank row cannot serve a 3-rank request, so it re-searched "
                         "and filed a new entry -- a hit would have written nothing")

    def test_a_cached_entry_naming_a_different_move_never_wins(self):
        """The curation guarantee, forced: deeper *and* wider, but a different rank 1."""
        fresh_book()
        pos_hash, move, score, depth, ev = self.book_row()
        others = self.other_legal_moves(move, 1)
        self.write_analysis(pos_hash, [
            (1, others[0], score + 50, depth + 4, ev),
            (2, move, score, depth + 4, ev),
        ])

        self.assertEqual(self.reload(), 0, "a different move must not displace the book")
        played = rs.find_best_move(initial_state(), depth, 1, None, False)
        self.assertEqual(str(played), move)

    def test_a_shallower_cached_entry_never_wins_even_with_more_ranks(self):
        """Mirrors `book_store`: never trade a deeper answer for a shallower one."""
        fresh_book()
        pos_hash, move, score, depth, ev = self.book_row()
        extra = self.other_legal_moves(move, 1)
        self.write_analysis(pos_hash, [
            (1, move, score, depth - 2, ev),
            (2, extra[0], score - 10, depth - 2, ev),
        ])

        self.assertEqual(self.reload(), 0, "extra ranks do not buy a shallower entry in")
        _, stored_depth = rs.find_best_move_with_score(initial_state(), depth, None, False)
        self.assertEqual(len(rows("SELECT 1 FROM book_move WHERE hash = ?", pos_hash)), 1)

    def test_a_row_from_another_eval_version_never_wins(self):
        fresh_book()
        pos_hash, move, score, depth, ev = self.book_row()
        extra = self.other_legal_moves(move, 1)
        self.write_analysis(pos_hash, [
            (1, move, score, depth + 4, ev + 1),
            (2, extra[0], score - 10, depth + 4, ev + 1),
        ])

        self.assertEqual(self.reload(), 0, "a stale evaluation is not more evidence")
