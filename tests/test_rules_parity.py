#!/usr/bin/env python3
"""
Differential tests: gamestate.py vs engine_rs must agree on the rules.

This repo carries two independent implementations of the same 6x6 crazyhouse
rules -- gamestate.py drives the GUI, the chess.com bot and training, while
engine_rs/src/gamestate.rs drives the search -- and nothing else detects drift
between them. A rules change that lands in only one is silent: the engine
searches a position the GUI does not agree with.

These tests play identical random games through both, comparing legal move sets,
terminal flags, ply counters and promotion state at every ply. Written after the
draw-rule work, they immediately caught a live bug: Rust emitted uppercase
promotion chars for both colours while gamestate.py validates them
case-sensitively, so every engine-chosen Black promotion was refused -- the GUI
fell back to a random move and self-play aborted the game.

Set RULES_PARITY_GAMES to sweep harder than the default:

    RULES_PARITY_GAMES=500 ./venv/bin/python -m pytest tests/test_rules_parity.py -q
"""

import os
import random
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gamestate import GameState
import minichess_engine as _rs

DEFAULT_GAMES = int(os.environ.get("RULES_PARITY_GAMES", "50"))
SEED = 20260826
MAX_PLY = 400

def _norm(moves):
    """Legal moves as a set -- the two generators need not agree on order."""
    out = set()
    for m in moves:
        if m[0] == 'drop':
            out.add(('drop', m[1], tuple(m[2])))
        else:
            out.add((tuple(m[0]), tuple(m[1]), m[2]))
    return out

def _flags(state):
    return (bool(state.checkmate), bool(state.stalemate), bool(state.is_draw))

def _play_paired(n_games=DEFAULT_GAMES, seed=SEED):
    """Play n_games through both implementations in lockstep.

    Returns (mismatches, reasons, lengths). A mismatch is
    (game, ply, kind, detail) and stops that game.
    """
    rng = random.Random(seed)
    mismatches, reasons, lengths = [], collections.Counter(), []

    for g in range(n_games):
        py = GameState()
        py.setup_initial_board()
        rs = _rs.GameState()
        rs.setup_initial_board()

        for ply in range(MAX_PLY):
            pm, rm = _norm(py.get_all_legal_moves()), _norm(rs.get_all_legal_moves())
            if pm != rm:
                mismatches.append((g, ply, "legal moves",
                                   f"python-only={sorted(pm - rm, key=repr)[:3]} "
                                   f"rust-only={sorted(rm - pm, key=repr)[:3]}"))
                break

            if _flags(py) != _flags(rs):
                mismatches.append((g, ply, "terminal flags",
                                   f"(checkmate,stalemate,is_draw) "
                                   f"python={_flags(py)} rust={_flags(rs)}"))
                break

            if py.ply_count != rs.ply:
                mismatches.append((g, ply, "ply counter",
                                   f"python={py.ply_count} rust={rs.ply}"))
                break

            if any(_flags(py)) or not pm:
                reasons["checkmate" if py.checkmate else
                        "stalemate" if py.stalemate else "draw"] += 1
                lengths.append(ply)
                break

            move = rng.choice(sorted(pm, key=repr))
            py.make_move(move, False)
            rs.make_move(move, False)

            if py.needs_promotion_choice != rs.needs_promotion_choice:
                mismatches.append((g, ply, "promotion pending",
                                   f"python={py.needs_promotion_choice} "
                                   f"rust={rs.needs_promotion_choice}"))
                break
            if py.needs_promotion_choice:
                piece = rng.choice(['R', 'N', 'B'])
                piece = piece if py.current_turn == 'b' else piece.lower()
                py.complete_promotion(piece)
                rs.complete_promotion(piece)

            py.check_game_over()
            rs.check_game_over()
        else:
            reasons["NO TERMINATION"] += 1
            lengths.append(MAX_PLY)

    return mismatches, reasons, lengths

_cached = None

def _paired_run():
    """Run once, share across the tests in this module."""
    global _cached
    if _cached is None:
        _cached = _play_paired()
    return _cached

def test_rule_implementations_agree():
    """gamestate.py and engine_rs agree at every ply of every game."""
    mismatches, _, _ = _paired_run()
    if mismatches:
        detail = "\n".join(f"  game {g} ply {p}: {kind} -- {d}"
                           for g, p, kind, d in mismatches[:10])
        raise AssertionError(
            f"{len(mismatches)} rule mismatch(es) between gamestate.py and "
            f"engine_rs (seed {SEED}):\n{detail}")

def test_every_game_terminates():
    """No game runs forever.

    Crazyhouse never removes material from play, so without repetition and
    ply-limit draws a position can repeat indefinitely -- 45% of random games
    used to run past 300 plies with no result, which self-play cannot label.
    """
    _, reasons, lengths = _paired_run()
    assert reasons.get("NO TERMINATION", 0) == 0, (
        f"{reasons['NO TERMINATION']} game(s) hit {MAX_PLY} plies without "
        f"terminating; termination breakdown: {dict(reasons)}")
    assert lengths, "no games were played"

def test_promotion_chars_are_colour_cased():
    """Both implementations case promotion chars by colour.

    gamestate.py validates against PROMOTION_PIECES_WHITE_STR / _BLACK_STR,
    which are case-sensitive, so a mis-cased char is refused outright and the
    move silently does not happen.
    """
    for turn, pawn, pawn_rank, expected in (('b', 'p', 4, str.islower),
                                            ('w', 'P', 1, str.isupper)):
        board = [['.'] * 6 for _ in range(6)]
        board[pawn_rank][5] = pawn
        board[5][0], board[0][0] = 'K', 'k'

        py = GameState()
        py.setup_initial_board()
        py.board = [row[:] for row in board]
        py.king_pos = {'w': (5, 0), 'b': (0, 0)}
        py.current_turn = turn
        py._all_legal_moves_cache = None

        rs = _rs.GameState()
        rs.setup_initial_board()
        rs.board = [row[:] for row in board]
        rs.king_pos = {'w': (5, 0), 'b': (0, 0)}
        rs.current_turn = turn

        py_proms = [m for m in py.get_all_legal_moves() if len(m) == 3 and m[2]]
        rs_proms = [m for m in rs.get_all_legal_moves() if len(m) == 3 and m[2]]

        assert py_proms, f"no promotion generated for {turn} in gamestate.py"
        assert rs_proms, f"no promotion generated for {turn} in engine_rs"
        assert _norm(py_proms) == _norm(rs_proms), (
            f"{turn}: gamestate.py {sorted(py_proms, key=repr)} != "
            f"engine_rs {sorted(rs_proms, key=repr)}")
        for m in rs_proms:
            assert expected(m[2]), f"{turn}: engine_rs promotion {m} wrongly cased"

        applied = py.make_move(rs_proms[0], False)
        assert applied, f"{turn}: engine_rs move {rs_proms[0]} refused by gamestate.py"

if __name__ == "__main__":
    ms, rs_, ls = _play_paired()
    print(f"{DEFAULT_GAMES} paired games, termination: {dict(rs_)}")
    print(f"mean length {sum(ls)/len(ls):.1f} plies")
    print("no drift" if not ms else f"DRIFT: {ms[:5]}")
