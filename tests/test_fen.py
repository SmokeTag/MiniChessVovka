#!/usr/bin/env python3
"""The FEN serializer must be exactly as expressive as the Zobrist hash.

The book is keyed by hash, and a hash is one-way: `position.fen` is the only thing that
can turn a book row back into a position. If the FEN drops something the hash counts --
a piece in hand, a promoted rook that reverts to a pawn when captured -- then two
different positions serialize the same and the FEN cannot be trusted to reproduce the
entry it sits beside. So the property under test is not "it round-trips" but "it round-
trips *and lands on the same hash*".

Random games are what get this into the corners: they reach hands, drops and promotions
within a few dozen plies (see tests/test_rules_parity.py, which relies on the same).

    FEN_ROUNDTRIP_GAMES=200 ./venv/bin/python -m pytest tests/test_fen.py -q
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai
import minichess_engine as rs
from gamestate import GameState

DEFAULT_GAMES = int(os.environ.get("FEN_ROUNDTRIP_GAMES", "20"))
SEED = 20260827
MAX_PLY = 120

def _sweep(n_games=DEFAULT_GAMES, seed=SEED):
    """Walk random games, checking every position on the way.

    Returns (checked, saw_hand, saw_promoted) so the tests can also assert the sweep
    actually visited the cases it exists to cover.
    """
    rng = random.Random(seed)
    checked = saw_hand = saw_promoted = 0

    for _ in range(n_games):
        gs = GameState()
        gs.setup_initial_board()

        for _ply in range(MAX_PLY):
            fen = ai.to_fen(gs)
            parsed = ai.from_fen(fen)

            assert rs.get_position_hash(parsed) == ai.get_position_hash(gs), (
                "FEN %r hashes differently than the position it came from" % fen
            )
            assert rs.to_fen(parsed) == fen, "FEN %r is not canonical" % fen
            checked += 1
            if "[]" not in fen:
                saw_hand += 1
            if "~" in fen:
                saw_promoted += 1

            if gs.checkmate or gs.stalemate or gs.is_draw:
                break
            moves = gs.get_all_legal_moves()
            if not moves:
                break

            if not gs.make_move(rng.choice(moves), False):
                break
            if gs.needs_promotion_choice:
                piece = rng.choice(['R', 'N', 'B'])
                gs.complete_promotion(piece if gs.current_turn == 'b' else piece.lower())
            gs.check_game_over()

    return checked, saw_hand, saw_promoted

@pytest.fixture(scope="module")
def sweep():
    return _sweep()

def test_random_positions_round_trip_to_the_same_hash(sweep):
    checked, _, _ = sweep
    assert checked > 100, "the sweep barely ran; %d positions checked" % checked

def test_the_sweep_covers_hands_and_promotions(sweep):
    """Guards the test itself: a sweep that never fills a hand proves nothing."""
    _, saw_hand, saw_promoted = sweep
    assert saw_hand > 0, "no position in the sweep had a piece in hand"
    assert saw_promoted > 0, "no position in the sweep had a promoted piece"

def test_initial_position():
    gs = GameState()
    gs.setup_initial_board()
    assert ai.to_fen(gs) == "2bnrk/5p/6/6/P5/KRNB2[] w"

def test_hand_and_promotion_encoding():
    parsed = ai.from_fen("2bnrk/5p/6/6/P5/KRNB1R~[PPn] b")
    assert rs.to_fen(parsed) == "2bnrk/5p/6/6/P5/KRNB1R~[PPn] b"

@pytest.mark.parametrize("bad", [
    "",
    "6/6/6/6/6/6[] w",
    "2bnrk/5p/6/6/P5/KRNB2[] x",
    "2bnrk/5p/6/6/P5[] w",
    "2bnrk/5p/6/6/P5/KRNB3[] w",
    "2bnrk/5p/6/6/P5/KRNB2[Pk] w",
    "~2bnrk/5p/6/6/P5/KRNB2[] w",
    "2bnrk/5p/6/6/P5/KRNB2[P w",
    "2bnrk/5p/6/6/P5/KRNB2[]",
])
def test_malformed_fens_are_rejected(bad):
    with pytest.raises(ValueError):
        ai.from_fen(bad)
