#!/usr/bin/env python3
"""The network's view of a position, and the map between moves and action indices.

This is bookkeeping, and a bug in it is close to undebuggable: a scrambled board or a
permuted policy still trains to a plausible-looking loss, and the failure only shows up
a week later as a network that never got stronger. So the encoding is fuzzed rather
than spot-checked -- every legal move of every position in thousands of random games
has to survive the round trip, and no two of them may land on the same index.

The two properties that matter:

  * **Injective within a position.** Two legal moves sharing an action index means the
    search can never distinguish them and the policy target is a lie.
  * **Exactly invertible.** `index_to_move(move_to_index(m)) == m`, so a visit
    distribution over indices can be read back as moves.

Canonicalisation is the third: the board is rotated 180 degrees and the colours swapped
when Black is to move, so the network only ever sees a position from the mover's side.
With no castling and no en passant that flip is exact, which
`test_black_to_move_mirrors_white_to_move` is here to hold.

    ENCODING_FUZZ_GAMES=1000 ./venv/bin/python -m pytest tests/test_encoding.py -q
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai
import minichess_engine as rs
from gamestate import GameState

DEFAULT_GAMES = int(os.environ.get("ENCODING_FUZZ_GAMES", "200"))
BOUNDARY_GAMES = int(os.environ.get("ENCODING_BOUNDARY_GAMES", "12"))
SEED = 20260902
MAX_PLY = 120

PLANE_SIZE = 36
OWN_PIECES, OPP_PIECES = 0, 5
OWN_PROMOTED, OPP_PROMOTED = 10, 11
OWN_HANDS, OPP_HANDS = 12, 16
SIDE_TO_MOVE, REPEAT_ONCE, REPEAT_TWICE, PROGRESS = 20, 21, 22, 23

def plane(planes, i):
    return planes[i * PLANE_SIZE:(i + 1) * PLANE_SIZE]

def at(planes, i, r, f):
    return planes[i * PLANE_SIZE + r * 6 + f]

def normalize(move):
    """Promotion case encodes colour, and the two generators pick it from different
    places. The action index carries the piece, not its case, so compare case-folded."""
    if move and len(move) == 3 and move[0] != 'drop' and move[2]:
        return (move[0], move[1], move[2].upper())
    return move

def _fuzz(n_games=DEFAULT_GAMES, seed=SEED):
    """Round-trip every legal move of every position in `n_games` random games.

    Returns a coverage dict so the tests can also assert the fuzz reached the cases it
    exists to cover -- a sweep that never sees a drop or a promotion proves nothing.
    """
    rng = random.Random(seed)
    seen = {
        'positions': 0, 'actions': 0,
        'drops': 0, 'promotions': 0, 'promoted_on_board': 0, 'repetitions': 0,
        'black_to_move': 0,
    }

    for _ in range(n_games):
        gs = rs.GameState()
        gs.setup_initial_board()

        for _ply in range(MAX_PLY):
            moves = gs.get_all_legal_moves()
            if not moves or gs.is_terminal_draw():
                break

            planes = rs.encode_position(gs)
            assert len(planes) == rs.ENCODE_INPUT_SIZE
            assert all(0.0 <= v <= 1.0 for v in planes), "a plane left [0, 1]"

            index_of = {}
            for m in moves:
                idx = rs.move_to_action_index(gs, m)
                assert 0 <= idx < rs.ACTION_SPACE, "index %d out of range" % idx
                assert idx not in index_of, (
                    "action %d is shared by %r and %r" % (idx, index_of[idx], m)
                )
                index_of[idx] = m
                assert normalize(rs.action_index_to_move(gs, idx)) == normalize(m), (
                    "index %d did not decode back to %r" % (idx, m)
                )
                if m[0] == 'drop':
                    seen['drops'] += 1
                elif m[2]:
                    seen['promotions'] += 1

            assert sorted(rs.legal_action_indices(gs)) == sorted(index_of), (
                "the legal mask disagrees with the moves it was built from"
            )

            seen['positions'] += 1
            seen['actions'] += len(moves)
            if gs.current_turn == 'b':
                seen['black_to_move'] += 1
            if sum(plane(planes, OWN_PROMOTED)) or sum(plane(planes, OPP_PROMOTED)):
                seen['promoted_on_board'] += 1
            if planes[REPEAT_ONCE * PLANE_SIZE]:
                seen['repetitions'] += 1

            gs.make_ai_move(rng.choice(moves))

    return seen

@pytest.fixture(scope="module")
def fuzz():
    return _fuzz()

def test_every_legal_move_round_trips(fuzz):
    """The exit criterion for phase 1: 10k positions, no collisions, no losses.

    Every assertion is inside `_fuzz`; this holds the sample size, because a sweep that
    quietly stopped after four positions would pass all of them.
    """
    assert fuzz['positions'] >= 10_000, (
        "only %d positions fuzzed; raise ENCODING_FUZZ_GAMES" % fuzz['positions']
    )
    assert fuzz['actions'] >= 100_000

def test_the_fuzz_reaches_the_hard_cases(fuzz):
    """Guards the test itself. Drops and promotions are where the encoding is unusual --
    drops index by target square rather than origin, promotions get their own planes."""
    assert fuzz['drops'] > 0, "no drop was ever encoded"
    assert fuzz['promotions'] > 0, "no promotion was ever encoded"
    assert fuzz['promoted_on_board'] > 0, "the promoted planes were never exercised"
    assert fuzz['black_to_move'] > 0, "canonicalisation was never exercised"

def test_action_space_shape():
    assert rs.ACTION_PLANES == 61, "40 ray + 8 knight + 9 promotion + 4 drop"
    assert rs.ACTION_SPACE == 61 * 36 == 2196
    assert rs.ENCODE_PLANES == 24
    assert rs.ENCODE_INPUT_SIZE == 24 * 36

def test_initial_position_planes():
    gs = rs.GameState()
    gs.setup_initial_board()
    planes = rs.encode_position(gs)

    # White to move, so no flip: kings sit where the board puts them.
    assert at(planes, OWN_PIECES + 4, 5, 0) == 1.0
    assert at(planes, OPP_PIECES + 4, 0, 5) == 1.0
    assert at(planes, OWN_PIECES + 0, 4, 0) == 1.0, "White pawn on (4, 0)"
    assert at(planes, OPP_PIECES + 0, 1, 5) == 1.0, "Black pawn on (1, 5)"

    assert sum(planes[:OWN_PROMOTED * PLANE_SIZE]) == 10.0, "ten men, one plane each"
    assert sum(planes[OWN_PROMOTED * PLANE_SIZE:OWN_HANDS * PLANE_SIZE]) == 0.0
    assert sum(planes[OWN_HANDS * PLANE_SIZE:SIDE_TO_MOVE * PLANE_SIZE]) == 0.0
    assert set(plane(planes, SIDE_TO_MOVE)) == {1.0}, "White to move"
    assert set(plane(planes, REPEAT_ONCE)) == {0.0}
    assert set(plane(planes, PROGRESS)) == {0.0}, "ply 0 of the cap"

def test_black_to_move_mirrors_white_to_move():
    """The canonical flip is a 180 degree rotation plus a colour swap. Build the mirror
    of the opening by hand and the two must encode identically -- side-to-move apart,
    which is kept precisely so a scrambled flip cannot hide."""
    white = rs.GameState()
    white.setup_initial_board()

    board = white.board
    mirrored = [['.' for _ in range(6)] for _ in range(6)]
    for r in range(6):
        for f in range(6):
            c = board[r][f]
            if c != '.':
                mirrored[5 - r][5 - f] = c.swapcase()

    black = rs.GameState()
    black.setup_initial_board()
    black.board = mirrored
    black.current_turn = 'b'
    black.find_kings()

    a = rs.encode_position(white)
    b = rs.encode_position(black)
    for i in range(rs.ENCODE_PLANES):
        if i == SIDE_TO_MOVE:
            continue
        assert plane(a, i) == plane(b, i), "plane %d survives the flip" % i
    assert plane(a, SIDE_TO_MOVE)[0] == 1.0
    assert plane(b, SIDE_TO_MOVE)[0] == 0.0

def test_hands_and_promotion_planes_carry_state():
    gs = rs.GameState()
    gs.setup_initial_board()
    gs.hands = {'w': {'N': 2}, 'b': {'P': 1, 'R': 3}}
    gs.promoted_pieces = [(5, 1)]
    planes = rs.encode_position(gs)

    assert set(plane(planes, OWN_HANDS + 1)) == {2.0 / 8.0}, "own knights, broadcast"
    assert set(plane(planes, OPP_HANDS + 0)) == {1.0 / 8.0}
    assert set(plane(planes, OPP_HANDS + 3)) == {3.0 / 8.0}
    assert set(plane(planes, OWN_HANDS + 0)) == {0.0}
    assert at(planes, OWN_PROMOTED, 5, 1) == 1.0, "White's rook on (5, 1) is an ex-pawn"
    assert sum(plane(planes, OPP_PROMOTED)) == 0.0

def test_drops_index_by_target_square():
    """Drops are the one action group indexed by where the piece lands rather than
    where it came from -- that is what lets them share the convolutional head."""
    gs = rs.GameState()
    gs.setup_initial_board()
    gs.hands = {'w': {'N': 1}, 'b': {}}

    drops = [m for m in gs.get_all_legal_moves() if m[0] == 'drop']
    assert drops, "a knight in hand must produce drops"
    for m in drops:
        idx = rs.move_to_action_index(gs, m)
        r, f = m[2]
        assert idx % PLANE_SIZE == r * 6 + f
        assert idx // PLANE_SIZE == 57 + 1, "knight drop plane"
        assert rs.action_index_to_move(gs, idx) == m

def test_progress_and_repetition_planes():
    gs = rs.GameState()
    gs.setup_initial_board()
    gs.ply = 50
    gs.ply_limit = 200
    planes = rs.encode_position(gs)
    assert set(plane(planes, PROGRESS)) == {0.25}

    # Shuffle the kings back and forth until the opening position recurs.
    gs = rs.GameState()
    gs.setup_initial_board()
    assert set(plane(rs.encode_position(gs), REPEAT_ONCE)) == {0.0}
    for m in [((5, 0), (4, 1), None), ((0, 5), (1, 4), None),
              ((4, 1), (5, 0), None), ((1, 4), (0, 5), None)]:
        gs.make_ai_move(m)
    planes = rs.encode_position(gs)
    assert set(plane(planes, REPEAT_ONCE)) == {1.0}, "this position has been here before"
    assert set(plane(planes, REPEAT_TWICE)) == {0.0}

def test_out_of_range_and_off_board_indices():
    gs = rs.GameState()
    gs.setup_initial_board()
    assert rs.action_index_to_move(gs, rs.ACTION_SPACE) is None
    # Ray plane 0 is "one square toward rank 6"; from the top rank it leaves the board.
    assert rs.action_index_to_move(gs, 0 * PLANE_SIZE + 0) is None
    assert rs.action_index_to_move(gs, 0 * PLANE_SIZE + 6) is not None

def test_the_python_gamestate_encodes_the_same_position():
    """`gamestate.py` owns the live game and `_sync_to_rust` is the only bridge to it.
    The encoder reads the Rust side, so what the GUI and the bot would hand a network
    has to agree with what the fuzz above walks."""
    rng = random.Random(SEED)
    checked = saw_hand = 0

    for _ in range(BOUNDARY_GAMES):
        gs = GameState()
        gs.setup_initial_board()

        for _ply in range(MAX_PLY):
            synced = ai._sync_to_rust(gs)
            planes = rs.encode_position(synced)
            assert len(planes) == rs.ENCODE_INPUT_SIZE

            expected_side = 1.0 if gs.current_turn == 'w' else 0.0
            assert plane(planes, SIDE_TO_MOVE)[0] == expected_side

            moves = gs.get_all_legal_moves()
            if not moves:
                break
            indices = {rs.move_to_action_index(synced, m) for m in moves}
            assert len(indices) == len(moves), "the Python move list collided"
            assert indices == set(rs.legal_action_indices(synced)), (
                "the Rust mask and the Python move list disagree"
            )
            checked += 1
            if any(gs.hands[c][p] for c in gs.hands for p in gs.hands[c]):
                saw_hand += 1

            if gs.checkmate or gs.stalemate or gs.is_draw:
                break
            if not gs.make_move(rng.choice(moves), False):
                break
            if gs.needs_promotion_choice:
                piece = rng.choice(['R', 'N', 'B'])
                gs.complete_promotion(piece if gs.current_turn == 'b' else piece.lower())
            gs.check_game_over()

    assert checked > 100, "the boundary sweep barely ran; %d positions" % checked
    assert saw_hand > 0, "no position in the boundary sweep had a piece in hand"
