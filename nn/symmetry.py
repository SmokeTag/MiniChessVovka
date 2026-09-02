"""File mirroring: an exact symmetry of this variant, and a free doubling of the data.

Reflecting the board across the file axis, `(r, f) -> (r, 5 - f)`, maps every legal
position to a legal position and every legal move to the corresponding legal move.
That holds here because the variant has **no castling and no en passant** -- the two
things that break left-right symmetry in ordinary chess -- and because every move offset
set in types.rs is closed under `df -> -df`: the knight jumps, the diagonals, the
straights, the king steps. Pawns move along ranks, so their pushes are untouched and
their two captures swap. Drops mirror trivially, and the one drop restriction is on the
rank. Promotions keep their rank and swap capture-left with capture-right.

So a labelled position and its mirror are two genuine training examples with the same
value and the corresponding best move.

**One permutation serves both colours.** The action index lives in the canonical frame,
which for Black is the board rotated 180 degrees: `(r, f) -> (5 - r, 5 - f)`. Mirroring
the board and then canonicalising gives `(5 - r, f)`, which is exactly the file flip of
the canonical `(5 - r, 5 - f)`. Rotation and reflection commute here, so the same index
permutation applies whoever is to move.

`tests/test_nn.py` fuzzes the claim rather than trusting the argument.
"""
import numpy as np

import minichess_engine as rs

BOARD = rs.BOARD_SIZE

def mirror_move(move):
    if move[0] == "drop":
        return ("drop", move[1], (move[2][0], BOARD - 1 - move[2][1]))
    (r1, f1), (r2, f2), promo = move
    return ((r1, BOARD - 1 - f1), (r2, BOARD - 1 - f2), promo)

def mirror_planes(x):
    """(..., 24, 6, 6) -> the same with files reversed.

    Correct for every plane, including the broadcast ones (hands, side to move,
    repetition, progress): those are constant across the board, so reversing them is a
    no-op, which is the right answer.
    """
    return np.ascontiguousarray(x[..., ::-1])

def _build_action_mirror():
    gs = rs.GameState()
    gs.setup_initial_board()  # White to move, so the canonical frame is the board frame
    perm = np.arange(rs.ACTION_SPACE, dtype=np.int64)
    for idx in range(rs.ACTION_SPACE):
        move = rs.action_index_to_move(gs, idx)
        if move is None:
            continue  # decodes off the board; no legal move ever maps here
        perm[idx] = rs.move_to_action_index(gs, mirror_move(move))
    return perm

ACTION_MIRROR = _build_action_mirror()

def mirror_actions(indices):
    return ACTION_MIRROR[indices]

def mirror_gamestate(gs):
    """The mirrored position, for testing the permutation against the engine itself."""
    board = gs.board
    out = rs.GameState()
    out.setup_initial_board()
    out.board = [[board[r][BOARD - 1 - f] for f in range(BOARD)] for r in range(BOARD)]
    out.current_turn = gs.current_turn
    out.hands = gs.hands
    out.promoted_pieces = [(r, BOARD - 1 - f) for (r, f) in gs.promoted_pieces]
    out.ply = gs.ply
    out.ply_limit = gs.ply_limit
    out.find_kings()
    return out
