# -*- coding: utf-8 -*-
"""Piece codes, promotion sets and move directions shared by the Python rules layer.

Pure data: the vector drawing primitives that used to live here (and the pygame and
config imports they needed) are gone -- sprites are drawn in gui.py. The piece *values*
are gone too; the numbers that decide moves are the consts at the top of
engine_rs/src/eval.rs, and a second set here was only ever misleading.
"""

# --- Piece Representation ---
EMPTY_SQUARE = '.'
PAWN = ['P', 'p']
KNIGHT = ['N', 'n']
BISHOP = ['B', 'b']
ROOK = ['R', 'r']
QUEEN = ['Q', 'q']
KING = ['K', 'k']

PROMOTION_PIECES_WHITE_STR = ['R', 'N', 'B']
PROMOTION_PIECES_BLACK_STR = ['r', 'n', 'b']

# Mapping of pieces to Unicode symbols (for text rendering, if needed)
PIECE_TO_SYMBOL = {
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'k': '♚',
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'K': '♔'
}

# --- Move Directions ---
KNIGHT_MOVES = [(1, 2), (1, -2), (-1, 2), (-1, -2),
                (2, 1), (2, -1), (-2, 1), (-2, -1)]
DIAGONAL_MOVES = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
STRAIGHT_MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]
KING_MOVES = DIAGONAL_MOVES + STRAIGHT_MOVES
