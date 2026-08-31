"""
Loader for bench/positions.json.

There is no FEN parser in this project, so a benchmark position is stored as a
name plus the exact list of moves to replay from a fresh GameState(). This
module turns that JSON back into a live GameState.

Move encoding in JSON (tuples are not JSON types, so everything is a list):
    normal move : [[r1, f1], [r2, f2], promotion_or_null]
    drop        : ["drop", "wN", [r, f]]

Usage:
    from positions import load_positions, build_gamestate
    positions = load_positions()
    gs = build_gamestate(positions["opening_book_exit"])
"""
import json
import os
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BENCH_DIR)
POSITIONS_JSON = os.path.join(BENCH_DIR, "positions.json")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def decode_move(m):
    """JSON list -> internal move tuple."""
    if isinstance(m, (list, tuple)) and len(m) == 3 and m[0] == "drop":
        return ("drop", m[1], (m[2][0], m[2][1]))
    (r1, f1), (r2, f2), promo = m[0], m[1], m[2]
    return ((r1, f1), (r2, f2), promo)

def encode_move(m):
    """Internal move tuple -> JSON-safe list."""
    if m[0] == "drop":
        return ["drop", m[1], [m[2][0], m[2][1]]]
    return [[m[0][0], m[0][1]], [m[1][0], m[1][1]], m[2]]

def load_positions(path=POSITIONS_JSON):
    """Return {name: position_dict} from the positions file."""
    with open(path, "r") as fh:
        data = json.load(fh)
    return {p["name"]: p for p in data["positions"]}

def load_positions_list(path=POSITIONS_JSON):
    """Return the positions in file order (opening -> endgame)."""
    with open(path, "r") as fh:
        data = json.load(fh)
    return data["positions"]

def build_gamestate(position):
    """
    Replay a stored position and return the resulting GameState.

    Raises RuntimeError if any move in the list is rejected, which is the
    signal that the stored line no longer matches the game rules.
    """
    from gamestate import GameState

    gs = GameState()
    gs.setup_initial_board()
    for idx, raw in enumerate(position["moves"]):
        move = decode_move(raw)
        if not gs.make_move(move):
            raise RuntimeError(
                "position %r: move %d (%r) was rejected during replay"
                % (position["name"], idx, move)
            )
        if gs.needs_promotion_choice:
            promo = position.get("promotions", {}).get(str(idx))
            if promo is None:
                promo = "R" if gs.current_turn == "w" else "r"
            gs.complete_promotion(promo)
    return gs
