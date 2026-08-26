# -*- coding: utf-8 -*-
"""
Regenerate bench/positions.json.

Determinism: every move is either
  (a) an index into gs.get_all_legal_moves(), whose order is fixed by the
      Python move generator, or
  (b) the engine's own choice at a fixed depth with parallel=False.
Both are reproducible, and the committed file is a literal move list anyway, so
the stored positions replay identically even if the engine later changes.

Run:  ./venv/bin/python bench/gen_positions.py
Never touches the SQLite move cache (no load_move_cache_from_db call).

Phase note: this is crazyhouse. Captured material comes back as drops, so an
"endgame" here is not "few pieces left in the game" but "few pieces left ON THE
BOARD", with the rest sitting in the hands. Positions are therefore selected by
measured properties (pieces on board, pieces in hand, side to move in check),
not by ply number alone.
"""
import json
import os
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, BENCH_DIR)

from positions import encode_move, POSITIONS_JSON  # noqa: E402

GEN_DEPTH = 5          # depth used to walk a game forward; low and fast
MIN_LEGAL_MOVES = 8    # below this, find_best_move short-circuits and measures nothing

PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}


def fresh_state():
    from gamestate import GameState
    gs = GameState()
    gs.setup_initial_board()
    return gs


def engine_move(gs, depth=GEN_DEPTH):
    import ai
    import minichess_engine as rs
    rust_gs = ai._sync_to_rust(gs)
    out = rs.find_best_move(rust_gs, depth, 2, None, False)
    if not out:
        return None
    return out[0][0]


def play(gs, move):
    if not gs.make_move(move):
        raise RuntimeError("generator produced an illegal move: %r" % (move,))
    if gs.needs_promotion_choice:
        # Should not happen: generated moves carry their promotion piece.
        raise RuntimeError("unexpected pending promotion after %r" % (move,))


def describe(gs):
    """Material / phase summary, also stored as position metadata."""
    on_board = 0
    material = 0
    for row in gs.board:
        for piece in row:
            if piece != "." and piece.upper() != "K":
                on_board += 1
                material += PIECE_VALUES.get(piece.upper(), 0)
    return {
        "side_to_move": gs.current_turn,
        "pieces_on_board": on_board,
        "material_on_board": material,
        "pieces_in_hand": sum(sum(h.values()) for h in gs.hands.values()),
        "hands": {c: {k: v for k, v in h.items() if v} for c, h in gs.hands.items()},
        "in_check": gs.is_in_check(gs.current_turn),
        "legal_moves": len(gs.get_all_legal_moves()),
        "game_over": bool(gs.checkmate or gs.stalemate),
    }


# --- selection criteria -----------------------------------------------------

def usable(info):
    return info["legal_moves"] >= MIN_LEGAL_MOVES and not info["game_over"]


CRITERIA = {
    # deepest usable ply, no further constraint
    "any": lambda info: True,
    # side to move is in check: forced, sharp, tactical
    "check": lambda info: info["in_check"],
    # thin board, material parked in the hands -> crazyhouse endgame
    "endgame": lambda info: info["pieces_on_board"] <= 4 and info["pieces_in_hand"] >= 3,
    # both sides still developed, some drop material available
    "hand": lambda info: info["pieces_in_hand"] >= 2 and info["pieces_on_board"] >= 5,
}


def walk(deviations, max_ply):
    """
    Play one line up to max_ply, returning [(moves_prefix, info), ...] for every
    ply reached, so a caller can pick the prefix matching its criterion.
    """
    gs = fresh_state()
    moves = []
    snapshots = [([], describe(gs))]
    for ply in range(max_ply):
        legal = gs.get_all_legal_moves()
        if not legal or gs.checkmate or gs.stalemate:
            break
        if ply in deviations:
            move = legal[deviations[ply] % len(legal)]
        else:
            move = engine_move(gs)
            if move is None:
                break
        play(gs, move)
        moves.append(move)
        snapshots.append((list(moves), describe(gs)))
    return snapshots


def pick(snapshots, criterion, min_ply=0):
    """Deepest snapshot at or past min_ply that is usable and matches criterion."""
    test = CRITERIA[criterion]
    candidates = [
        (mv, info) for mv, info in snapshots
        if len(mv) >= min_ply and usable(info) and test(info)
    ]
    if not candidates:
        return None, None
    return max(candidates, key=lambda pair: len(pair[0]))


# name, deviations, max_ply, min_ply, criterion, phase label
SPECS = [
    ("opening_start",       {},                  0,  0, "any",     "opening"),
    ("opening_early",       {0: 3, 1: 2},        6,  6, "any",     "opening"),
    ("middlegame_hand_a",   {0: 1, 1: 4},       16, 10, "hand",    "middlegame, pieces in hand"),
    ("middlegame_hand_b",   {0: 6, 1: 0, 2: 2}, 30, 12, "hand",    "middlegame, pieces in hand"),
    ("tactical_check_a",    {0: 2, 1: 6, 4: 1}, 24,  6, "check",   "sharp tactical, side to move in check"),
    ("tactical_check_b",    {0: 4, 1: 5},       34,  6, "check",   "sharp tactical, side to move in check"),
    ("endgame_thin_a",      {0: 0, 1: 1},       30, 12, "endgame", "endgame, thin board with drop material"),
    ("endgame_thin_b",      {0: 5, 1: 1, 2: 3}, 30, 12, "endgame", "endgame, thin board with drop material"),
]


def main():
    out = []
    failures = []
    for name, deviations, max_ply, min_ply, criterion, phase in SPECS:
        snapshots = walk(deviations, max_ply)
        moves, info = pick(snapshots, criterion, min_ply)
        if moves is None:
            failures.append((name, criterion, len(snapshots) - 1))
            sys.stderr.write(
                "FAIL %-20s no usable ply matching %r (line reached ply %d)\n"
                % (name, criterion, len(snapshots) - 1)
            )
            continue
        entry = {"name": name, "phase": phase, "plies": len(moves)}
        entry.update(info)
        entry["moves"] = [encode_move(m) for m in moves]
        out.append(entry)
        sys.stderr.write(
            "OK   %-20s ply=%-3d legal=%-3d board=%-2d hand=%-2d check=%-5s %s\n"
            % (name, len(moves), info["legal_moves"], info["pieces_on_board"],
               info["pieces_in_hand"], info["in_check"], phase)
        )

    if failures:
        raise SystemExit("refusing to write: %d spec(s) found no usable position" % len(failures))

    with open(POSITIONS_JSON, "w") as fh:
        json.dump(
            {
                "note": "Positions for the root-parallel search benchmark. Replay `moves` "
                        "from GameState() + setup_initial_board(); see bench/positions.py. "
                        "Crazyhouse: captured pieces return as drops, so 'endgame' means a "
                        "thin board with material in hand, not few pieces in the game.",
                "generated_with_depth": GEN_DEPTH,
                "min_legal_moves": MIN_LEGAL_MOVES,
                "positions": out,
            },
            fh,
            indent=2,
        )
    sys.stderr.write("wrote %s (%d positions)\n" % (POSITIONS_JSON, len(out)))


if __name__ == "__main__":
    main()
