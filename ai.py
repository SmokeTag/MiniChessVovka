"""
AI module - Rust-accelerated wrapper.
Delegates heavy search to the Rust minichess_engine module while maintaining
the same Python API for compatibility with gui.py, play_online.py, etc.
"""
import time
import minichess_engine as _rs

from config import BOARD_SIZE
from utils import get_piece_color, algebraic_to_coords

CHECKMATE_SCORE = 1000000
STALEMATE_SCORE = 0

move_cache = {}
tt = {}
DB_PATH = "book.db"

def setup_db():
    """Create the opening book schema in book.db if it is not already there.

    Raises RuntimeError if book.db was written by a different SCHEMA_VERSION. It never
    drops anything -- see rebuild_book().
    """
    _rs.setup_db()

def rebuild_book():
    """Drop both book tables and recreate them at the current SCHEMA_VERSION.

    Destructive, and nothing calls it for you. `rebuild_book.py` is the front door.
    """
    _rs.rebuild_book()

def load_move_cache_from_db():
    """Load the opening book from SQLite into the Rust engine."""
    global move_cache
    _rs.load_move_cache_from_db()

def save_move_cache_to_db(cache_to_save=None):
    """Flush **every** position searched since the last save back to book.db.

    Bulk and undiscriminating by design — the book builder wants exactly that, because
    everything it searches is a repertoire node it chose to search. Interactive front
    ends want the opposite: a GUI session searches whatever the user happens to look at,
    and this call cannot tell an opening line from idle exploration. `main.py` therefore
    does not call it at all; see `save_position_to_book`.
    """
    _rs.save_move_cache_to_db()

def save_position_to_book(gamestate):
    """File just this position's search result into book.db. Returns True if it wrote.

    The deliberate counterpart to `save_move_cache_to_db`: the caller names the one
    position it wants kept, and everything else the session searched stays in memory.
    False means nothing has searched this position in this process, so there was
    nothing to write — not an error.
    """
    return _rs.save_book_position(_sync_to_rust(gamestate))

def book_has_position(gamestate):
    """Whether this process holds a searched entry for the position — i.e. whether
    `save_position_to_book` would write anything."""
    return _rs.book_has_position(_sync_to_rust(gamestate))

def pending_book_writes():
    """How many searched positions are queued in memory and not on disk."""
    return _rs.pending_book_writes()

def discard_pending_book_writes():
    """Drop the unsaved queue. The entries stay in memory as analysis cache."""
    return _rs.discard_pending_book_writes()

def book_has_row(gamestate):
    """Whether this position already has a row in `book_move` **on disk**.

    Distinct from `book_has_position`, which asks whether anything is in memory for it.
    Once the analysis cache is loaded the two share one map, so only the table a row came
    from says which store it belongs to.
    """
    return _rs.book_has_row(_sync_to_rust(gamestate))

def load_analysis_from_db():
    """Merge the analysis cache into the in-memory book. Returns how many it added.

    The book wins every collision. Loading this means the engine will also *play* from
    cached analysis — the probe reads one map and cannot tell the stores apart — which is
    why only the GUI calls it.
    """
    return _rs.load_analysis_from_db()

def save_analysis_to_db():
    """Flush everything searched since the last flush into the analysis tables.

    The bulk write the book no longer gets. Nothing here can reach `book_move`.
    """
    return _rs.save_analysis_to_db()

def rebuild_analysis():
    """Drop the analysis cache and recreate it empty. The book is untouched."""
    _rs.rebuild_analysis()

def book_size():
    """How many positions the in-memory book holds."""
    return _rs.book_size()

def to_fen(gamestate):
    """Serialize a position to minihouse FEN (see engine_rs/src/fen.rs)."""
    return _rs.to_fen(_sync_to_rust(gamestate))

def from_fen(fen):
    """Parse a minihouse FEN into a Rust GameState. Raises ValueError if malformed.

    Returns the engine's own state object, not a `gamestate.GameState`: the book stores
    FENs so positions can be re-searched, and a search takes the Rust state anyway.
    """
    return _rs.from_fen(fen)

def _sync_to_rust(gamestate):
    """Create a Rust GameState from a Python GameState."""
    rs = _rs.GameState()
    rs.board = gamestate.board
    rs.current_turn = gamestate.current_turn
    rs.hands = gamestate.hands
    rs.king_pos = gamestate.king_pos
    rs.checkmate = gamestate.checkmate
    rs.stalemate = gamestate.stalemate
    rs.promoted_pieces = list(gamestate.promoted_pieces) if hasattr(gamestate, 'promoted_pieces') else []
    rs.ply = getattr(gamestate, 'ply_count', 0)
    return rs

def _normalize_promotion(move, color):
    """Case a promotion piece char to match the side that is moving.

    The Rust generator emits promotion pieces uppercase for both colours, while
    gamestate.py validates against PROMOTION_PIECES_WHITE_STR / _BLACK_STR, which
    are case-sensitive: 'R' for White, 'r' for Black. Feeding Rust's 'R' into a
    Black promotion makes make_move print "Invalid promotion choice" and return
    False, leaving the board untouched -- the GUI then falls back to a random
    move (main.py) rather than the engine's answer.

    Every engine move re-enters Python through this module, so this is the single
    place to reconcile the two conventions. Idempotent, and a no-op for drops.
    """
    if not move or len(move) != 3 or move[0] == 'drop' or not move[2]:
        return move
    start, end, promotion = move
    return (start, end, promotion.upper() if color == 'w' else promotion.lower())

def _normalize_result(result, color):
    """Apply _normalize_promotion across either shape find_best_move returns."""
    if result is None:
        return None
    if isinstance(result, list):
        return [(_normalize_promotion(m, color), score) for m, score in result]
    return _normalize_promotion(result, color)

def get_position_hash(gamestate):
    """Compute Zobrist hash for the position (via Rust)."""
    rs = _sync_to_rust(gamestate)
    return _rs.get_position_hash(rs)

def evaluate_position(gamestate):
    """Static evaluation of the position (via Rust)."""
    rs = _sync_to_rust(gamestate)
    return _rs.evaluate_position(rs)

def set_parallel_search(enabled, min_depth=None):
    """
    Turn the engine's root-level parallel search on or off for this process.

    Off by default, so self-play training stays single-threaded: training runs many
    independent games side by side across cores, which scales better than
    parallelising a single game. Interactive analysis of one position wants it on.

    Args:
        enabled: True to split the root move list across all cores, False for
                 single-threaded search.
        min_depth: First iterative-deepening depth that gets parallelised.
                   None keeps the current value (default 3).
    """
    _rs.set_parallel_search(enabled, min_depth)

def get_parallel_search():
    """Return (enabled, min_depth) for the engine's root-level parallel search."""
    return _rs.get_parallel_search()

def find_best_move(gamestate, depth=6, return_top_n=1, time_limit=None, parallel=None):
    """
    Find the best move using Rust engine's iterative deepening alpha-beta search.

    Args:
        gamestate: Python GameState object
        depth: Maximum search depth
        return_top_n: If > 1, returns a list of (move, score) tuples ranked best-first
                      for the side to move, and runs a MultiPV search so ranks 2+ carry
                      exact scores instead of alpha-beta bounds. Measured at depth 9:
                      about 2.4-3.0x a single-PV search for 2 moves, 3.8-4.4x for 3.
        time_limit: Max seconds for search. None = no limit.
        parallel: True forces the parallel root search for this call, False forces
                  single-threaded. None (default) uses set_parallel_search().

    Returns:
        If return_top_n == 1: best_move tuple or None
        If return_top_n > 1: list of (move, score) tuples, at most return_top_n long.
        Scores are white-relative, so the list runs non-increasing for White and
        non-decreasing for Black; rank 1 is the best move for whoever is to move.
    """
    print(f"AI ({gamestate.current_turn}) thinking with Rust engine, depth {depth}...")
    start_time = time.time()

    if gamestate.needs_promotion_choice:
        print("AI Error: Cannot find move, waiting for promotion choice.")
        return None if return_top_n == 1 else []

    legal_moves = gamestate.get_all_legal_moves()
    if not legal_moves:
        print("AI Error: No legal moves available!")
        return None if return_top_n == 1 else []

    if len(legal_moves) == 1:
        print("Only one legal move available.")
        elapsed = time.time() - start_time
        print(f"AI ({gamestate.current_turn}) finished (single move) in {elapsed:.2f}s.")
        return legal_moves[0] if return_top_n == 1 else [(legal_moves[0], 0)]

    rs = _sync_to_rust(gamestate)

    result = _rs.find_best_move(rs, depth, return_top_n, time_limit, parallel)

    elapsed = time.time() - start_time
    print(f"AI ({gamestate.current_turn}) finished in {elapsed:.2f}s.")

    return _normalize_result(result, gamestate.current_turn)

def find_best_move_with_score(gamestate, depth=6, time_limit=None, parallel=None):
    """Single-PV search returning `(move, white_relative_score)`.

    Same search as `find_best_move(..., return_top_n=1)` — no MultiPV premium — but it
    hands back the score too, which is what the GUI's eval readout shows while the
    engine is playing. The score is white-relative (see `search::find_best_move`), so
    positive always favours White whoever is to move.

    Returns `(None, None)` when there is nothing to search, and `(move, None)` for a
    forced move: one legal reply is answered without a search, so there is no score
    behind it — the same reason `search::find_best_move` refuses to file that row.
    """
    if gamestate.needs_promotion_choice:
        return None, None

    legal_moves = gamestate.get_all_legal_moves()
    if not legal_moves:
        return None, None
    if len(legal_moves) == 1:
        return legal_moves[0], None

    rs = _sync_to_rust(gamestate)
    move, score = _rs.find_best_move_with_score(rs, depth, time_limit, parallel)
    if move is None:
        return None, None
    return _normalize_promotion(move, gamestate.current_turn), int(score)

def minimax_alpha_beta(gamestate, depth, alpha, beta, maximizing_player, allow_null=True):
    """
    Compatibility wrapper. Runs a Rust search at the given depth and returns (score, best_move).
    Note: alpha/beta/allow_null params are handled internally by Rust.
    """
    rs = _sync_to_rust(gamestate)
    result = _rs.find_best_move(rs, depth)
    if result is None:
        score = evaluate_position(gamestate)
        return (score, None)
    return (0, _normalize_promotion(result, gamestate.current_turn))

def parse_move_string(move_str):
    """Convert move string (e.g., 'e2e4', 'N@c3') to internal move format."""
    try:
        if '@' in move_str:
            piece_char, sq_str = move_str.split('@')
            target_sq = algebraic_to_coords(sq_str)
            if target_sq:
                return ('drop', piece_char, target_sq)
        elif len(move_str) >= 4:
            start_sq = algebraic_to_coords(move_str[:2])
            end_sq = algebraic_to_coords(move_str[2:4])
            promotion_char = move_str[4] if len(move_str) == 5 else None
            if start_sq and end_sq:
                return (start_sq, end_sq, promotion_char)
    except Exception as e:
        print(f"[ERROR] Failed to parse move string '{move_str}': {e}")
    return None

def is_move_still_legal(gamestate, move):
    """Check if a move is legal in the current gamestate."""
    try:
        return move in gamestate.get_all_legal_moves()
    except Exception:
        return False
