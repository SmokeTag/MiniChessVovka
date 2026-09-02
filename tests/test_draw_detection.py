"""The alpha-beta search has to know a draw when it reaches one.

Every position here is the same one: White is a rook up, both kings have shuffled
back and forth twice, and Black -- to move -- can step into a position that has
already occurred twice. Threefold. Black's only good result in the game, and the
score the search has to report is 0 rather than the rook it is down.

`_position_key()` carries exactly what the Zobrist hash reads, so `ai` can turn the
game's history into hashes without replaying anything; these tests pin that
equivalence too, because everything else rests on it.
"""
import minichess_engine as _rs

import ai
from gamestate import GameState

REPEATING_MOVE = ((0, 4), (0, 5), None)

SHUFFLE = [((5, 0), (5, 1), None), ((0, 5), (0, 4), None),
           ((5, 1), (5, 0), None), ((0, 4), (0, 5), None),
           ((5, 0), (5, 1), None), ((0, 5), (0, 4), None),
           ((5, 1), (5, 0), None)]


def _shuffled_position(plies=len(SHUFFLE)):
    """White Ka1 Ra3 Nd2, Black Kf6 Nd5, after `plies` of king shuffling.

    Nothing is attacked and no pawn is on the board, so every shuffle move is
    reversible and the whole line stays inside one repetition window.
    """
    gs = GameState()
    gs.board = [['.' for _ in range(6)] for _ in range(6)]
    gs.board[5][0] = 'K'
    gs.board[3][0] = 'R'
    gs.board[4][3] = 'N'
    gs.board[0][5] = 'k'
    gs.board[1][3] = 'n'
    gs.hands = {'w': {}, 'b': {}}
    gs.current_turn = 'w'
    gs.find_kings()
    gs._reset_position_history()
    for move in SHUFFLE[:plies]:
        assert gs.make_move(move), move
    return gs


def test_the_repeating_move_really_is_a_threefold():
    gs = _shuffled_position()
    assert gs.current_turn == 'b'
    assert gs.make_move(REPEATING_MOVE)
    assert gs.is_draw
    assert gs.game_over_message == "Draw by repetition."


def test_search_takes_the_repetition_rather_than_the_lost_position():
    gs = _shuffled_position()
    assert ai.evaluate_position(gs) > 300, "White should be clearly winning statically"
    for depth in (4, 6):
        move, score = ai.find_best_move_with_score(gs, depth=depth)
        assert move == REPEATING_MOVE, (depth, move)
        assert score == 0, (depth, score)


def test_without_the_game_history_the_search_is_blind_to_it(monkeypatch):
    """What the fix buys: the same search with no history walks into the loss."""
    monkeypatch.setattr(ai, "_history_hashes", lambda gamestate: [])
    gs = _shuffled_position()
    move, score = ai.find_best_move_with_score(gs, depth=4)
    assert move != REPEATING_MOVE
    assert score > 300


def test_a_history_repetition_is_kept_out_of_the_book():
    """A draw that rests on the game's path must not be filed under the position hash.

    `probe_book` is keyed on the hash alone, so a row saying "this position is 0"
    would be served to every future game that reaches it by any other route.
    """
    ai.discard_pending_book_writes()
    ai.find_best_move_with_score(_shuffled_position(), depth=4)
    assert ai.pending_book_writes() == 0

    ordinary = GameState()
    ordinary.setup_initial_board()
    ai.find_best_move_with_score(ordinary, depth=3)
    assert ai.pending_book_writes() > 0, "an ordinary search should still file a row"
    ai.discard_pending_book_writes()


def test_history_hashes_reconstruct_the_live_position():
    gs = _shuffled_position()
    history = ai._history_hashes(gs)
    assert history, "a shuffled game should carry a reversible tail"
    assert history[-1] == int(ai.get_position_hash(gs))
    assert len(set(history)) < len(history), "the tail should contain repeats"


def test_the_reversible_tail_stops_at_the_last_irreversible_move():
    gs = GameState()
    gs.setup_initial_board()
    gs.make_move(((4, 0), (3, 0), None))     # a pawn move: nothing before it can recur
    assert len(ai._history_hashes(gs)) == 1


def test_a_search_leaves_the_history_exactly_as_it_found_it():
    """make_ai_move pushes and undo_ai_move pops; an imbalance would rot the counter."""
    gs = GameState()
    gs.setup_initial_board()
    rs = ai._sync_to_rust(gs)
    before = (list(rs.position_history), rs.ply)
    _rs.find_best_move_with_score(rs, 4, None, False)
    assert (list(rs.position_history), rs.ply) == before


def test_rust_reports_a_terminal_draw_without_python():
    """The verdict an MCTS leaf evaluator asks for, entirely inside Rust."""
    rs = ai._sync_to_rust(_shuffled_position(plies=0))
    for move in SHUFFLE:
        assert rs.make_move(move), move
        assert not rs.is_terminal_draw()
    assert rs.make_move(REPEATING_MOVE)
    assert rs.is_terminal_draw()


def test_the_ply_limit_is_a_draw_too():
    rs = ai._sync_to_rust(_shuffled_position(plies=0))
    assert not rs.is_terminal_draw()
    rs.ply = rs.ply_limit
    assert rs.is_terminal_draw()
