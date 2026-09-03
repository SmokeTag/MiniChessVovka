#!/usr/bin/env python3
"""The Rust tree, driven against a deliberately ignorant network.

Every test here uses a **uniform** evaluator: flat priors, zero values. That is the
point. With a trained network it is impossible to tell a working search from a working
policy — the network would find the mate on its own and every result would be evidence
about the weights rather than about the tree. Given a network that knows nothing, any
preference the search shows is the search's own.

These do not need torch. `nn.mcts.search` is pure tree driving and imports torch only
inside `Evaluator`, so the loop can be exercised with any callable.

    ./venv/bin/python -m pytest tests/test_mcts.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minichess_engine as rs

np = pytest.importorskip("numpy")

from nn import mcts

# Rf6 is mate: the rook seals the top rank and the white king covers the flight squares.
MATE_IN_ONE = "k5/6/1K4/6/6/5R w"
ALREADY_MATE = "k4R/6/1K4/6/6/6 b"

class Uniform:
    """Flat policy, zero value, and a call counter."""

    def __init__(self):
        self.calls = 0
        self.positions = 0
        self.batch_sizes = []

    def __call__(self, flat):
        n = len(flat) // rs.ENCODE_INPUT_SIZE
        self.calls += 1
        self.positions += n
        self.batch_sizes.append(n)
        return (np.full(n * rs.ACTION_SPACE, 1.0 / rs.ACTION_SPACE, dtype=np.float32),
                np.zeros(n, dtype=np.float32))

def opening():
    gs = rs.GameState()
    gs.setup_initial_board()
    return gs

def test_every_simulation_is_accounted_for():
    """Root visits must equal simulations minus the root's own first visit. A backup that
    walks the wrong path, or counts a node twice, shows up here and nowhere else."""
    tree = mcts.search(opening(), Uniform(), simulations=400, batch=8)
    assert tree.simulations == 400
    assert sum(v for _, v in tree.root_visits()) == tree.simulations - 1

def test_a_search_that_knows_nothing_still_finds_mate_in_one():
    """The load-bearing test for selection, backup and terminal scoring together.

    The evaluator supplies no signal at all, so the only way Rf6 can come out on top is
    that the tree noticed the position after it is over and propagated that back.
    """
    tree = mcts.search(rs.from_fen(MATE_IN_ONE), Uniform(), simulations=400, batch=8)
    played = rs.from_fen(MATE_IN_ONE)
    played.make_ai_move(tree.best_move())
    assert not played.get_all_legal_moves()
    assert played.is_in_check(played.current_turn), "the chosen move is stalemate, not mate"
    assert tree.value > 0.5, "a forced mate should not read as balanced: %.3f" % tree.value

def test_a_finished_game_costs_the_network_nothing():
    """A terminal root has no work to do. Getting this wrong is not a slow search, it is
    an expanded node with zero edges and then a selection out of an empty range."""
    ev = Uniform()
    tree = mcts.search(rs.from_fen(ALREADY_MATE), ev, simulations=50, batch=8)
    assert ev.calls == 0
    assert tree.simulations == 0
    assert tree.best_move() is None
    assert tree.nodes == 1

def test_terminal_children_never_reach_the_network():
    """Positions that end the game are scored by the rules, not evaluated. With mate
    available, a good part of the tree is terminal and must not be paid for."""
    ev = Uniform()
    tree = mcts.search(rs.from_fen(MATE_IN_ONE), ev, simulations=300, batch=8)
    assert ev.positions < tree.simulations, (
        "every simulation reached the network (%d positions for %d sims); terminals are "
        "not being resolved in place" % (ev.positions, tree.simulations))

def test_virtual_loss_actually_fills_the_batch():
    """Without virtual loss every descent in a batch returns the same leaf and the batch
    is worth one simulation. The evidence is the batch sizes the evaluator sees."""
    ev = Uniform()
    mcts.search(opening(), ev, simulations=400, batch=16)
    # The first call can only ever be the root alone; after that they should be full.
    assert ev.batch_sizes[0] == 1
    later = ev.batch_sizes[1:]
    assert later, "the search never got past the root"
    assert max(later) > 1, "no batch held more than one leaf: virtual loss is not working"
    assert sum(later) / len(later) > 8, (
        "batches averaged %.1f leaves of a requested 16" % (sum(later) / len(later)))

def _top_share(position, sims):
    tree = mcts.search(position, Uniform(), simulations=sims, batch=8)
    counts = sorted((v for _, v in tree.root_visits()), reverse=True)
    return counts[0] / max(1, sum(counts))

def test_search_concentrates_only_where_there_is_something_to_find():
    """Two halves of the same PUCT property, and the first is the easier one to get
    backwards.

    Given a uniform evaluator, an opening position carries *no* information: every leaf
    scores 0 and every prior is equal, so spreading visits evenly across the legal moves
    is correct behaviour and a sharpening distribution would mean something is leaking a
    preference it has not earned. Near a forced mate the terminal scores are real
    information, and there the search must concentrate.
    """
    flat = _top_share(opening(), 1200)
    even = 1.0 / len(opening().get_all_legal_moves())
    assert flat < 3 * even, (
        "with nothing to go on the search still favoured one move (%.3f vs %.3f even)"
        % (flat, even))

    mate = rs.from_fen(MATE_IN_ONE)
    assert _top_share(mate, 1200) > _top_share(mate, 60), (
        "more search did not concentrate on a mate that is there to be found")

def test_batching_does_not_change_what_is_searched():
    """Batch size is a throughput knob. With a deterministic evaluator the simulation
    count must not depend on it, or the batch is losing or double-counting work."""
    for batch in (1, 4, 32):
        tree = mcts.search(opening(), Uniform(), simulations=300, batch=batch)
        assert tree.simulations == 300, (batch, tree.simulations)
        assert sum(v for _, v in tree.root_visits()) == 299

def test_a_time_budget_stops_the_search():
    """The GUI's budget is a clock, not a count (docs/GUI.md).

    A deadline that is not honoured is the whole failure mode here, and it is one an
    interactive front end feels directly: the setting the user picked is the wait they
    were promised. The deadline is only read between batches, so the search may run one
    batch past it -- checked with a generous ceiling rather than a tight one, because a
    loaded machine makes a tight bound flaky without making it a better test.
    """
    import time

    slow = Uniform()
    started = time.monotonic()
    tree = mcts.search(opening(), slow, simulations=None, batch=8, time_limit=0.2)
    elapsed = time.monotonic() - started

    assert 0.2 <= elapsed < 1.0, "a 0.2s budget ran for %.3fs" % elapsed
    assert tree.simulations > 0, "the deadline stopped the search before it started"

def test_a_budget_too_small_to_honour_still_chooses_a_move():
    """A deadline already past must not return the root untouched.

    With only the root expanded every child has zero visits, so `best_move` falls through
    to whichever edge comes first -- a move nothing selected, handed back as though the
    search had picked it. Worse than slow, and invisible: it is a legal move with a
    plausible score. The first two rounds therefore run regardless of the clock.
    """
    tree = mcts.search(opening(), Uniform(), simulations=None, batch=8, time_limit=0.0)
    assert tree.simulations > 1
    assert max(v for _, v in tree.root_visits()) > 0, "no child was ever visited"
    assert tree.best_move() in opening().get_all_legal_moves()

def test_a_simulation_cap_still_wins_when_both_are_given():
    """Whichever budget binds first stops the search. A time limit far in the future
    must not turn a simulation count into an unbounded run, or the arena's numbers stop
    meaning what they say."""
    tree = mcts.search(opening(), Uniform(), simulations=200, batch=8, time_limit=60.0)
    assert tree.simulations == 200

def test_a_search_with_no_budget_is_refused():
    """Silently searching forever is the one outcome worse than an error."""
    with pytest.raises(ValueError, match="budget"):
        mcts.search(opening(), Uniform(), simulations=None, time_limit=None)

def test_expand_rejects_a_mismatched_batch():
    """The Rust side must not be fed rows it did not ask for."""
    tree = rs.Mcts(opening())
    planes = tree.collect(4)
    n = len(planes) // rs.ENCODE_INPUT_SIZE
    assert n == 1, "the first collect can only be the root"

    with pytest.raises(ValueError, match="values"):
        tree.expand([0.0] * rs.ACTION_SPACE, [0.0, 0.0])
    with pytest.raises(ValueError, match="priors"):
        tree.expand([0.0] * (rs.ACTION_SPACE - 1), [0.0])

def test_root_visits_agree_with_the_legal_moves():
    gs = opening()
    tree = mcts.search(gs, Uniform(), simulations=200, batch=8)
    moves = [m for m, _ in tree.root_visits()]
    assert sorted(moves) == sorted(gs.get_all_legal_moves())

def test_visit_distribution_temperatures():
    tree = mcts.search(opening(), Uniform(), simulations=300, batch=8)
    moves, probs = mcts.visit_distribution(tree, temperature=1.0)
    assert len(moves) == len(probs)
    assert probs.sum() == pytest.approx(1.0)

    _, greedy = mcts.visit_distribution(tree, temperature=0.0)
    assert greedy.sum() == pytest.approx(1.0)
    assert (greedy > 0).sum() == 1, "temperature 0 must be argmax"
    counts = np.array([v for _, v in tree.root_visits()], dtype=float)
    assert greedy.argmax() == counts.argmax()
