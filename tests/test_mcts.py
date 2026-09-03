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

    def __call__(self, planes):
        n = planes.size // rs.ENCODE_INPUT_SIZE
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
    planes = np.frombuffer(tree.collect(4), dtype=np.float32)
    n = planes.size // rs.ENCODE_INPUT_SIZE
    assert n == 1, "the first collect can only be the root"

    with pytest.raises(ValueError, match="values"):
        tree.expand([0.0] * rs.ACTION_SPACE, [0.0, 0.0])
    with pytest.raises(ValueError, match="priors"):
        tree.expand([0.0] * (rs.ACTION_SPACE - 1), [0.0])

def test_collect_hands_back_raw_float32_planes():
    """The boundary is the buffer protocol, not a list of boxed floats (docs/ZERO.md).

    A list would still work and would still be correct -- it would just cost most of a
    self-play ply. So the shape of what comes back is the thing to pin: 4 bytes a float,
    ENCODE_INPUT_SIZE floats a leaf, and the same numbers a `float32` view reads back.
    """
    tree = rs.Mcts(opening())
    raw = tree.collect(4)
    assert len(raw) == rs.ENCODE_INPUT_SIZE * 4, "not one f32 block per leaf"
    planes = np.frombuffer(raw, dtype=np.float32)
    assert planes.dtype == np.float32 and planes.size == rs.ENCODE_INPUT_SIZE
    assert np.array_equal(planes, np.asarray(rs.encode_position(opening()),
                                             dtype=np.float32))

def test_expand_takes_a_numpy_batch_and_a_list_alike():
    """Both paths into `expand` must agree, because only one of them is measured.

    The fast path extracts a contiguous float32 buffer in one copy; the list path is the
    fallback the tests and any ad-hoc caller use. A search driven either way has to build
    the same tree.
    """
    def run(as_array):
        tree = rs.Mcts(opening())
        for _ in range(20):
            raw = tree.collect(8)
            if not raw:
                break
            n = len(raw) // (rs.ENCODE_INPUT_SIZE * 4)
            priors = np.full(n * rs.ACTION_SPACE, 1.0 / rs.ACTION_SPACE, dtype=np.float32)
            values = np.full(n, 0.25, dtype=np.float32)
            if as_array:
                tree.expand(priors, values)
            else:
                tree.expand(priors.tolist(), values.tolist())
        return tree.simulations, sorted(tree.root_visits())

    assert run(True) == run(False)

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

# --- tree reuse ---------------------------------------------------------------
#
# Re-rooting is worth roughly a doubling of the search, and it is exactly the kind of
# optimisation that can be wrong without looking wrong: a tree with mismapped indices
# still returns a legal move with a plausible visit count. So these check the arithmetic
# (every retained visit still accounted for, siblings actually dropped) rather than that
# a move came back.

def _played(position, move):
    """`position` after `move`, as the caller's own state -- what a searcher is handed."""
    nxt = rs.from_fen(rs.to_fen(position))
    nxt.ply = position.ply
    nxt.ply_limit = position.ply_limit
    nxt.position_history = position.position_history
    nxt.make_move(move)
    return nxt

def test_advancing_keeps_the_subtree_under_the_move_played():
    """The whole claim of reuse, in one assertion: the visits survive the re-rooting.

    They are the same statistics -- the subtree under the move played *is* the tree for
    the position that follows -- so the count carried over must be exactly the count that
    child had, and the root's own accounting (one visit of its own, the rest through a
    child) must still balance afterwards.
    """
    gs = opening()
    tree = mcts.search(gs, Uniform(), simulations=600, batch=8)
    move, visits = max(tree.root_visits(), key=lambda mv: mv[1])
    assert visits > 1

    assert tree.advance_to(_played(gs, move), 2) == 1
    assert tree.root_total_visits == visits
    assert tree.simulations == 0, "a re-rooted tree must not claim work it did not do"
    assert sum(v for _, v in tree.root_visits()) == tree.root_total_visits - 1

def test_advancing_drops_everything_it_cannot_reach():
    """Pruning is not tidiness: without it the flat node/edge Vecs grow for the whole
    game, and a re-rooted tree would carry every line the game did not take."""
    gs = opening()
    tree = mcts.search(gs, Uniform(), simulations=600, batch=8)
    before = tree.nodes
    move, _ = max(tree.root_visits(), key=lambda mv: mv[1])

    tree.advance_to(_played(gs, move), 2)
    assert tree.nodes < before, "siblings of the played move were kept"
    # Every node kept is one the retained root reached, so there cannot be more of them
    # than that root has visits.
    assert tree.nodes <= tree.root_total_visits + 1

def test_advancing_finds_the_position_two_plies_down():
    """The opponent replies too. A searcher that only looked one ply would rebuild on
    every move of a game it is not playing both sides of."""
    gs = opening()
    tree = mcts.search(gs, Uniform(), simulations=800, batch=8)
    mine, _ = max(tree.root_visits(), key=lambda mv: mv[1])
    after_mine = _played(gs, mine)
    theirs = after_mine.get_all_legal_moves()[0]
    reply = _played(after_mine, theirs)

    assert tree.advance_to(reply, 2) == 2
    assert tree.simulations == 0
    assert sum(v for _, v in tree.root_visits()) == tree.root_total_visits - 1
    assert tree.advance_to(reply, 1) == 0, "already there: depth is measured from now"

def test_a_re_rooted_tree_is_rooted_where_the_caller_thinks():
    """`advance_to` matches on a Zobrist hash, and a tree rooted one move away from the
    real position still answers with a legal move and a plausible score. The only way to
    see that is to ask the tree where it thinks it is."""
    gs = opening()
    tree = mcts.search(gs, Uniform(), simulations=400, batch=8)
    assert tree.root_fen == rs.to_fen(gs)

    move, _ = max(tree.root_visits(), key=lambda mv: mv[1])
    nxt = _played(gs, move)
    tree.advance_to(nxt, 2)
    assert tree.root_fen == rs.to_fen(nxt)

def test_a_position_the_tree_never_saw_is_refused():
    """Reuse has to fail loudly-by-returning-None rather than re-rooting onto something
    plausible: a new game, a takeback, a position pasted into the analysis board."""
    tree = mcts.search(opening(), Uniform(), simulations=200, batch=8)
    assert tree.advance_to(rs.from_fen(MATE_IN_ONE), 2) is None
    assert tree.advance_to(opening(), 2) == 0, "the root itself is always reachable"

def test_a_reused_tree_still_finds_mate_in_one():
    """End to end through the re-rooting, on a position where the answer is known.

    The tree is built one ply *above* MATE_IN_ONE -- Black to move, walking into it --
    and then re-rooted onto the position Black's move produced. If `rebase` mismapped a
    single child index the continued search is reading another position's statistics,
    and a search that could find Rf6 from a standing start no longer does.
    """
    before_mate = rs.from_fen("1k4/6/1K4/6/6/5R b")
    tree = mcts.search(before_mate, Uniform(), simulations=600, batch=8)

    walk_in = ((0, 1), (0, 0), None)             # Kb6-a6, into the mating net
    assert walk_in in before_mate.get_all_legal_moves()
    mated = _played(before_mate, walk_in)
    assert rs.to_fen(mated).split("[")[0] == MATE_IN_ONE.split(" ")[0]

    assert tree.advance_to(mated, 1) == 1
    mcts.search(mated, Uniform(), simulations=400, batch=8, tree=tree)

    played = _played(mated, tree.best_move())
    assert not played.get_all_legal_moves()
    assert played.is_in_check(played.current_turn), "the chosen move is not mate"

def test_pending_leaves_do_not_survive_a_re_rooting():
    """A leaf left waiting on the network keeps a `pending` flag and a virtual loss on
    its whole path. Carried into the tree that is kept, the flag is never cleared and no
    descent can pass through that node again -- a search that quietly stops growing."""
    gs = opening()
    tree = mcts.search(gs, Uniform(), simulations=300, batch=8)
    tree.collect(8)          # deliberately not answered
    move, _ = max(tree.root_visits(), key=lambda mv: mv[1])
    tree.advance_to(_played(gs, move), 2)

    grown = mcts.search(gs, Uniform(), simulations=200, batch=8, tree=tree)
    assert grown.simulations == 200
    assert sum(v for _, v in grown.root_visits()) == grown.root_total_visits - 1

def test_root_priors_are_a_distribution_over_the_root_moves():
    """Self-play's Dirichlet noise goes in through here, so the contract is one entry
    per legal move, in `root_visits()` order, summing to one."""
    gs = opening()
    tree = mcts.search(gs, Uniform(), simulations=100, batch=8)
    priors = tree.root_priors()
    assert len(priors) == len(tree.root_visits())
    assert sum(priors) == pytest.approx(1.0)

    tree.set_root_priors([1.0] + [0.0] * (len(priors) - 1))
    assert tree.root_priors()[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="root moves"):
        tree.set_root_priors([1.0, 0.0])

def test_the_root_hook_fires_once_whether_the_root_is_fresh_or_reused():
    """The hook is the one moment a fresh tree and a re-rooted one look the same, and
    self-play's noise depends on it firing in both. A fresh tree's root is expanded by
    the first batch; a reused one arrives already expanded and must not be skipped."""
    gs = opening()
    seen = []
    tree = mcts.search(gs, Uniform(), simulations=200, batch=8,
                       root_hook=lambda t: seen.append(t.root_total_visits))
    assert len(seen) == 1 and seen[0] <= 1, "the hook fired late, or more than once"

    move, _ = max(tree.root_visits(), key=lambda mv: mv[1])
    tree.advance_to(_played(gs, move), 2)
    mcts.search(gs, Uniform(), simulations=200, batch=8, tree=tree,
                root_hook=lambda t: seen.append(t.root_total_visits))
    assert len(seen) == 2 and seen[1] > 1, "the hook was skipped on an expanded root"

def test_searcher_reuses_down_a_game_and_rebuilds_when_it_cannot():
    """The payoff, measured the way it is claimed: after a few moves the root stands at
    more visits than the budget bought, because the rest was banked."""
    searcher = mcts.Searcher(Uniform(), simulations=200, batch=8)
    gs = opening()
    for _ in range(4):
        tree = searcher.search(gs)
        assert tree.simulations == 200
        move, _ = max(tree.root_visits(), key=lambda mv: mv[1])
        gs = _played(gs, move)

    assert searcher.reused == 3 and searcher.rebuilt == 1
    assert tree.root_total_visits > 200, "nothing was carried over"

    searcher.search(rs.from_fen(MATE_IN_ONE))
    assert searcher.rebuilt == 2, "an unreachable position must start a new tree"

def test_searcher_budgets_can_be_removed_for_one_move():
    """`None` means "no budget of this kind", which is not the same as "not given" --
    the GUI hands over a time limit and no simulation cap."""
    searcher = mcts.Searcher(Uniform(), simulations=200, batch=8)
    tree = searcher.search(opening(), simulations=None, time_limit=0.05)
    assert tree.simulations > 1

def test_a_saturated_tree_stops_instead_of_counting_forever():
    """Every reachable leaf already resolved is the end of the search, not the start of
    a spin.

    A descent onto a terminal position backs up without producing a leaf to evaluate, so
    "no batch" and "no simulation" are different conditions, and a tree that can reach
    nothing new still increments the count on every pass. Under a time budget that is a
    won position spending its entire clock learning nothing and reporting the count as
    though it had: half a million simulations on a 254-node tree, in the GUI's readout.
    """
    class Saturated:
        """A tree that resolves terminals forever and never grows."""

        def __init__(self):
            self.simulations = 0
            self.nodes = 12
            self.root_expanded = True
            self.root_total_visits = 0

        def collect(self, _want):
            self.simulations += 1     # a terminal backed up, no leaf to evaluate
            return bytearray()

    tree = Saturated()
    out = mcts.search(None, Uniform(), simulations=None, batch=8, time_limit=30.0,
                      tree=tree)
    assert out.simulations < 10, (
        "the driver kept asking a tree that had nothing left: %d passes" % out.simulations)
