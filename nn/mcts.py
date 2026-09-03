"""Driving the Rust tree from Python: batch the leaves, run the network, hand them back.

The loop is deliberately thin. `engine_rs/src/mcts.rs` owns the tree, the PUCT selection,
virtual loss, terminal resolution and the backup; this module owns the GPU and nothing
else. That split is the phase-3 design decision from docs/ZERO.md -- Rust threads calling
into torch would serialise on the GIL, so instead one Python loop holds the GIL and does
the only work that needs it.

**The mask is implicit and cannot be forgotten.** Rust reads priors only at the legal
actions of the leaf it is expanding and renormalises them, so a plain softmax over all
2,196 logits here is *exactly* a masked softmax: `p_i / sum_{j in legal} p_j` cancels the
partition function and leaves `exp(z_i) / sum_{j in legal} exp(z_j)`. There is no masking
step to get wrong.

**Planes cross the boundary as bytes, not as a list.** `collect` hands back raw f32 that
`np.frombuffer` wraps without copying, and priors go back as a contiguous float32 array
that Rust takes through the buffer protocol in one memcpy. The list form was measured at
most of a self-play ply against ~3ms of GPU: 110k floats each becoming a boxed,
refcounted `PyObject`. Every evaluator therefore receives a numpy array and must return
numpy arrays -- `np.ndarray` in, `(priors, values)` as float32 arrays out.

**A tree can outlive the move it chose.** `Searcher` keeps one across the moves of a game
and re-roots it onto whatever position comes back, so the next search starts with the
subtree under the move played already visited instead of from an empty root. The budget
still counts new simulations only; `tree.root_total_visits` is what the root actually
stands at.
"""
import time

import numpy as np

import minichess_engine as rs

DEFAULT_SIMULATIONS = 800
DEFAULT_BATCH = 16
DEFAULT_C_PUCT = 1.5

# How far below the old root `Searcher` will look for the new position. Two plies is our
# move and the opponent's reply, which is every case a game actually produces; deeper
# would only ever match after a position the searcher was never asked about.
DEFAULT_ADVANCE_DEPTH = 2

class Evaluator:
    """A network wrapped so the search only ever hands it planes and takes back rows.

    torch is imported here rather than at module scope so that `search` -- which is pure
    tree driving -- can be exercised against any callable, including a deliberately
    ignorant one, without a GPU stack present.

    In and out are numpy arrays. `torch.from_numpy` wraps the buffer `collect` returned
    without copying it, and the priors going back are handed to Rust as a contiguous
    float32 block rather than a list; the round trip used to be the largest single cost
    in self-play.
    """

    def __init__(self, net, device, batch_limit=None):
        import torch
        self.torch = torch
        self.net = net.eval()
        self.device = device
        self.batch_limit = batch_limit
        self.calls = 0
        self.positions = 0

    def __call__(self, planes):
        torch = self.torch
        n = planes.size // rs.ENCODE_INPUT_SIZE
        with torch.inference_mode():
            x = torch.from_numpy(planes).view(
                n, rs.ENCODE_PLANES, rs.BOARD_SIZE, rs.BOARD_SIZE)
            x = x.to(self.device, non_blocking=True)
            logits, values = self.net(x)
            priors = logits.float().softmax(-1).reshape(-1).cpu().numpy()
            vals = values.float().reshape(-1).cpu().numpy()
        self.calls += 1
        self.positions += n
        return priors, vals

def search(position, evaluator, simulations=DEFAULT_SIMULATIONS,
           batch=DEFAULT_BATCH, c_puct=DEFAULT_C_PUCT, time_limit=None,
           tree=None, root_hook=None):
    """Search a Rust GameState until the budget runs out and return the finished tree.

    The budget is whichever of `simulations` and `time_limit` is given; either may be
    None, and with both the search stops at the first of the two. A fixed simulation
    count is what a reproducible measurement wants (the arena, the self-play loop); a
    fixed *time* is what an interactive caller wants, because the cost of a simulation
    is a property of the position -- a full-hand middlegame descends further and branches
    wider than an opening -- so the same count is not the same wait twice running.

    The deadline is only checked between batches, and never before the tree has visited
    a child. That is a real granularity, not an oversight: a batch of 16 is a few
    milliseconds, so the overshoot is bounded by one network call, and a search that
    stopped at the bare root would return an unchosen move rather than a fast one.

    `collect` returning nothing means the tree has no new work -- every leaf it can reach
    is terminal, or the batch stalled on a node already awaiting evaluation. Either way
    the loop must stop, or it spins.

    `tree` continues a tree the caller has already re-rooted (see `Searcher`), in which
    case `position` is ignored. `root_hook` is called once, as soon as the root is
    expanded and before any deeper descent -- the moment self-play's Dirichlet noise has
    to land, and the only moment at which a fresh tree and a reused one look the same.
    """
    if simulations is None and time_limit is None:
        raise ValueError("a search needs a budget: simulations, time_limit or both")
    deadline = None if time_limit is None else time.monotonic() + time_limit

    if tree is None:
        tree = rs.Mcts(position, c_puct)
    if root_hook is not None and tree.root_expanded:
        root_hook(tree)
        root_hook = None

    while simulations is None or tree.simulations < simulations:
        # The deadline may not cut the search before it has chosen anything. A tree with
        # only its root expanded has no visited children, so `best_move` falls through to
        # whichever edge comes first -- a worse move than the raw policy's, returned with
        # no sign that the budget was the problem. So the first two rounds always run: a
        # batch is a few milliseconds, and a budget that cannot afford one could not have
        # been honoured anyway.
        if deadline is not None and tree.simulations > 1 and time.monotonic() >= deadline:
            break
        want = batch if simulations is None else min(
            batch, max(1, simulations - tree.simulations))
        before = tree.simulations
        raw = tree.collect(want)
        if raw:
            # Zero-copy view of the bytes Rust just handed over; the array keeps them
            # alive for as long as the evaluator holds it.
            priors, values = evaluator(np.frombuffer(raw, dtype=np.float32))
            tree.expand(priors, values)
            if root_hook is not None and tree.root_expanded:
                root_hook(tree)
                root_hook = None
        elif tree.simulations == before:
            # No leaves to evaluate *and* nothing was resolved: the tree has nothing left
            # to do, so looping again would spin. An empty batch alone is not that signal
            # -- a descent ending on a terminal position backs up without producing work.
            break
    return tree

class Searcher:
    """One tree carried across the moves of a game, re-rooted rather than rebuilt.

    The subtree under the move that was played is already the correct tree for the
    position that follows, and throwing it away costs roughly half the search: the same
    budget then buys a root that stands at `simulations + whatever was banked` visits.

    Reuse is keyed on the **position**, not on a move the caller reports: `search` asks
    the tree whether the position it is handed lies within `advance_depth` plies of the
    current root and starts a new tree when it does not. That makes every misuse a
    rebuild rather than a wrong answer -- a new game, a takeback, a jump to an analysis
    position -- and means callers with no notion of "a game" (the GUI's per-move thread)
    need no extra bookkeeping.

    Not thread-safe, and deliberately not shared between concurrent games: one tree
    belongs to one line of play.
    """

    def __init__(self, evaluator, simulations=DEFAULT_SIMULATIONS, batch=DEFAULT_BATCH,
                 c_puct=DEFAULT_C_PUCT, time_limit=None,
                 advance_depth=DEFAULT_ADVANCE_DEPTH, reuse=True):
        self.evaluator = evaluator
        self.simulations = simulations
        self.batch = batch
        self.c_puct = c_puct
        self.time_limit = time_limit
        self.advance_depth = advance_depth
        self.reuse = reuse
        self.tree = None
        # Diagnostics: how often reuse actually happened and how many visits it saved.
        self.reused = 0
        self.rebuilt = 0
        self.banked = 0

    def reset(self):
        """Forget the tree. Not required for correctness -- a position the tree cannot
        reach rebuilds on its own -- but it releases the nodes at the end of a game."""
        self.tree = None

    def search(self, position, simulations=None, time_limit=None, root_hook=None):
        tree = None
        if self.reuse and self.tree is not None:
            if self.tree.advance_to(position, self.advance_depth) is not None:
                tree = self.tree
                self.reused += 1
                self.banked += tree.root_total_visits
        if tree is None:
            self.rebuilt += 1
        self.tree = search(
            position, self.evaluator,
            simulations=self.simulations if simulations is None else simulations,
            batch=self.batch, c_puct=self.c_puct,
            time_limit=self.time_limit if time_limit is None else time_limit,
            tree=tree, root_hook=root_hook)
        return self.tree

def best_move(position, evaluator, **kwargs):
    tree = search(position, evaluator, **kwargs)
    return tree.best_move(), tree

def visit_distribution(tree, temperature=1.0):
    """Root visit counts as a probability vector over `tree.root_visits()` order.

    Temperature 0 is argmax. Above 0 the counts are raised to `1/temperature` and
    normalised -- AlphaZero's rule, used for the opening plies of self-play so that games
    diverge, and for the policy target itself at temperature 1.
    """
    entries = tree.root_visits()
    counts = np.array([v for _, v in entries], dtype=np.float64)
    if counts.sum() == 0:
        counts = np.ones_like(counts)
    if temperature <= 0:
        out = np.zeros_like(counts)
        out[counts.argmax()] = 1.0
        return [m for m, _ in entries], out
    scaled = counts ** (1.0 / temperature)
    return [m for m, _ in entries], scaled / scaled.sum()
