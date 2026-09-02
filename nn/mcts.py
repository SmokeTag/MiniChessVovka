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
"""
import numpy as np

import minichess_engine as rs

DEFAULT_SIMULATIONS = 800
DEFAULT_BATCH = 16
DEFAULT_C_PUCT = 1.5

class Evaluator:
    """A network wrapped so the search only ever hands it planes and takes back rows.

    torch is imported here rather than at module scope so that `search` -- which is pure
    tree driving -- can be exercised against any callable, including a deliberately
    ignorant one, without a GPU stack present.
    """

    def __init__(self, net, device, batch_limit=None):
        import torch
        self.torch = torch
        self.net = net.eval()
        self.device = device
        self.batch_limit = batch_limit
        self.calls = 0
        self.positions = 0

    def __call__(self, flat_planes):
        torch = self.torch
        n = len(flat_planes) // rs.ENCODE_INPUT_SIZE
        with torch.inference_mode():
            x = torch.tensor(flat_planes, dtype=torch.float32)
            x = x.view(n, rs.ENCODE_PLANES, rs.BOARD_SIZE, rs.BOARD_SIZE).to(self.device)
            logits, values = self.net(x)
            priors = logits.float().softmax(-1)
            out = priors.reshape(-1).cpu().numpy(), values.float().cpu().numpy()
        self.calls += 1
        self.positions += n
        return out

def search(position, evaluator, simulations=DEFAULT_SIMULATIONS,
           batch=DEFAULT_BATCH, c_puct=DEFAULT_C_PUCT):
    """Run `simulations` simulations from a Rust GameState and return the finished tree.

    `collect` returning nothing means the tree has no new work -- every leaf it can reach
    is terminal, or the batch stalled on a node already awaiting evaluation. Either way
    the loop must stop, or it spins.
    """
    tree = rs.Mcts(position, c_puct)
    while tree.simulations < simulations:
        want = min(batch, max(1, simulations - tree.simulations))
        before = tree.simulations
        planes = tree.collect(want)
        if planes:
            priors, values = evaluator(planes)
            tree.expand(priors.tolist(), values.tolist())
        elif tree.simulations == before:
            # No leaves to evaluate *and* nothing was resolved: the tree has nothing left
            # to do, so looping again would spin. An empty batch alone is not that signal
            # -- a descent ending on a terminal position backs up without producing work.
            break
    return tree

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
