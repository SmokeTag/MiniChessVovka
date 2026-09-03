"""The network as an engine the rest of the app can play against.

This is the phase-5 seam: something with the same contract as
`ai.find_best_move_with_score` -- `(move, white_relative_score)` -- that the GUI can
select instead of the alpha-beta search.

**Importing this module does not import torch.** `available()` answers without it, and
the network is only loaded when a move is actually asked for, from inside the background
`AIThread`. So a user who never selects the network never pays the 3 GB of CUDA, and the
GUI's "the search never blocks the UI" invariant covers the several seconds of first
load for a user who does.

**The move comes from MCTS, not from the policy head alone.** `nn/mcts.py` drives the
Rust tree (`engine_rs/src/mcts.rs`) with this network as its leaf evaluator and the move
is the most-visited root child. The difference is not marginal: on the same weights
`docs/ZERO.md` measures the raw policy argmax at 0.185 against depth-2 alpha-beta and the
tree at 0.833, because in this variant one hanging piece loses the game and search is
what stops a policy that is right half the time from hanging one. `nn/arena.py`'s
`PolicyPlayer` is where the searchless policy still lives, for measuring the phase-2
criterion.

**The budget is time, not simulations.** A simulation costs whatever the position makes
it cost -- a full-hand middlegame descends further and branches wider than an opening --
so a fixed count is not a fixed wait, and a wait is the thing an interactive caller is
actually spending. The arena and the self-play loop keep counting simulations, which is
what a reproducible measurement needs; see `nn.mcts.search`, which takes either.

The score reported is the root's mean value, put back through the checkpoint's value
scale. It is **not a calibrated position assessment** -- the value head carries a
side-to-move bias from its teacher set (docs/ZERO.md) -- which is why the GUI labels the
readout `network` rather than naming a depth.
"""
import glob
import math
import os
import threading

import minichess_engine as rs

from nn import paths

ENV_CHECKPOINT = "MINIZERO_CHECKPOINT"

# atanh(1) is infinite, and the value head does reach +-1 on decided positions.
VALUE_CLAMP = 0.999

# Seconds of search per move when the caller names no budget. The GUI's ladder
# (`settings.NET_TIME_CHOICES`) carries the same value as its default: at the ~8k
# simulations/s this machine reaches from the opening, half a second buys a search past
# the 2,400 that drew level with depth-6 alpha-beta (docs/ZERO.md), at about the wall
# clock depth 6 costs.
DEFAULT_THINK_SECONDS = 0.5

# Leaves handed to the network per call. Virtual loss is what makes a batch worth more
# than one simulation, and it is worth ~8x at 16 (800 sims: 1.08s at batch 1, 0.13s at
# batch 16). Not raised further: the bottleneck above this size is marshalling the planes
# across PyO3 rather than the GPU -- this is a 0.5M-parameter network -- while a wider
# batch explores more of the tree under a loss that has not happened, which costs
# accuracy for no speed.
DEFAULT_BATCH = 16

_lock = threading.Lock()
_loaded = None

def find_checkpoint():
    """$MINIZERO_CHECKPOINT, else the most recently written best.pt under checkpoints/.

    Picking the newest by mtime rather than a configured run name means training a new
    checkpoint is enough to make the GUI play it -- there is no second place to update
    and therefore no way for the two to disagree.
    """
    override = os.environ.get(ENV_CHECKPOINT)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    root = paths.subdir("checkpoints", create=False)
    found = glob.glob(os.path.join(root, "*", "best.pt"))
    if not found:
        return None
    return max(found, key=os.path.getmtime)

def available():
    """Whether there is a network to play, answered **without importing torch.**"""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        return False
    path = find_checkpoint()
    return bool(path and os.path.exists(path))

def unavailable_reason():
    """Why `available()` is False, phrased for a toast. None when it is True.

    Must agree with `available()` on every branch: it asks whether the checkpoint
    *exists*, not merely whether one is configured, or a bad MINIZERO_CHECKPOINT would
    make the GUI format None into its message.
    """
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed — see requirements-nn.txt"
    path = find_checkpoint()
    if not path:
        return "no trained checkpoint — run `python -m nn.train` first"
    if not os.path.exists(path):
        return "no checkpoint at %s" % path
    return None

class _Network:
    def __init__(self, path):
        import torch
        from nn.model import load_checkpoint

        self.path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net, self.meta = load_checkpoint(path, device=self.device)
        self.net.eval()
        # The value head's output only means anything against the scale its targets were
        # built with, which is why the scale is stamped into the checkpoint.
        self.value_scale = float(self.meta.get("value_scale", 400.0))
        self.evaluator = None
        self.last_simulations = 0
        self.last_nodes = 0
        self._warm_up(torch)

    def _warm_up(self, torch):
        """Pay CUDA's first-call cost here rather than out of the first move's budget.

        The first forward pass on a fresh device builds the context and picks kernels,
        which took ~250ms on this machine -- an entire default think time, so the first
        network move came back after **one** simulation and was effectively unchosen.
        Loading already happens on the background `AIThread`, where the "search never
        blocks the UI" invariant covers a slow start; a move does not have that excuse.
        """
        shape = (1, rs.ENCODE_PLANES, rs.BOARD_SIZE, rs.BOARD_SIZE)
        with torch.inference_mode():
            self.net(torch.zeros(shape, dtype=torch.float32, device=self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def label(self):
        top1 = self.meta.get("metrics", {}).get("top1")
        run = self.meta.get("run", os.path.basename(os.path.dirname(self.path)))
        if top1 is None:
            return "network · %s" % run
        return "network · %s · top1 %.0f%%" % (run, 100 * top1)

    def best_move(self, gamestate, seconds=None, simulations=None, batch=None):
        """MCTS from this position: `(move, white_relative_score)`.

        `seconds` and `simulations` are both optional and both may be given, in which
        case the search stops at whichever comes first; with neither, it searches for
        `DEFAULT_THINK_SECONDS`.
        """
        import ai
        from nn import mcts

        if seconds is None and simulations is None:
            seconds = DEFAULT_THINK_SECONDS
        if self.evaluator is None:
            self.evaluator = mcts.Evaluator(self.net, self.device)

        synced = ai._sync_to_rust(gamestate)
        tree = mcts.search(synced, self.evaluator, simulations=simulations,
                           batch=batch or DEFAULT_BATCH, time_limit=seconds)
        self.last_simulations = tree.simulations
        self.last_nodes = tree.nodes

        move = tree.best_move()
        if move is None:
            return None, None
        # Promotion case encodes colour and the two generators pick it differently;
        # ai._normalize_promotion is the single reconciliation point (CLAUDE.md).
        move = ai._normalize_promotion(move, gamestate.current_turn)
        # The root's mean value is in the frame of the side to move there, exactly like
        # the value head's own output, so it takes the same flip into white-relative.
        return move, self._centipawns(float(tree.value), gamestate.current_turn)

    def _centipawns(self, value, turn):
        """Value head -> the white-relative centipawns everything else in the app speaks.

        The head answers in the frame of the side to move, because the input is
        canonicalised that way; scores are white-relative everywhere else (CLAUDE.md), so
        the sign follows the same flip the board does.
        """
        clamped = max(-VALUE_CLAMP, min(VALUE_CLAMP, value))
        centipawns = self.value_scale * math.atanh(clamped)
        return centipawns if turn == "w" else -centipawns

def load(path=None):
    """The loaded network, loading it on first use. Safe to call from any thread."""
    global _loaded
    with _lock:
        target = path or find_checkpoint()
        if target is None:
            raise RuntimeError(unavailable_reason() or "no checkpoint")
        if _loaded is None or _loaded.path != target:
            _loaded = _Network(target)
        return _loaded

def unload():
    """Drop the loaded network. Used when switching back to alpha-beta."""
    global _loaded
    with _lock:
        _loaded = None

def describe():
    try:
        return load().label()
    except Exception as exc:
        return "network unavailable — %s" % exc

def find_best_move_with_score(gamestate, depth=None, seconds=None, simulations=None):
    """Same contract as `ai.find_best_move_with_score`.

    `depth` is accepted and ignored so the caller does not have to know which engine it
    is talking to: an MCTS budget is time or simulations, and there is no depth to set.
    """
    return load().best_move(gamestate, seconds=seconds, simulations=simulations)

def find_best_move(gamestate, depth=None, seconds=None, simulations=None):
    return find_best_move_with_score(gamestate, seconds=seconds,
                                     simulations=simulations)[0]

def last_search_stats():
    """(simulations, nodes) from the most recent search, or (0, 0). For a readout that
    says what the time budget actually bought."""
    net = _loaded
    return (net.last_simulations, net.last_nodes) if net else (0, 0)
