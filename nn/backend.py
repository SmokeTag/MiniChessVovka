"""The network as an engine the rest of the app can play against.

This is the phase-5 seam in its smallest useful form: something with the same contract
as `ai.find_best_move_with_score` -- `(move, white_relative_score)` -- that the GUI can
select instead of the alpha-beta search.

**Importing this module does not import torch.** `available()` answers without it, and
the network is only loaded when a move is actually asked for, from inside the background
`AIThread`. So a user who never selects the network never pays the 3 GB of CUDA, and the
GUI's "the search never blocks the UI" invariant covers the several seconds of first
load for a user who does.

The move is a raw policy argmax over the legal actions -- no search. That is genuinely
weak: `docs/ZERO.md` measures it beating random 0.970 but scoring ~0.1 against depth-2
alpha-beta, because in this variant one hanging piece loses the game and a policy that is
right half the time hangs one soon enough. Phase 3 puts MCTS on top; until then this is
here to be played against and inspected, not to be competitive.
"""
import glob
import math
import os
import threading

from nn import paths

ENV_CHECKPOINT = "MINIZERO_CHECKPOINT"

# atanh(1) is infinite, and the value head does reach +-1 on decided positions.
VALUE_CLAMP = 0.999

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

        self.torch = torch
        self.path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net, self.meta = load_checkpoint(path, device=self.device)
        self.net.eval()
        # The value head's output only means anything against the scale its targets were
        # built with, which is why the scale is stamped into the checkpoint.
        self.value_scale = float(self.meta.get("value_scale", 400.0))

    def label(self):
        top1 = self.meta.get("metrics", {}).get("top1")
        run = self.meta.get("run", os.path.basename(os.path.dirname(self.path)))
        if top1 is None:
            return "network · %s" % run
        return "network · %s · top1 %.0f%%" % (run, 100 * top1)

    def best_move(self, gamestate):
        import ai
        import minichess_engine as rs
        from nn import features

        synced = ai._sync_to_rust(gamestate)
        indices = rs.legal_action_indices(synced)
        if not indices:
            return None, None

        x = self.torch.from_numpy(features.encode(synced)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            logits, value = self.net(x)
        logits = logits.float().squeeze(0)

        best = max(indices, key=lambda i: logits[i].item())
        move = rs.action_index_to_move(synced, best)
        # Promotion case encodes colour and the two generators pick it differently;
        # ai._normalize_promotion is the single reconciliation point (CLAUDE.md).
        move = ai._normalize_promotion(move, gamestate.current_turn)
        return move, self._centipawns(float(value.item()), gamestate.current_turn)

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

def find_best_move_with_score(gamestate, depth=None):
    """Same contract as `ai.find_best_move_with_score`. `depth` is accepted and ignored:
    a raw policy has no depth, and the caller should not have to care which engine it
    is talking to."""
    return load().best_move(gamestate)

def find_best_move(gamestate, depth=None):
    return load().best_move(gamestate)[0]
