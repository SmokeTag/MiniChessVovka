"""Background search threads.

Both threads copy the position before they start: the GUI keeps mutating the live
`GameState` while a search runs, and a search reading a board mid-move would answer a
question about a position that never existed.

Staleness is handled differently on the two paths. `AIThread` is a move the engine is
about to play, so `main.Game` stamps a `generation` on it and drops the result if the
position moved on. `HintPool` needs no such stamp: it keys every search by the position
it answers, so a late answer is not stale, it is simply an answer to a different
question — and one worth keeping.
"""

import collections
import copy
import threading
import time
import traceback

from ai import find_best_move, find_best_move_with_score
from utils import format_move_for_print

class AIThread(threading.Thread):
    """The engine's own move. Single-PV, so the score costs nothing extra."""

    def __init__(self, gamestate, depth):
        threading.Thread.__init__(self)
        self.gamestate = copy.deepcopy(gamestate)
        self.depth = depth
        self.best_move = None
        self.score = None
        self.done = False
        self.name = f"AIThread-{gamestate.current_turn}-D{depth:.0f}-Move-{time.time():.0f}"
        self.daemon = True

    def run(self):
        try:
            print(f"Starting AI move calculation in thread: {self.name}")
            self.best_move, self.score = find_best_move_with_score(self.gamestate, self.depth)
            move_str = format_move_for_print(self.best_move)
            print(f"AI thread {self.name} finished. Best move: {move_str} ({self.score})")
        except Exception as e:
            print(f"!!! EXCEPTION in AI thread {self.name}: {e}")
            traceback.print_exc()
            self.best_move, self.score = None, None
        finally:
            self.done = True

class HintThread(threading.Thread):
    """The human's hint: the top `lines` moves, ranked, with scores.

    `lines > 1` runs a MultiPV search — the only way ranks 2+ carry real scores rather
    than alpha-beta bounds, and 2.4-4.4x the cost of a single-PV search at the same
    depth. That is why the count is a user-facing setting rather than a constant.
    """

    def __init__(self, gamestate, depth=8, lines=1):
        threading.Thread.__init__(self)
        self.gamestate = copy.deepcopy(gamestate)
        self.depth = depth
        self.lines = max(1, int(lines))
        self.ranked = []
        self.done = False
        self.started_at = time.time()
        self.daemon = True
        self.name = f"HintThread-{gamestate.current_turn}-L{self.lines}-{time.time():.0f}"

    @property
    def best_move(self):
        return self.ranked[0][0] if self.ranked else None

    def run(self):
        try:
            if self.lines == 1:
                move, score = find_best_move_with_score(self.gamestate, self.depth)
                self.ranked = [(move, score)] if move else []
            else:
                ranked = list(find_best_move(self.gamestate, self.depth,
                                             return_top_n=self.lines) or [])
                if len(ranked) == 1 and len(self.gamestate.get_all_legal_moves()) == 1:
                    ranked = [(ranked[0][0], None)]
                self.ranked = ranked
        except Exception as e:
            print(f"HintThread error: {e}")
            self.ranked = []
        finally:
            self.done = True

class HintPool:
    """Hint searches for several positions at once, one thread — one core — each.

    The GUI used to run exactly one hint and tie it to a `generation` counter. Playing a
    move while that search was in flight was the worst of both worlds: the result was
    discarded when it landed, *and* the next hint could not start until it did. Three
    quick moves therefore cost three full searches and showed an answer for none of
    them, and because `find_best_move` held the Rust book lock for its whole duration,
    the main thread froze the moment it asked the book anything (measured: 46s).

    Here a search is keyed by the question it answers — `(fen, depth, lines)` — not by
    when it was started, and nothing is cancelled when the board moves on. The searches
    for the positions you walked past keep running on their own cores and file their
    answers under their own keys, so the analysis fans out ahead of you and walking back
    into one of those positions shows its lines immediately. Keying on the FEN also
    replaces the generation guard outright: an answer can only ever be displayed against
    the position it was computed for.

    A Python thread cannot be killed, so `workers` is a real budget, not a hint. Every
    slot in flight is a core busy until that search finishes; `submit` simply declines
    when they are all taken and the caller tries again next frame.
    """

    KEEP_RESULTS = 128

    def __init__(self, workers=1):
        self.workers = max(1, int(workers))
        self._running = {}
        self._results = collections.OrderedDict()

    def set_workers(self, workers):
        """Change the budget. Searches already in flight are left alone — they cannot
        be stopped, and lowering the number to abandon their results as well as their
        cores would waste the work twice."""
        self.workers = max(1, int(workers))

    def submit(self, key, gamestate, depth, lines):
        """Start a search for `key` if it is neither answered nor already running and a
        core is free. Returns whether a thread was started."""
        if key in self._results or key in self._running or len(self._running) >= self.workers:
            return False
        thread = HintThread(gamestate, depth=depth, lines=lines)
        thread.start()
        self._running[key] = thread
        return True

    def reap(self):
        """Move finished searches into the result cache. Returns the keys that landed."""
        finished = [key for key, thread in self._running.items() if thread.done]
        for key in finished:
            thread = self._running.pop(key)
            self._results[key] = [(move, score) for move, score in thread.ranked if move]
            self._results.move_to_end(key)
        while len(self._results) > self.KEEP_RESULTS:
            self._results.popitem(last=False)
        return finished

    def result(self, key):
        """The ranked lines for `key`, or None if nothing has answered it yet.

        An empty list is an answer — a position with no legal moves — and is not None.
        """
        if key not in self._results:
            return None
        self._results.move_to_end(key)
        return self._results[key]

    def is_running(self, key):
        return key in self._running

    def started_at(self, key):
        thread = self._running.get(key)
        return thread.started_at if thread else 0.0

    def active(self):
        return len(self._running)

    def forget_results(self):
        """Drop cached answers, keeping searches in flight. For a new game, where the
        old answers are still correct but nothing will ever ask for them again."""
        self._results.clear()
