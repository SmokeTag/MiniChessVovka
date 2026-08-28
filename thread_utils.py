# -*- coding: utf-8 -*-
"""Background search threads.

Both threads copy the position before they start: the GUI keeps mutating the live
`GameState` while a search runs, and a search reading a board mid-move would answer a
question about a position that never existed. The caller stamps a `generation` on the
thread and drops the result if the position has moved on since (see `main.Game`).
"""

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
        self.score = None          # white-relative; None for a forced move
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
        self.ranked = []           # [(move, white_relative_score)], best-first
        self.done = False
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
                # A forced move is answered without a search, and `find_best_move`
                # pads it with a placeholder 0. Showing that as "+0.00" would claim
                # an evaluation nothing computed, so it goes back to None.
                if len(ranked) == 1 and len(self.gamestate.get_all_legal_moves()) == 1:
                    ranked = [(ranked[0][0], None)]
                self.ranked = ranked
        except Exception as e:
            print(f"HintThread error: {e}")
            self.ranked = []
        finally:
            self.done = True
