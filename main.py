# -*- coding: utf-8 -*-
"""Pygame front end for Mini Crazyhouse 6×6.

The loop is: read events into intents, apply intents, redraw only if something
changed. Three things are deliberate and easy to undo by accident:

* **The engine never freezes the window.** A search runs in a worker thread and
  the UI keeps drawing, keeps responding to the mouse, and keeps showing how long
  the search has been going. Undo and New game stay live during a search; results
  that arrive for a position the user has since left are discarded by generation
  tag rather than applied to the wrong board.
* **Panel geometry is fixed.** See layout.py — controls must never move under the
  cursor as the game progresses.
* **Redraw is dirty-flagged.** A still board costs one idle tick, not a full
  re-render plus sprite rescaling at 30fps.
"""

import random
import sys
import time

import pygame

import ai
import settings
from config import FPS, IDLE_FPS, HINT_MAX_DEPTH, MIN_MOVE_DISPLAY, PANEL_COLORS
from gamestate import GameState
from gui import draw_frame, load_images
from layout import DEFAULT_WIN_H, DEFAULT_WIN_W, Layout, MIN_WIN_H, MIN_WIN_W
from pieces import EMPTY_SQUARE
from thread_utils import AIThread, HintThread
from utils import coords_to_algebraic, format_move_for_print, get_piece_color

DRAG_THRESHOLD = 5          # pixels before a press becomes a drag
FLASH_DURATION = 0.45       # seconds an illegal-click flash stays on screen
TOAST_DURATION = 3.5
PIECE_NAMES = {'P': 'Pawn', 'N': 'Knight', 'B': 'Bishop', 'R': 'Rook', 'Q': 'Queen'}


class BoardView:
    """Flat snapshot handed to the renderer — live position or a historical one."""

    def __init__(self, board, hands, turn, last_move, check_square, needs_promotion,
                 result_title="", result_detail="", result_color=PANEL_COLORS['text']):
        self.board = board
        self.hands = hands
        self.turn = turn
        self.last_move = last_move
        self.check_square = check_square
        self.needs_promotion = needs_promotion
        self.result_title = result_title
        self.result_detail = result_detail
        self.result_color = result_color


class UIState:
    """Everything the renderer needs that is not game state."""

    def __init__(self, prefs):
        self.flipped = False
        self.mouse_pos = (0, 0)
        self.drag = None
        self.selected_square = None
        self.selected_drop = None
        self.move_targets = []
        self.drop_targets = []
        self.hover_square = None
        self.flashes = []
        self.show_hint = bool(prefs['show_hint'])
        self.hint_move = None
        self.hint_pending = False
        self.thinking = False
        self.think_started = 0.0
        self.think_depth = prefs['ai_depth']
        self.history = []
        self.view_ply = 0
        self.live_ply = 0
        self.movelist_scroll = 0
        self.depth = prefs['ai_depth']
        self.ai_white = bool(prefs['white_ai'])
        self.ai_black = bool(prefs['black_ai'])
        self.can_undo = False
        self.anim_phase = 0.0
        self.toast = None
        self.game_over = False
        self.needs_promotion = False

    # --- derived values the renderer reads ---

    @property
    def browsing(self):
        return self.view_ply != self.live_ply

    @property
    def interactive(self):
        return not self.browsing and not self.game_over and not self.needs_promotion

    @property
    def think_elapsed(self):
        return time.time() - self.think_started if self.thinking else 0.0

    @property
    def depth_label(self):
        return settings.DEPTH_LABELS.get(self.depth, "")

    @property
    def can_depth_up(self):
        return self.depth < settings.DEPTH_CHOICES[-1]

    @property
    def can_depth_down(self):
        return self.depth > settings.DEPTH_CHOICES[0]

    @property
    def hint_text(self):
        return format_move_for_print(self.hint_move) if self.hint_move else ""

    @property
    def idle_status(self):
        if self.game_over:
            return "Game over"
        if self.show_hint:
            return "Hint on — no suggestion yet"
        human_turn = not (self.ai_white and self.ai_black)
        return "Your move" if human_turn else "Engine idle"

    def player_label(self, color):
        is_ai = self.ai_white if color == 'w' else self.ai_black
        return f"engine · depth {self.depth}" if is_ai else "you"

    def status_message(self):
        """(text, colour) for the bottom band, highest priority first."""
        if self.needs_promotion:
            return "Choose a promotion piece", PANEL_COLORS['accent']
        if self.browsing:
            return (f"Viewing move {self.view_ply} of {self.live_ply}\n"
                    "← → step · End returns to the game", PANEL_COLORS['warn'])
        if self.toast and self.toast[2] > time.time():
            return self.toast[0], self.toast[1]
        if self.selected_drop:
            name = PIECE_NAMES.get(self.selected_drop[1], self.selected_drop[1])
            return (f"Dropping {name}\nclick a highlighted square · Esc cancels",
                    PANEL_COLORS['accent'])
        if self.selected_square:
            return (f"{coords_to_algebraic(*self.selected_square)} selected\n"
                    "click a target, drag it, or press Esc", PANEL_COLORS['text_dim'])
        return ("← → history · U undo · F flip · H hint", PANEL_COLORS['text_faint'])


class Game:
    def __init__(self):
        self.prefs = settings.load()
        self.gs = GameState()
        self.gs.setup_initial_board()
        self.gs.white_ai_enabled = bool(self.prefs['white_ai'])
        self.gs.black_ai_enabled = bool(self.prefs['black_ai'])
        self.gs.ai_depth = self.prefs['ai_depth']

        self.ui = UIState(self.prefs)
        self.ui.flipped = self._default_orientation()

        self.ai_thread = None
        self.hint_thread = None
        self.generation = 0          # bumped whenever the position the engine was
                                     # asked about stops being the current one
        self.last_move_at = 0.0
        self.check_cache = {}
        self.running = True
        self.dirty = True

        self.screen = None
        self.layout = None
        self.hits = {'buttons': {}, 'hand': {}, 'promotion': {}, 'movelist': {}}

    # --- setup -------------------------------------------------------------

    def _default_orientation(self):
        """Face the board toward whichever side the human plays."""
        stored = self.prefs.get('board_flipped')
        if stored is not None:
            return bool(stored)
        human_is_black = self.prefs['white_ai'] and not self.prefs['black_ai']
        return bool(human_is_black)

    def initial_window_size(self):
        width, height = self.prefs['win_w'], self.prefs['win_h']
        if width is None or height is None:
            try:
                info = pygame.display.Info()
                width = int(info.current_w * 0.78)
                height = int(info.current_h * 0.82)
            except pygame.error:
                width, height = DEFAULT_WIN_W, DEFAULT_WIN_H
        return max(MIN_WIN_W, width), max(MIN_WIN_H, height)

    def resize(self, width, height):
        width, height = max(MIN_WIN_W, width), max(MIN_WIN_H, height)
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.layout = Layout(width, height)
        self.dirty = True

    def save_prefs(self):
        self.prefs.update({
            'win_w': self.layout.win_w if self.layout else None,
            'win_h': self.layout.win_h if self.layout else None,
            'ai_depth': self.ui.depth,
            'white_ai': self.ui.ai_white,
            'black_ai': self.ui.ai_black,
            'show_hint': self.ui.show_hint,
            'board_flipped': self.ui.flipped,
        })
        settings.save(self.prefs)

    # --- state helpers -----------------------------------------------------

    def is_ai_turn(self):
        return ((self.gs.current_turn == 'w' and self.ui.ai_white) or
                (self.gs.current_turn == 'b' and self.ui.ai_black))

    def game_finished(self):
        return self.gs.checkmate or self.gs.stalemate or self.gs.is_draw

    def invalidate(self, clear_hint=True):
        """The position changed: abandon in-flight engine work and selections."""
        self.generation += 1
        self.ui.selected_square = None
        self.ui.selected_drop = None
        self.ui.move_targets = []
        self.ui.drop_targets = []
        self.ui.drag = None
        if clear_hint:
            self.ui.hint_move = None
            self.ui.hint_pending = False
        self.check_cache.clear()
        self.dirty = True

    def toast(self, text, color=None):
        self.ui.toast = (text, color or PANEL_COLORS['text_dim'], time.time() + TOAST_DURATION)
        self.dirty = True

    def flash(self, square):
        self.ui.flashes.append((square, time.time() + FLASH_DURATION))
        self.dirty = True

    def sync_ui(self):
        self.ui.live_ply = max(0, len(self.gs.saved_states) - 1)
        self.ui.view_ply = min(self.ui.view_ply, self.ui.live_ply)
        self.ui.can_undo = self.ui.live_ply > 0
        self.ui.game_over = self.game_finished()
        self.ui.needs_promotion = self.gs.needs_promotion_choice
        self.ui.depth = self.gs.ai_depth
        self.ui.ai_white = self.gs.white_ai_enabled
        self.ui.ai_black = self.gs.black_ai_enabled

    # --- rendering data ----------------------------------------------------

    def check_square_for(self, board, king_pos, turn):
        """King square to flag as in check, or None.

        The old code swapped which side it tested when the board was flipped,
        which silently disabled the check highlight for a flipped board. Flip is
        a view transform; it must not change what is true about the position.
        """
        key = (tuple(tuple(row) for row in board), turn)
        cached = self.check_cache.get(key)
        if cached is None:
            probe = GameState()
            probe.board = [row[:] for row in board]
            probe.current_turn = turn
            probe.king_pos = dict(king_pos)
            probe.find_kings()
            cached = probe.king_pos.get(turn) if probe.is_in_check(turn) else False
            if len(self.check_cache) > 64:
                self.check_cache.clear()
            self.check_cache[key] = cached
        return cached or None

    def build_view(self):
        if (self.ui.browsing and not self.gs.needs_promotion_choice
                and 0 <= self.ui.view_ply < len(self.gs.saved_states)):
            snap = self.gs.saved_states[self.ui.view_ply]
            return BoardView(
                board=snap['board'], hands=snap['hands'], turn=snap['current_turn'],
                last_move=snap.get('last_move'),
                check_square=self.check_square_for(snap['board'], self.gs.king_pos,
                                                  snap['current_turn']),
                needs_promotion=False)

        title, detail, color = "", "", PANEL_COLORS['text']
        if self.gs.checkmate:
            winner = "Black" if self.gs.current_turn == 'w' else "White"
            title, color = f"Checkmate — {winner} wins", PANEL_COLORS['bad']
            detail = f"{'White' if self.gs.current_turn == 'w' else 'Black'} has no legal reply."
        elif self.gs.stalemate:
            title, color = "Stalemate — draw", PANEL_COLORS['warn']
            detail = "No legal moves, and the king is not in check."
        elif self.gs.is_draw:
            title, color = "Draw", PANEL_COLORS['warn']
            detail = self.gs.game_over_message.rstrip('.') or "Drawn game."

        return BoardView(
            board=self.gs.board, hands=self.gs.hands, turn=self.gs.current_turn,
            last_move=self.gs.last_move,
            check_square=self.check_square_for(self.gs.board, self.gs.king_pos,
                                               self.gs.current_turn),
            needs_promotion=self.gs.needs_promotion_choice,
            result_title=title, result_detail=detail, result_color=color)

    # --- move application --------------------------------------------------

    @staticmethod
    def notation(move):
        """Move list text. Drops read `N@a6`, not utils' `WN@a6`."""
        if move and move[0] == 'drop':
            return f"{move[1][1].upper()}@{coords_to_algebraic(*move[2])}"
        return format_move_for_print(move)

    def record_move(self, move):
        """Append notation for a completed move, with a check/mate suffix."""
        text = self.notation(move)
        if self.gs.checkmate:
            text += "#"
        elif self.gs.is_in_check(self.gs.current_turn):
            text += "+"
        self.ui.history.append(text)
        self.ui.view_ply = self.ui.live_ply = max(0, len(self.gs.saved_states) - 1)
        self.ui.movelist_scroll = 10 ** 6      # clamped on draw: pin to the newest

    def legal_moves_between(self, start, target):
        return [m for m in self.gs.get_all_legal_moves()
                if m[0] != 'drop' and m[0] == start and m[1] == target]

    def play_move(self, move):
        """Apply a move made by the human. Returns True if the board changed."""
        if not self.gs.make_move(move):
            self.toast("That move was rejected by the rules engine.", PANEL_COLORS['bad'])
            return False

        if self.gs.needs_promotion_choice:
            # Leave the modal up; the position is not committed until it is answered.
            self.invalidate()
            self.sync_ui()
            return True

        self.gs.save_state()
        self.record_move(move)
        self.last_move_at = time.time()
        self.invalidate()
        self.sync_ui()
        return True

    def finish_promotion(self, piece_char):
        if not self.gs.complete_promotion(piece_char):
            self.toast("Invalid promotion choice.", PANEL_COLORS['bad'])
            return
        self.gs.save_state()
        self.record_move(self.gs.last_move)
        self.last_move_at = time.time()
        self.invalidate()
        self.sync_ui()

    def undo_one_ply(self):
        """Take back exactly one half-move.

        The old build always undid two, which was wrong for human-vs-human, wrong
        at the first move, and silent when it failed.
        """
        if self.ui.live_ply <= 0:
            self.toast("Nothing to undo — this is the starting position.",
                       PANEL_COLORS['text_dim'])
            return
        if not self.gs.undo_move():
            self.toast("Could not undo.", PANEL_COLORS['bad'])
            return
        if self.ui.history:
            self.ui.history.pop()
        self.invalidate()
        self.sync_ui()
        self.ui.view_ply = self.ui.live_ply

        side = "White" if self.gs.current_turn == 'w' else "Black"
        if self.ui.thinking:
            # The in-flight search is now answering a question about a position
            # that no longer exists. invalidate() bumped the generation, so its
            # result will be dropped when it lands.
            self.ai_thread = None
            self.ui.thinking = False
            self.toast(f"Took back a move — {side} to play.\nThe engine's search was abandoned.",
                       PANEL_COLORS['warn'])
        else:
            self.toast(f"Took back a move — {side} to play.", PANEL_COLORS['text_dim'])

    def new_game(self):
        self.gs.setup_initial_board()
        self.gs.white_ai_enabled = self.ui.ai_white
        self.gs.black_ai_enabled = self.ui.ai_black
        self.gs.ai_depth = self.ui.depth
        self.ai_thread = None
        self.hint_thread = None
        self.ui.thinking = False
        self.ui.history = []
        self.ui.view_ply = self.ui.live_ply = 0
        self.ui.movelist_scroll = 0
        self.ui.flashes = []
        self.last_move_at = 0.0
        self.invalidate()
        self.sync_ui()
        self.toast("New game.", PANEL_COLORS['good'])

    # --- engine ------------------------------------------------------------

    def start_ai_if_needed(self):
        if (self.ui.thinking or self.ai_thread or self.gs.needs_promotion_choice
                or self.game_finished() or not self.is_ai_turn()):
            return
        # Keep the previous move on screen briefly so an engine-vs-engine game is
        # watchable. This never blocks input, unlike the old post-search stall.
        if time.time() - self.last_move_at < MIN_MOVE_DISPLAY:
            self.dirty = True
            return
        self.gs.ai_depth = self.ui.depth
        thread = AIThread(self.gs, self.ui.depth)
        thread.generation = self.generation
        thread.start()
        self.ai_thread = thread
        self.ui.thinking = True
        self.ui.think_started = time.time()
        self.ui.think_depth = self.ui.depth
        self.dirty = True

    def poll_ai(self):
        thread = self.ai_thread
        if not thread or not thread.done:
            return
        self.ai_thread = None
        self.ui.thinking = False
        self.dirty = True

        if thread.generation != self.generation:
            print("[engine] discarding a search result for a superseded position")
            return

        move = thread.best_move
        if move is None:
            legal = self.gs.get_all_legal_moves()
            if not legal:
                self.sync_ui()
                return
            move = random.choice(legal)
            print(f"[engine] returned no move; playing {format_move_for_print(move)} at random")
            self.toast("Engine returned no move — played a random legal move.",
                       PANEL_COLORS['warn'])

        if not self.gs.make_move(move):
            print(f"[engine] illegal move rejected: {format_move_for_print(move)}")
            self.toast("Engine proposed an illegal move; it was ignored.", PANEL_COLORS['bad'])
            self.sync_ui()
            return

        if self.gs.needs_promotion_choice:
            self.gs.complete_promotion('R' if self.gs.current_turn == 'w' else 'r')

        self.gs.save_state()
        self.record_move(self.gs.last_move)
        self.last_move_at = time.time()
        self.invalidate()
        self.sync_ui()
        ai.save_move_cache_to_db(ai.move_cache)

    def start_hint_if_needed(self):
        if (not self.ui.show_hint or self.hint_thread or self.ui.hint_move
                or self.gs.needs_promotion_choice or self.game_finished()
                or self.is_ai_turn()):
            return
        depth = min(self.ui.depth, HINT_MAX_DEPTH)
        thread = HintThread(self.gs, depth=depth)
        thread.generation = self.generation
        thread.start()
        self.hint_thread = thread
        self.ui.hint_pending = True
        self.dirty = True

    def poll_hint(self):
        thread = self.hint_thread
        if not thread or not thread.done:
            return
        self.hint_thread = None
        self.ui.hint_pending = False
        self.dirty = True
        if thread.generation == self.generation:
            self.ui.hint_move = thread.best_move

    # --- selection ---------------------------------------------------------

    def select_square(self, square):
        piece = self.gs.board[square[0]][square[1]]
        if piece == EMPTY_SQUARE or get_piece_color(piece) != self.gs.current_turn:
            return False
        self.ui.selected_square = square
        self.ui.selected_drop = None
        self.ui.drop_targets = []
        self.ui.move_targets = sorted({m[1] for m in self.gs.get_all_legal_moves()
                                       if m[0] != 'drop' and m[0] == square})
        self.dirty = True
        return True

    def select_drop(self, code):
        """`code` is colour+type, e.g. 'wN'."""
        if code[0] != self.gs.current_turn or self.gs.hands[code[0]].get(code[1], 0) <= 0:
            return False
        self.ui.selected_drop = code
        self.ui.selected_square = None
        self.ui.move_targets = []
        self.ui.drop_targets = sorted({m[2] for m in self.gs.get_all_legal_moves()
                                       if m[0] == 'drop' and m[1] == code})
        self.dirty = True
        if not self.ui.drop_targets:
            self.toast(f"No legal square to drop that {PIECE_NAMES.get(code[1], code[1])} on.",
                       PANEL_COLORS['warn'])
        return True

    def clear_selection(self):
        had = self.ui.selected_square or self.ui.selected_drop or self.ui.drag
        self.ui.selected_square = None
        self.ui.selected_drop = None
        self.ui.move_targets = []
        self.ui.drop_targets = []
        self.ui.drag = None
        if had:
            self.dirty = True

    # --- intents -----------------------------------------------------------

    def go_to_ply(self, ply):
        if self.gs.needs_promotion_choice:
            return          # the promotion must be answered before anything else
        ply = max(0, min(self.ui.live_ply, ply))
        if ply != self.ui.view_ply:
            self.ui.view_ply = ply
            self.clear_selection()
            self.dirty = True

    def set_depth(self, depth):
        if depth == self.ui.depth:
            return
        self.ui.depth = self.gs.ai_depth = depth
        # A hint computed at the old depth is no longer the answer to the
        # question the depth control now asks.
        self.ui.hint_move = None
        self.generation += 1
        self.hint_thread = None
        self.ui.hint_pending = False
        self.toast(f"Engine depth {depth} — {settings.DEPTH_LABELS.get(depth, '')}.",
                   PANEL_COLORS['text_dim'])
        self.dirty = True

    def toggle_ai(self, color):
        if color == 'w':
            self.ui.ai_white = self.gs.white_ai_enabled = not self.ui.ai_white
            state = self.ui.ai_white
        else:
            self.ui.ai_black = self.gs.black_ai_enabled = not self.ui.ai_black
            state = self.ui.ai_black
        side = "White" if color == 'w' else "Black"
        self.toast(f"{side} is now played by the {'engine' if state else 'human'}.",
                   PANEL_COLORS['text_dim'])
        if not state and self.ui.thinking and self.gs.current_turn == color:
            self.ai_thread = None       # generation bump below drops the result
            self.ui.thinking = False
        self.invalidate()
        self.sync_ui()

    def handle_button(self, name):
        if name == 'undo':
            self.undo_one_ply()
        elif name == 'new_game':
            self.new_game()
        elif name == 'toggle_white_ai':
            self.toggle_ai('w')
        elif name == 'toggle_black_ai':
            self.toggle_ai('b')
        elif name == 'toggle_hint':
            self.ui.show_hint = not self.ui.show_hint
            self.ui.hint_move = None
            self.ui.hint_pending = False
            self.hint_thread = None
            self.toast(f"Hints {'on' if self.ui.show_hint else 'off'}.", PANEL_COLORS['text_dim'])
        elif name == 'toggle_flip':
            self.ui.flipped = not self.ui.flipped
            self.dirty = True
        elif name == 'depth_up':
            self.set_depth(settings.cycle_depth(self.ui.depth, 1))
        elif name == 'depth_down':
            self.set_depth(settings.cycle_depth(self.ui.depth, -1))

    def try_board_action(self, square):
        """A click landed on `square` while the human is to move."""
        piece = self.gs.board[square[0]][square[1]]
        mine = piece != EMPTY_SQUARE and get_piece_color(piece) == self.gs.current_turn

        if self.ui.selected_drop:
            if square in self.ui.drop_targets:
                self.play_move(('drop', self.ui.selected_drop, square))
                return
            if mine:
                self.select_square(square)
                return
            self.flash(square)
            self.toast(f"A {PIECE_NAMES.get(self.ui.selected_drop[1], 'piece')} cannot be "
                       f"dropped on {coords_to_algebraic(*square)}.", PANEL_COLORS['bad'])
            self.clear_selection()
            return

        if self.ui.selected_square:
            start = self.ui.selected_square
            if square == start:
                return                       # keep the selection; may become a drag
            options = self.legal_moves_between(start, square)
            if options:
                # Several entries differ only by promotion piece: submit the move
                # without one so the picker opens instead of silently choosing.
                move = (start, square, None) if len(options) > 1 or options[0][2] else options[0]
                self.play_move(move)
                return
            if mine:
                self.select_square(square)
                return
            self.flash(square)
            self.toast(f"{coords_to_algebraic(*start)} → {coords_to_algebraic(*square)} "
                       "is not a legal move.", PANEL_COLORS['bad'])
            self.clear_selection()
            return

        if mine:
            self.select_square(square)
        elif piece != EMPTY_SQUARE:
            self.flash(square)
            side = "White" if self.gs.current_turn == 'w' else "Black"
            self.toast(f"That is not your piece — {side} to move.", PANEL_COLORS['warn'])

    def snap_to_live(self):
        if self.ui.browsing:
            self.go_to_ply(self.ui.live_ply)
            return True
        return False

    # --- events ------------------------------------------------------------

    def handle_mouse_down(self, pos, button):
        self.dirty = True

        if button in (4, 5):
            return

        # Promotion is modal: only its own buttons exist this frame (draw_frame
        # clears every other hit region while the picker is up).
        if self.hits['promotion']:
            for char, rect in self.hits['promotion'].items():
                if rect.collidepoint(pos):
                    self.finish_promotion(char)
            return

        for name, rect in self.hits['buttons'].items():
            if rect.collidepoint(pos):
                self.handle_button(name)
                return

        for ply, rect in self.hits['movelist'].items():
            if rect.collidepoint(pos):
                self.go_to_ply(ply)
                return

        for code, rect in self.hits['hand'].items():
            if rect.collidepoint(pos):
                if self.snap_to_live():
                    return
                if self.ui.selected_drop == code:
                    self.clear_selection()
                elif self.select_drop(code):
                    char = code[1].upper() if code[0] == 'w' else code[1].lower()
                    self.ui.drag = {'piece': char, 'from_hand': code, 'origin': None,
                                    'start': pos, 'active': False}
                return

        square = self.layout.square_at(pos, self.ui.flipped)
        if square is None:
            self.clear_selection()
            return
        if self.snap_to_live():
            return
        if not self.ui.interactive or self.is_ai_turn():
            if self.is_ai_turn() and not self.game_finished():
                self.toast("The engine is still thinking — you can undo or start a new game.",
                           PANEL_COLORS['text_dim'])
            return

        before = self.ui.selected_square
        self.try_board_action(square)
        if self.ui.selected_square == square and before != square:
            piece = self.gs.board[square[0]][square[1]]
            self.ui.drag = {'piece': piece, 'origin': square, 'from_hand': None,
                            'start': pos, 'active': False}

    def handle_mouse_up(self, pos):
        drag = self.ui.drag
        self.ui.drag = None
        self.dirty = True
        if not drag or not drag['active']:
            return

        square = self.layout.square_at(pos, self.ui.flipped)
        if square is None:
            self.clear_selection()
            return
        if drag['from_hand']:
            if square in self.ui.drop_targets:
                self.play_move(('drop', drag['from_hand'], square))
            else:
                self.flash(square)
                self.clear_selection()
            return
        if square == drag['origin']:
            return                              # dropped back where it started
        self.try_board_action(square)

    def handle_key(self, event):
        key = event.key
        ctrl = event.mod & pygame.KMOD_CTRL
        self.dirty = True

        if key == pygame.K_ESCAPE:
            if self.ui.browsing:
                self.go_to_ply(self.ui.live_ply)
            else:
                self.clear_selection()
        elif key == pygame.K_LEFT:
            self.go_to_ply(self.ui.view_ply - 1)
        elif key == pygame.K_RIGHT:
            self.go_to_ply(self.ui.view_ply + 1)
        elif key == pygame.K_HOME:
            self.go_to_ply(0)
        elif key == pygame.K_END:
            self.go_to_ply(self.ui.live_ply)
        elif key == pygame.K_PAGEUP:
            self.go_to_ply(self.ui.view_ply - 10)
        elif key == pygame.K_PAGEDOWN:
            self.go_to_ply(self.ui.view_ply + 10)
        elif key in (pygame.K_u, pygame.K_BACKSPACE) or (ctrl and key == pygame.K_z):
            self.undo_one_ply()
        elif ctrl and key == pygame.K_n:
            self.new_game()          # bare N is unbound: a new game is unrecoverable
        elif key == pygame.K_f:
            self.handle_button('toggle_flip')
        elif key == pygame.K_h:
            self.handle_button('toggle_hint')
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.set_depth(settings.cycle_depth(self.ui.depth, 1))
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.set_depth(settings.cycle_depth(self.ui.depth, -1))
        elif key in (pygame.K_r, pygame.K_n, pygame.K_b) and self.gs.needs_promotion_choice:
            upper = chr(key).upper()
            self.finish_promotion(upper if self.gs.current_turn == 'w' else upper.lower())
        elif ctrl and key == pygame.K_q:
            self.running = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.resize(event.w, event.h)
            elif event.type == pygame.MOUSEMOTION:
                self.ui.mouse_pos = event.pos
                square = self.layout.square_at(event.pos, self.ui.flipped)
                if square != self.ui.hover_square:
                    self.ui.hover_square = square
                    self.dirty = True
                if self.ui.drag and not self.ui.drag['active']:
                    dx = event.pos[0] - self.ui.drag['start'][0]
                    dy = event.pos[1] - self.ui.drag['start'][1]
                    if abs(dx) > DRAG_THRESHOLD or abs(dy) > DRAG_THRESHOLD:
                        self.ui.drag['active'] = True
                if self.ui.drag and self.ui.drag['active']:
                    self.dirty = True
                elif self.layout.panel.collidepoint(event.pos):
                    self.dirty = True       # hover feedback on the controls
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.ui.mouse_pos = event.pos
                if event.button in (4, 5):
                    if self.layout.movelist.collidepoint(event.pos):
                        step = -2 if event.button == 4 else 2
                        self.ui.movelist_scroll = max(0, self.ui.movelist_scroll + step)
                        self.dirty = True
                elif event.button == 1:
                    self.handle_mouse_down(event.pos, event.button)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.ui.mouse_pos = event.pos
                self.handle_mouse_up(event.pos)
            elif event.type == pygame.KEYDOWN:
                self.handle_key(event)

    # --- loop --------------------------------------------------------------

    def expire_transients(self):
        now = time.time()
        remaining = [(sq, until) for sq, until in self.ui.flashes if until > now]
        if len(remaining) != len(self.ui.flashes):
            self.ui.flashes = remaining
            self.dirty = True
        if self.ui.toast and self.ui.toast[2] <= now:
            self.ui.toast = None
            self.dirty = True

    def render(self):
        now = time.time()
        self.ui.anim_phase = now
        # Renderer wants (square, strength) so a flash can fade out.
        flashes = self.ui.flashes
        self.ui.flashes = [(sq, max(0.0, (until - now) / FLASH_DURATION)) for sq, until in flashes]
        try:
            self.hits = draw_frame(self.screen, self.layout, self.build_view(), self.ui)
        finally:
            self.ui.flashes = flashes
        pygame.display.flip()

    def run(self):
        ai.load_move_cache_from_db()

        pygame.display.set_caption("Mini Crazyhouse 6×6")
        self.resize(*self.initial_window_size())
        load_images()
        self.sync_ui()

        clock = pygame.time.Clock()
        while self.running:
            self.handle_events()
            self.expire_transients()
            self.poll_ai()
            self.poll_hint()
            self.start_ai_if_needed()
            self.start_hint_if_needed()

            animating = (self.ui.thinking or self.ui.hint_pending
                         or self.ui.flashes or (self.ui.drag and self.ui.drag['active']))
            if self.dirty or animating:
                self.render()
                self.dirty = False
            clock.tick(FPS if animating else IDLE_FPS)

        self.save_prefs()
        pygame.quit()


def main():
    pygame.init()
    Game().run()
    sys.exit()


if __name__ == "__main__":
    main()
