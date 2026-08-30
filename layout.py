# -*- coding: utf-8 -*-
"""Window geometry for the Pygame front end.

Everything the GUI draws is positioned from a `Layout` recomputed from the current
window size, so the whole app scales when the window is resized. Nothing outside
this module may assume a pixel size.

Two invariants matter for usability and are enforced here rather than by drawing
order:

1. **Panel zones never move.** The side panel is split into fixed-height bands
   (header / controls / toast) plus one elastic band (the move list). Content
   growing inside a band clips instead of pushing the bands below it around, so a
   button is always where the user last saw it.
2. **Hands live beside the board**, not in the panel — a strip above and a strip
   below. They follow board orientation, so the strip nearest you is always yours.

The one band whose height is not a constant is the analysis list, which is sized from
`analysis_rows` — how many hint lines the user asked for. That is a *setting*, not live
search state, so the band still cannot resize while a search is running; changing it
rebuilds the Layout, and only the move list below absorbs the difference.
"""

import pygame

from config import BOARD_SIZE

# Refuse to shrink past the point where the board stops being clickable.
MIN_WIN_W = 760
MIN_WIN_H = 500

# Window size used the very first time, before settings.json exists. Clamped to
# the actual desktop at startup (see main.initial_window_size).
DEFAULT_WIN_W = 1100
DEFAULT_WIN_H = 740

# Board squares plus the two hand strips, expressed in squares. The strips are
# 0.52 squares tall with a 0.06 square gap on each side of the board.
_HAND_H_RATIO = 0.52
_HAND_GAP_RATIO = 0.06
_VERTICAL_SQUARES = BOARD_SIZE + 2 * _HAND_H_RATIO + 2 * _HAND_GAP_RATIO


def clamp(value, low, high):
    return max(low, min(high, value))


class Layout:
    """Immutable geometry for one window size. Recreate it on resize."""

    def __init__(self, win_w, win_h, analysis_rows=0):
        self.win_w = win_w = max(MIN_WIN_W, int(win_w))
        self.win_h = win_h = max(MIN_WIN_H, int(win_h))

        # Global chrome scale: fonts, paddings, border radii. Independent of the
        # square size so text stays readable on a wide-but-short window.
        self.k = clamp(min(win_w / 1100.0, win_h / 740.0), 0.72, 2.0)

        self.margin = m = self.s(14)
        self.panel_w = int(clamp(win_w * 0.30, self.s(280), self.s(420)))

        # --- Board + hand strips occupy the left column ---
        col_w = win_w - self.panel_w - 2 * m
        col_h = win_h - 2 * m

        self.square = sq = max(24, int(min(col_w / BOARD_SIZE, col_h / _VERTICAL_SQUARES)))
        self.hand_h = hand_h = int(sq * _HAND_H_RATIO)
        self.hand_gap = gap = int(sq * _HAND_GAP_RATIO)

        board_px = sq * BOARD_SIZE
        stack_h = board_px + 2 * hand_h + 2 * gap
        col_x = m + max(0, (col_w - board_px) // 2)
        col_y = m + max(0, (col_h - stack_h) // 2)

        self.hand_top = pygame.Rect(col_x, col_y, board_px, hand_h)
        self.board = pygame.Rect(col_x, col_y + hand_h + gap, board_px, board_px)
        self.hand_bottom = pygame.Rect(col_x, self.board.bottom + gap, board_px, hand_h)

        # --- Side panel, pinned to the right edge ---
        self.panel = pygame.Rect(win_w - self.panel_w, 0, self.panel_w, win_h)
        pad = self.s(14)
        self.panel_pad = pad
        inner_x = self.panel.x + pad
        inner_w = self.panel_w - 2 * pad

        self.btn_h = btn_h = self.s(34)
        self.btn_gap = btn_gap = self.s(7)
        self.row_h = self.s(22)

        # Header: title, turn row, eval row, engine-status row.
        self.eval_h = eval_h = self.s(30)
        header_h = self.s(26) + self.s(34) + eval_h + self.s(30) + 4 * self.s(6)
        self.header = pygame.Rect(inner_x, pad, inner_w, header_h)

        # Controls: 7 button rows. Fixed height, so the move list below can never
        # push them and they can never push it.
        controls_h = 7 * btn_h + 6 * btn_gap
        self.controls = pygame.Rect(inner_x, self.header.bottom + self.s(10), inner_w, controls_h)

        # Analysis band: one row per requested hint line, plus a header. Zero rows
        # collapses it to nothing so the move list gets the pixels back.
        self.analysis_rows = max(0, int(analysis_rows))
        self.analysis_row_h = self.s(22)
        analysis_h = (self.s(22) + self.analysis_rows * self.analysis_row_h + self.s(8)
                      if self.analysis_rows else 0)
        self.analysis = pygame.Rect(inner_x, self.controls.bottom + self.s(10),
                                    inner_w, analysis_h)

        # Toast / status band, pinned to the bottom edge and always reserved even
        # when empty — otherwise the move list would resize as messages come and go.
        self.toast_h = toast_h = self.s(46)
        self.toast = pygame.Rect(inner_x, win_h - pad - toast_h, inner_w, toast_h)

        # Move list absorbs every remaining pixel.
        list_top = (self.analysis.bottom if analysis_h else self.controls.bottom) + self.s(10)
        self.movelist = pygame.Rect(inner_x, list_top, inner_w,
                                    max(self.s(40), self.toast.top - self.s(10) - list_top))
        self.movelist_row_h = self.s(21)

    # --- helpers -----------------------------------------------------------

    def s(self, value):
        """Scale a design-pixel value (authored at k == 1.0) to this window."""
        return max(1, int(round(value * self.k)))

    def font_size(self, value):
        return max(9, int(round(value * self.k)))

    def square_rect(self, r, f, flipped):
        """Screen rect of logical square (r, f). Row 0 is rank 6 (see utils)."""
        sr = BOARD_SIZE - 1 - r if flipped else r
        sf = BOARD_SIZE - 1 - f if flipped else f
        return pygame.Rect(self.board.x + sf * self.square,
                           self.board.y + sr * self.square,
                           self.square, self.square)

    def square_center(self, r, f, flipped):
        return self.square_rect(r, f, flipped).center

    def square_at(self, pos, flipped):
        """Logical (row, file) under a screen point, or None if off-board."""
        if not self.board.collidepoint(pos):
            return None
        sf = (pos[0] - self.board.x) // self.square
        sr = (pos[1] - self.board.y) // self.square
        sf = int(clamp(sf, 0, BOARD_SIZE - 1))
        sr = int(clamp(sr, 0, BOARD_SIZE - 1))
        if flipped:
            return BOARD_SIZE - 1 - sr, BOARD_SIZE - 1 - sf
        return sr, sf

    def hand_strips(self, flipped):
        """[(strip_rect, colour)] — which hand each strip shows.

        Unflipped the board has Black at the top, so the top strip is Black's
        hand. Flipping swaps them, keeping "the strip nearest me is mine" true.
        """
        if flipped:
            return [(self.hand_top, 'w'), (self.hand_bottom, 'b')]
        return [(self.hand_top, 'b'), (self.hand_bottom, 'w')]

    def button_grid(self, row, col, span=1, cols=2):
        """Rect for a control button. Row/col are 0-based inside `self.controls`."""
        gap = self.btn_gap
        total_w = self.controls.w
        cell_w = (total_w - gap * (cols - 1)) / cols
        x = self.controls.x + col * (cell_w + gap)
        w = cell_w * span + gap * (span - 1)
        y = self.controls.y + row * (self.btn_h + gap)
        return pygame.Rect(int(x), int(y), int(w), self.btn_h)

    def stepper_parts(self, row):
        """(whole_row, minus_rect, readout_rect, plus_rect) for a ‹ value › control.

        Both the renderer and the wheel hit-test need these, and they must agree —
        computing them twice is how a control ends up clickable where it is not drawn.
        """
        whole = self.button_grid(row, 0, span=2)
        step = self.btn_h
        minus = pygame.Rect(whole.x, whole.y, step, step)
        plus = pygame.Rect(whole.right - step, whole.y, step, step)
        middle = pygame.Rect(minus.right + self.s(6), whole.y,
                             plus.left - minus.right - self.s(12), step)
        return whole, minus, middle, plus
