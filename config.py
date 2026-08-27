# -*- coding: utf-8 -*-

# --- Game Constants ---
BOARD_SIZE = 6
FPS = 60              # cap while animating; the loop idles far below this
IDLE_FPS = 15         # cap when nothing on screen is changing

# Pixel geometry now lives in layout.Layout, which recomputes it from the current
# window size. SQUARE_SIZE remains only because pieces.py's (unused) vector
# drawing primitives import it; nothing in the live render path reads it.
SQUARE_SIZE = 100

# Minimum time an engine move stays visible before the game moves on. The old
# AI_MOVE_DELAY was a flat 1.5s stall added *after* the search returned and
# *before* the move was drawn, so it made nothing more visible — it only added
# latency. This pads fast moves up to a floor instead, and never blocks input.
MIN_MOVE_DISPLAY = 0.25

# Hints are advisory, so they are capped well below the engine's playing depth —
# a 30s "Thinking…" for a hint you asked for casually is not worth the wait.
HINT_MAX_DEPTH = 6

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
INFO_BG_COLOR = (30, 29, 27)
INFO_TEXT_COLOR = WHITE
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER_COLOR = (100, 100, 100)
BUTTON_TEXT_COLOR = WHITE

# --- UI Colors ---
BOARD_COLORS = {
    'light': (240, 217, 181),
    'dark': (181, 136, 99),
}

PANEL_COLORS = {
    'bg': (30, 29, 27),
    'raised': (44, 42, 40),        # cards: turn row, hand strips, move list
    'raised_alt': (38, 36, 34),    # alternating move-list rows
    'border': (62, 59, 55),
    'text': (226, 224, 220),
    'text_dim': (150, 146, 140),
    'text_faint': (110, 107, 102),
    'accent': (110, 175, 255),
    'good': (108, 196, 122),
    'warn': (232, 196, 92),
    'bad': (232, 96, 92),
}

HIGHLIGHT_COLORS = {
    'selected': (100, 150, 255, 150),
    'legal_move': (0, 0, 0, 90),
    'check': (235, 50, 50, 160),
    'previous_move': (255, 255, 0, 80),
    'move_origin': (200, 200, 0, 60),
    'illegal': (235, 70, 60, 150),      # transient flash on a rejected click
    'drop_target': (110, 200, 255, 52),
    'undo': (150, 82, 52),
    'toggle_ai': (66, 64, 72),
    'toggle_ai_active': (46, 139, 87),
    'trainer': (0, 150, 150),
    'hint': (86, 66, 130),
    'hint_active': (150, 108, 232),
    'hint_from': (80, 200, 255, 140),
    'hint_to': (80, 255, 140, 160),
    'hint_arrow': (80, 200, 255, 200),
    'button': (66, 64, 62),
    'button_hover': (86, 83, 80),
    'button_text': (255, 255, 255),
    'neutral': (60, 58, 56),
    'new_game': (34, 122, 64),
}

# Translucent color for legal moves
POSSIBLE_MOVE_COLOR = (80, 150, 105, 170)

# --- Dead weight, kept only so old imports keep resolving -------------------
# The numbers that actually decide moves are the consts at the top of
# engine_rs/src/eval.rs. Editing anything below changes nothing.
CENTER_SQUARES = [(2, 2), (2, 3), (3, 2), (3, 3)]
CENTER_BONUS = 25
DEVELOPMENT_PENALTY = -10
PAWN_STRUCTURE_BONUS = 15
KING_SAFETY_BONUS = 30
MOBILITY_BONUS = 8
ATTACK_BONUS = 15
OPENING_PHASE = 12
ENDGAME_PHASE = 4

# Background colors for pieces (used by the drawing primitives in pieces.py)
PIECE_BG_COLORS = {
    'K': (220, 220, 255), 'R': (220, 255, 220),
    'B': (255, 255, 220), 'N': (220, 255, 255), 'P': (255, 220, 255),
    'k': (100, 100, 180), 'r': (100, 180, 100),
    'b': (180, 180, 100), 'n': (100, 180, 180), 'p': (180, 100, 180)
}
