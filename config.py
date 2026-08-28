# -*- coding: utf-8 -*-

# --- Game Constants ---
BOARD_SIZE = 6
FPS = 60              # cap while animating; the loop idles far below this
IDLE_FPS = 15         # cap when nothing on screen is changing

# Minimum time an engine move stays visible before the game moves on. The old
# AI_MOVE_DELAY was a flat 1.5s stall added *after* the search returned and
# *before* the move was drawn, so it made nothing more visible — it only added
# latency. This pads fast moves up to a floor instead, and never blocks input.
MIN_MOVE_DISPLAY = 0.25

# HINT_MAX_DEPTH is gone. It silently clamped the hint to 6 whatever the depth control
# said, and with both AI toggles off the hint is the *only* search the GUI ever runs —
# so picking depth 10 changed nothing that ever executed, and the whole depth selector
# looked broken. The hint now searches at the selected depth: "very slow" is already
# written on the control, the search never blocks the UI, and the elapsed time is on
# screen while it runs. Wanting a faster hint is what lowering the depth is for.

# --- Score display ---
# Engine scores are white-relative integers in the same units as eval.rs, where a pawn
# is PIECE_VALUES[0] == 100. Divide by this to print pawns.
PAWN_UNIT = 100

# eval.rs returns a flat ±CHECKMATE_SCORE for a forced mate — it carries no distance,
# so the display can say "mate", never "M3". This is the same 9/10 fraction the search
# uses to decide an iteration found one.
MATE_SCORE_CUTOFF = 900_000

# The eval bar saturates here (±10 pawns). Past that the exact number stops meaning
# anything a bar could show, and the readout beside it still prints the real value.
EVAL_BAR_CLAMP = 1000

# --- UI Colors ---
# Plain black and white, used directly by gui.py for piece glyphs and button text.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

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
