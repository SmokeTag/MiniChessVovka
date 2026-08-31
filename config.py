
BOARD_SIZE = 6
FPS = 60
IDLE_FPS = 15

MIN_MOVE_DISPLAY = 0.25

PAWN_UNIT = 100

MATE_SCORE_CUTOFF = 900_000

EVAL_BAR_CLAMP = 1000

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

BOARD_COLORS = {
    'light': (240, 217, 181),
    'dark': (181, 136, 99),
}

PANEL_COLORS = {
    'bg': (30, 29, 27),
    'raised': (44, 42, 40),
    'raised_alt': (38, 36, 34),
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
    'illegal': (235, 70, 60, 150),
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
