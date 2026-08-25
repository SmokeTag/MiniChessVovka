# -*- coding: utf-8 -*-

# --- Game Constants ---
BOARD_SIZE = 6
SQUARE_SIZE = 100  # Larger square size, for looks
TOTAL_WIDTH = BOARD_SIZE * SQUARE_SIZE  # Total board width
SIDE_PANEL_WIDTH = 300  # Side panel
INFO_HEIGHT = 0  # Unused (everything lives in the side panel)
WIDTH = TOTAL_WIDTH + SIDE_PANEL_WIDTH  # Total window width
HEIGHT = TOTAL_WIDTH  # Height = board
FPS = 30
AI_MOVE_DELAY = 1.5  # Delay before the AI moves (seconds) — so the moves are visible

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
INFO_BG_COLOR = (38, 36, 33)
INFO_TEXT_COLOR = WHITE
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER_COLOR = (100, 100, 100)
BUTTON_TEXT_COLOR = WHITE

# --- UI Colors ---
BOARD_COLORS = {
    'light': (240, 217, 181),
    'dark': (181, 136, 99)
}
HIGHLIGHT_COLORS = {
    'selected': (100, 150, 255, 150),
    'legal_move': (0, 0, 0, 60),
    'check': (235, 50, 50, 160),
    'previous_move': (255, 255, 0, 80),
    'move_origin': (200, 200, 0, 60),
    'undo': (180, 80, 40),
    'toggle_ai': (70, 70, 80),
    'toggle_ai_active': (46, 139, 87),
    'trainer': (0, 150, 150),
    'hint': (130, 80, 220),
    'hint_active': (170, 120, 255),
    'hint_from': (80, 200, 255, 140),
    'hint_to': (80, 255, 140, 160),
    'hint_arrow': (80, 200, 255, 200),
    'button': (80, 140, 200),
    'button_hover': (100, 160, 220),
    'button_active': (120, 180, 240),
    'button_text': (255, 255, 255),
    'undo_hover': (170, 140, 240),
    'undo_active': (190, 160, 255),
    'toggle_ai_hover': (140, 220, 170)
}

# Translucent color for legal moves
POSSIBLE_MOVE_COLOR = (80, 150, 105, 170)  # Darker green for legal moves

# Positional bonuses/penalties (for the AI)
CENTER_SQUARES = [(2, 2), (2, 3), (3, 2), (3, 3)] # c4, d4, c3, d3
CENTER_BONUS = 25 # Increased bonus for center control
DEVELOPMENT_PENALTY = -10 # Penalty for B/N still on their starting squares after a few moves? (tricky) - Not directly used in evaluate_position currently
PAWN_STRUCTURE_BONUS = 15 # Increased bonus for good pawn structures
KING_SAFETY_BONUS = 30 # Increased bonus for king safety
MOBILITY_BONUS = 8  # Bonus for mobility (number of moves)
ATTACK_BONUS = 15  # Bonus for attacks/threats - Not directly used in evaluate_position currently

# Game phase affects position evaluation (for the AI)
OPENING_PHASE = 12  # Starting number of pieces (excluding pawns) on the board
ENDGAME_PHASE = 4   # Transition into the endgame

# Background colors for pieces (used by the drawing primitives in pieces.py)
PIECE_BG_COLORS = {
    'K': (220, 220, 255), 'R': (220, 255, 220),
    'B': (255, 255, 220), 'N': (220, 255, 255), 'P': (255, 220, 255),
    'k': (100, 100, 180), 'r': (100, 180, 100),
    'b': (180, 180, 100), 'n': (100, 180, 180), 'p': (180, 100, 180)
}