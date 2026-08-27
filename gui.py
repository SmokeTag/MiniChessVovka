# -*- coding: utf-8 -*-
"""All rendering for the Pygame front end.

Every function takes a `layout.Layout` and derives its pixel sizes from it, so the
window is freely resizable. Two caches keep that affordable: fonts and scaled
piece sprites are memoised per pixel size, so a steady-state frame does no
scaling work at all (the old code ran `smoothscale` on every hand sprite on every
one of 30 frames per second).

Drawing never reads UI state off the GameState. Callers pass a `BoardView`
(a plain snapshot: board, hands, turn, last move) plus a `ui` object, which lets
the same code render either the live position or a historical one while the user
browses with the arrow keys.
"""

import math
import os

import pygame

from config import (BOARD_SIZE, BOARD_COLORS, HIGHLIGHT_COLORS, PANEL_COLORS, WHITE, BLACK)
from pieces import EMPTY_SQUARE, PROMOTION_PIECES_WHITE_STR, PROMOTION_PIECES_BLACK_STR, PIECE_TO_SYMBOL
from utils import get_piece_color, piece_to_lower

pygame.init()

# --- Caches -----------------------------------------------------------------

_ORIGINALS = {}      # piece char -> unscaled Surface
_SCALED = {}         # (piece char, px) -> Surface
_FONTS = {}          # (family, px, bold) -> Font

_HAND_ORDER = ['P', 'N', 'B', 'R', 'Q']


def get_font(px, bold=False, family='segoeui'):
    key = (family, px, bold)
    font = _FONTS.get(key)
    if font is None:
        font = pygame.font.SysFont(family, px, bold=bold)
        _FONTS[key] = font
    return font


def piece_image(char, px):
    """Piece sprite scaled to `px`, memoised. None if the sprite never loaded."""
    key = (char, px)
    surface = _SCALED.get(key)
    if surface is None:
        original = _ORIGINALS.get(char)
        if original is None:
            return None
        if len(_SCALED) > 240:      # bound growth during a resize drag
            _SCALED.clear()
        surface = pygame.transform.smoothscale(_fit_square(original, px), (px, px))
        _SCALED[key] = surface
    return surface


def _fit_square(surface, px):
    """Letterbox `surface` into a transparent px-by-px square, preserving aspect."""
    w, h = surface.get_size()
    if w == 0 or h == 0:
        return pygame.Surface((px, px), pygame.SRCALPHA)
    scale = min(px / w, px / h)
    scaled = pygame.transform.smoothscale(surface, (max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = pygame.Surface((px, px), pygame.SRCALPHA)
    canvas.blit(scaled, scaled.get_rect(center=(px // 2, px // 2)))
    return canvas


def _invert_colors(surface):
    """Black sprite -> white sprite, preserving alpha."""
    out = surface.copy()
    out.lock()
    for x in range(out.get_width()):
        for y in range(out.get_height()):
            r, g, b, a = out.get_at((x, y))
            if a > 0:
                out.set_at((x, y), (255 - r, 255 - g, 255 - b, a))
    out.unlock()
    return out


def load_images(image_dir="assets/sprites"):
    """Load piece sprites once at full resolution. Scaling happens in piece_image."""
    _ORIGINALS.clear()
    _SCALED.clear()
    piece_files = {
        'pawn': ('pawn.png', 'p', 'P'),
        'knight': ('horse.png', 'n', 'N'),
        'bishop': ('bishop.png', 'b', 'B'),
        'rook': ('rookie.png', 'r', 'R'),
        'king': ('king.png', 'k', 'K'),
    }
    loaded = 0
    for _name, (filename, black_char, white_char) in piece_files.items():
        path = os.path.join(image_dir, filename)
        if not os.path.exists(path):
            print(f"[gui] sprite missing: {path}")
            continue
        try:
            image = pygame.image.load(path).convert_alpha()
        except pygame.error as exc:
            print(f"[gui] could not load {path}: {exc}")
            continue
        _ORIGINALS[black_char] = image
        _ORIGINALS[white_char] = _invert_colors(image)
        loaded += 1
    if loaded == 0:
        print("[gui] no sprites loaded; falling back to text pieces")
    return True     # text fallback keeps the game playable either way


# --- Small drawing helpers --------------------------------------------------

def _alpha_rect(screen, rect, color):
    surface = pygame.Surface(rect.size, pygame.SRCALPHA)
    surface.fill(color)
    screen.blit(surface, rect.topleft)


def _text(screen, text, font, color, *, center=None, midleft=None, midright=None, topleft=None):
    surface = font.render(text, True, color)
    if center is not None:
        rect = surface.get_rect(center=center)
    elif midleft is not None:
        rect = surface.get_rect(midleft=midleft)
    elif midright is not None:
        rect = surface.get_rect(midright=midright)
    else:
        rect = surface.get_rect(topleft=topleft)
    screen.blit(surface, rect)
    return rect


def _clip_text(text, font, max_w):
    """Truncate with an ellipsis so long labels clip instead of overflowing."""
    if font.size(text)[0] <= max_w:
        return text
    while text and font.size(text + "…")[0] > max_w:
        text = text[:-1]
    return text + "…"


def draw_button(screen, layout, rect, label, *, bg, fg=WHITE, hovered=False,
                enabled=True, badge=None, badge_color=None):
    """One control button. Hover is a flat lightening — no extra frames, no cost."""
    if not enabled:
        bg = tuple(int(c * 0.45) for c in bg[:3])
        fg = PANEL_COLORS['text_faint']
    elif hovered:
        bg = tuple(min(255, int(c * 1.28) + 12) for c in bg[:3])

    radius = layout.s(6)
    pygame.draw.rect(screen, bg, rect, border_radius=radius)
    if enabled and hovered:
        pygame.draw.rect(screen, PANEL_COLORS['text'], rect, 1, border_radius=radius)

    font = get_font(layout.font_size(14), bold=True)
    if badge is not None:
        # Label left, state pill right — reads faster than "AI White: ON".
        pad = layout.s(8)
        pill_font = get_font(layout.font_size(12), bold=True)
        pill_w = pill_font.size(badge)[0] + layout.s(12)
        pill = pygame.Rect(rect.right - pad - pill_w, rect.centery - layout.s(9), pill_w, layout.s(18))
        _text(screen, _clip_text(label, font, pill.left - rect.left - 2 * pad), font, fg,
              midleft=(rect.x + pad, rect.centery))
        pygame.draw.rect(screen, badge_color or PANEL_COLORS['border'], pill, border_radius=layout.s(9))
        _text(screen, badge, pill_font, WHITE, center=pill.center)
    else:
        _text(screen, _clip_text(label, font, rect.w - layout.s(12)), font, fg, center=rect.center)
    return rect


# --- Board ------------------------------------------------------------------

def draw_board(screen, layout, flipped):
    """Checkerboard plus edge coordinates."""
    coord_font = get_font(layout.font_size(12), bold=True, family='consolas')
    inset = layout.s(3)
    for r in range(BOARD_SIZE):
        for f in range(BOARD_SIZE):
            light = (r + f) % 2 == 0
            rect = layout.square_rect(r, f, flipped)
            pygame.draw.rect(screen, BOARD_COLORS['light' if light else 'dark'], rect)

            # Ranks label the left file on screen, files label the bottom rank.
            screen_col = BOARD_SIZE - 1 - f if flipped else f
            screen_row = BOARD_SIZE - 1 - r if flipped else r
            label_color = BOARD_COLORS['dark' if light else 'light']
            if screen_col == 0:
                _text(screen, str(BOARD_SIZE - r), coord_font, label_color,
                      topleft=(rect.x + inset, rect.y + inset))
            if screen_row == BOARD_SIZE - 1:
                glyph = chr(ord('a') + f)
                surface = coord_font.render(glyph, True, label_color)
                screen.blit(surface, (rect.right - surface.get_width() - inset,
                                      rect.bottom - surface.get_height() - inset))

    pygame.draw.rect(screen, PANEL_COLORS['border'], layout.board, layout.s(2))


def draw_pieces(screen, layout, board, flipped, skip=None):
    """Blit pieces. `skip` omits one square — the piece being dragged."""
    px = int(layout.square * 0.9)
    offset = (layout.square - px) // 2
    fallback = get_font(layout.font_size(44), bold=True)
    for r in range(BOARD_SIZE):
        for f in range(BOARD_SIZE):
            piece = board[r][f]
            if piece == EMPTY_SQUARE or (skip is not None and (r, f) == skip):
                continue
            rect = layout.square_rect(r, f, flipped)
            image = piece_image(piece, px)
            if image:
                screen.blit(image, (rect.x + offset, rect.y + offset))
            else:
                color = WHITE if get_piece_color(piece) == 'w' else BLACK
                _text(screen, PIECE_TO_SYMBOL.get(piece, piece), fallback, color, center=rect.center)


def draw_board_overlays(screen, layout, view, ui):
    """Last move, check, selection, legal targets, illegal-click flashes."""
    flipped = ui.flipped

    if view.last_move:
        if view.last_move[0] == 'drop':
            _alpha_rect(screen, layout.square_rect(*view.last_move[2], flipped),
                        HIGHLIGHT_COLORS['previous_move'])
        else:
            _alpha_rect(screen, layout.square_rect(*view.last_move[0], flipped),
                        HIGHLIGHT_COLORS['move_origin'])
            _alpha_rect(screen, layout.square_rect(*view.last_move[1], flipped),
                        HIGHLIGHT_COLORS['previous_move'])

    if view.check_square:
        _alpha_rect(screen, layout.square_rect(*view.check_square, flipped), HIGHLIGHT_COLORS['check'])

    if ui.selected_square:
        _alpha_rect(screen, layout.square_rect(*ui.selected_square, flipped), HIGHLIGHT_COLORS['selected'])

    # Move targets: a dot for a quiet move, a ring for a capture.
    for target in ui.move_targets:
        rect = layout.square_rect(*target, flipped)
        occupied = view.board[target[0]][target[1]] != EMPTY_SQUARE
        surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        if occupied:
            pygame.draw.circle(surface, (0, 0, 0, 70), (rect.w // 2, rect.h // 2),
                               rect.w // 2 - layout.s(3), layout.s(5))
        else:
            pygame.draw.circle(surface, HIGHLIGHT_COLORS['legal_move'],
                               (rect.w // 2, rect.h // 2), max(3, rect.w // 7))
        screen.blit(surface, rect.topleft)

    # Drop targets get a tinted square rather than a dot: dropping is a different
    # action from moving and should not look the same.
    for target in ui.drop_targets:
        _alpha_rect(screen, layout.square_rect(*target, flipped), HIGHLIGHT_COLORS['drop_target'])

    if ui.hover_square and ui.hover_square in (set(ui.move_targets) | set(ui.drop_targets)):
        pygame.draw.rect(screen, WHITE, layout.square_rect(*ui.hover_square, flipped), layout.s(3))

    for square, strength in ui.flashes:
        color = HIGHLIGHT_COLORS['illegal']
        _alpha_rect(screen, layout.square_rect(*square, flipped),
                    (color[0], color[1], color[2], int(color[3] * strength)))


def draw_hint(screen, layout, hint_move, flipped):
    if not hint_move:
        return
    if hint_move[0] == 'drop':
        rect = layout.square_rect(*hint_move[2], flipped)
        _alpha_rect(screen, rect, HIGHLIGHT_COLORS['hint_to'])
        pygame.draw.rect(screen, HIGHLIGHT_COLORS['hint_arrow'][:3], rect, layout.s(3))
        return

    _alpha_rect(screen, layout.square_rect(*hint_move[0], flipped), HIGHLIGHT_COLORS['hint_from'])
    _alpha_rect(screen, layout.square_rect(*hint_move[1], flipped), HIGHLIGHT_COLORS['hint_to'])

    sx, sy = layout.square_center(*hint_move[0], flipped)
    ex, ey = layout.square_center(*hint_move[1], flipped)
    dx, dy = ex - sx, ey - sy
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    trim = layout.square * 0.3
    sx, sy = sx + ux * trim, sy + uy * trim
    ex, ey = ex - ux * trim, ey - uy * trim

    color = HIGHLIGHT_COLORS['hint_arrow'][:3]
    pygame.draw.line(screen, color, (int(sx), int(sy)), (int(ex), int(ey)), layout.s(6))
    head = layout.s(15)
    angle = math.atan2(dy, dx)
    pygame.draw.polygon(screen, color, [
        (int(ex), int(ey)),
        (int(ex - head * math.cos(angle - 0.5)), int(ey - head * math.sin(angle - 0.5))),
        (int(ex - head * math.cos(angle + 0.5)), int(ey - head * math.sin(angle + 0.5))),
    ])


def draw_drag(screen, layout, ui):
    """The dragged piece follows the cursor, centred under it."""
    if not ui.drag:
        return
    px = int(layout.square * 0.9)
    image = piece_image(ui.drag['piece'], px)
    if image:
        screen.blit(image, image.get_rect(center=ui.mouse_pos))
    else:
        _text(screen, PIECE_TO_SYMBOL.get(ui.drag['piece'], ui.drag['piece']),
              get_font(layout.font_size(44), bold=True),
              WHITE if get_piece_color(ui.drag['piece']) == 'w' else BLACK, center=ui.mouse_pos)


# --- Hands (drop inventory, beside the board) --------------------------------

def draw_hands(screen, layout, view, ui, hits):
    """Draw both hand strips and register their click targets.

    Hands sit next to the board rather than in the side panel for two reasons:
    they are game state you read while reading the board, and keeping them out of
    the panel is what lets every panel zone hold a fixed position.
    """
    label_font = get_font(layout.font_size(12), bold=True)
    count_font = get_font(layout.font_size(13), bold=True)
    cell = layout.hand_h
    sprite_px = int(cell * 0.80)

    for strip, color in layout.hand_strips(ui.flipped):
        is_turn = view.turn == color and ui.interactive
        pygame.draw.rect(screen, PANEL_COLORS['raised'], strip, border_radius=layout.s(6))
        pygame.draw.rect(screen,
                         PANEL_COLORS['accent'] if is_turn else PANEL_COLORS['border'],
                         strip, layout.s(2) if is_turn else 1, border_radius=layout.s(6))

        label = "White" if color == 'w' else "Black"
        label_rect = _text(screen, label, label_font,
                           PANEL_COLORS['text'] if is_turn else PANEL_COLORS['text_faint'],
                           midleft=(strip.x + layout.s(10), strip.centery))

        x = label_rect.right + layout.s(12)
        hand = view.hands.get(color, {})
        drawn = 0
        for piece_type in _HAND_ORDER:
            count = hand.get(piece_type, 0)
            if count <= 0:
                continue
            drawn += 1
            char = piece_type.upper() if color == 'w' else piece_to_lower(piece_type)
            rect = pygame.Rect(x, strip.y + (strip.h - cell) // 2, cell, cell)
            if rect.right > strip.right - layout.s(8):
                break

            selected = ui.selected_drop == (color + piece_type) and is_turn
            hovered = is_turn and rect.collidepoint(ui.mouse_pos) and not ui.drag

            if selected:
                pygame.draw.rect(screen, PANEL_COLORS['accent'], rect.inflate(layout.s(6), layout.s(6)),
                                 border_radius=layout.s(5))
            elif hovered:
                pygame.draw.rect(screen, HIGHLIGHT_COLORS['button_hover'],
                                 rect.inflate(layout.s(6), layout.s(6)), border_radius=layout.s(5))

            dragging_this = ui.drag and ui.drag.get('from_hand') == (color + piece_type)
            if not dragging_this:
                image = piece_image(char, sprite_px)
                if image:
                    screen.blit(image, image.get_rect(center=rect.center))
                else:
                    _text(screen, PIECE_TO_SYMBOL.get(char, char), count_font,
                          WHITE if color == 'w' else BLACK, center=rect.center)

            # Count as a numeral, not repeated sprites: a full hand used to wrap
            # onto extra rows and shove every control below it down the panel.
            if count > 1:
                badge = f"×{count}"
                bw, bh = count_font.size(badge)
                badge_rect = pygame.Rect(rect.right - bw - layout.s(3),
                                         rect.bottom - bh, bw + layout.s(3), bh)
                pygame.draw.rect(screen, PANEL_COLORS['bg'], badge_rect, border_radius=layout.s(3))
                _text(screen, badge, count_font, PANEL_COLORS['text'], center=badge_rect.center)

            if is_turn:
                hits['hand'][color + piece_type] = rect
            x = rect.right + layout.s(10)

        if drawn == 0:
            _text(screen, "empty", get_font(layout.font_size(12)), PANEL_COLORS['text_faint'],
                  midleft=(x, strip.centery))


# --- Side panel --------------------------------------------------------------

def _draw_spinner(screen, layout, center, radius, phase, color):
    box = pygame.Rect(0, 0, radius * 2, radius * 2)
    box.center = center
    start = phase % (2 * math.pi)
    pygame.draw.arc(screen, color, box, start, start + 1.9, max(2, layout.s(3)))


def _draw_header(screen, layout, view, ui):
    panel_x, width = layout.header.x, layout.header.w
    y = layout.header.y

    _text(screen, "MINI CRAZYHOUSE 6×6", get_font(layout.font_size(12), bold=True),
          PANEL_COLORS['text_faint'], midleft=(panel_x, y + layout.s(13)))
    y += layout.s(26) + layout.s(6)

    # Turn card
    card = pygame.Rect(panel_x, y, width, layout.s(34))
    pygame.draw.rect(screen, PANEL_COLORS['raised'], card, border_radius=layout.s(6))
    dot_r = layout.s(6)
    dot_x = card.x + layout.s(14)
    pygame.draw.circle(screen, (245, 245, 245) if view.turn == 'w' else (26, 26, 26),
                       (dot_x, card.centery), dot_r)
    pygame.draw.circle(screen, PANEL_COLORS['text_faint'], (dot_x, card.centery), dot_r, 1)

    side = "White" if view.turn == 'w' else "Black"
    who = ui.player_label(view.turn)
    _text(screen, f"{side} to move", get_font(layout.font_size(15), bold=True), PANEL_COLORS['text'],
          midleft=(dot_x + dot_r + layout.s(9), card.centery))
    _text(screen, who, get_font(layout.font_size(12)), PANEL_COLORS['text_dim'],
          midright=(card.right - layout.s(12), card.centery))
    y = card.bottom + layout.s(6)

    # Engine row — the fix for "the app looks frozen". Always says what the
    # engine is doing and, once past a second, how long it has been doing it.
    row = pygame.Rect(panel_x, y, width, layout.s(30))
    if ui.thinking:
        pygame.draw.rect(screen, (46, 52, 64), row, border_radius=layout.s(6))
        _draw_spinner(screen, layout, (row.x + layout.s(16), row.centery), layout.s(8),
                      ui.anim_phase * 3.0, PANEL_COLORS['accent'])
        elapsed = ui.think_elapsed
        text = f"Thinking… {elapsed:0.1f}s" if elapsed >= 1.0 else "Thinking…"
        _text(screen, text, get_font(layout.font_size(14), bold=True), PANEL_COLORS['accent'],
              midleft=(row.x + layout.s(32), row.centery))
        _text(screen, f"depth {ui.think_depth}", get_font(layout.font_size(12)),
              PANEL_COLORS['text_dim'], midright=(row.right - layout.s(10), row.centery))
    elif ui.hint_pending:
        pygame.draw.rect(screen, (52, 44, 66), row, border_radius=layout.s(6))
        _draw_spinner(screen, layout, (row.x + layout.s(16), row.centery), layout.s(8),
                      ui.anim_phase * 3.0, HIGHLIGHT_COLORS['hint_active'])
        _text(screen, "Finding a hint…", get_font(layout.font_size(14)),
              HIGHLIGHT_COLORS['hint_active'], midleft=(row.x + layout.s(32), row.centery))
    elif ui.show_hint and ui.hint_move:
        pygame.draw.rect(screen, (52, 44, 66), row, border_radius=layout.s(6))
        _text(screen, f"Hint: {ui.hint_text}", get_font(layout.font_size(14), bold=True),
              HIGHLIGHT_COLORS['hint_active'], midleft=(row.x + layout.s(12), row.centery))
    else:
        pygame.draw.circle(screen, PANEL_COLORS['text_faint'],
                           (row.x + layout.s(6), row.centery), layout.s(3))
        _text(screen, ui.idle_status, get_font(layout.font_size(13)), PANEL_COLORS['text_dim'],
              midleft=(row.x + layout.s(16), row.centery))


def _draw_controls(screen, layout, ui, hits):
    mouse = ui.mouse_pos

    def button(name, rect, label, bg, *, enabled=True, badge=None, badge_color=None):
        draw_button(screen, layout, rect, label, bg=bg,
                    hovered=rect.collidepoint(mouse), enabled=enabled,
                    badge=badge, badge_color=badge_color)
        if enabled:
            hits['buttons'][name] = rect

    button('undo', layout.button_grid(0, 0), "Undo", HIGHLIGHT_COLORS['undo'],
           enabled=ui.can_undo)
    button('new_game', layout.button_grid(0, 1), "New game", HIGHLIGHT_COLORS['new_game'])

    on, off = HIGHLIGHT_COLORS['toggle_ai_active'], HIGHLIGHT_COLORS['toggle_ai']
    button('toggle_white_ai', layout.button_grid(1, 0), "AI White",
           on if ui.ai_white else off, badge="ON" if ui.ai_white else "OFF",
           badge_color=(24, 84, 48) if ui.ai_white else (54, 52, 58))
    button('toggle_black_ai', layout.button_grid(1, 1), "AI Black",
           on if ui.ai_black else off, badge="ON" if ui.ai_black else "OFF",
           badge_color=(24, 84, 48) if ui.ai_black else (54, 52, 58))

    button('toggle_hint', layout.button_grid(2, 0), "Hint",
           HIGHLIGHT_COLORS['hint_active'] if ui.show_hint else HIGHLIGHT_COLORS['hint'],
           badge="ON" if ui.show_hint else "OFF",
           badge_color=(74, 48, 120) if ui.show_hint else (54, 52, 58))
    button('toggle_flip', layout.button_grid(2, 1), "Flip board", HIGHLIGHT_COLORS['neutral'])

    # Depth stepper: two square steps around a wide readout.
    row = layout.button_grid(3, 0, span=2)
    step = layout.btn_h
    minus = pygame.Rect(row.x, row.y, step, step)
    plus = pygame.Rect(row.right - step, row.y, step, step)
    middle = pygame.Rect(minus.right + layout.s(6), row.y,
                         plus.left - minus.right - layout.s(12), step)

    button('depth_down', minus, "‹", HIGHLIGHT_COLORS['neutral'], enabled=ui.can_depth_down)
    button('depth_up', plus, "›", HIGHLIGHT_COLORS['neutral'], enabled=ui.can_depth_up)
    pygame.draw.rect(screen, PANEL_COLORS['raised'], middle, border_radius=layout.s(6))
    _text(screen, f"Depth {ui.depth}", get_font(layout.font_size(14), bold=True),
          PANEL_COLORS['text'], midleft=(middle.x + layout.s(10), middle.centery))
    _text(screen, ui.depth_label, get_font(layout.font_size(12)), PANEL_COLORS['text_dim'],
          midright=(middle.right - layout.s(10), middle.centery))


def _draw_movelist(screen, layout, ui, hits):
    area = layout.movelist
    pygame.draw.rect(screen, PANEL_COLORS['raised_alt'], area, border_radius=layout.s(6))
    pygame.draw.rect(screen, PANEL_COLORS['border'], area, 1, border_radius=layout.s(6))

    head_h = layout.s(24)
    _text(screen, "MOVES", get_font(layout.font_size(11), bold=True), PANEL_COLORS['text_faint'],
          midleft=(area.x + layout.s(10), area.y + head_h // 2))
    if ui.view_ply != ui.live_ply:
        _text(screen, f"{ui.view_ply}/{ui.live_ply}", get_font(layout.font_size(11), bold=True),
              PANEL_COLORS['warn'], midright=(area.right - layout.s(10), area.y + head_h // 2))

    body = pygame.Rect(area.x + layout.s(4), area.y + head_h,
                       area.w - layout.s(8), area.h - head_h - layout.s(4))
    if body.h <= 0:
        return

    row_h = layout.movelist_row_h
    rows = max(1, body.h // row_h)
    pairs = [(i, ui.history[i], ui.history[i + 1] if i + 1 < len(ui.history) else None)
             for i in range(0, len(ui.history), 2)]

    first = max(0, min(ui.movelist_scroll, max(0, len(pairs) - rows)))
    ui.movelist_scroll = first
    visible = pairs[first:first + rows]

    num_font = get_font(layout.font_size(12))
    move_font = get_font(layout.font_size(13), bold=True)
    num_w = layout.s(28)
    cell_w = (body.w - num_w - layout.s(8)) // 2

    previous = pygame.Rect(0, 0, 0, 0)
    screen.set_clip(body)
    for index, (ply_index, white_move, black_move) in enumerate(visible):
        y = body.y + index * row_h
        _text(screen, f"{ply_index // 2 + 1}.", num_font, PANEL_COLORS['text_faint'],
              midleft=(body.x + layout.s(6), y + row_h // 2))
        for half, move in ((0, white_move), (1, black_move)):
            if move is None:
                continue
            ply = ply_index + half + 1      # ply N == the position after N moves
            cell = pygame.Rect(body.x + num_w + half * cell_w, y, cell_w, row_h)
            if ply == ui.view_ply:
                pygame.draw.rect(screen, PANEL_COLORS['accent'], cell, border_radius=layout.s(3))
                color = (18, 20, 24)
                previous = cell
            elif cell.collidepoint(ui.mouse_pos):
                pygame.draw.rect(screen, HIGHLIGHT_COLORS['button_hover'], cell,
                                 border_radius=layout.s(3))
                color = PANEL_COLORS['text']
            else:
                color = PANEL_COLORS['text']
            _text(screen, _clip_text(move, move_font, cell.w - layout.s(8)), move_font, color,
                  midleft=(cell.x + layout.s(5), cell.centery))
            hits['movelist'][ply] = cell
    screen.set_clip(None)

    if not ui.history:
        _text(screen, "no moves yet", get_font(layout.font_size(12)), PANEL_COLORS['text_faint'],
              center=(body.centerx, body.y + row_h))
    elif len(pairs) > rows:
        # Scroll indicator: proportional thumb on the right edge.
        track = pygame.Rect(area.right - layout.s(6), body.y, layout.s(3), body.h)
        pygame.draw.rect(screen, PANEL_COLORS['border'], track, border_radius=layout.s(2))
        thumb_h = max(layout.s(16), int(track.h * rows / len(pairs)))
        thumb_y = track.y + int((track.h - thumb_h) * first / max(1, len(pairs) - rows))
        pygame.draw.rect(screen, PANEL_COLORS['text_faint'],
                         pygame.Rect(track.x, thumb_y, track.w, thumb_h), border_radius=layout.s(2))
    _ = previous


def _draw_toast(screen, layout, ui):
    """Bottom band. Height is reserved even when empty so nothing above shifts."""
    area = layout.toast
    message, color = ui.status_message()
    if not message:
        return
    pygame.draw.rect(screen, PANEL_COLORS['raised'], area, border_radius=layout.s(6))
    pygame.draw.rect(screen, color, pygame.Rect(area.x, area.y, layout.s(3), area.h),
                     border_radius=layout.s(2))

    font = get_font(layout.font_size(13), bold=True)
    small = get_font(layout.font_size(12))
    inner_w = area.w - layout.s(20)
    lines = message.split("\n")
    _text(screen, _clip_text(lines[0], font, inner_w), font, color,
          midleft=(area.x + layout.s(12), area.y + area.h // 3))
    if len(lines) > 1:
        _text(screen, _clip_text(lines[1], small, inner_w), small, PANEL_COLORS['text_dim'],
              midleft=(area.x + layout.s(12), area.y + 2 * area.h // 3))


def draw_panel(screen, layout, view, ui, hits):
    pygame.draw.rect(screen, PANEL_COLORS['bg'], layout.panel)
    pygame.draw.line(screen, PANEL_COLORS['border'],
                     (layout.panel.x, 0), (layout.panel.x, layout.win_h), 1)
    _draw_header(screen, layout, view, ui)
    _draw_controls(screen, layout, ui, hits)
    _draw_movelist(screen, layout, ui, hits)
    _draw_toast(screen, layout, ui)


# --- Overlays ----------------------------------------------------------------

def draw_promotion(screen, layout, view, hits):
    """Modal promotion picker.

    The caller must hit-test `hits['promotion']` *before* the panel buttons and
    ignore everything else while this is up. Previously the panel was tested
    first, so a dimmed, apparently-disabled "Undo" underneath the overlay was
    still live and won the click.
    """
    overlay = pygame.Surface((layout.win_w, layout.win_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 190))
    screen.blit(overlay, (0, 0))

    color = view.turn
    choices = PROMOTION_PIECES_WHITE_STR if color == 'w' else PROMOTION_PIECES_BLACK_STR
    cell = int(layout.square * 1.05)
    gap = layout.s(10)
    box_w = len(choices) * cell + (len(choices) + 1) * gap
    box_h = cell + layout.s(98)
    box = pygame.Rect(0, 0, box_w, box_h)
    box.center = layout.board.center

    pygame.draw.rect(screen, PANEL_COLORS['raised'], box, border_radius=layout.s(10))
    pygame.draw.rect(screen, PANEL_COLORS['accent'], box, layout.s(2), border_radius=layout.s(10))

    side = "White" if color == 'w' else "Black"
    _text(screen, f"{side} — promote to", get_font(layout.font_size(17), bold=True),
          PANEL_COLORS['text'], center=(box.centerx, box.y + layout.s(24)))

    sprite_px = int(cell * 0.86)
    key_font = get_font(layout.font_size(12), bold=True)
    mouse = pygame.mouse.get_pos()
    for index, upper in enumerate(choices):
        char = upper if color == 'w' else piece_to_lower(upper)
        rect = pygame.Rect(box.x + gap + index * (cell + gap), box.y + layout.s(42), cell, cell)
        hovered = rect.collidepoint(mouse)
        pygame.draw.rect(screen, HIGHLIGHT_COLORS['button_hover'] if hovered else PANEL_COLORS['raised_alt'],
                         rect, border_radius=layout.s(6))
        pygame.draw.rect(screen, PANEL_COLORS['accent'] if hovered else PANEL_COLORS['border'],
                         rect, layout.s(2) if hovered else 1, border_radius=layout.s(6))
        image = piece_image(char, sprite_px)
        if image:
            screen.blit(image, image.get_rect(center=rect.center))
        else:
            _text(screen, PIECE_TO_SYMBOL.get(char, char), get_font(layout.font_size(40), bold=True),
                  WHITE if color == 'w' else BLACK, center=rect.center)
        _text(screen, upper, key_font, PANEL_COLORS['text_dim'],
              center=(rect.centerx, rect.bottom + layout.s(13)))
        hits['promotion'][char] = rect

    _text(screen, "click, or press R / N / B", get_font(layout.font_size(12)),
          PANEL_COLORS['text_faint'], center=(box.centerx, box.bottom - layout.s(15)))


def draw_game_over(screen, layout, view, ui, hits):
    """Banner across the board. The old build only whispered this in the panel."""
    banner_h = layout.s(96)
    banner = pygame.Rect(layout.board.x, layout.board.centery - banner_h // 2,
                         layout.board.w, banner_h)
    shade = pygame.Surface(banner.size, pygame.SRCALPHA)
    shade.fill((16, 16, 18, 232))
    screen.blit(shade, banner.topleft)
    pygame.draw.rect(screen, view.result_color, banner, layout.s(2))

    _text(screen, view.result_title, get_font(layout.font_size(26), bold=True),
          view.result_color, center=(banner.centerx, banner.y + layout.s(30)))
    _text(screen, view.result_detail, get_font(layout.font_size(14)), PANEL_COLORS['text_dim'],
          center=(banner.centerx, banner.y + layout.s(56)))

    button = pygame.Rect(0, 0, layout.s(150), layout.s(32))
    button.center = (banner.centerx, banner.bottom - layout.s(20))
    draw_button(screen, layout, button, "New game", bg=HIGHLIGHT_COLORS['new_game'],
                hovered=button.collidepoint(ui.mouse_pos))
    hits['buttons']['new_game'] = button


# --- Frame -------------------------------------------------------------------

def draw_frame(screen, layout, view, ui):
    """Render one frame. Returns the hit-test map for this frame's geometry.

    Hit regions are produced by the same code that draws them, so a control can
    never be clickable somewhere it is not visible.
    """
    hits = {'buttons': {}, 'hand': {}, 'promotion': {}, 'movelist': {}}
    screen.fill(PANEL_COLORS['bg'])

    draw_board(screen, layout, ui.flipped)
    if ui.show_hint and ui.hint_move and not ui.browsing:
        draw_hint(screen, layout, ui.hint_move, ui.flipped)
    draw_board_overlays(screen, layout, view, ui)
    draw_pieces(screen, layout, view.board, ui.flipped,
                skip=ui.drag['origin'] if ui.drag and ui.drag.get('origin') else None)
    draw_hands(screen, layout, view, ui, hits)
    draw_panel(screen, layout, view, ui, hits)

    if view.result_title and not view.needs_promotion:
        draw_game_over(screen, layout, view, ui, hits)
    if view.needs_promotion:
        # Modal: drop every hit region collected so far so nothing underneath is
        # clickable while the picker is up.
        hits = {'buttons': {}, 'hand': {}, 'promotion': {}, 'movelist': {}}
        draw_promotion(screen, layout, view, hits)

    draw_drag(screen, layout, ui)
    return hits
