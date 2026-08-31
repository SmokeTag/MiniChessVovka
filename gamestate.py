import copy
from config import BOARD_SIZE
from pieces import (EMPTY_SQUARE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
                    PROMOTION_PIECES_WHITE_STR, PROMOTION_PIECES_BLACK_STR,
                    KNIGHT_MOVES, DIAGONAL_MOVES, STRAIGHT_MOVES, KING_MOVES)
from utils import (get_piece_color, is_on_board, get_opposite_color,
                   piece_to_lower, piece_to_upper)

class GameState:
    """Represents the state of a game."""
    def __init__(self):
        """Initialize a new game."""
        self.board = [[EMPTY_SQUARE for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_turn = 'w'
        self.hands = {'w': {}, 'b': {}}
        self.king_pos = {'w': None, 'b': None}
        self.checkmate = False
        self.stalemate = False
        self.last_move = None
        self.move_log = []
        self.game_over_message = ""
        self.saved_states = []
        self.selected_square = None
        self.selected_drop_piece = None
        self.highlighted_moves = []
        self.needs_promotion_choice = False
        self.promotion_square = None
        self.last_move_for_promotion = None
        self.white_ai_enabled = False
        self.black_ai_enabled = False
        self.ai_depth = 10
        self.show_hint = False
        self._all_legal_moves_cache = None
        self._is_check_cache = None
        self._hash_cache = None
        self.ai_history = []
        self.promoted_pieces = set()
        self.is_draw = False
        self.ply_limit = 200
        self.ply_count = 0
        self.position_history = []
        self.position_counts = {}
        self._push_position()

    def _position_key(self):
        """Immutable identity of the current position for repetition detection.

        Identity = (board, side to move, both hands, set of promoted squares).
        Hand entries with a zero count are dropped so that an explicit ``'Q': 0``
        key is indistinguishable from the piece simply being absent.
        """
        board_key = tuple(tuple(row) for row in self.board)
        white_hand = tuple(sorted((p, c) for p, c in self.hands.get('w', {}).items() if c > 0))
        black_hand = tuple(sorted((p, c) for p, c in self.hands.get('b', {}).items() if c > 0))
        return (board_key, self.current_turn, white_hand, black_hand,
                frozenset(self.promoted_pieces))

    def _push_position(self):
        """Records the current position as having occurred."""
        key = self._position_key()
        self.position_history.append(key)
        self.position_counts[key] = self.position_counts.get(key, 0) + 1
        return key

    def _pop_position(self):
        """Removes the most recently recorded position."""
        if not self.position_history:
            return None
        key = self.position_history.pop()
        remaining = self.position_counts.get(key, 0) - 1
        if remaining > 0:
            self.position_counts[key] = remaining
        else:
            self.position_counts.pop(key, None)
        return key

    def _truncate_position_history(self, length):
        """Rewinds the position history to exactly `length` entries."""
        while len(self.position_history) > length:
            self._pop_position()

    def _reset_position_history(self):
        """Clears history/ply counter and records the current position as the first one."""
        self.position_history = []
        self.position_counts = {}
        self.ply_count = 0
        self.is_draw = False
        self._push_position()

    def save_state(self):
        """Saves the current game state so a move can be undone."""
        state = {
            'board': copy.deepcopy(self.board),
            'hands': copy.deepcopy(self.hands),
            'current_turn': self.current_turn,
            'king_pos': copy.deepcopy(self.king_pos),
            'checkmate': self.checkmate,
            'stalemate': self.stalemate,
            'last_move': self.last_move,
            'game_over_message': self.game_over_message,
            'needs_promotion_choice': self.needs_promotion_choice,
            'promotion_square': self.promotion_square,
            'promoted_pieces': set(self.promoted_pieces),
            'is_draw': self.is_draw,
            'ply_count': self.ply_count,
            'history_len': len(self.position_history)
        }
        self.saved_states.append(state)

    def undo_move(self):
        """Undoes the last move."""
        if len(self.saved_states) <= 1:
            print("Cannot undo further.")
            return False

        self.saved_states.pop()
        if not self.saved_states:
             print("Error: No previous state to restore.")
             self.setup_initial_board()
             self.save_state()
             return False

        prev_state = self.saved_states[-1]
        self.board = copy.deepcopy(prev_state['board'])
        self.hands = copy.deepcopy(prev_state['hands'])
        self.current_turn = prev_state['current_turn']
        self.king_pos = copy.deepcopy(prev_state['king_pos'])
        self.checkmate = prev_state['checkmate']
        self.stalemate = prev_state['stalemate']
        self.last_move = prev_state.get('last_move')
        self.game_over_message = prev_state['game_over_message']
        self.needs_promotion_choice = prev_state['needs_promotion_choice']
        self.promotion_square = prev_state['promotion_square']
        self.promoted_pieces = set(prev_state.get('promoted_pieces', set()))
        self.is_draw = prev_state.get('is_draw', False)
        self.ply_count = prev_state.get('ply_count', self.ply_count)
        self._truncate_position_history(prev_state.get('history_len', len(self.position_history)))

        self.selected_square = None
        self.selected_drop_piece = None
        self.highlighted_moves = []
        self._all_legal_moves_cache = None

        print(f"Move undone. Current turn: {self.current_turn}")
        if self.move_log:
            undone_move = self.move_log.pop()
        self.find_kings()

        return True

    def reset_board(self):
        """Resets the game state to the initial one by calling __init__."""
        self.__init__()

    def copy(self):
        """Creates and returns a deep copy of this GameState object."""
        try:
            hands_copy = copy.deepcopy(self.hands)
            board_copy = copy.deepcopy(self.board)
            king_pos_copy = copy.deepcopy(self.king_pos)
        except Exception as e:
            print(f"[ERROR GameState.copy] Deepcopy failed: {e}")
            hands_copy = {'w':{}, 'b':{}}
            board_copy = [[EMPTY_SQUARE for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            king_pos_copy = {'w': None, 'b': None}

        new_state = GameState()

        new_state.board = board_copy
        new_state.current_turn = self.current_turn
        new_state.hands = hands_copy
        new_state.king_pos = king_pos_copy

        new_state.checkmate = self.checkmate
        new_state.stalemate = self.stalemate
        new_state.last_move = self.last_move
        new_state.game_over_message = self.game_over_message
        new_state.selected_square = self.selected_square
        new_state.selected_drop_piece = self.selected_drop_piece
        new_state.highlighted_moves = copy.deepcopy(self.highlighted_moves)
        new_state.needs_promotion_choice = self.needs_promotion_choice
        new_state.promotion_square = self.promotion_square
        new_state.last_move_for_promotion = self.last_move_for_promotion
        new_state.white_ai_enabled = self.white_ai_enabled
        new_state.black_ai_enabled = self.black_ai_enabled
        new_state.ai_depth = self.ai_depth
        new_state.promoted_pieces = set(self.promoted_pieces)
        new_state._all_legal_moves_cache = None

        new_state.is_draw = self.is_draw
        new_state.ply_limit = self.ply_limit
        new_state.ply_count = self.ply_count
        new_state.position_history = list(self.position_history)
        new_state.position_counts = dict(self.position_counts)

        return new_state

    def fast_copy_for_simulation(self):
        """Fast copy for AI simulation only - skips history and UI state."""
        new_state = GameState()
        
        new_state.board = [row[:] for row in self.board]
        
        new_state.hands = {
            'w': dict(self.hands.get('w', {})),
            'b': dict(self.hands.get('b', {}))
        }
        
        new_state.king_pos = dict(self.king_pos)
        
        new_state.current_turn = self.current_turn
        new_state.checkmate = self.checkmate
        new_state.stalemate = self.stalemate
        new_state.needs_promotion_choice = self.needs_promotion_choice
        new_state.promotion_square = self.promotion_square
        new_state.promoted_pieces = set(self.promoted_pieces)

        new_state.ply_limit = self.ply_limit
        new_state._reset_position_history()
        new_state.ply_count = self.ply_count
        new_state.is_draw = self.is_draw

        return new_state

    def find_kings(self):
        """Explicitly finds and updates king positions."""
        self.king_pos = {'w': None, 'b': None}
        for r in range(BOARD_SIZE):
            for f in range(BOARD_SIZE):
                piece = self.board[r][f]
                if piece == KING[0]:
                    self.king_pos['w'] = (r, f)
                elif piece == KING[1]:
                    self.king_pos['b'] = (r, f)

    def setup_initial_board(self):
        """Sets up the starting position on the board."""
        self.board = [
            ['.', '.', 'b', 'n', 'r', 'k'],
            ['.', '.', '.', '.', '.', 'p'],
            ['.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.'],
            ['P', '.', '.', '.', '.', '.'],
            ['K', 'R', 'N', 'B', '.', '.']
        ]
        self.king_pos = {'w': (5, 0), 'b': (0, 5)}
        self.current_turn = 'w'
        self.hands = {'w': {}, 'b': {}}
        for p_upper in "PNBR":
             self.hands['w'][p_upper] = 0
             self.hands['b'][p_upper] = 0

        self.checkmate = False
        self.stalemate = False
        self.last_move = None
        self.move_log = []
        self.game_over_message = ""
        self.saved_states = []
        self.selected_square = None
        self.selected_drop_piece = None
        self.highlighted_moves = []
        self.needs_promotion_choice = False
        self.promotion_square = None
        self.last_move_for_promotion = None
        self._all_legal_moves_cache = None
        self._reset_position_history()
        self.save_state()

    def make_move(self, move, is_check_game_over=True):
        """Makes a move, switches the side to move and checks for game over."""
        self._all_legal_moves_cache = None

        if self.needs_promotion_choice:
            print("Error: Cannot make move, must choose promotion first.")
            return False

        if move[0] == 'drop':
            _, piece_code, (r, f) = move
            color = piece_code[0]
            piece_type_upper = piece_code[1]

            if self.board[r][f] != EMPTY_SQUARE:
                print(f"Error (drop): Target square {r},{f} not empty.")
                return False
            if color != self.current_turn:
                 print(f"Error (drop): Trying to drop {color} piece ('{piece_code}') on {self.current_turn}'s turn.")
                 return False
            if self.hands[color].get(piece_type_upper, 0) <= 0:
                 print(f"Error (drop): No {piece_type_upper} in {color}'s hand. Hand: {self.hands[color]}")
                 return False
            if piece_type_upper == 'P':
                 promotion_rank = 0 if color == 'w' else BOARD_SIZE - 1
                 if r == promotion_rank:
                     print(f"Error (drop): Cannot drop pawn {piece_code} on promotion rank {r}.")
                     return False

            correct_piece_char = piece_code[1].upper() if color == 'w' else piece_code[1].lower()
            self.board[r][f] = correct_piece_char 
            self.hands[color][piece_type_upper] -= 1

            self.last_move = move
            self.move_log.append(move)
            self.current_turn = get_opposite_color(self.current_turn)
            self.selected_square = None
            self.selected_drop_piece = None
            self.highlighted_moves = []

            self.ply_count += 1
            self._push_position()

            if is_check_game_over:
                 self.check_game_over()
            return True

        if len(move) != 3 or not isinstance(move[0], tuple) or not isinstance(move[1], tuple):
             print(f"Error: Invalid move format for regular move: {move}")
             return False

        (r1, f1), (r2, f2), promotion_choice = move
        piece = self.board[r1][f1]

        if piece == EMPTY_SQUARE:
            print(f"Error (move): Start square {r1},{f1} is empty.")
            return False
        moving_color = get_piece_color(piece)
        if moving_color != self.current_turn:
             print(f"Error (move): Trying to move {moving_color} piece on {self.current_turn}'s turn.")
             return False

        target_piece = self.board[r2][f2]
        is_capture = target_piece != EMPTY_SQUARE

        self.board[r1][f1] = EMPTY_SQUARE

        if is_capture:
            captured_type = target_piece.upper()
            if captured_type == 'K':
                 print("Error: King capture detected - should not happen in legal moves.")
            else:
                 if (r2, f2) in self.promoted_pieces:
                     captured_type = 'P'
                     self.promoted_pieces.discard((r2, f2))
                 if moving_color not in self.hands: self.hands[moving_color] = {}
                 for p_upper in "PNBRQ":
                      if p_upper not in self.hands[moving_color]: self.hands[moving_color][p_upper] = 0

                 self.hands[moving_color][captured_type] = self.hands[moving_color].get(captured_type, 0) + 1

        if (r1, f1) in self.promoted_pieces:
            self.promoted_pieces.discard((r1, f1))
            self.promoted_pieces.add((r2, f2))

        if piece.upper() == 'K':
            self.king_pos[moving_color] = (r2, f2)

        is_pawn_move = piece.upper() == 'P'
        promotion_rank = 0 if moving_color == 'w' else BOARD_SIZE - 1

        if is_pawn_move and r2 == promotion_rank:
            if promotion_choice:
                valid_promotions = PROMOTION_PIECES_WHITE_STR if moving_color == 'w' else PROMOTION_PIECES_BLACK_STR
                if promotion_choice not in valid_promotions:
                     print(f"Error: Invalid promotion choice '{promotion_choice}' for {moving_color}.")
                     self.board[r1][f1] = piece
                     if is_capture and captured_type != 'K':
                          self.hands[moving_color][captured_type] -= 1
                     return False
                self.board[r2][f2] = promotion_choice
                self.promoted_pieces.add((r2, f2))
                self.needs_promotion_choice = False
                self.promotion_square = None
                self.last_move_for_promotion = None
            else:
                self.board[r2][f2] = piece
                self.needs_promotion_choice = True
                self.promotion_square = (r2, f2)
                self.last_move_for_promotion = ((r1, f1), (r2, f2), None)
                self.last_move = self.last_move_for_promotion
                self.move_log.append(self.last_move_for_promotion)
                print(f"Pawn reached promotion rank at {r2},{f2}. Waiting for choice.")
                return True
        else:
            self.board[r2][f2] = piece
            self.needs_promotion_choice = False

        self.last_move = move
        self.move_log.append(move)
        self.current_turn = get_opposite_color(self.current_turn)
        self.selected_square = None
        self.selected_drop_piece = None
        self.highlighted_moves = []

        self.ply_count += 1
        self._push_position()

        if is_check_game_over:
            self.check_game_over()

        return True

    def complete_promotion(self, chosen_piece_char):
        """Completes a promotion move after the player/AI has chosen a piece."""
        if not self.needs_promotion_choice or not self.promotion_square:
            print("Error: Not in promotion choice state.")
            return False

        r, f = self.promotion_square
        original_player_color = self.current_turn

        valid_promotions = PROMOTION_PIECES_WHITE_STR if original_player_color == 'w' else PROMOTION_PIECES_BLACK_STR
        if chosen_piece_char not in valid_promotions:
             print(f"Error: Invalid promotion choice '{chosen_piece_char}' for {original_player_color}.")
             return False

        self.board[r][f] = chosen_piece_char
        self.promoted_pieces.add((r, f))

        if self.move_log and self.last_move_for_promotion:
            if self.move_log[-1] == self.last_move_for_promotion:
                 self.move_log.pop()
                 (r1,f1),(r2,f2),_ = self.last_move_for_promotion
                 completed_move = ((r1,f1),(r2,f2), chosen_piece_char)
                 self.move_log.append(completed_move)
                 self.last_move = completed_move
            else:
                 print("Warning: Could not find base promotion move in log to update.")
                 (r1,f1),(r2,f2),_ = self.last_move_for_promotion
                 completed_move = ((r1,f1),(r2,f2), chosen_piece_char)
                 self.last_move = completed_move

        else:
             print("Warning: Move log or last_move_for_promotion missing during promotion completion.")
             if self.last_move and self.last_move[0] != 'drop':
                 (r1,f1),(r2,f2),_ = self.last_move
                 if (r2, f2) == (r, f):
                     completed_move = ((r1,f1),(r2,f2), chosen_piece_char)
                     self.last_move = completed_move
                 else: print("Error: Cannot reliably update last move for promotion.")
             else: print("Error: Cannot reliably update last move for promotion.")

        self.needs_promotion_choice = False
        self.promotion_square = None
        self.last_move_for_promotion = None
        self.current_turn = get_opposite_color(self.current_turn)

        self.selected_square = None
        self.selected_drop_piece = None
        self.highlighted_moves = []
        self._all_legal_moves_cache = None

        self.ply_count += 1
        self._push_position()

        print(f"Promotion to {chosen_piece_char} completed. Turn: {self.current_turn}")

        self.check_game_over()

        return True

    def get_pawn_moves(self, r, f, color):
        moves = []
        direction = -1 if color == 'w' else 1
        promotion_rank = 0 if color == 'w' else BOARD_SIZE - 1
        prom_pieces = PROMOTION_PIECES_WHITE_STR if color == 'w' else PROMOTION_PIECES_BLACK_STR

        nr, nf = r + direction, f
        if is_on_board(nr, nf) and self.board[nr][nf] == EMPTY_SQUARE:
            if nr == promotion_rank:
                for prom_piece in prom_pieces:
                    moves.append(((r, f), (nr, nf), prom_piece))
            else:
                moves.append(((r, f), (nr, nf), None))

        for df in [-1, 1]:
            nr, nf = r + direction, f + df
            if is_on_board(nr, nf):
                target_piece = self.board[nr][nf]
                if target_piece != EMPTY_SQUARE and get_piece_color(target_piece) != color:
                    if nr == promotion_rank:
                        for prom_piece in prom_pieces:
                            moves.append(((r, f), (nr, nf), prom_piece))
                    else:
                        moves.append(((r, f), (nr, nf), None))
        return moves

    def get_knight_moves(self, r, f, color):
        moves = []
        for dr, df in KNIGHT_MOVES:
            nr, nf = r + dr, f + df
            if is_on_board(nr, nf):
                target_piece = self.board[nr][nf]
                if target_piece == EMPTY_SQUARE or get_piece_color(target_piece) != color:
                    moves.append(((r, f), (nr, nf), None))
        return moves

    def get_sliding_moves(self, r, f, color, directions):
        moves = []
        for dr, df in directions:
            nr, nf = r + dr, f + df
            while is_on_board(nr, nf):
                target_piece = self.board[nr][nf]
                if target_piece == EMPTY_SQUARE:
                    moves.append(((r, f), (nr, nf), None))
                elif get_piece_color(target_piece) != color:
                    moves.append(((r, f), (nr, nf), None))
                    break
                else:
                    break
                nr, nf = nr + dr, nf + df
        return moves

    def get_bishop_moves(self, r, f, color):
        return self.get_sliding_moves(r, f, color, DIAGONAL_MOVES)

    def get_rook_moves(self, r, f, color):
         return self.get_sliding_moves(r, f, color, STRAIGHT_MOVES)

    def get_queen_moves(self, r, f, color):
        return self.get_sliding_moves(r, f, color, DIAGONAL_MOVES + STRAIGHT_MOVES)

    def get_king_moves(self, r, f, color):
        moves = []
        for dr, df in KING_MOVES:
            nr, nf = r + dr, f + df
            if is_on_board(nr, nf):
                target_piece = self.board[nr][nf]
                if target_piece == EMPTY_SQUARE or get_piece_color(target_piece) != color:
                     moves.append(((r, f), (nr, nf), None))
        return moves

    def generate_all_pseudo_legal_moves(self, color):
        """Generates all possible moves for 'color' without checking for check.
           Uses the current object's state (self.board, self.hands).
        """
        moves = []
        drop_moves_generated = []
            
        for r in range(BOARD_SIZE):
            for f in range(BOARD_SIZE):
                piece = self.board[r][f]
                piece_color = get_piece_color(piece)

                if piece != EMPTY_SQUARE and piece_color == color:

                    piece_type = piece.upper()
                    move_func = None
                    if piece_type == PAWN[0]: move_func = self.get_pawn_moves
                    elif piece_type == KNIGHT[0]: move_func = self.get_knight_moves
                    elif piece_type == BISHOP[0]: move_func = self.get_bishop_moves
                    elif piece_type == ROOK[0]: move_func = self.get_rook_moves
                    elif piece_type == QUEEN[0]: move_func = self.get_queen_moves
                    elif piece_type == KING[0]: move_func = self.get_king_moves

                    if move_func:

                        piece_moves = move_func(r, f, color)

                        moves.extend(piece_moves)

        player_hand = self.hands.get(color, {})
        if player_hand:
            for piece_type_upper, count in player_hand.items():
                if count > 0:
                    piece_code = color + piece_type_upper
                    promotion_rank = 0 if color == 'w' else BOARD_SIZE - 1
                    is_pawn = piece_type_upper == 'P'

                    for r in range(BOARD_SIZE):
                        if is_pawn and r == promotion_rank:
                            continue
                        for f in range(BOARD_SIZE):
                            target_cell = self.board[r][f]
                            if target_cell == EMPTY_SQUARE:
                                drop_move = ('drop', piece_code, (r, f))
                                moves.append(drop_move)
                                drop_moves_generated.append(drop_move)

        return moves

    def get_all_legal_moves(self):
        """Generates all LEGAL moves for the CURRENT player. Caches result."""
        if self._all_legal_moves_cache is not None:
            return self._all_legal_moves_cache

        if self.needs_promotion_choice:
            self._all_legal_moves_cache = []
            return []

        legal_moves = []
        current_color = self.current_turn
        pseudo_legal_moves = self.generate_all_pseudo_legal_moves(current_color)

        for move in pseudo_legal_moves:
            self.make_ai_move(move)
            is_check_after = self.is_in_check(current_color)
            self.undo_ai_move()
            
            if not is_check_after:
                legal_moves.append(move)

        self._all_legal_moves_cache = legal_moves
        return legal_moves

    def is_in_check(self, color):
        """Checks whether the king of the given color is in check."""
        king_pos = self.king_pos.get(color)
        if not king_pos:
             self.find_kings()
             king_pos = self.king_pos.get(color)
             if not king_pos:
                  print(f"Warning (is_in_check): the {color} king was not found on the board!")
                  return False
        return self._internal_is_square_attacked(king_pos[0], king_pos[1], get_opposite_color(color))

    def check_game_over(self):
        """Checks and sets checkmate/stalemate/draw flags."""
        if self.needs_promotion_choice: return False

        current_player_color = self.current_turn
        legal_moves = self.get_all_legal_moves()

        if not legal_moves:
            if self.is_in_check(current_player_color):
                self.checkmate = True
                winner = "Black" if current_player_color == 'w' else "White"
                self.game_over_message = f"Checkmate! {winner} wins."
                print(self.game_over_message)
            else:
                self.stalemate = True
                self.game_over_message = "Stalemate! Draw."
                print(self.game_over_message)
            return True

        current_key = self.position_history[-1] if self.position_history else self._position_key()
        if self.position_counts.get(current_key, 0) >= 3:
            self.is_draw = True
            self.game_over_message = "Draw by repetition."
            print(self.game_over_message)
            return True

        if self.ply_count >= self.ply_limit:
            self.is_draw = True
            self.game_over_message = "Draw by move limit."
            print(self.game_over_message)
            return True

        self.checkmate = False
        self.stalemate = False
        self.is_draw = False
        self.game_over_message = ""
        return False

    def _internal_is_square_attacked(self, r, f, attacker_color):
        """Checks whether square (r, f) is attacked by pieces of attacker_color.
           Uses self.board.
        """
        opponent_color = get_opposite_color(attacker_color)

        pawn_piece = PAWN[0] if attacker_color == 'w' else PAWN[1]
        pawn_dir = -1 if attacker_color == 'w' else 1
        for df_attack in [-1, 1]:
            pr, pf = r - pawn_dir, f + df_attack
            if is_on_board(pr, pf) and self.board[pr][pf] == pawn_piece:
                 return True

        knight_piece = KNIGHT[0] if attacker_color == 'w' else KNIGHT[1]
        for dr, df in KNIGHT_MOVES:
            nr, nf = r + dr, f + df
            if is_on_board(nr, nf) and self.board[nr][nf] == knight_piece:
                 return True

        bishop_piece = BISHOP[0] if attacker_color == 'w' else BISHOP[1]
        rook_piece = ROOK[0] if attacker_color == 'w' else ROOK[1]
        queen_piece = QUEEN[0] if attacker_color == 'w' else QUEEN[1]

        for dr, df in DIAGONAL_MOVES:
            cr, cf = r + dr, f + df
            while is_on_board(cr, cf):
                piece = self.board[cr][cf]
                if piece != EMPTY_SQUARE:
                    if piece == bishop_piece or piece == queen_piece:
                         return True
                    break
                cr, cf = cr + dr, cf + df

        for dr, df in STRAIGHT_MOVES:
            cr, cf = r + dr, f + df
            while is_on_board(cr, cf):
                piece = self.board[cr][cf]
                if piece != EMPTY_SQUARE:
                    if piece == rook_piece or piece == queen_piece:
                         return True
                    break
                cr, cf = cr + dr, cf + df

        king_piece = KING[0] if attacker_color == 'w' else KING[1]
        for dr, df in KING_MOVES:
            kr, kf = r + dr, f + df
            if is_on_board(kr, kf) and self.board[kr][kf] == king_piece:
                 return True

        return False

    def make_ai_move(self, move):
        """
        Fast move application for AI search.
        Assumes move is legal.
        Saves state to self.ai_history for undo.
        """
        undo_info = {
            'move': move,
            'captured': None,
            'prev_king_pos': None,
            'prev_checkmate': self.checkmate,
            'prev_stalemate': self.stalemate,
            'prev_last_move': self.last_move,
        }
        
        if move[0] == 'drop':
            _, piece_code, (r, f) = move
            color = piece_code[0]
            piece_char = piece_code[1].upper() if color == 'w' else piece_code[1].lower()
            piece_type_upper = piece_code[1]
            
            self.board[r][f] = piece_char
            self.hands[color][piece_type_upper] -= 1
            
            self.last_move = move
            self.current_turn = get_opposite_color(self.current_turn)
            
        else:
            (r1, f1), (r2, f2), promotion = move
            piece = self.board[r1][f1]
            target = self.board[r2][f2]
            color = get_piece_color(piece)
            
            if target != EMPTY_SQUARE:
                undo_info['captured'] = target
                captured_type = target.upper()
                if (r2, f2) in self.promoted_pieces:
                    captured_type = 'P'
                    self.promoted_pieces.discard((r2, f2))
                    undo_info['was_promoted'] = True
                self.hands[color][captured_type] = self.hands[color].get(captured_type, 0) + 1
            
            if (r1, f1) in self.promoted_pieces:
                self.promoted_pieces.discard((r1, f1))
                self.promoted_pieces.add((r2, f2))
                undo_info['moved_promoted'] = True

            self.board[r1][f1] = EMPTY_SQUARE
            if promotion:
                self.board[r2][f2] = promotion
                self.promoted_pieces.add((r2, f2))
                undo_info['new_promotion'] = True
            else:
                self.board[r2][f2] = piece
                
            if piece.upper() == 'K':
                undo_info['prev_king_pos'] = self.king_pos[color]
                self.king_pos[color] = (r2, f2)
                
            self.last_move = move
            self.current_turn = get_opposite_color(self.current_turn)

        self.ai_history.append(undo_info)
        
        self._all_legal_moves_cache = None
        self._hash_cache = None
        self._is_check_cache = None
        
        return True

    def undo_ai_move(self):
        """
        Reverts the last move made by make_ai_move.
        """
        if not self.ai_history:
            return False
            
        undo_info = self.ai_history.pop()
        move = undo_info['move']
        
        self.current_turn = get_opposite_color(self.current_turn)
        
        if move[0] == 'drop':
            _, piece_code, (r, f) = move
            color = piece_code[0]
            piece_type_upper = piece_code[1]
            
            self.board[r][f] = EMPTY_SQUARE
            self.hands[color][piece_type_upper] += 1
            
        else:
            (r1, f1), (r2, f2), promotion = move
            moved_piece = self.board[r2][f2]
            color = get_piece_color(moved_piece)
            
            if undo_info.get('new_promotion'):
                self.promoted_pieces.discard((r2, f2))
            
            if promotion:
                moved_piece = 'P' if color == 'w' else 'p'
            
            if undo_info.get('moved_promoted'):
                self.promoted_pieces.discard((r2, f2))
                self.promoted_pieces.add((r1, f1))
            
            self.board[r1][f1] = moved_piece
            
            captured = undo_info['captured']
            if captured:
                self.board[r2][f2] = captured
                if undo_info.get('was_promoted'):
                    self.hands[color]['P'] -= 1
                    self.promoted_pieces.add((r2, f2))
                else:
                    self.hands[color][captured.upper()] -= 1
            else:
                self.board[r2][f2] = EMPTY_SQUARE
                
            if undo_info['prev_king_pos']:
                color = get_piece_color(moved_piece)
                self.king_pos[color] = undo_info['prev_king_pos']

        self.checkmate = undo_info['prev_checkmate']
        self.stalemate = undo_info['prev_stalemate']
        self.last_move = undo_info['prev_last_move']
        
        self._all_legal_moves_cache = None
        self._hash_cache = None
        self._is_check_cache = None
        return True