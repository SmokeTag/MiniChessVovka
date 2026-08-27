#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-play training mode for the Mini Chess AI.
The AI plays against itself and 20% of the time picks the second-best move
to explore alternative branches of the game tree.
"""

import argparse
import os
import signal
import sys
import time
import random
from datetime import datetime

# Running this as a script puts src/ on sys.path, not the repo root, where
# gamestate/ai/utils live. The DB path in the Rust engine is relative too, so the
# repo root must also be the working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

from gamestate import GameState
import ai
from utils import format_move_for_print


# Global flag for graceful shutdown
shutdown_requested = False

# Per-move chatter is suppressed under --quiet; summaries always print.
VERBOSE = True


def vprint(*a, **kw):
    if VERBOSE:
        print(*a, **kw)


def signal_handler(signum, frame):
    """Signal handler for graceful shutdown."""
    global shutdown_requested
    print("\n\n" + "="*60)
    print("Interrupt received. Shutting down...")
    print("="*60)
    shutdown_requested = True


def setup_signal_handlers():
    """Installs the signal handlers."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def choose_move_with_exploration(gamestate: GameState, depth: int, exploration_rate: float = 0.2):
    """
    Picks a move, exploring alternatives.
    
    Args:
        gamestate: Current game state
        depth: Search depth
        exploration_rate: Probability of picking the second-best move (default 20%)
    
    Returns:
        The chosen move
    """
    # Get the top 2 moves
    # parallel=False: training is parallelised across games, not inside one search.
    top_moves = ai.find_best_move(gamestate, depth=depth, return_top_n=2, parallel=False)
    
    if not top_moves:
        print("No moves available!")
        return None
    
    # If there is only one move, take it
    if len(top_moves) == 1:
        move, score = top_moves[0]
        vprint(f"  Only move: {format_move_for_print(move)}, score: {score:.1f}")
        return move
    
    # Decide whether to explore
    explore = random.random() < exploration_rate
    
    best_move, best_score = top_moves[0]
    second_move, second_score = top_moves[1]
    
    # Do not explore if the best move is mate
    is_mate = abs(best_score) >= ai.CHECKMATE_SCORE * 0.9
    
    if explore and not is_mate:
        vprint(f"  EXPLORING: taking the 2nd-best move")
        vprint(f"    1st: {format_move_for_print(best_move)}, score: {best_score:.1f}")
        vprint(f"    2nd: {format_move_for_print(second_move)}, score: {second_score:.1f} <- CHOSEN")
        return second_move
    else:
        reason = "mate found" if is_mate else "standard choice"
        vprint(f"  Best move ({reason}): {format_move_for_print(best_move)}, score: {best_score:.1f}")
        if not is_mate:
            vprint(f"    2nd: {format_move_for_print(second_move)}, score: {second_score:.1f}")
        return best_move


def play_random_opening(gamestate: GameState, plies: int):
    """Plays `plies` uniformly random legal moves to spread workers over the tree.

    Twenty workers starting from the same position and the same deterministic engine
    would search largely the same nodes; a short random prefix is what makes the
    extra processes cover new ground instead of duplicating each other.
    """
    opening = []
    for _ in range(plies):
        moves = gamestate.get_all_legal_moves()
        if not moves:
            break
        move = random.choice(moves)
        if not gamestate.make_move(move):
            break
        if gamestate.needs_promotion_choice:
            gamestate.complete_promotion('R' if gamestate.current_turn == 'w' else 'r')
        opening.append(move)
    return opening


def play_self_game(depth: int = 6, exploration_rate: float = 0.2, max_moves: int = 200,
                   random_plies: int = 0):
    """
    Plays a single AI-vs-itself game.
    
    Args:
        depth: AI search depth
        exploration_rate: Probability of picking the 2nd-best move
        max_moves: Maximum number of moves in a game
    
    Returns:
        dict with the game results
    """
    global shutdown_requested
    
    gamestate = GameState()
    gamestate.setup_initial_board()
    
    game_start = time.time()
    move_count = 0
    move_times = []
    
    print("\n" + "="*60)
    print(f"Starting a new game (depth: {depth}, exploration: {exploration_rate*100:.0f}%)")
    print("="*60 + "\n")

    if random_plies > 0:
        opening = play_random_opening(gamestate, random_plies)
        print(f"Random opening ({len(opening)} plies): "
              + ", ".join(format_move_for_print(m) for m in opening))
    
    while not shutdown_requested:
        move_count += 1
        current_player = "White" if gamestate.current_turn == 'w' else "Black"
        
        vprint(f"\n--- Move {move_count} ({current_player}) ---")
        
        # Check for a terminal state
        if gamestate.checkmate:
            winner = "Black" if gamestate.current_turn == 'w' else "White"
            print(f"\nCHECKMATE! {winner} wins!")
            return {
                'result': 'checkmate',
                'winner': winner,
                'moves': move_count - 1,
                'duration': time.time() - game_start,
                'avg_move_time': sum(move_times) / len(move_times) if move_times else 0
            }
        
        if gamestate.stalemate:
            print("\nSTALEMATE! Draw.")
            return {
                'result': 'stalemate',
                'winner': None,
                'moves': move_count - 1,
                'duration': time.time() - game_start,
                'avg_move_time': sum(move_times) / len(move_times) if move_times else 0
            }

        if gamestate.is_draw:
            # Threefold repetition or the ply limit; game_over_message says which.
            print(f"\n{gamestate.game_over_message}")
            return {
                'result': 'draw',
                'winner': None,
                'moves': move_count - 1,
                'duration': time.time() - game_start,
                'avg_move_time': sum(move_times) / len(move_times) if move_times else 0
            }
        
        # Check the move limit
        if move_count > max_moves:
            print(f"\nMove limit reached ({max_moves}). Draw.")
            return {
                'result': 'max_moves',
                'winner': None,
                'moves': move_count - 1,
                'duration': time.time() - game_start,
                'avg_move_time': sum(move_times) / len(move_times) if move_times else 0
            }
        
        # Choose and play a move
        move_start = time.time()
        move = choose_move_with_exploration(gamestate, depth, exploration_rate)
        move_time = time.time() - move_start
        move_times.append(move_time)
        
        if move is None:
            print("Error: no moves available!")
            return {
                'result': 'error',
                'winner': None,
                'moves': move_count - 1,
                'duration': time.time() - game_start,
                'avg_move_time': sum(move_times) / len(move_times) if move_times else 0
            }
        
        vprint(f"  Think time: {move_time:.1f}s")
        
        # Make the move
        if not gamestate.make_move(move):
            print(f"Error: cannot play move {format_move_for_print(move)}")
            return {
                'result': 'error',
                'winner': None,
                'moves': move_count - 1,
                'duration': time.time() - game_start,
                'avg_move_time': sum(move_times) / len(move_times) if move_times else 0
            }
        
        # Handle promotion if needed
        if gamestate.needs_promotion_choice:
            promo_piece = 'R' if gamestate.current_turn == 'w' else 'r'
            gamestate.complete_promotion(promo_piece)
        
        # Save the cache after every move
        ai.save_move_cache_to_db(ai.move_cache)
    
    # If an interrupt was received
    return {
        'result': 'interrupted',
        'winner': None,
        'moves': move_count - 1,
        'duration': time.time() - game_start,
        'avg_move_time': sum(move_times) / len(move_times) if move_times else 0
    }


def run_self_play_training(num_games: int = None, depth: int = 6, exploration_rate: float = 0.2,
                           random_plies: int = 0):
    """
    Runs the self-play training mode.
    
    Args:
        num_games: Number of games to play (None = unlimited)
        depth: AI search depth
        exploration_rate: Probability of picking the 2nd-best move
        random_plies: Random legal moves played before each game starts
    """
    global shutdown_requested
    
    print("\n" + "="*60)
    print("SELF-PLAY TRAINING MODE")
    print("="*60)
    print(f"Parameters:")
    print(f"  - Search depth: {depth}")
    print(f"  - Exploration probability: {exploration_rate*100:.0f}%")
    print(f"  - Number of games: {'∞' if num_games is None else num_games}")
    print(f"  - Random opening plies: {random_plies}")
    print(f"  - PID: {os.getpid()}")
    print(f"  - Search threads: 1 (single-threaded; run several games in parallel instead)")
    print(f"\nPress Ctrl+C to stop")
    print("="*60)

    # Training deliberately runs a single-threaded search: many independent games across
    # cores scale better than parallelising one game.
    ai.set_parallel_search(False)
    
    # Load the cache from the DB
    print("\nLoading the move cache from the database...")
    ai.load_move_cache_from_db()
    
    stats = {
        'total_games': 0,
        'checkmate': 0,
        'stalemate': 0,
        'draw': 0,
        'max_moves': 0,
        'interrupted': 0,
        'errors': 0,
        'white_wins': 0,
        'black_wins': 0,
        'total_moves': 0,
        'total_time': 0
    }
    
    game_num = 0
    training_start = time.time()
    
    try:
        while not shutdown_requested:
            if num_games is not None and game_num >= num_games:
                print(f"\nPlayed {num_games} games. Finishing...")
                break
            
            game_num += 1
            print(f"\n{'='*60}")
            print(f"GAME {game_num}" + (f" / {num_games}" if num_games else ""))
            print(f"{'='*60}")
            
            result = play_self_game(depth, exploration_rate, random_plies=random_plies)
            
            # Update the statistics
            stats['total_games'] += 1
            stats[result['result']] = stats.get(result['result'], 0) + 1
            stats['total_moves'] += result['moves']
            stats['total_time'] += result['duration']
            
            if result.get('winner') == 'White':
                stats['white_wins'] += 1
            elif result.get('winner') == 'Black':
                stats['black_wins'] += 1
            
            # Print interim statistics
            print("\n" + "-"*60)
            print("Current game statistics:")
            print(f"  Result: {result['result']}")
            if result.get('winner'):
                print(f"  Winner: {result['winner']}")
            print(f"  Moves: {result['moves']}")
            print(f"  Duration: {result['duration']:.1f}s")
            print(f"  Average move time: {result['avg_move_time']:.1f}s")
            
            print("\nOverall statistics:")
            print(f"  Total games: {stats['total_games']}")
            print(f"  Checkmate: {stats['checkmate']} ({stats['checkmate']/stats['total_games']*100:.1f}%)")
            print(f"    - White wins: {stats['white_wins']}")
            print(f"    - Black wins: {stats['black_wins']}")
            print(f"  Stalemate: {stats['stalemate']} ({stats['stalemate']/stats['total_games']*100:.1f}%)")
            print(f"  Draw (repetition / ply limit): {stats['draw']} ({stats['draw']/stats['total_games']*100:.1f}%)")
            print(f"  Move limit: {stats['max_moves']}")
            print(f"  Average moves per game: {stats['total_moves']/stats['total_games']:.1f}")
            print(f"  Average game time: {stats['total_time']/stats['total_games']:.1f}s")
            print(f"  Book size: {ai.book_size()} positions")
            print("-"*60)
            
            if result['result'] == 'interrupted':
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupt received...")
    
    finally:
        # Final cache save
        print("\n" + "="*60)
        print("Saving the cache to the database...")
        ai.save_move_cache_to_db(ai.move_cache)
        
        training_duration = time.time() - training_start
        
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"Total games: {stats['total_games']}")
        print(f"Training time: {training_duration/60:.1f} minutes")
        if stats['total_games'] > 0:
            print(f"\nResults:")
            print(f"  Checkmate: {stats['checkmate']} ({stats['checkmate']/stats['total_games']*100:.1f}%)")
            print(f"    - White wins: {stats['white_wins']}")
            print(f"    - Black wins: {stats['black_wins']}")
            print(f"  Stalemate: {stats['stalemate']} ({stats['stalemate']/stats['total_games']*100:.1f}%)")
            print(f"  Draw (repetition / ply limit): {stats['draw']} ({stats['draw']/stats['total_games']*100:.1f}%)")
            print(f"  Move limit: {stats['max_moves']}")
            print(f"  Errors: {stats['errors']}")
            print(f"  Interrupted: {stats['interrupted']}")
            print(f"\nGame statistics:")
            print(f"  Average moves: {stats['total_moves']/stats['total_games']:.1f}")
            print(f"  Average game time: {stats['total_time']/stats['total_games']:.1f}s")
            print(f"  Total moves: {stats['total_moves']}")
        print(f"\nBook size: {ai.book_size()} positions")
        print("="*60)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Self-play training. Searches are single-threaded by design; "
                    "run several of these processes to use more cores.")
    p.add_argument("--depth", type=int, default=6,
                   help="search depth (default: 6). Iterative deepening also caches "
                        "every shallower depth along the way.")
    p.add_argument("--games", type=int, default=None,
                   help="number of games to play (default: unlimited)")
    p.add_argument("--exploration", type=float, default=0.2,
                   help="probability of taking the 2nd-best move (default: 0.2)")
    p.add_argument("--random-plies", type=int, default=0,
                   help="random legal moves before each game, to spread parallel "
                        "workers across the tree (default: 0)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed; omit for OS entropy (distinct per process)")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-move output, keep per-game summaries")
    return p.parse_args(argv)


def main():
    """Entry point."""
    global VERBOSE
    args = parse_args()
    VERBOSE = not args.quiet
    if args.seed is not None:
        random.seed(args.seed)

    setup_signal_handlers()

    run_self_play_training(args.games, args.depth, args.exploration,
                           random_plies=args.random_plies)


if __name__ == "__main__":
    main()
