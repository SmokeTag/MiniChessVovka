#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended self-play training mode with detailed logging.
Logs every move with a timestamp so progress can be tracked.
"""

import signal
import sys
import time
import random
import os
import threading
from datetime import datetime, timezone
from gamestate import GameState
import ai
from utils import format_move_for_print


# Global flag for graceful shutdown
shutdown_requested = False
health_updater_running = False

# Log file
LOG_FILE = "training_log.txt"
PROGRESS_FILE = "training_progress.txt"
HEALTH_FILE = "training.health"
PID_FILE = "training.pid"


def signal_handler(signum, frame):
    """Signal handler for graceful shutdown."""
    global shutdown_requested
    log_message("\n" + "="*60)
    log_message("Interrupt received. Shutting down...")
    log_message("="*60)
    shutdown_requested = True
    
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except Exception:
            pass

def write_pid():
    """Writes current PID to file"""
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

def setup_signal_handlers():
    """Installs the signal handlers."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    write_pid()


def update_health():
    """Updates the health file with the current timestamp, for monitoring."""
    try:
        with open(HEALTH_FILE, 'w') as f:
            f.write(str(int(time.time())))
    except Exception as e:
        # Do not abort if the health file could not be written
        pass


def health_updater_thread():
    """Background thread that refreshes the health file periodically."""
    global health_updater_running
    health_updater_running = True
    
    while not shutdown_requested and health_updater_running:
        update_health()
        # Refresh every 30 seconds
        for _ in range(30):
            if shutdown_requested:
                break
            time.sleep(1)
    
    health_updater_running = False


def start_health_updater():
    """Starts the background health-update thread."""
    thread = threading.Thread(target=health_updater_thread, daemon=True)
    thread.start()
    return thread


def log_message(message, console=True, file=True):
    """
    Logs a message to the file and/or the console.
    
    Args:
        message: The message to log
        console: Whether to print it to the console
        file: Whether to write it to the file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if console:
        print(message)
    
    if file:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            # Prefix every line with a timestamp
            for line in message.split('\n'):
                if line.strip():
                    f.write(f"[{timestamp}] {line}\n")
                else:
                    f.write("\n")


def update_progress(stats, training_start_time):
    """
    Updates the training-progress file.
    
    Args:
        stats: Statistics dictionary
        training_start_time: When training started
    """
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"Mini Chess AI training progress\n")
        f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        training_duration = time.time() - training_start_time
        hours = int(training_duration // 3600)
        minutes = int((training_duration % 3600) // 60)
        
        f.write(f"Uptime: {hours}h {minutes}min\n")
        f.write(f"Total games: {stats['total_games']}\n")
        f.write(f"Cache size: {len(ai.move_cache)} positions\n\n")
        
        if stats['total_games'] > 0:
            f.write("Results:\n")
            f.write(f"  - Checkmate: {stats['checkmate']} ({stats['checkmate']/stats['total_games']*100:.1f}%)\n")
            f.write(f"    - White wins: {stats['white_wins']}\n")
            f.write(f"    - Black wins: {stats['black_wins']}\n")
            f.write(f"  - Stalemate: {stats['stalemate']} ({stats['stalemate']/stats['total_games']*100:.1f}%)\n")
            f.write(f"  - Move limit: {stats['max_moves']}\n\n")
            
            f.write("Statistics:\n")
            f.write(f"  - Average moves per game: {stats['total_moves']/stats['total_games']:.1f}\n")
            f.write(f"  - Average game time: {stats['total_time']/stats['total_games']:.1f}s\n")
            f.write(f"  - Total moves played: {stats['total_moves']}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("To stop training press Ctrl+C in the terminal\n")
        f.write("or send SIGTERM to the process\n")
        f.write("="*60 + "\n")


def choose_move_with_exploration(gamestate: GameState, depth: int, exploration_rate: float, game_num: int, move_num: int):
    """
    Picks a move, exploring alternatives, with logging.
    
    Args:
        gamestate: Current game state
        depth: Search depth
        exploration_rate: Probability of picking the second-best move
        game_num: Current game number
        move_num: Current move number
    
    Returns:
        The chosen move
    """
    move_calc_start = time.time()
    
    # Get the top 2 moves
    # parallel=False: training is parallelised across games, not inside one search.
    top_moves = ai.find_best_move(gamestate, depth=depth, return_top_n=2, parallel=False)
    
    if not top_moves:
        log_message("No moves available!")
        return None
    
    # If there is only one move, take it
    if len(top_moves) == 1:
        move, score = top_moves[0]
        calc_time = time.time() - move_calc_start
        log_message(f"  Only move: {format_move_for_print(move)}, score: {score:.1f}, time: {calc_time:.1f}s")
        return move
    
    # Decide whether to explore
    explore = random.random() < exploration_rate
    
    best_move, best_score = top_moves[0]
    second_move, second_score = top_moves[1]
    
    # Do not explore if the best move is mate
    is_mate = abs(best_score) >= ai.CHECKMATE_SCORE * 0.9
    
    calc_time = time.time() - move_calc_start
    
    if explore and not is_mate:
        log_message(f"  EXPLORING: taking the 2nd-best move")
        log_message(f"    1st: {format_move_for_print(best_move)}, score: {best_score:.1f}")
        log_message(f"    2nd: {format_move_for_print(second_move)}, score: {second_score:.1f} <- CHOSEN")
        log_message(f"    Think time: {calc_time:.1f}s")
        return second_move
    else:
        reason = "mate found" if is_mate else "standard choice"
        log_message(f"  Best move ({reason}): {format_move_for_print(best_move)}, score: {best_score:.1f}")
        if not is_mate:
            log_message(f"    2nd option: {format_move_for_print(second_move)}, score: {second_score:.1f}")
        log_message(f"    Think time: {calc_time:.1f}s")
        return best_move


def get_current_utc_hour():
    """Returns the current hour in UTC."""
    return datetime.now(timezone.utc).hour

def is_training_time():
    """Checks whether we are inside the 02:00-10:00 UTC training window."""
    current_hour = get_current_utc_hour()
    return 2 <= current_hour < 10

def is_before_training_window():
    """Checks whether we are BEFORE the training window (00:00-02:00 UTC)."""
    current_hour = get_current_utc_hour()
    return current_hour < 2

def should_exit():
    """Checks whether we should shut down (10:00 UTC reached)."""
    current_hour = get_current_utc_hour()
    # Exit only if the hour is >= 10 (past the training window)
    return current_hour >= 10

def wait_for_training_window():
    """
    Waits for the training window to open (02:00 UTC).
    Returns True if it opened, False if a shutdown signal arrived.
    """
    global shutdown_requested
    
    if is_training_time():
        return True  # Already inside the window
    
    if should_exit():
        # After 10:00 UTC - exit; the timer will restart us tomorrow at 00:00
        log_message(f"Current time is past the training window (UTC hour: {get_current_utc_hour()}). Exiting; the timer will restart tomorrow.")
        return False
    
    # We are in the 00:00-02:00 UTC period - wait for the window to open
    log_message(f"\nWaiting for the training window to open (02:00 UTC)...")
    log_message(f"Current UTC hour: {get_current_utc_hour()}")
    
    while is_before_training_window() and not shutdown_requested:
        # Work out how long is left to wait
        now = datetime.now(timezone.utc)
        minutes_to_wait = (2 - now.hour) * 60 - now.minute
        
        if minutes_to_wait > 0:
            log_message(f"Training starts in ~{minutes_to_wait} minutes. Waiting...", console=True, file=False)
        
        # Sleep in 60-second slices, checking for shutdown
        for _ in range(60):
            if shutdown_requested:
                log_message("Shutdown signal received while waiting.")
                return False
            time.sleep(1)
            update_health()  # Refresh health to show the process is alive
    
    if shutdown_requested:
        return False
    
    log_message(f"\nTraining window is open! (UTC hour: {get_current_utc_hour()})")
    return True


def play_self_game(depth: int, exploration_rate: float, max_moves: int, game_num: int):
    """
    Plays a single AI-vs-itself game with detailed logging.
    
    Args:
        depth: AI search depth
        exploration_rate: Probability of picking the 2nd-best move
        max_moves: Maximum number of moves in a game
        game_num: Current game number
    
    Returns:
        dict with the game results
    """
    global shutdown_requested
    
    # Check the time window before starting the game
    if should_exit():
        log_message(f"\nEnd of the training window reached (10:00 UTC). Shutting down.")
        ai.save_move_cache_to_db(ai.move_cache)
        sys.exit(0)

    
    gamestate = GameState()
    gamestate.setup_initial_board()
    
    game_start = time.time()
    move_count = 0
    move_times = []
    
    log_message("\n" + "="*60)
    log_message(f"Game {game_num} (depth: {depth}, exploration: {exploration_rate*100:.0f}%)")
    log_message("="*60 + "\n")
    
    while not shutdown_requested:
        # Check the time window before every move
        if should_exit():
            log_message(f"\nEnd of the training window reached (10:00 UTC). Shutting down.")
            ai.save_move_cache_to_db(ai.move_cache)
            sys.exit(0)

        
        move_count += 1
        current_player = "White" if gamestate.current_turn == 'w' else "Black"
        
        log_message(f"\n--- Move {move_count} ({current_player}) ---")
        
        # Check for a terminal state
        if gamestate.checkmate:
            winner = "Black" if gamestate.current_turn == 'w' else "White"
            log_message(f"\nCHECKMATE! {winner} wins!")
            return {
                'result': 'checkmate',
                'winner': winner,
                'moves': move_count - 1,
                'duration': time.time() - game_start,
                'avg_move_time': sum(move_times) / len(move_times) if move_times else 0
            }
        
        if gamestate.stalemate:
            log_message("\nSTALEMATE! Draw.")
            return {
                'result': 'stalemate',
                'winner': None,
                'moves': move_count - 1,
                'duration': time.time() - game_start,
                'avg_move_time': sum(move_times) / len(move_times) if move_times else 0
            }
        
        # Check the move limit
        if move_count > max_moves:
            log_message(f"\nMove limit reached ({max_moves}). Draw.")
            return {
                'result': 'max_moves',
                'winner': None,
                'moves': move_count - 1,
                'duration': time.time() - game_start,
                'avg_move_time': sum(move_times) / len(move_times) if move_times else 0
            }
        
        # Choose and play a move
        move_start = time.time()
        move = choose_move_with_exploration(gamestate, depth, exploration_rate, game_num, move_count)
        move_time = time.time() - move_start
        move_times.append(move_time)
        
        if move is None:
            log_message("Error: no moves available!")
            return {
                'result': 'error',
                'winner': None,
                'moves': move_count - 1,
                'duration': time.time() - game_start,
                'avg_move_time': sum(move_times) / len(move_times) if move_times else 0
            }
        
        # Make the move
        if not gamestate.make_move(move):
            log_message(f"Error: cannot play move {format_move_for_print(move)}")
            return {
                'result': 'error',
                'winner': None,
                'moves': move_count - 1,
                'duration': time.time() - game_start,
                'avg_move_time': sum(move_times) / len(move_times) if move_times else 0
            }
        
        # Handle promotion if needed
        if gamestate.needs_promotion_choice:
            promo_piece = 'R' if gamestate.current_turn == 'b' else 'r'
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


def run_self_play_training(num_games: int = None, depth: int = 5, exploration_rate: float = 0.2):
    """
    Runs the self-play training mode with full logging.
    
    Args:
        num_games: Number of games to play (None = unlimited)
        depth: AI search depth
        exploration_rate: Probability of picking the 2nd-best move
    """
    global shutdown_requested
    
    # Create/truncate the log file for the new session
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"New training session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
    
    log_message("\n" + "="*60)
    log_message("SELF-PLAY TRAINING MODE WITH EXTENDED LOGGING")
    log_message("="*60)
    log_message(f"Parameters:")
    log_message(f"  - Search depth: {depth}")
    log_message(f"  - Exploration probability: {exploration_rate*100:.0f}%")
    log_message(f"  - Number of games: {'∞' if num_games is None else num_games}")
    log_message(f"  - Log file: {LOG_FILE}")
    log_message(f"  - Progress file: {PROGRESS_FILE}")
    log_message(f"  - Search threads: 1 (single-threaded; run several games in parallel instead)")
    log_message(f"\nPress Ctrl+C or send SIGTERM to stop")
    log_message("="*60)

    # Training deliberately runs a single-threaded search: many independent games across
    # cores scale better than parallelising one game.
    ai.set_parallel_search(False)
    
    # Load the cache from the DB
    log_message("\nLoading the move cache from the database...")
    ai.load_move_cache_from_db()
    log_message(f"Loaded {len(ai.move_cache)} positions from the cache")
    
    stats = {
        'total_games': 0,
        'checkmate': 0,
        'stalemate': 0,
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
                log_message(f"\nPlayed {num_games} games. Finishing...")
                break
            
            game_num += 1
            
            result = play_self_game(depth, exploration_rate, 200, game_num)
            
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
            log_message("\n" + "-"*60)
            log_message("Current game statistics:")
            log_message(f"  Result: {result['result']}")
            if result.get('winner'):
                log_message(f"  Winner: {result['winner']}")
            log_message(f"  Moves: {result['moves']}")
            log_message(f"  Duration: {result['duration']:.1f}s ({result['duration']/60:.1f} min)")
            log_message(f"  Average move time: {result['avg_move_time']:.1f}s")
            
            log_message("\nOverall statistics:")
            log_message(f"  Total games: {stats['total_games']}")
            log_message(f"  Checkmate: {stats['checkmate']} ({stats['checkmate']/stats['total_games']*100:.1f}%)")
            log_message(f"    - White wins: {stats['white_wins']}")
            log_message(f"    - Black wins: {stats['black_wins']}")
            log_message(f"  Stalemate: {stats['stalemate']} ({stats['stalemate']/stats['total_games']*100:.1f}%)")
            log_message(f"  Move limit: {stats['max_moves']}")
            log_message(f"  Average moves per game: {stats['total_moves']/stats['total_games']:.1f}")
            log_message(f"  Average game time: {stats['total_time']/stats['total_games']:.1f}s")
            log_message(f"  Cache size: {len(ai.move_cache)} positions")
            log_message("-"*60)
            
            # Update the progress file
            update_progress(stats, training_start)
            
            if result['result'] == 'interrupted':
                break
    
    except KeyboardInterrupt:
        log_message("\n\nInterrupt received...")
    
    finally:
        # Final cache save
        log_message("\n" + "="*60)
        log_message("Saving the cache to the database...")
        ai.save_move_cache_to_db(ai.move_cache)
        
        training_duration = time.time() - training_start
        
        log_message("\n" + "="*60)
        log_message("FINAL STATISTICS")
        log_message("="*60)
        log_message(f"Total games: {stats['total_games']}")
        log_message(f"Training time: {training_duration/3600:.2f} hours ({training_duration/60:.1f} minutes)")
        if stats['total_games'] > 0:
            log_message(f"\nResults:")
            log_message(f"  Checkmate: {stats['checkmate']} ({stats['checkmate']/stats['total_games']*100:.1f}%)")
            log_message(f"    - White wins: {stats['white_wins']}")
            log_message(f"    - Black wins: {stats['black_wins']}")
            log_message(f"  Stalemate: {stats['stalemate']} ({stats['stalemate']/stats['total_games']*100:.1f}%)")
            log_message(f"  Move limit: {stats['max_moves']}")
            log_message(f"  Errors: {stats['errors']}")
            log_message(f"  Interrupted: {stats['interrupted']}")
            log_message(f"\nGame statistics:")
            log_message(f"  Average moves: {stats['total_moves']/stats['total_games']:.1f}")
            log_message(f"  Average game time: {stats['total_time']/stats['total_games']:.1f}s")
            log_message(f"  Total moves: {stats['total_moves']}")
        log_message(f"\nCache size: {len(ai.move_cache)} positions")
        log_message("="*60)
        
        # Final progress update
        update_progress(stats, training_start)


def main():
    """Entry point."""
    setup_signal_handlers()
    
    # Start the background thread that refreshes health every 30s
    health_thread = start_health_updater()
    
    log_message("\n" + "="*60)
    log_message("Mini Chess AI Self-Play Training")
    log_message(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log_message(f"Training window: 02:00-10:00 UTC (8 hours)")
    log_message("="*60)
    
    # Wait for the training window to open if needed
    if not wait_for_training_window():
        log_message("Training did not start - exiting.")
        return
    
    # Self-play parameters
    num_games = None  # None = unlimited, or give a number
    depth = 6         # Search depth (optimized: ~0.5s/move at depth 6)
    exploration_rate = 0.2  # 20% chance of picking the 2nd-best move
    
    run_self_play_training(num_games, depth, exploration_rate)


if __name__ == "__main__":
    main()
