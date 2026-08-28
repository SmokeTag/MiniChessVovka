# Future Improvements for Mini Chess 6x6 Crazyhouse

This document lists recommended improvements that could be implemented to increase AI strength, execution speed, and code quality.

---

## 🚀 Urgent improvements (quick to implement + large effect)

### 1. Opening Book — ✅ **DONE**
**Problem:** the AI spent 10-60 seconds on the first 5-8 moves of positions that are well studied.

**What shipped** is not what this section originally proposed. It is not "50-100 popular openings"
keyed by frequency — there is no corpus of minihouse games to take frequencies from. It is a
**repertoire computed by deep search**: `build_book.py` walks only the positions where the engine is
to move, one search and one child at our nodes, every legal reply at the opponent's. Two SQLite
tables (`book_move`, `position`), not one, because a Zobrist hash cannot be turned back into a
position without a FEN stored beside it. See **Filling the opening book** in `CLAUDE.md`.

**Effect, measured:** first moves answer in ~0.00s off a `[BOOK HIT]` instead of 3-30s.

---

### 2. Aspiration Windows
**Problem:** iterative deepening runs without aspiration windows, which slows the search down.

**Solution:**
- Add aspiration windows based on the previous iteration
- Use a narrow [alpha, beta] range around the previous result
- On a fail high/low, widen the window

**Implementation:**
```python
# In find_best_move, inside iterative deepening
aspiration_window = 50
for d in range(1, depth + 1):
    if d >= 4 and best_score is not None:
        alpha = best_score - aspiration_window
        beta = best_score + aspiration_window
    else:
        alpha, beta = -float('inf'), float('inf')
    
    score, move = minimax_alpha_beta(gamestate, d, alpha, beta, is_maximizing)
    
    # If we fell outside the window - repeat with full bounds
    if score <= alpha or score >= beta:
        score, move = minimax_alpha_beta(gamestate, d, -float('inf'), float('inf'), is_maximizing)
```

**Effect:**
- ⚡ +20-30% speed in the middlegame
- 📉 Fewer nodes to explore

**Time estimate:** 1-2 hours

---

### 3. Null-Move Pruning — ✅ **DONE**
Implemented in `engine_rs/src/search.rs` (`allow_null`, `null_move_r = 2`), gated on not being in
check and on the opponent holding an empty hand — a side with pieces to drop is never in zugzwang,
and dropping is exactly the counterplay a null move assumes away.

One caveat, documented on `search_worker` and in `CLAUDE.md`: it returns a hard `beta`, which the TT
store site then classifies against the original window. That is the source of the score drift
between single-PV and MultiPV roots. Fixing it is a search change needing its own bench pass.

---

### 4. Code cleanup and optimization
**Problem:** ~~Many DEBUG prints slow things down~~ ✅ **FIXED**

**Additionally:**
- Add type hints to all functions
- Write docstrings in a consistent style
- Profile the code to find bottlenecks

**Implementation:**
```python
# Profiling
import cProfile
cProfile.run('ai.find_best_move(gamestate, depth=16)', 'ai_profile.stats')

# Analysis
import pstats
p = pstats.Stats('ai_profile.stats')
p.sort_stats('cumulative').print_stats(20)
```

**Effect:**
- ~~⚡ +5-10% speed~~ ✅ Achieved
- 🧹 Readable code
- 📊 Understanding of the bottlenecks

**Time estimate:** 2-4 hours

---

## 🎯 Medium improvements (several days of work)

### 5. Late Move Reduction (LMR) — ✅ **DONE**
In `engine_rs/src/search.rs`: `lmr_full_depth = 4`, `lmr_reduction_limit = 3`, skipped for noisy
moves and for moves that give check, with a full-depth re-search when a reduced search beats alpha.

---

### 6. Principal Variation Search (PVS) — ✅ **DONE**
Also in `minimax_ab` — see the header comment on it: "alpha-beta, PVS, LMR, null-move, check
extensions, TT". Note the interaction with the book: ranks below 1 cannot be read off a PVS root,
because every move after the first returns a *bound* rather than a value. That is why
`multipv_root` exists and why it costs 2.4-4.4x.

---

### 7. Improved evaluation function
**Current:** basic evaluation (material + position + center + king)

**To add:**
- Mobility (piece mobility)
- King safety (advanced king-safety evaluation)
- Piece coordination
- Pawn shield (pawn cover in front of the king)
- Outposts (for knights)

**Effect:**
- 📈 +100-200 ELO
- 🎯 More accurate positional play

**Time estimate:** 8-12 hours

---

## 🔥 Long-term improvements (weeks of work)

### 8. Bitboards
**Problem:** the 2D array `board[r][c]` is slow for these operations.

**Solution:**
- Represent the board as a set of 64-bit integers (bitboards)
- One bitboard per piece type
- Use bitwise operations (&, |, ^, ~, <<, >>)

**Example:**
```python
class Bitboard:
    def __init__(self):
        self.white_pawns = 0b0000000000000000  # 6x6 = 36 bits
        self.black_pawns = 0b0000000000000000
        # ... one per piece
    
    def get_piece_attacks(self, square):
        """Returns the attacked squares in O(1)"""
        return ATTACK_TABLES[piece_type][square]
```

**Effect:**
- ⚡ **10-20x speedup** of operations
- 🎯 Depth 20-24 becomes realistic
- 📈 +300-500 ELO from the extra depth

**Time estimate:** 40-80 hours (full rewrite)

---

### 9. Neural Network Evaluation
**Current:** hand-written evaluation function

**Solution:**
- Train a neural network on self-play games
- Architecture: CNN or Transformer
- Input: board state (6x6x12 channels, one per piece)
- Output: value (-1 to +1) + policy (move probabilities)

**Implementation:**
```python
# No stub exists: the old nn/model.py was deleted (its action space was
# board_size**4, which cannot express a drop or a promotion choice).
# Start from the Rust GameState and an encoding that covers drops.
model = PolicyValueNet()          # 24 input planes at 6x6, 2196 policy logits
model.load_state_dict(torch.load('best_model.pth'))

def nn_evaluate(board_state):
    tensor = board_to_tensor(board_state)
    with torch.no_grad():
        value, policy = model(tensor)
    return value.item()
```

**Training:**
1. Self-play (the AI plays against itself)
2. Save the games
3. Train on the results
4. Iterative improvement (AlphaZero style)

**Effect:**
- 📈 +300-500 ELO
- 🎯 More accurate evaluation of complex positions
- 🧠 Understanding of subtle nuances

**Time estimate:** 60-120 hours + a GPU for training

---

### 10. Monte Carlo Tree Search (MCTS) + NN
**Solution:**
- Combine MCTS with neural-network evaluation (as in AlphaZero)
- Write it against the Rust `GameState` (make/undo, no allocation). The old `nn/mcts.py` stub was
  deleted: it deep-copied a state per child and never negated the value between players.

**Algorithm:**
1. Selection: pick a node by UCB
2. Expansion: expand the node
3. Evaluation: evaluate via the NN
4. Backpropagation: update the statistics

**Effect:**
- 📈 +500-800 ELO
- 🎯 ~2800-3000 ELO (top level)
- 🧠 AlphaZero-like strength

**Time estimate:** 80-150 hours + a GPU

---

### 11. Endgame Tablebase
**Solution:**
- Precompute all positions with ≤4 pieces
- Store them in a database (SQLite or RocksDB)
- Use it for perfect endgame play

**Format:**
```
position_hash -> {result: 'WIN/LOSS/DRAW', moves_to_mate: 12}
```

**Effect:**
- ♟️ Perfect endgame play
- 📈 +50-100 ELO in the endgame
- ⏱️ Instant moves in simple positions

**Time estimate:** 30-50 hours + compute power

---

## 📊 Implementation priorities

### Phase 1: quick wins (1-2 weeks)
1. ✅ DEBUG code cleanup
2. ✅ Opening Book — shipped as a repertoire, see above
3. Aspiration Windows — **the only one of these still open**
4. ✅ Null-Move Pruning

---

### Phase 2: algorithmic improvements (2-4 weeks)
5. ✅ Late Move Reduction
6. ✅ Principal Variation Search
7. Improved evaluation function — open

---

### Phase 3: large-scale changes (2-4 months)
8. Bitboards (full rewrite)
9. Neural Network Evaluation
10. MCTS + NN (AlphaZero approach)
11. Endgame Tablebase

**Result:** ~2800-3200 ELO, professional level

---

## 🛠️ Tools and libraries

### For training neural networks:
- PyTorch (already in requirements.txt)
- TensorBoard for visualization
- CUDA for GPU acceleration

### For profiling:
```bash
# Profiling Python
python -m cProfile -o profile.stats main.py

# Analysis
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(30)"

# Visualization
pip install snakeviz
snakeviz profile.stats
```

### For strength testing:
```python
# AI vs AI tournament
def tournament(ai_versions, games_per_pair=10):
    for ai1, ai2 in combinations(ai_versions, 2):
        results = play_matches(ai1, ai2, games_per_pair)
        print(f"{ai1.name} vs {ai2.name}: {results}")
```

---

## 📚 Resources for further reading

### Chess programming:
- Chess Programming Wiki: https://www.chessprogramming.org
- Stockfish source code: https://github.com/official-stockfish/Stockfish
- Sunfish (a simple engine in Python): https://github.com/thomasahle/sunfish

### AlphaZero / neural networks:
- AlphaZero paper: https://arxiv.org/abs/1712.01815
- Leela Chess Zero: https://lczero.org
- PyTorch tutorial: https://pytorch.org/tutorials/

### Optimization:
- Python Performance Tips: https://wiki.python.org/moin/PythonSpeed
- Numba (JIT compilation): http://numba.pydata.org

---

## 📝 Notes

- **Current strength:** ~2000-2200 ELO (depth 16)
- **Potential without neural networks:** ~2600-2800 ELO (bitboards + optimizations)
- **Potential with neural networks:** ~3000-3200 ELO (MCTS + NN)

**Priority:** start with Phase 1 (Opening Book, Aspiration, Null-Move) - it gives the largest effect for the least time.

---

*Document created: 2024*  
*Last updated: after the DEBUG code cleanup*
