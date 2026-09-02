# The alpha-beta search: draws, MultiPV, and how it is measured

> Extracted from CLAUDE.md.

## Draws in the search

The search scores a repetition as a draw, and the repetition it looks for spans **both** the real
game and its own tree. `GameState.position_history` is one `Vec<u64>` of Zobrist hashes carrying
both: `make_ai_move` pushes and `undo_ai_move` pops, so the search extends the same list the game
built. `history_root` is where the game stops and the tree starts, pinned by `set_search_root()` at
the top of `find_best_move`.

- **A match at or below the root is a draw immediately; a match above it needs a true threefold.**
  Inside the tree either side can usually repeat again, which is why one recurrence is enough there —
  the usual convention, and the same one `build_book.py` searches under. Against game history the
  rule is the game's own: three occurrences.
- **The scan is backwards, steps by two, and stops at the last irreversible move.** Only the same
  side to move can match, and no position survives a capture, a drop, a pawn move or a promotion —
  all four change the board-plus-hands identity for good. `reversible_plies` is that counter, kept by
  `make_ai_move` / `record_position` and restored from `UndoInfo` on the way back, capped at
  `MAX_REPETITION_SCAN`. In Crazyhouse it collapses to zero within a ply or two of almost any node,
  which is why the check does not show up in `bench/`.
- **A null move pushes `NULL_MOVE_SENTINEL`, not the flipped hash.** The null position is not one the
  game can reach, and the two plies after it can walk the board back onto it. A sentinel keeps the
  scan's parity and stops it seeing through the null.
- **A draw score is never written to the TT and never keyed on the hash.** Both checks run before the
  TT probe and before quiescence, and return without storing: whether a position is drawn depends on
  the path to it, and the hash does not carry the path.
- **A draw that rests on *game* history keeps the whole search out of the book** (`DrawKind::
  FromHistory`, `SearchState::draw_used_history`). `probe_book` is keyed on the hash alone, so a row
  saying "this position is 0" would be served to every future game reaching it by another route. An
  in-tree draw is still stored — the builder has no game history, so every draw it finds is that
  kind, and refusing to store them would re-search those nodes forever.

**Carrying the history across the boundary.** `gamestate.py` keys its own history on
`_position_key()` tuples, which hold exactly what the Zobrist hash reads, so `ai._history_hashes`
rebuilds Rust hashes from them without replaying anything and memoises the result. It sends only the
reversible tail (`_reversible_tail`, capped at `HISTORY_LIMIT`) — everything before the last
irreversible move can never recur, and a shorter list is a shorter scan. `ai._sync_to_rust` assigns
it **last**, after board/turn/hands/promoted, because every one of those setters recomputes the hash.
A history whose last entry is not the live hash is discarded by `set_search_root()` rather than
trusted.

**For an MCTS leaf evaluator**: `GameState::is_terminal_draw()` is the game-rule verdict — threefold
or `ply >= ply_limit`, no search heuristic, matching `check_game_over` exactly — and is exposed to
Python as `is_terminal_draw()`. `search_draw()` is the alpha-beta one and is not the same question.

## MultiPV and parallelism

`find_best_move(..., return_top_n=K)` with `K > 1` runs a full-window search of the top K root moves
at the final depth, which is the only way ranks 2+ get scores rather than fail-low bounds. There is no
ply or depth gate: the caller decides where to spend. It is not free — several times a single-PV
search at the same depth.

"Exact" here means procedural: never a fail-low or fail-high bound. It is *not* a claim that the
number is the true depth-N minimax value. Score every root move with its own independent search and
the ranking comes out different, and so does the ordinary single-PV root. That drift comes from
null-move pruning returning a hard `beta` while the TT store site classifies flags against the
original window. Fixing it is a search change needing its own bench and parity pass.

**Root-parallel search (`minimax_parallel`) is off by default** and `build_book.py` passes
`parallel=False` explicitly: the build scales better running many workers on disjoint subtrees than
parallelising one search. The parallel path is for interactive analysis of a single position.
`find_best_move(..., parallel=True/False)` overrides per call; `ai.set_parallel_search(enabled,
min_depth)` sets the process default (`min_depth` = first iterative-deepening depth that gets split,
default 3). There is a `/parallel-search-eval` skill and `.claude/workflows/parallel-search-eval.js`
that automate a measure → propose → implement → verify pass over this path.

## Benchmarking

`bench/` measures the search honestly: one fresh `venv/bin/python` subprocess per measurement, never
loads the on-disk book, reduces repeats with **min**, and flags cells measured under high load.

```bash
./venv/bin/python bench/run_bench.py --depths 6,7,8 --repeats 3 --modes seq,par --out bench/mine.json
./venv/bin/python bench/compare.py bench/results/baseline.json bench/mine.json
./venv/bin/python bench/gen_positions.py     # regenerate bench/positions.json
```

In `compare.py`, best-move **agreement** is the headline, not speed: a parallel run returning a
different move with a different score is a regression regardless of how fast it was, and exits
non-zero.

### Measuring accuracy, not just speed

`run_bench.py` says how fast, and on its own it cannot judge LMR, null move or delta pruning at all:
each of those is faster *because* it is wrong more often. `bench/accuracy.py` says how wrong, against
an exact reference — the same search with null move, LMR, delta pruning and the TT switched off,
which is plain alpha-beta over the same depth-N + quiescence tree and so returns that tree's true
minimax value.

```bash
./venv/bin/python bench/accuracy.py reference --depth 8 --jobs 4     # fill the reference cache
./venv/bin/python bench/accuracy.py run --depth 8 --seeds 8 --configs shipped,lmr_off
./venv/bin/python bench/accuracy.py report bench/results/lmr_confirm.json
./venv/bin/python bench/accuracy.py selfcheck                        # reference is ordering-invariant
./venv/bin/python bench/gen_accuracy_positions.py --append --scale 2 # widen the suite
```

- **Pair over move orderings AND over the whole position sample.** `--seeds N` re-runs every position
  under N value-neutral orderings — a seeded tiebreak between moves the ordering scores equally, which
  an exact search cannot notice and only the heuristics can. One ordering cannot resolve an accuracy
  change and neither can a handful of positions: the same ablation has read −13.5 ± 9.4 on 16
  positions and −82.9 ± 4.3 on 48. Error bars are the standard error over **positions**, and deltas
  are paired per position.
- **Regret is the metric, not score error**: the cp the chosen move gives away against the exact value
  of the position. A search that misreports the score and still plays the best move has lost nothing.
  `best moves` counts regret == 0, and it is by far the most stable of the three — regret itself is
  heavy-tailed, so read its error bar before believing a difference.
- **Node counts are the speed number here**, not seconds: many searches share one process, and
  `--jobs` > 1 makes the clock meaningless. Wall clock is `run_bench.py`'s job.
- **The reference costs ~1 minute of CPU per position at depth 8**, so it is cached by FEN in the
  committed `bench/results/reference_d8.json`. It carries `eval_version` and refuses to load against
  another one; delete it if anything below the search changes — eval, quiescence, move generation.
- **Positions whose exact value is a mate are dropped** (an error against ±1e6 is not a number you can
  average) and per-cell error is capped at 2000cp, which is why `accuracy_positions.json` holds 112
  positions to leave 77 measurable ones.
- The book is off in every search here (`use_book=False`), so nothing reads or writes `book.db`.

`search::Knobs` is what makes any of it possible: six switches — `null_move`, `lmr`, `use_tt`,
`use_book`, `delta_margin`, `order_seed` — snapshotted into `SearchState` once per search and read
from there, never per node and never from a lock. `DEFAULT_KNOBS` is exactly what ships. From Python:
`set_search_knobs({...})`, `reset_search_knobs()`, `get_search_knobs()`, and `last_search_nodes()`.
**They are ablations, not tuning.** The numbers that shape the search are `const`s at the top of
`search.rs` — `DELTA_MARGIN`, `LMR_MIN_MOVE`, `LMR_MIN_DEPTH` — exactly as the eval's numbers are
`const`s at the top of `eval.rs`, so sweeping one means editing it, rebuilding, and reporting against
a results file saved from the old build. Nothing in the front ends touches a knob.

LMR itself, measured that way over 77 positions × 16 orderings at depth 8: it reduces from the
**9th** move (`LMR_MIN_MOVE`), and `is_noisy_move` — which exists only to say what LMR may not
reduce — counts a drop as noisy only **next to the enemy king**, the same test quiescence uses. Both
numbers were measured, and both are crazyhouse-specific: reducing from the 5th move cost 2.75 ± 1.64
best moves of 77, and exempting every drop meant LMR did almost nothing in exactly the positions
where the branching factor is worst.

