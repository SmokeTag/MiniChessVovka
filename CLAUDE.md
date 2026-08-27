# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A 6×6 Crazyhouse ("minihouse") chess engine. The search/eval core is Rust (`engine_rs/`), exposed to
Python as the `minichess_engine` extension module via PyO3. Python drives it from three front ends:
a Pygame GUI (`main.py` + `gui.py`), a Chess.com browser bot (`play_online.py`), and self-play
training (`src/self_play.py`, `src/scheduled_self_play.py`).

## Build & run

The Rust extension is installed into `./venv` by maturin. **Rust edits do not take effect until you
rebuild** — Python keeps importing the stale `.so` and gives no warning:

```bash
source venv/bin/activate
cd engine_rs && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release && cd ..
```

The env var is required here: the venv is on Python 3.14 and PyO3 0.24 refuses anything past 3.13
without it. Plain `maturin develop --release` fails with "configured Python interpreter version
(3.14) is newer than PyO3's maximum supported version".

Always invoke Python through the project venv (`./venv/bin/python`, or activate it) — that is where
`minichess_engine` lives.

```bash
./play.sh                          # Pygame GUI (venv/bin/python main.py)
./train.sh                         # self-play training, Ctrl+C stops gracefully
./train_parallel.sh 20 10          # 20 self-play workers at depth 10 (see below)
./train_status.sh ; ./train_stop.sh
./bot_start.sh casual|rated        # chess.com bot (needs .env + playwright chromium)
./bot_stop.sh ; ./monitor_bot.sh
./venv/bin/python precalc_openings.py   # fill the opening cache by deep search
```

## Filling the move cache in parallel

`src/self_play.py` takes `--depth/--games/--exploration/--random-plies/--seed/--quiet`. Throughput
comes from **many single-threaded workers**, never from `parallel=True` — see the parallelism policy
below.

```bash
./train_parallel.sh 20 10     # 20 workers, depth 10; logs in logs/selfplay/, PIDs in training_workers.pid
./train_status.sh             # workers alive + row counts per depth
./train_stop.sh               # SIGTERM, waits for each worker to flush, SIGKILL after 60s
```

Three things about this that are easy to get wrong:

- **Workers must run from the repo root.** `DB_PATH` in `cache.rs` is the relative string
  `"move_cache.db"`, so a worker started elsewhere silently creates its own DB. `self_play.py`
  now `os.chdir`s to the repo root itself (and puts it on `sys.path`; without that,
  `python src/self_play.py` cannot import `gamestate`).
- **Saves are incremental.** `save_move_cache_to_db` used to re-`INSERT OR REPLACE` the process's
  entire cache after *every move* — ~5µs/row, so ~1.3s per move once a worker holds 250k entries,
  times N workers on one write lock. `search::find_best_move` now returns the keys it inserted via
  a `dirty` out-param, `lib.rs` accumulates them in `DIRTY_KEYS`, and only those rows get written.
  Any new `move_cache.insert` in `search.rs` **must** push its key onto `dirty` or the entry never
  reaches disk.
- **The DB is in WAL mode** with a 30s busy timeout (`cache::open_db`). Sidecar files
  `move_cache.db-wal` / `-shm` are gitignored.

Depth 10 costs roughly: median ~3s/move, p90 ~12s, worst seen ~31s — crazyhouse midgames with full
hands are far more expensive than the opening. `--random-plies` puts each worker in a different
position so N workers cover new ground instead of re-searching the same tree. Workers load the DB
only at startup, so they do not see each other's entries until restarted.

Note that **purging shallow rows is a one-off, not a steady state**: iterative deepening caches the
best move at every depth it passes through, and the TT dump persists anything at depth ≥ 4, so a
depth-10 run repopulates depths 1–9 within minutes. `find_best_move`'s `time_limit` is also a trap
here — a search cut short still stores its result under the *requested* depth, so time-limited
training would file shallow answers as deep ones.

## Tests

`tests/` mixes unittest classes and bare pytest functions; pytest runs both.

```bash
./venv/bin/python -m pytest tests/ -q
./venv/bin/python -m pytest tests/test_e2e.py::TestE2EGameFlow::test_promotion_flow -q
./venv/bin/python -m pytest tests/test_ai_optimization.py -q -s
```

Tests are slow by design — several play full AI-vs-AI games. Prefer running a single test while
iterating.

**No test may write to the repo-root `move_cache.db`** — that file is the live training cache, and
`test_nightly.py` drops and recreates the table. `DB_PATH` in `cache.rs` is the relative string
`"move_cache.db"` with no override, so the only seam is the CWD: `tests/cache_isolation.py` gives an
`isolated_cache_db()` context manager and an `IsolatedCacheDB` TestCase base that run a test in a
throwaway directory and reload the real cache afterwards (the Rust cache is process-global, so a
test that skipped the reload would leave its scratch rows to be written back by a later test).
Anything the NN work adds — replay buffers, checkpoints — goes under a configurable path outside the
repo root for the same reason. A full `pytest tests/ -q` should leave `move_cache.db` byte-identical;
`md5sum` it before and after if you touch this.

Rust has no test suite; `cargo build --release` inside `engine_rs/` only checks that it compiles —
use `maturin develop --release` for anything you intend to run.

## Architecture

**Two independent implementations of the same rules.** `gamestate.py` (Python) owns the live game —
GUI, bot, and training all mutate it — while `engine_rs/src/gamestate.rs` re-implements move
generation for the search. `ai._sync_to_rust()` copies board/turn/hands/king_pos/promoted_pieces into
a fresh Rust `GameState` on *every* `find_best_move` call. A rules change (a new drop restriction,
promotion behaviour, check detection) must land in **both** or the engine will search a position the
GUI does not agree with. `tests/test_rules_parity.py` is what catches that drift: it plays identical
random games through both and compares legal-move sets, terminal flags, ply counters and promotion
state at every ply. Run it with `RULES_PARITY_GAMES=500` after any rules change — the 50-game default
is a smoke test, not a sweep.

**The Pygame front end is `main.py` + `gui.py` + `layout.py` + `settings.py`.**
`layout.Layout` recomputes every rectangle from the current window size — nothing else may
assume a pixel dimension, and `config.SQUARE_SIZE` survives only because `pieces.py`'s unused
vector primitives import it. Four invariants are load-bearing and easy to undo by accident:

- **The search never blocks the UI.** `main.Game` polls a worker thread and keeps drawing;
  Undo / New game / Flip / Hint stay live during a search. Results carry a `generation` tag
  and are discarded if the position moved on, so taking back a move mid-search cannot apply a
  stale answer to the wrong board. Anything that changes the position must call
  `Game.invalidate()` (which bumps the generation) or that guarantee breaks.
- **Panel zones have fixed heights** (`layout.py`): header, controls and the bottom status band
  are reserved, and only the move list flexes. Hands live in strips beside the board rather
  than in the panel, and show a `×N` count instead of one sprite per copy — both so that a
  filling hand can never push a button out from under the cursor.
- **Redraw is dirty-flagged.** `Game.dirty` gates rendering; the loop idles at `IDLE_FPS`
  and only rises to `FPS` while something animates. Fonts and scaled sprites are memoised in
  `gui.py`, so a still board costs almost nothing. Any new per-frame `smoothscale` undoes this.
- **Flip is a view transform only.** It must never change what is true about the position —
  the previous build swapped which king it tested for check, which silently disabled the check
  highlight on a flipped board.

Drawing takes a `BoardView` snapshot rather than reading the live `GameState`, which is what
lets the arrow keys render any past ply from `gamestate.saved_states`. UI state lives in
`main.UIState`; the `selected_square` / `highlighted_moves` fields on `GameState` are legacy and
the GUI no longer uses them. Hit regions are produced by the code that draws each control, so a
control can never be clickable where it is not visible — and the promotion picker clears every
other region, which is what stops clicks reaching the buttons underneath it.

`gui_settings.json` (gitignored) remembers window size, depth, AI toggles, hints and orientation.

**`ai.py` is a thin shim, not the AI.** It exists so `gui.py`, `play_online.py`, and the training
scripts keep a stable Python API while all real work happens in Rust. Its module-level `move_cache`
and `tt` dicts are vestigial — the Rust side owns both caches.

**Move representation** (same tuples on both sides of the PyO3 boundary):
- normal: `((r1, f1), (r2, f2), promotion_char_or_None)`
- drop: `('drop', 'wN', (r, f))` — colour+type code, target square

Coordinates are `(row, file)` with **row 0 = rank 6** (top). `utils.coords_to_algebraic` /
`algebraic_to_coords` are the only correct converters; don't hand-roll the flip.

**Caching.** `engine_rs/src/cache.rs` persists a `(position hash, depth) → best move` table to
`move_cache.db` (SQLite, repo root, gitignored) and there is a separate in-memory transposition table
per search. The persistent cache is process-global and filled as you search, so **searching the same
(position, depth) twice in one process is a cache hit that reports ~0s** — the reason `bench/` forks a
fresh process per measurement.

**Parallelism policy — deliberate, don't "fix" it.** Root-parallel search
(`minimax_parallel` in `search.rs`) is **off by default**. Self-play training explicitly passes
`parallel=False` and calls `ai.set_parallel_search(False)`: training scales better by running many
independent games across cores than by parallelising one search. The parallel path is for interactive
analysis of a single position. `find_best_move(..., parallel=True/False)` overrides per call;
`ai.set_parallel_search(enabled, min_depth)` sets the process default (`min_depth` = first
iterative-deepening depth that gets split, default 3).

## Benchmarking the engine

`bench/` measures the search honestly: one fresh `venv/bin/python` subprocess per measurement, never
loads the on-disk move cache, reduces repeats with **min** (background load is additive noise), and
flags cells measured under high load average.

```bash
./venv/bin/python bench/run_bench.py --depths 6,7,8 --repeats 3 --modes seq,par --out bench/mine.json
./venv/bin/python bench/compare.py bench/results/baseline.json bench/mine.json
./venv/bin/python bench/gen_positions.py     # regenerate bench/positions.json
```

Positions are stored as replayed move lists (there is no FEN parser here). In `compare.py`, best-move
**agreement** is the headline, not speed: a parallel run returning a different move with a different
score is a regression regardless of how fast it was, and exits non-zero.

There is a `/parallel-search-eval` skill and `.claude/workflows/parallel-search-eval.js` that automate
a full measure → propose → implement-in-worktrees → verify pass over this path.

## Gotchas

- `config.AI_MOVE_DELAY` is gone. It was a flat 1.5s stall imposed *after* the search returned
  and *before* the move was drawn, with input blocked throughout — so it made nothing more
  visible and only added latency. `MIN_MOVE_DISPLAY` replaces it: a floor on how long a move
  stays on screen before the *next* search starts, which never blocks input.
- `bot_start.sh` rewrites `play_online.py` in place with `sed -i ''` to flip rated/casual — BSD syntax
  that fails on Linux (GNU sed reads `''` as the script). Set `rated=` in `create_minihouse_game` by
  hand, or fix the sed, rather than assuming the mode switched.
- `autorun_training.sh`, `minichesstrain.service`, and `minichesstrain.timer` are pinned to a
  deployment path (`/srv/MiniChessVovka`, user `ubuntu`) and are not runnable as-is from a checkout.
- `nn/` and `engine/env.py` are **deleted**. They were an unfinished torch/RL experiment that nothing
  imported, and they were actively misleading as a starting point for the planned AlphaZero work:
  `nn/model.py` sized the action space as `board_size**4`, which cannot express a drop or a promotion
  choice, and `nn/mcts.py` never negated the value between alternating players. Anything neural starts
  from the Rust `GameState`, not from those files (`git show 709f963:nn/model.py` if you want them back).
- `config.py`'s eval constants (`CENTER_BONUS`, `KING_SAFETY_BONUS`, `PIECE_VALUES` in `pieces.py`, …)
  are dead weight from the pre-Rust era — the numbers that actually decide moves are the `const`s at
  the top of `engine_rs/src/eval.rs`. Editing the Python ones changes nothing.
- `docs/ARCHITECTURE.md` is stale and wrong (claims bitboards; the board is a list of chars).
  `docs/evaluation_strategy.md` and the README are accurate.
- Console output across the Python layer is heavily print-based and partly emoji-formatted; the bot
  and training scripts are read via their logs (`/tmp/minichess_bot.log`, `training_log.txt`).
