# CLAUDE.md

## What this is

A 6×6 Crazyhouse ("minihouse") chess engine. The search/eval core is Rust (`engine_rs/`), exposed to
Python as the `minichess_engine` extension module via PyO3. Python drives it from three front ends:
a Pygame GUI (`main.py` + `gui.py`), a Chess.com browser bot (`play_online.py`), and the opening
book builder (`build_book.py`).

Deeper notes live in `docs/`, and are worth loading before working in the area they cover:

| doc | covers |
| --- | --- |
| `docs/BOOK.md` | filling the opening book, the staged parallel build, the on-disk schema, export/import |
| `docs/BOOK_FILES.md` | which book file is which, as a cheat sheet |
| `docs/ENCODING.md` | the network's 24 input planes and 2,196-action policy map, and the fuzz that holds them |
| `docs/GUI.md` | the Pygame front end's invariants and the user-visible search settings |
| `docs/SEARCH.md` | draws in the search, MultiPV/parallelism, `bench/`, and the accuracy harness |
| `docs/ZERO.md` | the neural engine: dependencies, teacher data, the network, and the arena |
| `docs/evaluation_strategy.md` | what the evaluation measures and why |

## Build & run

**Rust edits do not take effect until you rebuild** — Python keeps importing the stale `.so` and
gives no warning:

```bash
source venv/bin/activate
cd engine_rs && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release && cd ..
```

The env var is required: the venv is on Python 3.14 and PyO3 0.24 refuses anything past 3.13
without it. Always invoke Python through the project venv — that is where `minichess_engine` lives.

```bash
./play.sh                          # Pygame GUI
./bot_start.sh casual|rated        # chess.com bot (needs .env + playwright chromium)
./bot_stop.sh ; ./monitor_bot.sh
./build_book_parallel.sh 20 6 10   # fill the opening book — read docs/BOOK.md first
```

## Tests

`tests/` mixes unittest classes and bare pytest functions; pytest runs both.

```bash
./venv/bin/python -m pytest tests/ -q
./venv/bin/python -m pytest tests/test_e2e.py::TestE2EGameFlow::test_promotion_flow -q
```

Tests are slow by design — several play full AI-vs-AI games. Prefer running a single test while
iterating.

**No test may write to the repo-root `book.db`** — that file is the live book, and `test_nightly.py`
drops and recreates its tables. `DB_PATH` is the relative string `"book.db"` with no override, so the
only seam is the CWD: `tests/cache_isolation.py` gives an `isolated_cache_db()` context manager and
an `IsolatedCacheDB` base that run a test in a throwaway directory and reload the real book
afterwards (the Rust book is process-global, so a test skipping the reload leaves its scratch rows to
be written back by a later test). **Never call `ai.setup_db()` outside that isolation.** Anything new
that persists — replay buffers, checkpoints, self-play games — goes under a configurable path outside
the repo root for the same reason. A full `pytest tests/ -q` must create no repo-root `book.db`.

Rust has a small unit suite (`fen.rs`): `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --release`
inside `engine_rs/`. `cargo build --release` only checks that it compiles — use
`maturin develop --release` for anything you intend to run.

## Architecture

**Two independent implementations of the same rules.** `gamestate.py` owns the live game — GUI, bot
and book builder all mutate it — while `engine_rs/src/gamestate.rs` re-implements move generation for
the search. `ai._sync_to_rust()` copies board/turn/hands/king_pos/promoted_pieces into a fresh Rust
`GameState` on *every* `find_best_move` call. A rules change (drop restriction, promotion behaviour,
check detection) must land in **both** or the engine searches a position the GUI does not agree with.
`tests/test_rules_parity.py` catches that drift: identical random games through both, comparing legal
moves, terminal flags, ply counters and promotion state at every ply. Run it with
`RULES_PARITY_GAMES=500` after any rules change — the 50-game default is a smoke test.

### The PyO3 boundary

**`ai.py` is a thin shim, not the AI.** It exists so `gui.py`, `play_online.py` and `build_book.py`
keep a stable Python API while all real work happens in Rust. Its module-level `move_cache` and `tt`
dicts are vestigial — the Rust side owns both; use `ai.book_size()` for a count that is true.
`_sync_to_rust` rebuilds the Rust state on every call, so anything the book records about a position
(currently `ply`) has to be copied there or it arrives as a default.

**Move representation** (same tuples on both sides):
- normal: `((r1, f1), (r2, f2), promotion_char_or_None)`
- drop: `('drop', 'wN', (r, f))` — colour+type code, target square

Coordinates are `(row, file)` with **row 0 = rank 6** (top). `utils.coords_to_algebraic` /
`algebraic_to_coords` are the only correct converters; don't hand-roll the flip. **Case encodes
colour throughout** — board chars, `'wN'` drop codes, promotion chars. Rust's `Move` carries a bare
`PieceType` with no colour, so `rust_move_to_py` derives promotion case from the destination rank.
Any new Rust→Python move conversion must follow that.

`lib.rs`'s book accessors take the `BOOK` mutex **without releasing the GIL**, so a main-thread
`ai.book_has_position` blocks on any lock a background search holds. `find_best_move` therefore takes
`&Mutex<Book>` and locks at exactly two points — the probe at the top, the store at the bottom. **Do
not put the lock back around the search**, and do not add a book accessor to `lib.rs` that holds it
across a DB write without `py.allow_threads`.

### FEN

`engine_rs/src/fen.rs` carries exactly what the Zobrist hash reads — board, side to move, both hands,
promoted squares — and nothing path-dependent. That equivalence is the point: `from_fen(to_fen(gs))`
must hash back to `gs.hash`, which `tests/test_fen.py` asserts over random games. Conventions at 6×6:
`2bnrk/5p/6/6/P5/KRNB2[] w`, hands in brackets (uppercase White), `~` marking a promoted piece, ranks
top row first. Exposed as `ai.to_fen` / `ai.from_fen`.

### Scores

**Scores are white-relative everywhere.** `search::find_best_move`, `eval::evaluate_position` and
`ai.find_best_move_with_score` all return positive-favours-White whichever side is to move;
`utils.format_score` / `score_advantage` are the only formatters. `eval.rs` encodes a mate as flat
±`CHECKMATE_SCORE` carrying no mate distance, so the display says "White mates", never "M3".

### Draws

The search scores a repetition as a draw, and the repetition it looks for spans **both** the real
game and its own tree — `GameState.position_history` is one `Vec<u64>` of Zobrist hashes holding
both, with `history_root` pinned by `set_search_root()` at the top of `find_best_move`. A match at or
below the root is a draw immediately; a match above it needs a true threefold. A draw score is never
written to the TT (it depends on the path, and the hash does not carry the path), and a draw resting
on *game* history keeps the whole search out of the book.

**For an MCTS leaf evaluator**: `GameState::is_terminal_draw()` is the game-rule verdict — threefold
or `ply >= ply_limit`, no search heuristic, matching `check_game_over` exactly — and is exposed to
Python as `is_terminal_draw()`. `search_draw()` is the alpha-beta one and is not the same question.
Full mechanism in `docs/SEARCH.md`.

### The network encoding

`engine_rs/src/encode.rs` turns a `GameState` into the tensors a policy-value network reads and
writes — phase 1 of Minihouse Zero. **Every tensor is in the frame of the side to move**: with
Black to move the board is rotated 180° and the colours swapped (`s -> 35 - s`), which is exact
here because there is no castling and no en passant. 24 input planes at 6×6 (864 floats,
`plane * 36 + r * 6 + f`); 61 action planes over the board = 2,196 logits, `plane * 36 + square`,
where rays/knights/promotions index the **origin** square and drops index the **target**.

**A queen cannot arise** — none in the initial array, and `PROMOTION_PIECES` is `[R, N, B]` — so
nothing encodes one, which is why the piece planes are 5 wide and the hand planes 4.

The two invariants are properties of a position: no two legal moves of one position share an
index, and `index_to_move(move_to_index(m)) == m`. Both are held by a round-trip fuzz on each
side of the boundary (`encode.rs` tests, `tests/test_encoding.py`) rather than by inspection — a
scrambled encoding still trains to a plausible loss curve. **A change to the plane layout or the
action map invalidates every trained checkpoint**, so it is a versioned decision, not a tweak.

From Python: `encode_position`, `move_to_action_index`, `action_index_to_move`,
`legal_action_indices`, plus `ENCODE_PLANES` / `ENCODE_INPUT_SIZE` / `ACTION_PLANES` /
`ACTION_SPACE`. All four take the **Rust** `GameState`; from a Python one go through
`ai._sync_to_rust`. They return plain lists, not numpy arrays, deliberately — `numpy` stays out
of the GUI's and the bot's import path. Full layout in `docs/ENCODING.md`.

### The neural engine (`nn/`)

Minihouse Zero — an AlphaZero-style policy-value net and MCTS on top of this core.
Phases 0 (draw rules) and 1 (encoding) are done; phase 2 is the network and its
alpha-beta teacher data. Full account in `docs/ZERO.md`.

**Only `nn/` may import torch or numpy.** The GUI, the bot, the book and the test suite
run on `requirements.txt` alone, and `tests/test_nn.py` skips itself when torch is
absent, so a plain `pytest tests/ -q` stays green without it. The NN deps live in the
**same venv** (`requirements-nn.txt`, `torch==2.11.0+cu128` — cu128 is required for this
machine's Blackwell GPU): a separate venv-nn would mean `maturin develop` into two
interpreters, and a stale `.so` is this project's worst failure mode.

**Nothing generated goes in the repo root.** `nn/paths.py` is the only answer —
`$MINIZERO_DATA`, else `~/.local/share/minihouse-zero`. Same rule, same reason as
`tests/cache_isolation.py`.

```bash
./venv/bin/python -m nn.teacher generate --positions 50000 --jobs 20 --depth 8
./venv/bin/python -m nn.train --epochs 40
./venv/bin/python -m nn.arena --opponent alphabeta --depth 2 --games 200
```

Three things that are not guessable from the code:

- **Teacher records store the position, not its encoding** (FEN + `ply` + `reps`), so a
  plane-layout change costs a re-encode rather than hours of regeneration. The generator
  asserts per position that `teacher.restore()` reproduces the live encoding — without
  it the progress and repetition planes differ between training and play, silently.
- **Walks run on the Rust `GameState`, never `gamestate.py`.** The Python one costs
  milliseconds a ply against ~126k plies/s, which made the walk — not the depth-8
  label — the bottleneck, and the generator 15x slower.
- **Every teacher search runs `use_book=False`**, which gates the probe and the store, so
  bulk generation never reads or writes `book.db`.

Checkpoints carry the encoding they were trained against and refuse to load against a
different one: otherwise a plane-layout change loads cleanly and plays nonsense.

**MCTS lives in Rust, the batch loop in Python** (`engine_rs/src/mcts.rs`, `nn/mcts.py`).
Rust descends until it has a batch of leaves needing evaluation, hands their planes across
one call, and takes back priors and values; Rust threads calling *into* torch would
serialise on the GIL. Three things are not guessable from the code:

- **The batch exists because of virtual loss.** A descent that reaches an unevaluated
  leaf marks its path as a loss so the next descent in the same `collect` goes elsewhere.
  Without it every descent returns the same leaf and a batch is worth one simulation.
- **The mask is implicit.** Rust reads priors only at the leaf's legal actions and
  renormalises, so a plain softmax in Python *is* a masked softmax — the partition
  function cancels. There is no masking step to forget.
- **An empty `collect` is not the stop condition.** A descent ending on a terminal
  position backs up without producing work, so the driver stops only when a `collect`
  returns no leaves *and* the simulation count did not move. Terminal positions are
  scored by the rules and never cost an evaluation; the root gets that verdict too, or a
  search from a finished game expands a node with no edges.

Values are in the frame of the side to move at each node, so `backup` flips sign every
step up the path. `tests/test_mcts.py` drives all of it against a deliberately uniform
evaluator — with a trained network you cannot tell a working search from a working
policy.

**The GUI can play the network** — the Engine button, via `nn/backend.py`, and it
searches: `nn/mcts.py` drives the Rust tree and plays the most-visited root move.
Selecting it is what imports torch: `thread_utils.AIThread` imports the backend inside
`run()`, never at module scope, and `tests/test_nn.py` checks in a subprocess that
importing the front ends pulls no torch. **Its budget is a time, not a simulation
count** — a simulation costs whatever the position makes it cost, so only the clock is a
promise a GUI can keep; `nn.mcts.search` takes `simulations`, `time_limit` or both, and
the arena and self-play keep counting simulations because a measurement wants
reproducibility. The readout says `network`, not `depth N`. See `docs/GUI.md`.

### The opening book, in one paragraph

`engine_rs/src/cache.rs` persists the book to `book.db` (SQLite, repo root, gitignored); `book.tsv`
is the versioned form and is what goes in git. It is read in exactly one place — `probe_book`, at the
root of `find_best_move` — so a row is only ever consulted where **the engine is to move**. The GUI
never writes to it on its own; `build_book.py` and `play_online.py` write every search they run.
`eval_version` guards the rows and is bumped by hand when an eval constant changes. Everything
else — the staged parallel build, the schema, export/import, the analysis-cache merge rule — is in
`docs/BOOK.md`, and reading it is not optional before touching the builder.

### Search knobs and measurement

`search::Knobs` — `null_move`, `lmr`, `use_tt`, `use_book`, `delta_margin`, `order_seed` — is
snapshotted into `SearchState` once per search and read from there. From Python:
`set_search_knobs({...})`, `reset_search_knobs()`, `get_search_knobs()`, `last_search_nodes()`.
**They are ablations, not tuning**: the numbers that shape the search are `const`s at the top of
`search.rs`, as the eval's numbers are `const`s at the top of `eval.rs`. Nothing in the front ends
touches a knob.

```bash
./venv/bin/python bench/run_bench.py --depths 6,7,8 --repeats 3 --out bench/mine.json
./venv/bin/python bench/compare.py bench/results/baseline.json bench/mine.json
./venv/bin/python bench/accuracy.py run --depth 8 --seeds 8 --configs shipped,lmr_off
```

`run_bench.py` says how fast; on its own it cannot judge LMR, null move or delta pruning at all,
since each is faster *because* it is wrong more often. `bench/accuracy.py` says how wrong, against an
exact reference. Read `docs/SEARCH.md` before trusting a number out of either — in particular, why
best-move **agreement** is the headline in `compare.py` and why **regret** is the metric in
`accuracy.py`.

Root-parallel search is **off by default**; it is for interactive analysis of one position, not for
throughput (many workers on disjoint subtrees scales better). MultiPV (`return_top_n=K`) costs
several times a single-PV search at the same depth.

## Gotchas

- `bot_start.sh` rewrites `play_online.py` in place with `sed -i ''` to flip rated/casual — BSD syntax
  that fails on Linux. Set `rated=` in `create_minihouse_game` by hand rather than assuming the mode
  switched.
- The numbers that decide moves are the `const`s at the top of `engine_rs/src/eval.rs`
  (`CENTER_BONUS`, `KING_SAFETY_BONUS`, `PIECE_VALUES`); there is no Python copy of them any more.
  Editing one means bumping `EVAL_VERSION` in the same file, or the book keeps serving scores from
  the evaluation you just replaced.
- Console output across the Python layer is heavily print-based and partly emoji-formatted; the bot
  and the book builder are read via their logs (`/tmp/minichess_bot.log`, `logs/book/`).
