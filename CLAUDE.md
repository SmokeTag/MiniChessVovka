# CLAUDE.md

## What this is

A 6×6 Crazyhouse ("minihouse") chess engine. The search/eval core is Rust (`engine_rs/`), exposed to
Python as the `minichess_engine` extension module via PyO3. Python drives it from three front ends:
a Pygame GUI (`main.py` + `gui.py`), a Chess.com browser bot (`play_online.py`), and the opening
book builder (`build_book.py`).

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
./build_book_parallel.sh 20 6 10   # fill the opening book: 20 workers, to ply 6, depth 10
./build_status.sh ; ./build_stop.sh
./venv/bin/python export_book.py        # book.db -> book.tsv, the versionable form
./venv/bin/python import_book.py        # book.tsv -> book.db; --merge keeps the deeper row
./venv/bin/python migrate_book.py       # carry book.db to the current schema, keeping every row
./venv/bin/python rebuild_book.py       # the only thing that deletes book rows; asks first
```

## Filling the opening book

**The book is an opening repertoire, not a record of games.** It is read in exactly one place —
`probe_book`, at the root of `find_best_move` — so it is only ever read where **the engine is the
side to move**. That makes the tree asymmetric, and the asymmetry is what makes it affordable:

| at a node where | the builder does | children expanded |
| --- | --- | --- |
| **we** are to move | one search, one entry | **one** — the move we will actually play |
| the **opponent** is to move | nothing: no search, no entry | **all** legal replies |

The tree grows like `B^(plies/2)` rather than `B^plies`. Measured branching is ~15–17.

```bash
./build_book_parallel.sh 20 6 10    # 20 workers, up to ply 6, depth 10
./build_status.sh                   # entries per ply
./build_stop.sh                     # SIGTERM, waits for the search in flight to flush
RESIGN=1500 BREADTH=0-8:all,9-16:3 ./build_book_parallel.sh 20 12
```

Easy to get wrong:

- **Build depth must be at least the depth you play at.** The probe rejects any row shallower than
  what it was asked for, so a tail built at depth 8 is invisible to a GUI searching at 10. Build
  flat. "Deep near the root, shallow in the tail" is exactly backwards.
- **Build with rank 1 only.** MultiPV costs multiples of a single-PV search, *and* the probe rejects
  an entry holding fewer ranks than the caller asked for.
- **`--max-ply` is capped at 20** (`HARD_PLY_CAP`).
- **The resign cutoff is free and does most of the pruning.** `--resign` (default 1200) stops
  expansion once `|score|` says the line is decided, using the score the search just returned. A bad
  opponent reply costs one search instead of a subtree.
- **`--opponent-breadth` narrows replies where full breadth stops being affordable.**
  `"0-8:all,9-16:3"` keeps every reply through ply 8 and the best 3 after; the ply in the spec is the
  **opponent node's** ply. Ranking is one shallow MultiPV search *at the opponent node*
  (`--scan-depth`), not one search per reply — the latter files a row at every reply, all of them at
  opponent-turn hashes the runtime never probes.
- **A shallower search never replaces a deeper entry.** `book_store` returns early if the stored
  rank-1 row is deeper at the same `eval_version`. Re-searching at equal depth or deeper still
  stores, which is what lets a later pass deepen the book in place.
- **A mate row is accepted at any depth.** The mate break exits iterative deepening early, so `depth`
  records the iteration that found the mate, not a limit on what was proved. `CHECKMATE_SCORE` is
  flat and carries no mate distance, so a deeper search has nothing to add. Rejecting these on depth
  means they are re-derived on every walk, forever.
- **Do not add a time limit.** `build_book.py` passes `time_limit=None` deliberately. A deadline
  breaks the iterative-deepening loop and `book_store` then files the row at `depth_completed` — the
  last iteration that *finished*. `probe_book` rejects it as too shallow on every future walk, so the
  node is re-searched forever. A capped search does not make the tail cheaper, it makes it permanent.
- **Costs vary by orders of magnitude with how full the hands are**, and the opening is no guide to
  the tail. Single depth-10 nodes running for hours are sightings, not outliers. Budget for the tail.

### How the parallel build coordinates

No work queue, no IPC. `build_book_parallel.sh` walks one our-turn tier per **stage**, and each stage
is a barrier:

```
stage 2    one process, plies 0-2
stage 4    N shards, --split-ply 2       disjoint subtrees
stage 6    N shards, --split-ply 4       ...two plies per stage
```

Everything a stage needs from the tier above is already in `book.db`, so the next stage's workers
re-walk that prefix as **book hits returning in ~0s** — that is how a worker learns the move another
worker searched. Below `--split-ply` each worker owns whole subtrees (`subtree_id % N == shard`) in
BFS order, which every worker reproduces identically. The same property makes the build
**idempotent and resumable**: re-running the same command picks up where it left off, and is the
only resume mechanism there is.

Workers load the DB only at startup, so they do not see each other's entries until restarted — which
is why the build is staged rather than one long run. A stage ends when its slowest shard does, so one
very long node idles every other worker. That is a real price of the staging and it is accepted; the
alternative is an unusable book row.

- **Workers must run from the repo root.** `DB_PATH` in `cache.rs` is the relative string
  `"book.db"`, so a worker started elsewhere silently creates its own DB. `build_book.py` does the
  `os.chdir` itself in `main()` rather than at import, which keeps the module importable from a test
  (the CWD is the only seam for isolating the book — see **Tests**).
- **`build_stop.sh` signals a recorded pid, never a name.** `pkill -f build_book_parallel.sh` also
  matches the shell you typed the command into. The driver writes `$$` to `book_build.pid`.
- **Worker stderr is kept and a dead shard is named.** Each worker writes `stage-P-worker-N.log` and
  `stage-P-worker-N.err` under `logs/`. The `.err` file is the *only* place a worker's traceback
  lands. The stage waits per pid, so a non-zero exit is printed with the tail of its `.err` and the
  tier is called **INCOMPLETE** rather than folded into the stage's entry count. Exit 130 is
  `build_book.py`'s own "stopped by SIGTERM" and is not a failure.
- **Saves are incremental.** `find_best_move` returns the hashes it wrote via a `dirty` out-param,
  `lib.rs` accumulates them in `DIRTY_KEYS`, and only those rows are written. Anything new that
  inserts into the book **must** go through `search::book_store` or the entry never reaches disk.
- **Both tables are written in one `BEGIN IMMEDIATE` transaction on one connection**
  (`cache::save_book_entries`). Two transactions would double the lock acquisitions under N workers
  and could leave a `book_move` row with no `position` row beside it.
- **The DB is in WAL mode** with a 30s busy timeout. `book.db-wal` / `-shm` are gitignored.
- **Nothing drops the book on its own.** `cache::load_book` will not create the file. `setup_db`
  creates tables when absent and **refuses** a `SCHEMA_VERSION` mismatch; a save into a foreign
  schema raises instead of writing, leaving the rows in memory and `DIRTY_KEYS` intact.

**One search writes one entry** — one `book_move` row per rank plus one `position` row, nothing
else. `depth` is the depth that **completed**. Forced moves (one legal reply) are answered without
being stored, since nothing was searched.

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
be written back by a later test). **Never call `ai.setup_db()` outside that isolation.** Anything the
NN work adds — replay buffers, checkpoints — goes under a configurable path outside the repo root for
the same reason. A full `pytest tests/ -q` must create no repo-root `book.db`.

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

### The Pygame front end

`main.py` + `gui.py` + `layout.py` + `settings.py`. `layout.Layout` recomputes every rectangle from
the current window size; nothing else may assume a pixel dimension. Four invariants are load-bearing:

- **The search never blocks the UI.** `main.Game` polls worker threads and keeps drawing; Undo / New
  game / Flip / Hint stay live during a search. The engine's own move carries a `generation` tag and
  is discarded if the position moved on, so taking back a move mid-search cannot apply a stale answer
  to the wrong board. Anything that changes the position must call `Game.invalidate()`. Hints do not
  use the counter — see below.

  This invariant is breakable from Rust with no sign of it in Python: `lib.rs`'s book accessors take
  the `BOOK` mutex **without releasing the GIL**, so a main-thread `ai.book_has_position` blocks on
  any lock a background search holds. The book is touched exactly twice in a search (probe at the
  top, one store at the bottom), so `find_best_move` takes `&Mutex<Book>` and locks at those two
  points only. **Do not put the lock back around the search**, and do not add a book accessor to
  `lib.rs` that holds it across a DB write without `py.allow_threads`.
- **Panel zones have fixed heights** (`layout.py`): header, controls and the bottom status band are
  reserved, and only the move list flexes. Hands live in strips beside the board and show a `×N`
  count rather than one sprite per copy, so a filling hand can never push a button out from under the
  cursor. The hint-lines list is sized from `Layout(..., analysis_rows=n)` — a *setting*, never live
  search state, so it cannot resize mid-search. Changing it calls `Game.relayout()`.
- **Redraw is dirty-flagged.** `Game.dirty` gates rendering; the loop idles at `IDLE_FPS` and rises
  to `FPS` only while something animates. Fonts and scaled sprites are memoised in `gui.py`. Any new
  per-frame `smoothscale` undoes this.
- **Flip is a view transform only.** It must never change what is true about the position.

Drawing takes a `BoardView` snapshot rather than reading the live `GameState`, which is what lets the
arrow keys render any past ply from `gamestate.saved_states`. UI state lives in `main.UIState`; the
`selected_square` / `highlighted_moves` fields on `GameState` are legacy. Hit regions are produced by
the code that draws each control, so a control can never be clickable where it is not visible — and
the promotion picker clears every other region, which stops clicks reaching the buttons underneath.

`gui_settings.json` (gitignored) remembers window size, depth, AI toggles, hints, hint-line count,
`hint_workers`, and orientation. Both AI toggles default to **off**.

### Writing to the book

**The GUI never writes to the book on its own** — it only reads (`load_move_cache_from_db` at
startup). A row goes in when the user presses **Save position to book** (Ctrl+S), which writes
exactly one position. The button's badge counts what the session has searched and not kept, so
"explored a lot, saved none" is visible rather than taken on trust.

| call | writes | for |
| --- | --- | --- |
| `ai.save_move_cache_to_db()` | everything queued in `DIRTY_KEYS` | `build_book.py`, `play_online.py` — every search they run is one they chose |
| `ai.save_position_to_book(gs)` | one named position | interactive front ends, where the search list is whatever the user glanced at |

`ai.book_has_position(gs)` says whether the second would write anything; `pending_book_writes()` /
`discard_pending_book_writes()` report and drop the queue. The in-memory book is still filled by
every search and is still the analysis cache — only the disk write is gated.

**The analysis cache may widen a book entry, never redirect it.** `load_analysis_from_db` merges
`analysis_move` into the same in-memory map, and the book wins every collision it is not *strictly
beaten* on. The one exception: a cached entry naming the **same rank-1 move** under the **same
`eval_version`**, with more depth or more ranks and never less of either, replaces the book's —
whole, never spliced, since ranks 2.. only mean anything as one MultiPV pass. This matters because
the builder files rank 1 only, so a curated row is routinely narrower than what a GUI session
produced, and `probe_book` rejects an entry holding fewer ranks than the caller asked for.

What it deliberately does not fix: a single-PV root and a MultiPV root can name different moves at
the same depth, and a builder row comes from the former while a hint at 2+ lines comes from the
latter. When they disagree the book's move stands and that position keeps re-searching — the
alternative is letting exploration silently redirect a curated line. Curate the position at the width
you read it at. `tests/test_book.py::TestAnalysisMergeKeepsTheBetterEntry` pins this, writing the
analysis rows by hand.

Since `probe_book` is read at the root of `find_best_move`, a row is only ever *consulted* where the
engine is to move. Saving a position where *you* are to move files a row the runtime will never probe.

### Search settings the user can see

**The depth control governs every search, including the hint.** There is no private cap on any search
path; a setting the user can see must be the one that runs. Wanting a faster hint is what lowering
the depth is for, and the header shows the hint's elapsed time so a deep one is visible rather than
mistaken for a hang.

**Controls that cannot change anything still answer.** The depth and hint-line steppers keep their end
buttons in `hits` at the top and bottom of their ladders, and `set_depth` / `set_hint_lines` toast
"Depth 10 is the deepest setting" rather than returning silently. The same rects are registered under
`hits['wheel']` so the mouse wheel nudges whichever stepper it is over.

**Scores are white-relative everywhere.** `search::find_best_move`, `eval::evaluate_position` and
`ai.find_best_move_with_score` all return positive-favours-White whichever side is to move;
`utils.format_score` / `score_advantage` are the only formatters. The readout falls back to the static
evaluation on every position change and is overwritten by a completed search labelling itself
`depth N` — "static" and "depth 10" are different claims about the same number, so the source is
always shown. `eval.rs` encodes a mate as flat ±`CHECKMATE_SCORE`, so the display says "White mates",
never "M3".

**Hints run several at a time, one core each.** `thread_utils.HintPool` keys every hint search by the
question it answers — `(fen, depth, lines)` — and starts up to `hint_workers` at once. Nothing is
cancelled when the board moves on, so playing three moves quickly searches all three positions side
by side. Answers are cached (`KEEP_RESULTS`), so stepping back into a position shows its lines with
no search.

- **A Python thread cannot be stopped**, so `hint_workers` is a hard budget. `submit` declines when
  every slot is taken and the caller retries next frame; lowering the setting does not free a core
  that is already busy.
- **`hint_key` folds depth and line count in.** That is what makes `drop_hint` cheap — a search
  started under the old setting files its answer under the old key and stays out of the way. It is
  also why `set_depth` bumps `generation` itself: `drop_hint` no longer does, and the engine's own
  move still has to be invalidated.
- **`poll_hint` gates the display on `show_hint` and whose turn it is.** The pool keeps answering
  regardless; without the gate a cached hint would reappear the moment hints were switched off.
- **The ladder is capped to the machine** (`settings.hint_worker_choices`, `cpu_count - 1`). A hint
  search is single-threaded, so N workers means N cores busy. Turning `set_parallel_search` on as
  well would oversubscribe.
- **What a high setting costs is memory, not stability.** `SearchState::new()` pre-allocates a
  transposition table *per search*, so every concurrent worker is tens of MB of RSS — the one
  consequence the window cannot otherwise show, which is why `set_hint_workers` names it in its toast.

**More than one hint line means MultiPV.** `HintThread(lines=n)` with `n > 1` calls
`find_best_move(..., return_top_n=n)`, several times the cost of a single-PV search at the same depth
— that cost is why the count is a user-facing setting and why the label names the multiplier. `n == 1`
routes through `find_best_move_with_score` instead. A forced move carries no search behind it, so its
placeholder `0` is turned back into `None` rather than shown as `+0.00`.

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

### The opening book on disk

`engine_rs/src/cache.rs` persists two tables to `book.db` (SQLite, repo root, gitignored). The
transposition table is per-search and never persisted.

```sql
book_move(hash, rank, move, score, depth, eval_version)   -- PK (hash, rank); rank 1 = best
                                                          -- FK hash -> position(hash) ON DELETE CASCADE
position(hash, fen, ply)                                  -- PK hash
```

- `book_move` is the hot path: everything a probe needs is in the row, so the runtime lookup **never
  joins `position`**. A probe rejects a row shallower than the depth asked for, one written under a
  different `eval_version`, and any move not legal in the position (a collision or a stale row).
- `position` is what makes a hash mean something: the hash is one-way, so without a FEN a row can
  never be re-opened or expanded from. Written `INSERT OR IGNORE`; a **differing FEN on an existing
  hash is a Zobrist collision** and is logged loudly. `ply` is path-dependent, so the minimum wins.
- `eval_version` is `EVAL_VERSION` in `eval.rs` and is **bumped by hand whenever an eval constant
  changes**. Nothing detects a stale value; bumping invalidates the book in place rather than
  dropping it.
- **The `hash` foreign key is what stops an unreadable row.** `save_entries` writes the `position`
  row **first**; reverse that and every move insert fails on a missing parent. Enforcement is
  per-connection and OFF by default in SQLite, so `open_db` sets `PRAGMA foreign_keys` — without that
  line the `REFERENCES` clause is inert. What it does *not* cover: a `position` with no `book_move`
  is legal. That direction is guarded in `save_entries`, which refuses to run its `DELETE` when the
  entry holds no moves — that statement is a *replacement*, and with nothing to insert afterwards it
  is a plain deletion.
- `SCHEMA_VERSION` lives in `PRAGMA user_version` and is **2**. A mismatch makes `setup_db`
  **refuse**, naming both versions and the ways out. It never drops, because "recreate the schema"
  runs on paths nobody thinks of as destructive. Two front doors:
  - `migrate_book.py` **keeps every row** — the documented 12-step rebuild in one transaction, with
    `foreign_keys` OFF during the swap, `legacy_alter_table` ON, and `PRAGMA foreign_key_check`
    *inside* the transaction so a violation rolls back. Snapshots to `backups/` first. Orphaned move
    rows make it refuse; `--drop-orphans` discards them after printing each.
  - `rebuild_book.py` **discards** them, shows what is there and asks first (`--yes` skips).
- The book is process-global and filled as you search, so **searching the same position twice in one
  process is a hit reporting ~0s** — the reason `bench/` forks a fresh process per measurement.

**`book.db` is gitignored; `book.tsv` is what goes in git.** The DB rewrites every page on a VACUUM or
migration, so tracking the binary means a fresh multi-MB blob per commit and an unresolvable binary
merge conflict the first time two branches both add rows. The text form delta-compresses, merges line
by line, and is reviewable.

```bash
./venv/bin/python export_book.py           # book.db -> book.tsv
./venv/bin/python import_book.py           # book.tsv -> book.db (refuses a non-empty book)
./venv/bin/python import_book.py --merge   # combine, keeping the deeper row
./venv/bin/python import_book.py --check   # verify the file, write nothing
```

- **The export is keyed by FEN, not by hash, and that is the whole point.** A Zobrist hash is one-way,
  so an artifact keyed on one dies the moment the hashing changes. `import_book.py` re-derives every
  hash through `from_fen` → `get_position_hash`, so the file survives changes the DB cannot. That
  makes the import an engine run rather than a plain SQL load.
- **Diff stability is designed in, and easy to undo.** Rows sort by `(fen, rank)` and nothing else —
  *not* by ply, which is path-dependent. The header carries **no timestamp**, so an unchanged book
  re-exports byte-identically. Add one and every export becomes a diff.
- **The import verifies before it writes.** Every FEN must satisfy `to_fen(from_fen(fen)) == fen`, and
  no two distinct FENs may share a hash. Either fails the whole import. `position` rows are written
  before `book_move` rows — the foreign key makes that a precondition.
- **`--merge` keeps the deeper row**, the rule `book_store` already applies. Equal depth lets the
  imported row win, which is how you overwrite in place.
- A `position` with no `book_move` is **not exported**. The analysis cache is not exported either: it
  is exploration, not a curated repertoire.
- **`build_book_parallel.sh` refreshes `book.tsv` on the way out**, on both exit paths. It **never
  commits**. `EXPORT=0` skips it. Because the export is byte-stable, "book.tsv is unchanged" means the
  build added nothing — while `book.db` still shows as modified, since SQLite churns pages on every
  open.
- `tests/test_book_export.py` asserts the hashes come back, not merely that the file parses.

### FEN

`engine_rs/src/fen.rs` carries exactly what the Zobrist hash reads — board, side to move, both hands,
promoted squares — and nothing path-dependent. That equivalence is the point: `from_fen(to_fen(gs))`
must hash back to `gs.hash`, which `tests/test_fen.py` asserts over random games. Conventions at 6×6:
`2bnrk/5p/6/6/P5/KRNB2[] w`, hands in brackets (uppercase White), `~` marking a promoted piece, ranks
top row first. Exposed as `ai.to_fen` / `ai.from_fen`.

### Draws in the search

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

### MultiPV and parallelism

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

## Gotchas

- `bot_start.sh` rewrites `play_online.py` in place with `sed -i ''` to flip rated/casual — BSD syntax
  that fails on Linux. Set `rated=` in `create_minihouse_game` by hand rather than assuming the mode
  switched.
- The numbers that decide moves are the `const`s at the top of `engine_rs/src/eval.rs`
  (`CENTER_BONUS`, `KING_SAFETY_BONUS`, `PIECE_VALUES`); there is no Python copy of them any more.
  Editing one means bumping `EVAL_VERSION` in the same file, or the book keeps serving scores from
  the evaluation you just replaced.
- **There is no self-play.** `src/self_play.py`, `src/scheduled_self_play.py` and the `train*.sh`
  runners are deleted; nothing in the repo plays AI-vs-AI from a script. The closest things left are
  `tests/test_ai_optimization.py` and `bench/`.
- `docs/evaluation_strategy.md` and the README are accurate. `docs/PARALLEL_SEARCH.md` is a dated
  record of one evaluation — read it as history; it still names files that no longer exist.
- Console output across the Python layer is heavily print-based and partly emoji-formatted; the bot
  and the book builder are read via their logs (`/tmp/minichess_bot.log`, `logs/book/`).
