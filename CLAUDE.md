# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A 6×6 Crazyhouse ("minihouse") chess engine. The search/eval core is Rust (`engine_rs/`), exposed to
Python as the `minichess_engine` extension module via PyO3. Python drives it from three front ends:
a Pygame GUI (`main.py` + `gui.py`), a Chess.com browser bot (`play_online.py`), and the opening
book builder (`build_book.py`).

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
./bot_start.sh casual|rated        # chess.com bot (needs .env + playwright chromium)
./bot_stop.sh ; ./monitor_bot.sh
./build_book_parallel.sh 20 6 10   # fill the opening book: 20 workers, to ply 6, depth 10
./build_status.sh ; ./build_stop.sh
./venv/bin/python export_book.py        # book.db -> book.tsv, the versionable form (book.db is gitignored)
./venv/bin/python import_book.py        # book.tsv -> book.db; --merge keeps the deeper row
./venv/bin/python migrate_book.py       # carry book.db to the current schema, keeping every row
./venv/bin/python rebuild_book.py       # the only thing that deletes book rows; asks first
```

## Filling the opening book

**The book is an opening repertoire, not a record of games.** `build_book.py` is what fills it;
self-play does not, and should not be pointed at it.

The reason is that the book is read in exactly one place — `probe_book`, at the root of
`find_best_move` — which means it is only ever read in a position where **the engine is the side to
move**. That makes the tree it needs asymmetric, and the asymmetry is what makes it affordable:

| at a node where | the builder does | children expanded |
| --- | --- | --- |
| **we** are to move | one search, one entry | **one** — the move we will actually play |
| the **opponent** is to move | nothing: no search, no entry | **all** legal replies |

We never play rank 2, so rank 2's subtree is dead weight; the opponent chooses, so every reply of
theirs has to be answered. The tree grows like `B^(plies/2)` rather than `B^plies`, and half of what
survives is never searched at all. Measured branching here is ~15–17, giving searched-node counts of:

```
ply    0     2      4       6        8
W      1    15    240   ~4,100   ~74,000      (Black is identical, shifted one ply)
```

Those are the counts with no pruning. What a real build costs is a good deal less, because
`--resign` cuts about half of it. The first full build, **ply 6 at depth 10, 20 workers, 65 minutes**,
came out at 1,997 entries:

```
ply  0     1     2     3     4      5     6
      1    15    15   173    90   1295   408
     W     B     W     B     W      B     W
```

Note the asymmetry at an even `--max-ply`: White gets answers for four of its own moves (plies
0/2/4/6) and Black for three (1/3/5), so the engine leaves book one move earlier as Black. Evening
that up means `--max-ply 7`, which adds Black's ply-7 tier — on the order of 70k searches.

About 18% of the rows land at a depth below the one asked for (`depth 1..9` in `build_status.sh`).
Every one of them is a **mate score**: the mate break exits iterative deepening early, so `depth` is
the depth that completed. A depth-10 probe rejects them and re-searches, which costs nothing — a
mate found at depth 3 is found again in milliseconds.

A whole self-play game, by contrast, writes ~200 entries of which maybe 8 are inside repertoire
range and the rest are midgame positions no game will reach twice. `--random-plies` is worse than
useless for this: a random prefix produces positions that never occur.

```bash
./build_book_parallel.sh 20 6 10    # 20 workers, up to ply 6, depth 10
./build_status.sh                   # entries per ply, and which repertoire each ply belongs to
./build_stop.sh                     # SIGTERM, waits for the search in flight to flush
RESIGN=1500 BREADTH=0-8:all,9-16:3 ./build_book_parallel.sh 20 12
./venv/bin/python build_book.py --max-ply 6 --depth 10   # the same thing, one process
```

Things about this that are easy to get wrong:

- **Build depth must be at least the depth you play at.** The probe rejects any row shallower than
  what it was asked for, so a tail built at depth 8 is invisible to a GUI searching at 10 — the work
  is simply wasted. Build flat at 10 or deeper. "Deep near the root, shallow in the tail" is exactly
  backwards.
- **Build with rank 1 only.** MultiPV costs 2.4–4.4x, *and* the probe rejects an entry holding fewer
  ranks than the caller asked for. Ranks 2+ only ever paid for themselves in self-play exploration,
  which is not how the book gets filled any more.
- **`--max-ply` is capped at 20** (`HARD_PLY_CAP`). Past that it is not a repertoire, and the tree
  is not affordable.
- **The resign cutoff is free, and does most of the pruning.** `--resign` (default 1200) stops
  expansion once `|score|` says the line is decided; the score is the one the search just returned,
  so it costs nothing. A bad opponent reply therefore costs exactly one search instead of a whole
  subtree. Measured at depth 10 the score distribution is bimodal — balanced lines sit under 800,
  decided ones over 1550, and nothing lands in between — so the exact threshold barely matters.
- **`--opponent-breadth` is for depths where full breadth stops being affordable.** `"0-8:all,9-16:3"`
  keeps every reply through ply 8 and the best 3 after. The ply in the spec is the **opponent node's**
  ply. Ranking is one shallow MultiPV search *at the opponent node*, not one search per reply: the
  latter costs 17x instead of 4.4x and files a row at all seventeen, which is how the pre-book
  `move_cache` ended up ~97% unreachable. The one entry it does write sits at an opponent-turn hash
  that the runtime never probes.
- **A shallower search never replaces a deeper entry.** `book_store` returns early if the stored
  rank-1 row is deeper at the same `eval_version`. Without it a narrowing scan could transpose onto
  a position another line had already searched deep and silently overwrite it, because an entry is
  replaced wholesale. Re-searching at equal depth or deeper still stores, which is what lets a later
  pass deepen the book in place.

### How the parallel build coordinates

There is no work queue and no IPC. `build_book_parallel.sh` walks one our-turn tier per **stage**,
and each stage is a barrier:

```
stage 2    one process, plies 0-2            ~31 searches
stage 4    N shards, --split-ply 2           disjoint subtrees
stage 6    N shards, --split-ply 4           ...two plies per stage
```

Everything a stage needs from the tier above is already in `book.db`, so when the next stage's
workers re-walk that prefix every node up there is a **book hit returning in ~0s** — that is how a
worker learns the move another worker searched, without being told. Below `--split-ply` each worker
owns whole subtrees (`subtree_id % N == shard`), assigned in BFS order, which every worker
reproduces identically because they all walk the prefix. Verified: a sharded build produces a
`book.db` byte-identical to the serial one. The only duplicated work is where two subtrees
transpose, ~1% of a tier.

The same property makes the build **idempotent and resumable** — stopping and re-running the same
command picks up where it left off, and is the only "resume" mechanism there is.

Things about *running* it that are easy to get wrong:

- **Workers must run from the repo root.** `DB_PATH` in `cache.rs` is the relative string
  `"book.db"`, so a worker started elsewhere silently creates its own DB. `build_book.py` does the
  `os.chdir` itself, in `main()` rather than at import — that keeps the module importable from a
  test, since the CWD is the only seam there is for isolating the book (see **Tests**).
- **`build_stop.sh` signals a recorded pid, never a name.** `pkill -f build_book_parallel.sh`
  also matches the shell you typed the command into, and kills it. The driver writes `$$` to
  `book_build.pid` for exactly this reason.
- **Saves are incremental.** `save_move_cache_to_db` used to re-`INSERT OR REPLACE` the process's
  entire cache after *every move* — ~5µs/row, so ~1.3s per move once a worker holds 250k entries,
  times N workers on one write lock. `search::find_best_move` now returns the position hashes it
  wrote via a `dirty` out-param, `lib.rs` accumulates them in `DIRTY_KEYS`, and only those rows get
  written. Anything new that inserts into the book **must** go through `search::book_store` (which
  pushes onto `dirty`) or the entry never reaches disk.
- **Both tables are written in one `BEGIN IMMEDIATE` transaction on one connection**
  (`cache::save_book_entries`). Twenty workers share one write lock, so two transactions would
  double the lock acquisitions and could leave a `book_move` row with no `position` row beside it.
  `IMMEDIATE` also avoids `SQLITE_BUSY_SNAPSHOT` on the read-then-write inside it.
- **The DB is in WAL mode** with a 30s busy timeout (`cache::open_db`). Sidecar files
  `book.db-wal` / `-shm` are gitignored.
- **Nothing drops the book on its own.** `cache::load_book` will not even create the file — a
  missing or stale book loads empty and says so. `setup_db` creates the tables when they are
  absent and **refuses** a `SCHEMA_VERSION` mismatch, naming both versions and the way out; a save
  into a foreign schema raises instead of writing, leaving the rows in memory and `DIRTY_KEYS`
  intact so the work is still recoverable. `rebuild_book.py` is the only thing that deletes rows
  wholesale; `migrate_book.py` deletes none unless `--drop-orphans` says to, and refuses otherwise.

Depth 10 costs roughly: median ~3s/move, p90 ~12s, worst seen ~31s — crazyhouse midgames with full
hands are far more expensive than the opening. Measured on the ply-6 build: ~3.5s/search at plies
0-2, but ~20s/search at plies 5-6, where both sides have pieces in hand. Budget for the tail, not
for the root. Workers
load the DB only at startup, so they do not see each other's entries until restarted — which is why
the build is staged rather than one long run.

**There is no self-play any more.** `src/self_play.py`, `src/scheduled_self_play.py` and the
`train*.sh` runners are deleted. They filled the book by playing games out, which is the thing the
repertoire replaced: ~97% of what they wrote was midgame positions no game reaches twice, and every
run would now corrupt a curated book. Nothing in the repo plays AI-vs-AI from a script; the closest
things left are `tests/test_ai_optimization.py`, which plays full games, and `bench/`, which
measures the search. (`git show 4fe0d1e:src/self_play.py` if you want it back.)

**One search writes one entry.** A depth-10 search leaves exactly one `book_move` row per rank plus
one `position` row, and nothing else. The per-iteration stores and the TT dump that used to fill
~97% of the DB with rows no depth-10 probe could accept are both gone — the TT dump for good, since
its interior nodes have no completed root depth behind their scores and no position to write a FEN
from. `depth` is the depth that **completed**: the mate break and `time_limit` both exit the
iterative-deepening loop early, and a result salvaged from an aborted iteration is returned to the
caller but never filed. Forced moves (one legal reply) are answered without being stored — nothing
was searched.

## Tests

`tests/` mixes unittest classes and bare pytest functions; pytest runs both.

```bash
./venv/bin/python -m pytest tests/ -q
./venv/bin/python -m pytest tests/test_e2e.py::TestE2EGameFlow::test_promotion_flow -q
./venv/bin/python -m pytest tests/test_ai_optimization.py -q -s
```

Tests are slow by design — several play full AI-vs-AI games. Prefer running a single test while
iterating.

**No test may write to the repo-root `book.db`** — that file is the live book, and `test_nightly.py`
drops and recreates its tables. `DB_PATH` in `cache.rs` is the relative string `"book.db"` with no
override, so the only seam is the CWD: `tests/cache_isolation.py` gives an `isolated_cache_db()`
context manager and an `IsolatedCacheDB` TestCase base that run a test in a throwaway directory and
reload the real book afterwards (the Rust book is process-global, so a test that skipped the reload
would leave its scratch rows to be written back by a later test). **Never call `ai.setup_db()`
outside that isolation** — it builds the schema in the CWD, which creates a `book.db` wherever the
test happens to be standing. It will not drop an existing one (that is `rebuild_book.py`, which
asks first), but a test has no business creating one in the repo root either. Anything the NN work
adds — replay buffers, checkpoints — goes under a
configurable path outside the repo root for the same reason. A full `pytest tests/ -q` must create
no repo-root `book.db` at all; check with `ls` before and after.

Rust has a small unit suite (`fen.rs`): `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --release`
inside `engine_rs/`. Everything else is tested from Python, and `cargo build --release` only checks
that it compiles — use `maturin develop --release` for anything you intend to run.

## Architecture

**Two independent implementations of the same rules.** `gamestate.py` (Python) owns the live game —
GUI, bot, and book builder all mutate it — while `engine_rs/src/gamestate.rs` re-implements move
generation for the search. `ai._sync_to_rust()` copies board/turn/hands/king_pos/promoted_pieces into
a fresh Rust `GameState` on *every* `find_best_move` call. A rules change (a new drop restriction,
promotion behaviour, check detection) must land in **both** or the engine will search a position the
GUI does not agree with. `tests/test_rules_parity.py` is what catches that drift: it plays identical
random games through both and compares legal-move sets, terminal flags, ply counters and promotion
state at every ply. Run it with `RULES_PARITY_GAMES=500` after any rules change — the 50-game default
is a smoke test, not a sweep.

**The Pygame front end is `main.py` + `gui.py` + `layout.py` + `settings.py`.**
`layout.Layout` recomputes every rectangle from the current window size — nothing else may assume a
pixel dimension, and nothing does any more: `config.SQUARE_SIZE` is gone along with `pieces.py`'s
commented-out vector primitives, which were its only importer. Four invariants are load-bearing and
easy to undo by accident:

- **The search never blocks the UI.** `main.Game` polls a worker thread and keeps drawing;
  Undo / New game / Flip / Hint stay live during a search. Results carry a `generation` tag
  and are discarded if the position moved on, so taking back a move mid-search cannot apply a
  stale answer to the wrong board. Anything that changes the position must call
  `Game.invalidate()` (which bumps the generation) or that guarantee breaks.
- **Panel zones have fixed heights** (`layout.py`): header, controls and the bottom status band
  are reserved, and only the move list flexes. Hands live in strips beside the board rather
  than in the panel, and show a `×N` count instead of one sprite per copy — both so that a
  filling hand can never push a button out from under the cursor. The one band sized from
  something other than the window is the hint-lines list, whose height comes from
  `Layout(..., analysis_rows=n)` — a *setting*, never live search state, so it still cannot
  resize mid-search; it sits below the controls, so only the move list absorbs the change.
  Changing that setting calls `Game.relayout()`, which rebuilds the geometry without touching
  the window.
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

`gui_settings.json` (gitignored) remembers window size, depth, AI toggles, hints, how many hint
lines to show, and orientation. Both AI toggles default to **off** — a fresh start is
human-vs-human, and each side is opted in by its own button.

**The GUI never writes to the book on its own.** `poll_ai` used to call
`save_move_cache_to_db` after every engine move, which flushed the *entire* `DIRTY_KEYS`
queue — so one AI move filed every position the session had searched since the last one,
hints and idle exploration included. That is the opposite of what a curated repertoire
wants, and it is invisible while it happens. The GUI now only ever *reads* the book
(`load_move_cache_from_db` at startup). A row goes in when the user presses **Save position
to book** (Ctrl+S), which calls `ai.save_position_to_book(gs)` and writes exactly one
position. The button's badge counts what the session has searched and not kept, so
"explored a lot, saved none" is visible rather than something to take on trust.

Two save paths now exist and they are not interchangeable:

| call | writes | for |
| --- | --- | --- |
| `ai.save_move_cache_to_db()` | everything queued in `DIRTY_KEYS` | `build_book.py`, `play_online.py` — every search they run is one they chose |
| `ai.save_position_to_book(gs)` | one named position | interactive front ends, where the search list is whatever the user glanced at |

`ai.book_has_position(gs)` says whether the second would write anything;
`pending_book_writes()` / `discard_pending_book_writes()` report and drop the queue. The
in-memory book is still filled by every search and is still the analysis cache — a position
searched twice in one process is a ~0s hit whether or not it was ever saved. Only the disk
write is gated.

Worth knowing when curating: `probe_book` is read at the root of `find_best_move`, so a row
is only ever *consulted* at a position where the engine is to move. Saving a position where
you are to move files a perfectly good row that the runtime will never probe in that
configuration.

**The depth control governs every search, including the hint.** `HINT_MAX_DEPTH` is deleted: it
clamped the hint to 6 whatever the selector said, and with both AI toggles off the hint is the
*only* search the GUI runs — so choosing depth 10 changed nothing that ever executed and the
selector looked broken while working exactly as written. Do not reintroduce a private cap on any
search path; a setting the user can see must be the one that runs. Wanting a faster hint is what
lowering the depth is for, and the header shows the hint's elapsed time so a deep one is visible
rather than mistaken for a hang.

**Controls that cannot change anything still answer.** The depth and hint-line steppers keep
their end buttons in `hits` at the top and bottom of their ladders (`gui._draw_stepper`), and
`set_depth` / `set_hint_lines` toast "Depth 10 is the deepest setting" rather than returning
silently. Dropping them from `hits` is what made the depth control read as broken: at the saved
default of 10 the "›" was both greyed and unregistered, so clicking it did nothing at all. The
same rects are registered under `hits['wheel']` so the mouse wheel nudges whichever stepper it
is over.

**Scores are white-relative everywhere.** `search::find_best_move`, `eval::evaluate_position`
and therefore `ai.find_best_move_with_score` all return positive-favours-White, whichever side
is to move; `utils.format_score` / `score_advantage` are the only formatters, and the eval
readout never flips sign per turn. The readout falls back to the static evaluation on every
position change (`Game.refresh_static_score`) and is overwritten by a completed search, which
labels itself `depth N` — "static" and "depth 10" are different claims about the same number,
so the source is always shown. `eval.rs` encodes a mate as a flat ±`CHECKMATE_SCORE` with no
distance in it, so the display says "White mates", never "M3".

**More than one hint line means MultiPV.** `HintThread(lines=n)` with `n > 1` calls
`find_best_move(..., return_top_n=n)`, which is 2.4–4.4x a single-PV search at the same depth —
that cost is why the count is a user-facing setting and why the label beside it names the
multiplier. `n == 1` routes through `find_best_move_with_score` instead, which is a plain
single-PV search that happens to hand back its score. A forced move carries no search behind it,
so its placeholder `0` is turned back into `None` rather than shown as `+0.00`.

**`ai.py` is a thin shim, not the AI.** It exists so `gui.py`, `play_online.py` and `build_book.py`
keep a stable Python API while all real work happens in Rust. Its module-level `move_cache`
and `tt` dicts are vestigial — the Rust side owns both the book and the TT; use `ai.book_size()` for
a count that is actually true. `_sync_to_rust` rebuilds the Rust state on every call, so anything the
book records about a position (currently `ply`) has to be copied there or it arrives as a default.

**Move representation** (same tuples on both sides of the PyO3 boundary):
- normal: `((r1, f1), (r2, f2), promotion_char_or_None)`
- drop: `('drop', 'wN', (r, f))` — colour+type code, target square

Coordinates are `(row, file)` with **row 0 = rank 6** (top). `utils.coords_to_algebraic` /
`algebraic_to_coords` are the only correct converters; don't hand-roll the flip.

**The opening book.** `engine_rs/src/cache.rs` persists two tables to `book.db` (SQLite, repo root,
gitignored); there is a separate in-memory transposition table per search, which is never persisted.

```sql
book_move(hash, rank, move, score, depth, eval_version)   -- PK (hash, rank); rank 1 = best
                                                          -- FK hash -> position(hash) ON DELETE CASCADE
position(hash, fen, ply)                                  -- PK hash
```

- `book_move` is the hot path: everything a probe needs is in the row, so the runtime lookup
  **never joins `position`**. A probe rejects a row shallower than the depth asked for, one written
  under a different `eval_version`, and any move that is not legal in the position (a collision or a
  stale row). Deeper-than-requested rows are accepted — same question, more evidence.
- `position` is what makes a hash mean something: the hash is one-way, so without a FEN a row can
  never be re-opened or expanded from. Written `INSERT OR IGNORE`; a **differing FEN on an existing
  hash is a Zobrist collision** and is logged as loudly as a log line can be. `ply` is
  path-dependent, so the minimum ever seen wins.
- `eval_version` is `EVAL_VERSION` in `eval.rs` and is **bumped by hand whenever an eval constant
  changes**. Nothing detects a stale value; bumping invalidates the book in place rather than
  dropping it.
- **The `hash` foreign key is what stops an unreadable row.** A `book_move` under a hash with no
  `position` beside it can never be turned back into a position — a Zobrist hash is one-way — which
  is exactly what made `move_cache` unmigratable. `save_entries` therefore writes the `position`
  row **first**; reverse that ordering and every move insert fails on a missing parent. Enforcement
  is per-connection and OFF by default in SQLite, so `open_db` sets `PRAGMA foreign_keys` — without
  that line the `REFERENCES` clause is inert decoration. The analysis pair carries the same key.
  Note what it does *not* cover: a `position` with no `book_move` is legal and does happen (SQL
  cannot require a parent to have children). That direction is guarded in `save_entries`, which
  refuses to run its `DELETE` when the entry holds no moves — that statement is a *replacement*,
  and with nothing to insert afterwards it is a plain deletion. It is how the ply-0 root row went
  missing from a 10,000-entry book while the save still reported success.
- `SCHEMA_VERSION` lives in `PRAGMA user_version` and is **2**. A mismatch makes `setup_db`
  **refuse** — it returns an error naming the on-disk version, the expected one, and both ways out,
  and touches nothing. It never drops, because "recreate the schema" runs on paths nobody thinks of
  as destructive (a worker's first save, a stray call from a test), and a version bump would turn
  any of them into a silent wipe of the book. There are two front doors, and reaching for the wrong
  one costs CPU-days:
  - `./venv/bin/python migrate_book.py` **keeps every row**. SQLite has no
    `ALTER TABLE ... ADD CONSTRAINT`, so it runs the documented 12-step rebuild in one transaction,
    with `foreign_keys` OFF during the swap (on, a mid-rebuild parent drop cascades into children
    still being copied), `legacy_alter_table` ON so the RENAME stays literal, and
    `PRAGMA foreign_key_check` **inside** the transaction — a violation rolls back rather than
    leaving a file that claims v2 and is not. It snapshots to `backups/` first. Orphaned move rows
    make it refuse; `--drop-orphans` discards them after printing each. Nothing is re-searched: the
    constraint changes what the schema permits, not what a row means.
  - `./venv/bin/python rebuild_book.py` **discards** them, shows what is there and asks before
    dropping both tables and stamping the new version (`--yes` skips the prompt). This replaced the
    old "no depth column → DROP TABLE" column-sniffing, which was both a guess *and* destructive.
- The book is process-global and filled as you search, so **searching the same position twice in one
  process is a hit that reports ~0s** — the reason `bench/` forks a fresh process per measurement and
  calls `find_best_move_with_score` (a plain single-PV search) rather than asking for two moves.

**`book.db` is gitignored; `book.tsv` is what goes in git.** The DB rewrites every page on a VACUUM
or a migration — measured, the version-2 migration took it 1.86 → 2.76 → 1.76 MB with identical
logical content — so tracking the binary means a fresh multi-MB blob per commit forever, and a
binary merge conflict with no resolution the first time two branches both add rows. "It is small"
also expires: ply 6 held 1,997 rows and ply 8 holds 10,001, ~5x per two plies.

```bash
./venv/bin/python export_book.py           # book.db -> book.tsv
./venv/bin/python import_book.py           # book.tsv -> book.db (refuses a non-empty book)
./venv/bin/python import_book.py --merge   # combine, keeping the deeper row
./venv/bin/python import_book.py --check   # verify the file, write nothing
```

Measured at 10,001 rows: 1.76 MB binary, 0.68 MB text, **0.09 MB once git packs it** — and unlike
the binary it delta-compresses between versions, merges line by line, and is reviewable.

- **The export is keyed by FEN, not by hash, and that is the whole point.** A Zobrist hash is
  one-way, so an artifact keyed on one dies the moment the hashing changes — which is exactly why
  nothing was salvaged from `move_cache`. `import_book.py` re-derives every hash through
  `from_fen` → `get_position_hash`, so the file survives changes the DB cannot. That also makes the
  import an engine run rather than a plain SQL load.
- **Diff stability is designed in, and easy to undo.** Rows sort by `(fen, rank)` and by nothing
  else — *not* by ply, which is path-dependent (the minimum ever seen wins), so sorting on it would
  let a later build that reaches a position by a shorter route reshuffle the file for no semantic
  reason. The header carries **no timestamp**, so an unchanged book re-exports byte-identically.
  Add one and every export becomes a diff.
- **The import verifies before it writes.** Every FEN must satisfy `to_fen(from_fen(fen)) == fen`
  (a FEN that does not round-trip would hash as something else), and no two distinct FENs may share
  a hash. Either fails the whole import rather than filing a row against the wrong position.
  `position` rows are written before `book_move` rows — the foreign key makes that a precondition.
- **`--merge` keeps the deeper row**, the rule `search::book_store` already applies. Equal depth
  lets the imported row win, which is how you overwrite in place.
- A `position` with no `book_move` is **not exported** — it carries no searched work and the builder
  re-derives it for free, so a round trip is not expected to reproduce it. The analysis cache is not
  exported by default either: it is exploration, not a curated repertoire.
- `tests/test_book_export.py` is what keeps this honest. It asserts the hashes come back, not merely
  that the file parses.

`move_cache.db` was the predecessor and is **gone**. Nothing was migrated out of it: its hashes
could not be turned back into FENs, and ~97% of its rows were unreachable by the depth-10 probe
training actually runs.

**FEN.** `engine_rs/src/fen.rs` carries exactly what the Zobrist hash reads — board, side to move,
both hands, promoted squares — and nothing path-dependent. That equivalence is the point:
`from_fen(to_fen(gs))` must hash back to `gs.hash`, which `tests/test_fen.py` asserts over random
games. Crazyhouse conventions at 6x6: `2bnrk/5p/6/6/P5/KRNB2[] w`, hands in brackets (uppercase
White), `~` marking a promoted piece, ranks top row first. Exposed as `ai.to_fen` / `ai.from_fen`.

**MultiPV.** `find_best_move(..., return_top_n=K)` with `K > 1` runs a full-window search of the top
K root moves at the final depth, which is the only way ranks 2+ get scores rather than fail-low
bounds — read those off an ordinary alpha-beta search and the book fills with numbers that mean
nothing. There is no ply or depth gate: the caller decides where to spend. It is not free. Measured
at depth 9 against the same search asked for one move: **K=2 costs 2.4x from the initial position
and 3.0x six plies in; K=3 costs 4.4x and 3.8x.** Self-play pays this on every move — that is what
made `choose_move_with_exploration`'s 20% exploration start working after never having run.

"Exact" here means procedural: never a fail-low or fail-high bound. It is *not* a claim that the
number is the true depth-N minimax value. Score every root move with its own independent search and
the ranking comes out different — and so does the ordinary single-PV root, which disagrees with both.
That drift is the engine's, not the book's: it comes from null-move pruning returning a hard `beta`
while the TT store site classifies flags against the original window (the same hazard documented on
`search_worker`). Fixing it is a search change needing its own bench and parity pass.

**Parallelism policy — deliberate, don't "fix" it.** Root-parallel search
(`minimax_parallel` in `search.rs`) is **off by default**, and `build_book.py` passes
`parallel=False` explicitly: the build scales better by running many workers on disjoint subtrees
than by parallelising one search. The parallel path is for interactive analysis of a single
position. `find_best_move(..., parallel=True/False)` overrides per call;
`ai.set_parallel_search(enabled, min_depth)` sets the process default (`min_depth` = first
iterative-deepening depth that gets split, default 3).

## Benchmarking the engine

`bench/` measures the search honestly: one fresh `venv/bin/python` subprocess per measurement, never
loads the on-disk book, reduces repeats with **min** (background load is additive noise), and flags
cells measured under high load average.

```bash
./venv/bin/python bench/run_bench.py --depths 6,7,8 --repeats 3 --modes seq,par --out bench/mine.json
./venv/bin/python bench/compare.py bench/results/baseline.json bench/mine.json
./venv/bin/python bench/gen_positions.py     # regenerate bench/positions.json
```

Positions are stored as replayed move lists, which predates `fen.rs` — either works now. In
`compare.py`, best-move
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
- `nn/` and `engine/env.py` are **deleted**. They were an unfinished torch/RL experiment that nothing
  imported, and they were actively misleading as a starting point for the planned AlphaZero work:
  `nn/model.py` sized the action space as `board_size**4`, which cannot express a drop or a promotion
  choice, and `nn/mcts.py` never negated the value between alternating players. Anything neural starts
  from the Rust `GameState`, not from those files (`git show 709f963:nn/model.py` if you want them back).
- `move_cache.db` and `precalc_openings.py` are **deleted**. The DB was the pre-book
  `(hash, depth)` table; nothing had read it since the book landed, and it was unmigratable —
  a Zobrist hash cannot be turned back into a position without the FEN the `position` table now
  stores beside it. The script hardcoded a three-level opening walk, threw the score away
  (`return best, 0, elapsed`), and never called `set_ply`, so every row it wrote would have
  claimed the position first occurs at ply 0. **Nothing fills the book on purpose right now** —
  that is the book builder's job, and it does not exist yet
  (`git show 75c03d4:precalc_openings.py` for the old script).
- `config.py`'s eval constants (`CENTER_BONUS`, `KING_SAFETY_BONUS`, `PIECE_VALUES` in `pieces.py`, …)
  are dead weight from the pre-Rust era — the numbers that actually decide moves are the `const`s at
  the top of `engine_rs/src/eval.rs`. Editing the Python ones changes nothing — and editing the Rust
  ones means bumping `EVAL_VERSION` in the same file, or the book keeps serving scores from the
  evaluation you just replaced.
- `docs/evaluation_strategy.md` and the README are accurate. `docs/PARALLEL_SEARCH.md` is a dated
  record of one evaluation — its §0 describes the code as it stands, §§1-5 are how the problem was
  found, and it still names `src/self_play.py`, which no longer exists. Read it as history.
- Console output across the Python layer is heavily print-based and partly emoji-formatted; the bot
  and the book builder are read via their logs (`/tmp/minichess_bot.log`, `logs/book/`).
