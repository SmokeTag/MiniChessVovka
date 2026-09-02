# The opening book

> Extracted from CLAUDE.md. `docs/BOOK_FILES.md` is the which-file-is-which cheat sheet;
> this is the mechanism and the traps.

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
  (the CWD is the only seam for isolating the book — see the **Tests** section of `CLAUDE.md`).
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

## Writing to the book

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

## The opening book on disk

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

