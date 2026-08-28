# Opening book: which file is which

> `book.db` is the live book and is **gitignored**. `book.tsv` is the versioned copy and
> **is** in git. Everything below is about keeping those two in step.

Run everything from the repo root, through the venv.

---

## Cheat sheet

| I want to… | Command |
|---|---|
| Fill the book | `./build_book_parallel.sh 20 6 10` |
| See what is in it | `./build_status.sh` |
| Save it to git | `./venv/bin/python export_book.py` → commit `book.tsv` |
| Restore it on a fresh clone | `./venv/bin/python import_book.py` |
| Combine two books | `./venv/bin/python import_book.py --merge` |
| Check a `.tsv` without writing | `./venv/bin/python import_book.py --check` |
| Move it to a new schema version | `./venv/bin/python migrate_book.py` |
| Throw it away and start over | `./venv/bin/python rebuild_book.py` |
| Snapshot before something risky | `./venv/bin/python export_book.py --out backups/book-$(date +%F).tsv` |

---

## The normal loop

**After a build, save the work.** `build_book_parallel.sh` already runs the export for you
on the way out — whether it finished or you stopped it — and prints the commit line if
anything changed. All you do is:

```bash
git add book.tsv && git commit -m "Extend the book to ply 8"
```

Run it by hand after a single-process `build_book.py`, or any time you are unsure:

```bash
./venv/bin/python export_book.py     # book.db -> book.tsv
```

`EXPORT=0 ./build_book_parallel.sh ...` skips the automatic refresh.

**On a fresh clone, get a book:**

```bash
./venv/bin/python import_book.py     # book.tsv -> book.db
```

That is the whole workflow. The rest of this page is the things that bite.

---

## Things that bite

**Do not run `import_book.py` while a build is running.** Twenty workers hold the write
lock, and a build that finishes afterwards flushes its in-memory book over the top of
whatever you imported. Check with `pgrep -f build_book` first.

**`import_book.py` refuses a non-empty book.** That is deliberate — it would otherwise
silently replace a book you spent CPU-days on. Use `--merge` to combine, or
`rebuild_book.py` to clear it first.

**`--merge` keeps the *deeper* row.** Same rule the engine uses internally: never trade a
deeper answer for a shallower one. On equal depth the imported row wins, so re-importing
the same file is how you overwrite in place.

**Re-exporting an unchanged book produces an identical file.** If `git diff book.tsv` is
empty after a build, the build genuinely added nothing. There is no timestamp in the
header, on purpose. `book.db` *will* still show as modified — SQLite churns pages on every
open — which is the whole reason the binary is not the thing in git.

**The build exports but never commits.** What goes into a commit is not a build script's
call, and a build that committed on its own would eventually commit a half-finished tier.

**A stale `eval_version` is reported, not blocked.** If you changed an eval constant and
bumped `EVAL_VERSION`, the import warns how many rows were scored by the old evaluation.
They still import; the probe rejects them until re-searched.

**Never commit `book.db`.** It rewrites every page on a VACUUM or a migration, so git
stores a whole new ~2MB blob each time and two branches adding rows conflict with no way
to merge them.

---

## Why text, and why keyed on FEN

`book.tsv` is one ranked move per line:

```
fen <TAB> ply <TAB> rank <TAB> move <TAB> score <TAB> depth <TAB> eval_version
```

At 10,001 rows: **1.76 MB** as `book.db`, **0.68 MB** as text, **0.09 MB** once git packs
it — and unlike the binary it delta-compresses between commits, merges line by line, and
is readable in a diff.

The important part is the key. A Zobrist hash is one-way, so a file keyed on hashes is
dead the moment the hashing changes — that is why nothing was rescued from the old
`move_cache.db`. `book.tsv` is keyed on **FEN**, and `import_book.py` re-derives each hash
through the engine (`from_fen` → `get_position_hash`). That is why importing takes a few
seconds instead of being an instant SQL load, and it is what lets the file outlive schema
and hash changes that would strand the DB.

The import verifies before writing anything: every FEN must survive
`to_fen(from_fen(fen))` unchanged, and no two FENs may share a hash. Either failure aborts
the whole import rather than filing rows against the wrong position.

Not exported: positions with no stored move (no searched work in them; rebuilt for free),
and the analysis cache (`--store analysis` if you ever need it).

---

## If something goes wrong

`backups/` holds `.db` snapshots; `migrate_book.py` writes one automatically before it
touches anything. To restore, stop everything touching the book and copy the snapshot over
`book.db`.

Nothing deletes book rows except `rebuild_book.py`, which asks first. `migrate_book.py`
deletes none unless you pass `--drop-orphans`.

Related: `CLAUDE.md` for the schema and the build, `tests/test_book_export.py` for what the
round trip guarantees.
