# Opening book: which file is which

> `book.db` is the live book and is **gitignored**. `book.tsv` is the versioned copy and
> **is** in git.

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

