//! The opening book: a `(position hash) -> ranked moves` table on disk.
//!
//! This replaces the old `move_cache` table, which stored one row per *(hash, depth)*
//! and was ~97% junk: iterative deepening filed a row for every depth it passed through
//! and the transposition-table dump filed internal nodes at depth >= 4, none of which a
//! depth-10 probe would ever accept. A book row is now the finished product of one
//! search -- the depth that actually completed, an exact score, and as many ranked
//! alternatives as the search was asked to produce.
//!
//! Two *pairs* of tables. Each pair is written together in one transaction:
//!
//! - `book_move` is the hot path. Everything a probe needs is in the row, so the
//!   runtime lookup never joins `position`.
//! - `position` is what makes a hash mean something again. A Zobrist hash is one-way, so
//!   without a FEN beside it a book entry can never be re-opened, re-searched, or
//!   expanded from. See `fen.rs`.
//!
//! `analysis_move` / `analysis_position` are the same two tables again, holding the
//! *analysis cache*: everything an interactive session searched, kept so that revisiting
//! a line is a probe hit rather than a re-search. They exist because the book is a
//! curated repertoire and exploration is not -- filing idle analysis into `book_move`
//! is how a repertoire stops meaning anything. Nothing writes across the pair boundary:
//! `Store::Book` is only ever written by an explicit save, `Store::Analysis` only by a
//! front end flushing what it searched.
//!
//! The analysis pair was added at version 1 without a `SCHEMA_VERSION` bump, because it
//! was purely additive: an older build never looks for those tables and reads the book
//! exactly as before. Version 2 is not additive. It puts a foreign key on each move
//! table's `hash`, referencing the position table beside it, so a ranked move can no
//! longer be filed under a hash with no FEN to re-open it -- the defect that made the old
//! `move_cache` unmigratable and left ~97% of it dead. Rewriting a table is not something
//! `CREATE TABLE IF NOT EXISTS` can do, so the bump is real and `ensure_schema` refuses a
//! version-1 file. What makes that affordable is `migrate_book.py`, which copies every
//! row forward: the constraint changes what the schema permits, not what any row means,
//! so nothing is re-searched.
//!
//! The constraint only guards the direction it names. `book_move` without `position` is
//! now impossible; `position` without `book_move` is still legal and still happens -- SQL
//! cannot require a parent to have children. The guard for that direction is in
//! [`save_entries`], which refuses to run its DELETE with nothing to insert afterwards.
//!
//! The DB lives in `book.db`. It did not replace the old `move_cache.db` in place: that
//! file's hashes could not be turned back into FENs, so nothing in it was migratable, and
//! it has since been deleted.
//!
//! `DB_PATH` is a relative string, so the file is resolved against the process CWD:
//! workers must run from the repo root, and tests must not (see
//! `tests/cache_isolation.py`).

use rusqlite::{Connection, params};
use std::collections::HashMap;
use std::fmt;

const DB_PATH: &str = "book.db";

/// Bumped whenever the table definitions below change.
///
/// Stored in `PRAGMA user_version`. The old code sniffed the columns of `move_cache`
/// ("no depth column -> DROP TABLE"), a heuristic that cannot tell a schema this module
/// has never seen from one it wrote itself. A version integer can.
pub const SCHEMA_VERSION: i32 = 2;

/// Which pair of tables a read or write targets.
///
/// The two pairs are structurally identical and deliberately never merged: a row's table
/// *is* its provenance. `Book` is curated and small; `Analysis` is whatever a session
/// happened to look at and grows without anyone deciding it should.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Store {
    Book,
    Analysis,
}

impl Store {
    pub fn moves_table(self) -> &'static str {
        match self {
            Store::Book => "book_move",
            Store::Analysis => "analysis_move",
        }
    }

    pub fn positions_table(self) -> &'static str {
        match self {
            Store::Book => "position",
            Store::Analysis => "analysis_position",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Store::Book => "book",
            Store::Analysis => "analysis cache",
        }
    }
}

/// What can go wrong opening or preparing the book.
#[derive(Debug)]
pub enum BookError {
    Sqlite(rusqlite::Error),
    /// The file on disk was written by a different schema version. Nothing has been
    /// changed; the caller has to decide whether to throw the book away.
    SchemaMismatch { found: i32, expected: i32 },
}

impl fmt::Display for BookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BookError::Sqlite(e) => write!(f, "{}", e),
            BookError::SchemaMismatch { found, expected } => write!(
                f,
                "{} is at schema version {}, but this build expects {}. \
                 Nothing was changed: the book is left exactly as it is. \
                 To carry the existing rows forward, run \
                 `./venv/bin/python migrate_book.py` (copies every row into the new \
                 schema; no re-searching, since the scores are unchanged). To discard the \
                 book instead, `./venv/bin/python rebuild_book.py` drops book_move and \
                 position and stamps version {}.",
                DB_PATH, found, expected, expected
            ),
        }
    }
}

impl From<rusqlite::Error> for BookError {
    fn from(e: rusqlite::Error) -> Self {
        BookError::Sqlite(e)
    }
}

/// One ranked move for one position. `rank` 1 is the best move.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BookMove {
    pub rank: i32,
    /// Python move repr, the same string `search::format_move_repr` produces.
    pub move_repr: String,
    /// White-relative and always exact -- never an alpha-beta bound. Ranks 2+ are only
    /// ever filled from a full-window MultiPV search for that reason.
    pub score: i32,
    /// The depth that actually *completed*, not the one that was requested.
    pub depth: i32,
    pub eval_version: i32,
}

/// One position's book entry.
#[derive(Clone, Debug)]
pub struct BookEntry {
    /// Ranked best-first, `rank` ascending and contiguous from 1.
    pub moves: Vec<BookMove>,
    /// Set only for entries this process searched; `None` for entries loaded from disk,
    /// because the load deliberately does not read the `position` table (the hot path
    /// has no use for a FEN, and loading them would cost memory per row for nothing).
    pub fen: Option<String>,
    /// Plies from the initial position. Path-dependent -- the same position can be
    /// reached by different move orders -- so the DB keeps the minimum ever seen.
    pub ply: Option<i32>,
}

/// Hash string -> ranked moves. Process-global in `lib.rs`.
pub type Book = HashMap<String, BookEntry>;

/// Opens the book DB in WAL mode with a generous busy timeout.
///
/// Both matter when many self-play workers share one `book.db`: WAL keeps readers off
/// the writer's back, and the timeout makes a concurrent writer wait its turn instead of
/// failing the transaction outright.
fn open_db() -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open(DB_PATH)?;
    let _ = conn.pragma_update(None, "journal_mode", "WAL");
    let _ = conn.pragma_update(None, "synchronous", "NORMAL");
    // SQLite defaults foreign key enforcement to OFF, per connection, and the setting is
    // not stored in the file -- so the REFERENCES clauses in `create_tables` are inert
    // decoration unless every connection opts in here. Must be issued outside a
    // transaction: inside one it is silently a no-op.
    let _ = conn.pragma_update(None, "foreign_keys", true);
    conn.busy_timeout(std::time::Duration::from_secs(30))?;
    Ok(conn)
}

fn user_version(conn: &Connection) -> Result<i32, rusqlite::Error> {
    conn.query_row("PRAGMA user_version", [], |row| row.get(0))
}

/// Creates the book schema if it is absent, and refuses to touch a foreign one.
///
/// **This never drops a table.** A `SCHEMA_VERSION` mismatch is reported as
/// [`BookError::SchemaMismatch`] and the file is left exactly as it was, because
/// "recreate the schema" runs on paths nobody thinks of as destructive -- the first save
/// of a worker, a stray call from a test, a script that opens the book to look at it --
/// and a version bump would turn any of them into a silent wipe of the training data.
/// Discarding the book is [`rebuild_db`], which only runs when a human asks for it.
pub fn setup_db() -> Result<(), BookError> {
    let conn = open_db()?;
    ensure_schema(&conn)
}

/// Schema creation against an already-open connection. Split out so a write transaction
/// can check the version without opening a second connection to the same file.
fn ensure_schema(conn: &Connection) -> Result<(), BookError> {
    let version = user_version(conn).unwrap_or(0);
    let has_tables: bool = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('book_move','position')")?
        .query_map([], |_| Ok(()))?
        .count()
        > 0;

    if has_tables && version != SCHEMA_VERSION {
        return Err(BookError::SchemaMismatch { found: version, expected: SCHEMA_VERSION });
    }

    create_tables(conn)
}

/// Drops both tables and recreates them at the current `SCHEMA_VERSION`.
///
/// The one destructive path in this module, and it exists only to be called explicitly
/// -- `rebuild_book.py`, which asks before it runs. Nothing calls it on your behalf.
pub fn rebuild_db() -> Result<(), BookError> {
    let conn = open_db()?;
    let version = user_version(&conn).unwrap_or(0);
    eprintln!(
        "[BOOK] rebuilding {}: dropping book_move and position (was schema version {})",
        DB_PATH, version
    );
    conn.execute("DROP TABLE IF EXISTS book_move", [])?;
    conn.execute("DROP TABLE IF EXISTS position", [])?;
    create_tables(&conn)
}

/// Drops the analysis cache and recreates it empty. Leaves the book untouched.
///
/// Separate from [`rebuild_db`] on purpose: discarding a session's exploration and
/// discarding a curated repertoire are not the same decision, and one must never be a
/// side effect of the other.
pub fn rebuild_analysis() -> Result<(), BookError> {
    let conn = open_db()?;
    eprintln!("[BOOK] clearing the analysis cache; {} and {} are untouched",
              Store::Book.moves_table(), Store::Book.positions_table());
    conn.execute("DROP TABLE IF EXISTS analysis_move", [])?;
    conn.execute("DROP TABLE IF EXISTS analysis_position", [])?;
    create_tables(&conn)
}

fn create_tables(conn: &Connection) -> Result<(), BookError> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS book_move (
            hash         TEXT NOT NULL,
            rank         INTEGER NOT NULL,
            move         TEXT NOT NULL,
            score        INTEGER NOT NULL,
            depth        INTEGER NOT NULL,
            eval_version INTEGER NOT NULL,
            PRIMARY KEY (hash, rank),
            FOREIGN KEY (hash) REFERENCES position(hash) ON DELETE CASCADE
        )",
        [],
    )?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS position (
            hash TEXT PRIMARY KEY,
            fen  TEXT NOT NULL,
            ply  INTEGER
        )",
        [],
    )?;

    // The analysis pair. Added at version 1 without a bump, because a build that predates
    // these tables reads book_move and position exactly as it always did. It carries the
    // same foreign key as the book pair from version 2 on: exploration is filed by the
    // same code path, so it can go wrong in exactly the same way.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS analysis_move (
            hash         TEXT NOT NULL,
            rank         INTEGER NOT NULL,
            move         TEXT NOT NULL,
            score        INTEGER NOT NULL,
            depth        INTEGER NOT NULL,
            eval_version INTEGER NOT NULL,
            PRIMARY KEY (hash, rank),
            FOREIGN KEY (hash) REFERENCES analysis_position(hash) ON DELETE CASCADE
        )",
        [],
    )?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS analysis_position (
            hash TEXT PRIMARY KEY,
            fen  TEXT NOT NULL,
            ply  INTEGER
        )",
        [],
    )?;

    conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    Ok(())
}

/// Loads `book_move` into memory. Does not touch `position` -- see [`BookEntry::fen`].
///
/// Reading never writes. It will not create the file, create the tables, or rebuild a
/// foreign schema: a missing or unreadable book loads as an empty one and says so. That
/// is what lets `tests/cache_isolation.py` reload the real book on the way out of a test
/// without the test suite creating -- or, after a `SCHEMA_VERSION` bump, dropping -- the
/// live `book.db` in the repo root. Writing is where the schema gets made.
pub fn load_book() -> Book {
    load_store(Store::Book)
}

/// Loads one store's move table into memory. `Store::Analysis` is read the same way and
/// under the same rules; a missing table (an older book.db that predates the analysis
/// pair) simply loads as empty.
pub fn load_store(store: Store) -> Book {
    let mut book: Book = HashMap::new();
    if !std::path::Path::new(DB_PATH).exists() {
        return book;
    }

    match open_db() {
        Ok(conn) => {
            let version = user_version(&conn).unwrap_or(0);
            if version != SCHEMA_VERSION {
                eprintln!(
                    "[BOOK] {}",
                    BookError::SchemaMismatch { found: version, expected: SCHEMA_VERSION }
                );
                eprintln!("[BOOK] loading it as empty; nothing on disk was touched");
                return book;
            }
            let sql = format!(
                "SELECT hash, rank, move, score, depth, eval_version
                 FROM {} ORDER BY hash, rank",
                store.moves_table()
            );
            match conn.prepare(&sql) {
                Ok(mut stmt) => {
                    let rows = stmt.query_map([], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            BookMove {
                                rank: row.get(1)?,
                                move_repr: row.get(2)?,
                                score: row.get(3)?,
                                depth: row.get(4)?,
                                eval_version: row.get(5)?,
                            },
                        ))
                    });
                    if let Ok(rows) = rows {
                        for (hash, bm) in rows.flatten() {
                            book.entry(hash)
                                .or_insert_with(|| BookEntry {
                                    moves: Vec::new(),
                                    fen: None,
                                    ply: None,
                                })
                                .moves
                                .push(bm);
                        }
                    }
                }
                // A missing analysis table is ordinary on a book.db written before the
                // pair existed: nothing to load, nothing to report as a failure.
                Err(e) => eprintln!("Error preparing {} query: {}", store.label(), e),
            }
            eprintln!("Loaded {} positions from the {}.", book.len(), store.label());
        }
        Err(e) => eprintln!("Error opening book DB: {}", e),
    }
    book
}

/// Writes the given positions to disk: their ranked moves and, when known, their FEN.
///
/// Both tables go through one connection inside one `BEGIN IMMEDIATE` transaction. With
/// twenty self-play workers contending for a single write lock, splitting this into two
/// transactions would double the lock acquisitions for no gain and could leave a
/// `book_move` row with no `position` row beside it.
///
/// `IMMEDIATE` matters in WAL: taking the write lock up front is what stops a
/// read-then-write transaction from failing with `SQLITE_BUSY_SNAPSHOT` when another
/// worker commits underneath it.
pub fn save_book_entries(book: &Book, hashes: &[String]) -> Result<(), BookError> {
    save_entries(book, hashes, Store::Book)
}

/// As [`save_book_entries`], but naming which pair of tables to write.
pub fn save_entries(book: &Book, hashes: &[String], store: Store) -> Result<(), BookError> {
    if hashes.is_empty() {
        return Ok(());
    }

    let conn = open_db()?;

    // A process that searched without ever loading the book -- a bench child, a test, a
    // worker on a fresh checkout -- would otherwise reach the INSERTs with
    // no tables to insert into and lose the whole run's work to an error log. A foreign
    // schema stops the save here instead, with the rows still in memory and the caller
    // told why, rather than writing this build's rows into someone else's tables.
    ensure_schema(&conn)?;

    conn.execute_batch("BEGIN IMMEDIATE")?;

    let mut written = 0usize;
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for hash in hashes {
        if !seen.insert(hash.as_str()) {
            continue;
        }
        let Some(entry) = book.get(hash) else { continue };

        // A hash with no FEN can never be turned back into a position -- it is the defect
        // that made `move_cache` unmigratable. Every live writer backfills the FEN before
        // queueing (`book_store` sets it, `save_position_to_book` fills it in from the
        // position the caller is looking at), so reaching this is a bug upstream; skip the
        // row rather than filing an entry nothing can ever re-open.
        let Some(fen) = &entry.fen else {
            eprintln!(
                "Skipping {} rows for {}: no FEN, so the hash could never be re-opened.",
                store.label(), hash
            );
            continue;
        };

        // An entry with no moves must not reach the DELETE below. That statement is a
        // *replacement*, and with nothing to insert afterwards it degrades into a plain
        // deletion -- which is how the ply-0 root row went missing from a 10,000-entry
        // book while the save still reported success.
        if entry.moves.is_empty() {
            eprintln!("Skipping {} rows for {}: entry holds no moves.", store.label(), hash);
            continue;
        }

        // The position row is the foreign key parent, so it has to be on disk before any
        // move row references it. It is written first for that reason -- with enforcement
        // on and this ordering reversed, every insert below fails on a missing parent.
        write_position(&conn, hash, fen, entry.ply, store);

        // Replace the whole rank list rather than upserting row by row: a re-search that
        // returns fewer ranks than last time must not leave the old tail behind, where it
        // would sit at a rank whose score came from a different search.
        let delete_sql = format!("DELETE FROM {} WHERE hash = ?1", store.moves_table());
        if let Err(e) = conn.execute(&delete_sql, params![hash]) {
            eprintln!("Error clearing {} rows for {}: {}", store.label(), hash, e);
            continue;
        }
        let mut ok = true;
        for bm in &entry.moves {
            let insert_sql = format!(
                "INSERT INTO {} (hash, rank, move, score, depth, eval_version)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                store.moves_table()
            );
            if let Err(e) = conn.execute(
                &insert_sql,
                params![hash, bm.rank, bm.move_repr, bm.score, bm.depth, bm.eval_version],
            ) {
                eprintln!("Error writing {} row for {}: {}", store.label(), hash, e);
                ok = false;
                break;
            }
        }
        if ok {
            written += 1;
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT") {
        let _ = conn.execute_batch("ROLLBACK");
        return Err(e.into());
    }
    eprintln!("Saved {} positions to the {}.", written, store.label());
    Ok(())
}

/// Inserts the `position` row for `hash`, or reconciles it with the row already there.
///
/// Two things can differ on an existing row. A differing **FEN** means two distinct
/// positions hashed to the same 64-bit value -- a Zobrist collision, which silently
/// corrupts every book entry involved, so it is logged as loudly as a log line can be
/// rather than swallowed. A differing **ply** is ordinary: the position was reached by a
/// different move order. The minimum wins, because that is how early the position can
/// actually appear in a game, which is what a book-expansion frontier wants to sort by.
fn write_position(conn: &Connection, hash: &str, fen: &str, ply: Option<i32>, store: Store) {
    let table = store.positions_table();
    let inserted = conn.execute(
        &format!("INSERT OR IGNORE INTO {} (hash, fen, ply) VALUES (?1, ?2, ?3)", table),
        params![hash, fen, ply],
    );
    match inserted {
        Ok(1) => return, // fresh row, nothing to reconcile
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error writing position row for {}: {}", hash, e);
            return;
        }
    }

    let existing: Result<(String, Option<i32>), _> = conn.query_row(
        &format!("SELECT fen, ply FROM {} WHERE hash = ?1", table),
        params![hash],
        |row| Ok((row.get(0)?, row.get(1)?)),
    );
    let Ok((existing_fen, existing_ply)) = existing else { return };

    if existing_fen != fen {
        eprintln!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        eprintln!("!! ZOBRIST COLLISION on hash {}", hash);
        eprintln!("!!   stored: {}", existing_fen);
        eprintln!("!!   new:    {}", fen);
        eprintln!("!! Both positions share one book entry; its moves are unsound for");
        eprintln!("!! at least one of them. Keeping the stored FEN.");
        eprintln!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        return;
    }

    let improves = match (ply, existing_ply) {
        (Some(new), Some(old)) => new < old,
        (Some(_), None) => true,
        _ => false,
    };
    if improves {
        if let Err(e) = conn.execute(
            &format!("UPDATE {} SET ply = ?1 WHERE hash = ?2", table),
            params![ply, hash],
        ) {
            eprintln!("Error updating ply for {}: {}", hash, e);
        }
    }
}
