
use rusqlite::{Connection, params};
use std::collections::HashMap;
use std::fmt;

const DB_PATH: &str = "book.db";

pub const SCHEMA_VERSION: i32 = 2;

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

#[derive(Debug)]
pub enum BookError {
    Sqlite(rusqlite::Error),
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BookMove {
    pub rank: i32,
    pub move_repr: String,
    pub score: i32,
    pub depth: i32,
    pub eval_version: i32,
}

#[derive(Clone, Debug)]
pub struct BookEntry {
    pub moves: Vec<BookMove>,
    pub fen: Option<String>,
    pub ply: Option<i32>,
}

pub type Book = HashMap<String, BookEntry>;

fn open_db() -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open(DB_PATH)?;
    let _ = conn.pragma_update(None, "journal_mode", "WAL");
    let _ = conn.pragma_update(None, "synchronous", "NORMAL");
    let _ = conn.pragma_update(None, "foreign_keys", true);
    conn.busy_timeout(std::time::Duration::from_secs(30))?;
    Ok(conn)
}

fn user_version(conn: &Connection) -> Result<i32, rusqlite::Error> {
    conn.query_row("PRAGMA user_version", [], |row| row.get(0))
}

pub fn setup_db() -> Result<(), BookError> {
    let conn = open_db()?;
    ensure_schema(&conn)
}

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

pub fn load_book() -> Book {
    load_store(Store::Book)
}

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
                Err(e) => eprintln!("Error preparing {} query: {}", store.label(), e),
            }
            eprintln!("Loaded {} positions from the {}.", book.len(), store.label());
        }
        Err(e) => eprintln!("Error opening book DB: {}", e),
    }
    book
}

pub fn save_book_entries(book: &Book, hashes: &[String]) -> Result<(), BookError> {
    save_entries(book, hashes, Store::Book)
}

pub fn save_entries(book: &Book, hashes: &[String], store: Store) -> Result<(), BookError> {
    if hashes.is_empty() {
        return Ok(());
    }

    let conn = open_db()?;

    ensure_schema(&conn)?;

    conn.execute_batch("BEGIN IMMEDIATE")?;

    let mut written = 0usize;
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for hash in hashes {
        if !seen.insert(hash.as_str()) {
            continue;
        }
        let Some(entry) = book.get(hash) else { continue };

        let Some(fen) = &entry.fen else {
            eprintln!(
                "Skipping {} rows for {}: no FEN, so the hash could never be re-opened.",
                store.label(), hash
            );
            continue;
        };

        if entry.moves.is_empty() {
            eprintln!("Skipping {} rows for {}: entry holds no moves.", store.label(), hash);
            continue;
        }

        write_position(&conn, hash, fen, entry.ply, store);

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

fn write_position(conn: &Connection, hash: &str, fen: &str, ply: Option<i32>, store: Store) {
    let table = store.positions_table();
    let inserted = conn.execute(
        &format!("INSERT OR IGNORE INTO {} (hash, fen, ply) VALUES (?1, ?2, ?3)", table),
        params![hash, fen, ply],
    );
    match inserted {
        Ok(1) => return,
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
