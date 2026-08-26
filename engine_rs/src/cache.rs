use rusqlite::{Connection, params};
use std::collections::HashMap;

const DB_PATH: &str = "move_cache.db";

/// Opens the cache DB in WAL mode with a generous busy timeout.
///
/// Both matter when many self-play workers share one `move_cache.db`: WAL keeps
/// readers off the writer's back, and the timeout makes a concurrent writer wait
/// its turn instead of failing the transaction outright.
fn open_db() -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open(DB_PATH)?;
    let _ = conn.pragma_update(None, "journal_mode", "WAL");
    let _ = conn.pragma_update(None, "synchronous", "NORMAL");
    conn.busy_timeout(std::time::Duration::from_secs(30))?;
    Ok(conn)
}

pub fn setup_db() -> Result<(), rusqlite::Error> {
    let conn = open_db()?;
    
    // Check if table exists and has correct schema
    let has_depth: bool = {
        let mut stmt = conn.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='move_cache'")?;
        let exists = stmt.query_map([], |_| Ok(()))?.count() > 0;
        if exists {
            let mut info = conn.prepare("PRAGMA table_info(move_cache)")?;
            let cols: Vec<String> = info.query_map([], |row| row.get::<_, String>(1))?.filter_map(|r| r.ok()).collect();
            cols.contains(&"depth".to_string())
        } else {
            false
        }
    };
    
    if !has_depth {
        conn.execute("DROP TABLE IF EXISTS move_cache", [])?;
    }
    
    conn.execute(
        "CREATE TABLE IF NOT EXISTS move_cache (
            hash TEXT NOT NULL,
            depth INTEGER NOT NULL,
            best_move_repr TEXT NOT NULL,
            PRIMARY KEY (hash, depth)
        )",
        [],
    )?;
    
    Ok(())
}

pub fn load_move_cache() -> HashMap<(String, i32), String> {
    let mut cache = HashMap::new();
    if let Err(e) = setup_db() {
        eprintln!("Error setting up DB: {}", e);
        return cache;
    }
    
    match open_db() {
        Ok(conn) => {
            match conn.prepare("SELECT hash, depth, best_move_repr FROM move_cache") {
                Ok(mut stmt) => {
                    let rows = stmt.query_map([], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, i32>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    });
                    if let Ok(rows) = rows {
                        for row in rows.flatten() {
                            cache.insert((row.0, row.1), row.2);
                        }
                    }
                }
                Err(e) => eprintln!("Error preparing query: {}", e),
            }
            eprintln!("Loaded {} entries from move cache.", cache.len());
        }
        Err(e) => eprintln!("Error opening DB: {}", e),
    }
    cache
}

pub fn save_move_cache(cache: &HashMap<(String, i32), String>) {
    if cache.is_empty() {
        return;
    }
    
    match open_db() {
        Ok(conn) => {
            let tx = conn.unchecked_transaction().ok();
            let mut count = 0;
            for ((hash, depth), move_repr) in cache {
                if conn.execute(
                    "INSERT OR REPLACE INTO move_cache (hash, depth, best_move_repr) VALUES (?1, ?2, ?3)",
                    params![hash, depth, move_repr],
                ).is_ok() {
                    count += 1;
                }
            }
            if let Some(tx) = tx {
                let _ = tx.commit();
            }
            eprintln!("Saved {} entries to move cache.", count);
        }
        Err(e) => eprintln!("Error saving cache: {}", e),
    }
}

/// Writes only the given keys back to the DB.
///
/// `save_move_cache` rewrites every entry the process holds, which is O(cache)
/// per call — fine for one process, ruinous once twenty self-play workers each
/// re-send a six-figure cache after every move. Callers that track which keys
/// they touched should use this instead.
pub fn save_move_cache_entries(cache: &HashMap<(String, i32), String>, keys: &[(String, i32)]) {
    if keys.is_empty() {
        return;
    }

    match open_db() {
        Ok(conn) => {
            let tx = conn.unchecked_transaction().ok();
            let mut count = 0;
            for key in keys {
                let Some(move_repr) = cache.get(key) else { continue };
                if conn.execute(
                    "INSERT OR REPLACE INTO move_cache (hash, depth, best_move_repr) VALUES (?1, ?2, ?3)",
                    params![key.0, key.1, move_repr],
                ).is_ok() {
                    count += 1;
                }
            }
            if let Some(tx) = tx {
                let _ = tx.commit();
            }
            eprintln!("Saved {} new entries to move cache.", count);
        }
        Err(e) => eprintln!("Error saving cache: {}", e),
    }
}
