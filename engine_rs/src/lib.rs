mod types;
mod zobrist;
mod gamestate;
mod eval;
mod fen;
mod search;
mod cache;

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple, PyDict};
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use std::collections::HashMap;
use std::sync::Mutex;

use types::*;
use gamestate::GameState as RustGameState;

// The opening book, process-global and thread-safe.
lazy_static::lazy_static! {
    static ref BOOK: Mutex<cache::Book> = Mutex::new(HashMap::new());
    /// Position hashes written since the last save. Lets save_move_cache_to_db push only
    /// the new entries instead of rewriting the whole book — see cache::save_book_entries.
    static ref DIRTY_KEYS: Mutex<Vec<String>> = Mutex::new(Vec::new());
    /// Hashes known to have a row in `book_move` on disk: everything `load_book` read,
    /// plus everything `save_book_position` has written this session. The in-memory book
    /// cannot answer this on its own once the analysis cache is loaded into it — both
    /// stores share one map, and only the table a row came from says which it is.
    static ref BOOK_HASHES: Mutex<std::collections::HashSet<String>> =
        Mutex::new(std::collections::HashSet::new());
}

/// Convert a Python move tuple to our internal Move
fn py_move_to_rust(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Move> {
    // Check if it's a drop: ('drop', 'wN', (r, f))
    if let Ok(tup) = obj.downcast::<PyTuple>() {
        if tup.len() >= 1 {
            if let Ok(first) = tup.get_item(0)?.extract::<String>() {
                if first == "drop" {
                    let code: String = tup.get_item(1)?.extract()?;
                    let target = tup.get_item(2)?;
                    let target = target.downcast::<PyTuple>()?;
                    let r: usize = target.get_item(0)?.extract()?;
                    let f: usize = target.get_item(1)?.extract()?;
                    
                    let color = match code.chars().next() {
                        Some('w') => Color::White,
                        Some('b') => Color::Black,
                        _ => return Err(PyValueError::new_err("Invalid drop color")),
                    };
                    let pt = match code.chars().nth(1) {
                        Some('P') => PieceType::Pawn,
                        Some('N') => PieceType::Knight,
                        Some('B') => PieceType::Bishop,
                        Some('R') => PieceType::Rook,
                        Some('Q') => PieceType::Queen,
                        _ => return Err(PyValueError::new_err("Invalid drop piece")),
                    };
                    return Ok(Move::new_drop(sq(r, f), pt, color));
                }
            }
            // Normal move: ((r1,f1), (r2,f2), promotion)
            if tup.len() >= 3 {
                let from = tup.get_item(0)?;
                let from = from.downcast::<PyTuple>()?;
                let r1: usize = from.get_item(0)?.extract()?;
                let f1: usize = from.get_item(1)?.extract()?;
                
                let to = tup.get_item(1)?;
                let to = to.downcast::<PyTuple>()?;
                let r2: usize = to.get_item(0)?.extract()?;
                let f2: usize = to.get_item(1)?.extract()?;
                
                let promo_obj = tup.get_item(2)?;
                let promotion = if promo_obj.is_none() {
                    None
                } else {
                    let promo_str: String = promo_obj.extract()?;
                    match promo_str.to_uppercase().as_str() {
                        "R" => Some(PieceType::Rook),
                        "N" => Some(PieceType::Knight),
                        "B" => Some(PieceType::Bishop),
                        _ => None,
                    }
                };
                
                return Ok(Move::new_normal(sq(r1, f1), sq(r2, f2), promotion));
            }
        }
    }
    Err(PyValueError::new_err("Cannot parse move"))
}

/// Convert our internal Move to a Python tuple
fn rust_move_to_py(py: Python<'_>, m: Move) -> PyObject {
    if m.is_null() {
        return py.None();
    }
    if m.is_drop() {
        let to = m.to_sq();
        let r = sq_row(to);
        let f = sq_file(to);
        let color_c = match m.drop_color() {
            Color::White => 'w',
            Color::Black => 'b',
        };
        let pt_c = match m.drop_piece_type() {
            PieceType::Pawn => 'P',
            PieceType::Knight => 'N',
            PieceType::Bishop => 'B',
            PieceType::Rook => 'R',
            PieceType::Queen => 'Q',
            _ => '?',
        };
        let code = format!("{}{}", color_c, pt_c);
        let target = PyTuple::new(py, &[r, f]).unwrap();
        PyTuple::new(py, &[
            "drop".into_pyobject(py).unwrap().into_any(),
            code.into_pyobject(py).unwrap().into_any(),
            target.into_any(),
        ]).unwrap().into()
    } else {
        let from = m.from_sq();
        let to = m.to_sq();
        let from_tup = PyTuple::new(py, &[sq_row(from), sq_file(from)]).unwrap();
        let to_tup = PyTuple::new(py, &[sq_row(to), sq_file(to)]).unwrap();
        let promo: PyObject = match m.promotion() {
            Some(pt) => {
                // Case encodes colour everywhere else in this codebase (board chars,
                // 'wN'/'bN' drop codes), and gamestate.py validates promotion chars
                // case-sensitively against PROMOTION_PIECES_WHITE_STR/_BLACK_STR.
                // Move carries no colour, but the destination rank is unambiguous:
                // White promotes on row 0, Black on the last row.
                let white = sq_row(to) == 0;
                let c = match (pt, white) {
                    (PieceType::Rook, true) => "R",
                    (PieceType::Rook, false) => "r",
                    (PieceType::Knight, true) => "N",
                    (PieceType::Knight, false) => "n",
                    (PieceType::Bishop, true) => "B",
                    (PieceType::Bishop, false) => "b",
                    _ => "?",
                };
                c.into_pyobject(py).unwrap().into_any().unbind()
            }
            None => py.None(),
        };
        PyTuple::new(py, &[
            from_tup.into_any().unbind(),
            to_tup.into_any().unbind(),
            promo,
        ]).unwrap().into()
    }
}

#[pyclass(name = "GameState")]
pub struct PyGameState {
    inner: RustGameState,
    // Store extra Python-compatible fields
    #[pyo3(get, set)]
    white_ai_enabled: bool,
    #[pyo3(get, set)]
    black_ai_enabled: bool,
    #[pyo3(get, set)]
    ai_depth: i32,
    #[pyo3(get, set)]
    show_hint: bool,
    #[pyo3(get, set)]
    selected_square: Option<(usize, usize)>,
    #[pyo3(get, set)]
    selected_drop_piece: Option<String>,
    #[pyo3(get, set)]
    highlighted_moves: PyObject,
    // For undo (simplified: just track states)
    saved_states_count: usize,
}

#[pymethods]
impl PyGameState {
    #[new]
    fn new(py: Python<'_>) -> Self {
        PyGameState {
            inner: RustGameState::new(),
            white_ai_enabled: false,
            black_ai_enabled: false,
            ai_depth: 6,
            show_hint: false,
            selected_square: None,
            selected_drop_piece: None,
            highlighted_moves: PyList::empty(py).into(),
            saved_states_count: 0,
        }
    }

    fn setup_initial_board(&mut self) {
        self.inner.setup_initial_board();
        self.saved_states_count = 1;
    }

    fn save_state(&mut self) {
        self.saved_states_count += 1;
    }

    #[pyo3(signature = (m, is_check_game_over=None))]
    fn make_move(&mut self, py: Python<'_>, m: &Bound<'_, PyAny>, is_check_game_over: Option<bool>) -> PyResult<bool> {
        let check = is_check_game_over.unwrap_or(true);
        let rm = py_move_to_rust(py, m)?;
        Ok(self.inner.make_move(rm, check))
    }

    fn undo_move(&mut self) -> bool {
        self.inner.undo_move()
    }

    fn make_ai_move(&mut self, py: Python<'_>, m: &Bound<'_, PyAny>) -> PyResult<bool> {
        let rm = py_move_to_rust(py, m)?;
        self.inner.make_ai_move(rm);
        Ok(true)
    }

    fn undo_ai_move(&mut self) -> bool {
        self.inner.undo_ai_move();
        true
    }

    fn get_all_legal_moves<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let moves = self.inner.get_legal_moves_vec();
        let py_moves: Vec<PyObject> = moves.iter().map(|&m| rust_move_to_py(py, m)).collect();
        Ok(PyList::new(py, &py_moves)?)
    }

    fn is_in_check(&self, color: &str) -> bool {
        let c = match color {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return false,
        };
        self.inner.is_in_check(c)
    }

    fn check_game_over(&mut self) -> bool {
        self.inner.check_game_over()
    }

    fn complete_promotion(&mut self, piece_char: &str) -> bool {
        let pt = match piece_char.to_uppercase().as_str() {
            "R" => PieceType::Rook,
            "N" => PieceType::Knight,
            "B" => PieceType::Bishop,
            _ => return false,
        };
        self.inner.complete_promotion(pt)
    }

    fn copy(&self, py: Python<'_>) -> PyGameState {
        PyGameState {
            inner: self.inner.clone(),
            white_ai_enabled: self.white_ai_enabled,
            black_ai_enabled: self.black_ai_enabled,
            ai_depth: self.ai_depth,
            show_hint: self.show_hint,
            selected_square: None,
            selected_drop_piece: None,
            highlighted_moves: PyList::empty(py).into(),
            saved_states_count: self.saved_states_count,
        }
    }

    fn fast_copy_for_simulation(&self, py: Python<'_>) -> PyGameState {
        PyGameState {
            inner: self.inner.fast_copy(),
            white_ai_enabled: false,
            black_ai_enabled: false,
            ai_depth: self.ai_depth,
            show_hint: false,
            selected_square: None,
            selected_drop_piece: None,
            highlighted_moves: PyList::empty(py).into(),
            saved_states_count: 0,
        }
    }

    fn find_kings(&mut self) {
        self.inner.find_kings();
    }

    fn generate_all_pseudo_legal_moves<'py>(&self, py: Python<'py>, color: &str) -> PyResult<Bound<'py, PyList>> {
        let c = match color {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(PyValueError::new_err("Invalid color")),
        };
        let moves = self.inner.generate_pseudo_legal_moves(c);
        let py_moves: Vec<PyObject> = moves.iter().map(|&m| rust_move_to_py(py, m)).collect();
        Ok(PyList::new(py, &py_moves)?)
    }

    fn is_move_legal(&mut self, py: Python<'_>, m: &Bound<'_, PyAny>) -> PyResult<bool> {
        let rm = py_move_to_rust(py, m)?;
        let legal = self.inner.get_legal_moves_vec();
        Ok(legal.contains(&rm))
    }

    fn reset_board(&mut self) {
        self.inner = RustGameState::new();
        self.saved_states_count = 0;
    }

    // Properties
    #[getter]
    fn board<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut rows = Vec::with_capacity(BOARD_SIZE);
        for r in 0..BOARD_SIZE {
            let mut row = Vec::with_capacity(BOARD_SIZE);
            for f in 0..BOARD_SIZE {
                row.push(self.inner.board[sq(r, f)].to_char().to_string());
            }
            rows.push(PyList::new(py, &row)?);
        }
        Ok(PyList::new(py, &rows)?)
    }

    #[setter]
    fn set_board(&mut self, _py: Python<'_>, value: &Bound<'_, PyList>) -> PyResult<()> {
        for r in 0..BOARD_SIZE {
            let row = value.get_item(r)?;
            let row = row.downcast::<PyList>()?;
            for f in 0..BOARD_SIZE {
                let cell: String = row.get_item(f)?.extract()?;
                self.inner.board[sq(r, f)] = Piece::from_char(cell.chars().next().unwrap_or('.'));
            }
        }
        self.inner.find_kings();
        self.inner.hash = self.inner.compute_hash();
        self.inner.invalidate_cache();
        Ok(())
    }

    #[getter]
    fn current_turn(&self) -> &str {
        match self.inner.current_turn {
            Color::White => "w",
            Color::Black => "b",
        }
    }

    #[setter]
    fn set_current_turn(&mut self, value: &str) {
        self.inner.current_turn = match value {
            "b" => Color::Black,
            _ => Color::White,
        };
        self.inner.hash = self.inner.compute_hash();
        self.inner.invalidate_cache();
    }

    #[getter]
    fn hands<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let d = PyDict::new(py);
        for (ci, color_str) in [(0usize, "w"), (1, "b")] {
            let hand = PyDict::new(py);
            for (pi, piece_str) in [(0, "P"), (1, "N"), (2, "B"), (3, "R"), (4, "Q")] {
                hand.set_item(piece_str, self.inner.hands[ci][pi] as i32)?;
            }
            d.set_item(color_str, hand)?;
        }
        Ok(d.into())
    }

    #[setter]
    fn set_hands(&mut self, _py: Python<'_>, value: &Bound<'_, PyDict>) -> PyResult<()> {
        for (ci, color_str) in [(0usize, "w"), (1usize, "b")] {
            if let Some(hand_obj) = value.get_item(color_str)? {
                let hand = hand_obj.downcast::<PyDict>()?;
                for (pi, piece_str) in [(0usize, "P"), (1, "N"), (2, "B"), (3, "R"), (4, "Q")] {
                    if let Some(count) = hand.get_item(piece_str)? {
                        self.inner.hands[ci][pi] = count.extract::<i32>()? as u8;
                    }
                }
            }
        }
        self.inner.hash = self.inner.compute_hash();
        Ok(())
    }

    #[getter]
    fn king_pos<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let d = PyDict::new(py);
        for (ci, color_str) in [(0usize, "w"), (1, "b")] {
            let s = self.inner.king_pos[ci];
            let tup = PyTuple::new(py, &[sq_row(s), sq_file(s)])?;
            d.set_item(color_str, tup)?;
        }
        Ok(d.into())
    }

    #[setter]
    fn set_king_pos(&mut self, _py: Python<'_>, value: &Bound<'_, PyDict>) -> PyResult<()> {
        for (ci, color_str) in [(0usize, "w"), (1, "b")] {
            if let Some(pos_obj) = value.get_item(color_str)? {
                if !pos_obj.is_none() {
                    let tup = pos_obj.downcast::<PyTuple>()?;
                    let r: usize = tup.get_item(0)?.extract()?;
                    let f: usize = tup.get_item(1)?.extract()?;
                    self.inner.king_pos[ci] = sq(r, f);
                }
            }
        }
        Ok(())
    }

    #[getter]
    fn checkmate(&self) -> bool { self.inner.checkmate }

    #[setter]
    fn set_checkmate(&mut self, v: bool) { self.inner.checkmate = v; }

    #[getter]
    fn stalemate(&self) -> bool { self.inner.stalemate }

    #[setter]
    fn set_stalemate(&mut self, v: bool) { self.inner.stalemate = v; }

    #[getter]
    fn last_move<'py>(&self, py: Python<'py>) -> PyObject {
        rust_move_to_py(py, self.inner.last_move)
    }

    #[setter]
    fn set_last_move(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            self.inner.last_move = Move::NULL;
        } else {
            self.inner.last_move = py_move_to_rust(py, value)?;
        }
        Ok(())
    }

    #[getter]
    fn game_over_message(&self) -> &str { &self.inner.game_over_message }

    #[getter]
    fn is_draw(&self) -> bool { self.inner.is_draw }

    #[setter]
    fn set_is_draw(&mut self, v: bool) { self.inner.is_draw = v; }

    #[getter]
    fn ply(&self) -> u32 { self.inner.ply }

    /// Settable because `ai._sync_to_rust` builds a fresh Rust state per search: without
    /// this, every book row would claim the position first occurs at ply 0.
    #[setter]
    fn set_ply(&mut self, v: u32) { self.inner.ply = v; }

    #[getter]
    fn ply_limit(&self) -> u32 { self.inner.ply_limit }

    #[setter]
    fn set_ply_limit(&mut self, v: u32) { self.inner.ply_limit = v; }

    /// How many times the current position has occurred in this game.
    fn repetition_count(&self) -> usize { self.inner.repetition_count() }

    #[getter]
    fn needs_promotion_choice(&self) -> bool { self.inner.needs_promotion_choice }

    #[setter]
    fn set_needs_promotion_choice(&mut self, v: bool) { self.inner.needs_promotion_choice = v; }

    #[getter]
    fn promotion_square<'py>(&self, py: Python<'py>) -> PyObject {
        match self.inner.promotion_square {
            Some(s) => PyTuple::new(py, &[sq_row(s), sq_file(s)]).unwrap().into(),
            None => py.None(),
        }
    }

    #[getter]
    fn promoted_pieces<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let pyset = pyo3::types::PySet::empty(py)?;
        for s in 0..NUM_SQUARES {
            if self.inner.promoted_pieces & (1u64 << s) != 0 {
                let tup = PyTuple::new(py, &[sq_row(s), sq_file(s)])?;
                pyset.add(tup)?;
            }
        }
        Ok(pyset.into())
    }

    #[setter]
    fn set_promoted_pieces(&mut self, _py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.promoted_pieces = 0;
        let iter = value.try_iter()?;
        for item in iter {
            let item = item?;
            let tup = item.downcast::<PyTuple>()?;
            let r: usize = tup.get_item(0)?.extract()?;
            let f: usize = tup.get_item(1)?.extract()?;
            self.inner.promoted_pieces |= 1u64 << sq(r, f);
        }
        // Promoted squares are hashed (zobrist::get_position_hash), so skipping this
        // left every position synced from Python with a promoted piece carrying a hash
        // that ignored it -- and the book is keyed by that hash. Caught by
        // tests/test_fen.py: the FEN written beside such an entry did not hash back to
        // the row it belonged to.
        self.inner.hash = self.inner.compute_hash();
        Ok(())
    }

    #[getter]
    fn move_log<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Return empty list (move_log is tracked in Python frontends)
        Ok(PyList::empty(py))
    }

    #[getter]
    fn last_move_for_promotion<'py>(&self, py: Python<'py>) -> PyObject {
        py.None()
    }

    #[getter]
    fn saved_states<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Return dummy list with correct length for undo logic
        let items: Vec<i32> = (0..self.saved_states_count as i32).collect();
        Ok(PyList::new(py, &items)?)
    }

    #[getter]
    fn ai_history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<i32> = (0..self.inner.ai_history.len() as i32).collect();
        Ok(PyList::new(py, &items)?)
    }

    // Extra getters that Python code accesses
    #[getter]
    fn _all_legal_moves_cache(&self) -> Option<bool> {
        None  // Always return None to match Python behavior
    }

    #[setter]
    fn set__all_legal_moves_cache(&mut self, _v: Option<bool>) {
        self.inner.invalidate_cache();
    }

    #[getter]
    fn _hash_cache(&self) -> Option<bool> { None }

    #[setter]
    fn set__hash_cache(&mut self, _v: Option<bool>) {}

    #[getter]
    fn _is_check_cache(&self) -> Option<bool> { None }

    #[setter]
    fn set__is_check_cache(&mut self, _v: Option<bool>) {}
}

// Module-level AI functions

/// `return_top_n == 1` gives back the bare move tuple, as the GUI, the bot and the
/// training scripts all expect. Anything higher gives back a list of `(move, score)`
/// ranked best-first for the side to move, and asks the engine for a MultiPV search so
/// those scores are real values rather than alpha-beta bounds — this is the call that
/// `self_play.choose_move_with_exploration` has always made and never got a second move
/// out of.
#[pyfunction]
#[pyo3(signature = (gs, depth=None, return_top_n=None, time_limit=None, parallel=None))]
fn find_best_move(py: Python<'_>, gs: &mut PyGameState, depth: Option<i32>, return_top_n: Option<i32>, time_limit: Option<f64>, parallel: Option<bool>) -> PyResult<PyObject> {
    let d = depth.unwrap_or(6);
    let top_n = return_top_n.unwrap_or(1);

    let ranked = py.allow_threads(|| {
        let mut book = BOOK.lock().unwrap();
        let mut dirty = Vec::new();
        let res = search::find_best_move(&mut gs.inner, d, top_n, &mut book, &mut dirty, time_limit, parallel);
        DIRTY_KEYS.lock().unwrap().append(&mut dirty);
        res
    });

    if top_n <= 1 {
        let best = ranked.first().map(|&(m, _)| m).unwrap_or(Move::NULL);
        return Ok(rust_move_to_py(py, best));
    }

    let list = PyList::empty(py);
    for &(m, score) in &ranked {
        let tup = PyTuple::new(py, &[
            rust_move_to_py(py, m),
            score.into_pyobject(py)?.into_any().unbind(),
        ])?;
        list.append(tup)?;
    }
    Ok(list.into())
}

/// The single-PV search plus its score, without asking for a second rank.
///
/// `bench/` wants exactly this: `return_top_n=2` would report a score, but it would also
/// switch the engine into MultiPV and stop measuring the path training actually runs.
#[pyfunction]
#[pyo3(signature = (gs, depth=None, time_limit=None, parallel=None))]
fn find_best_move_with_score(py: Python<'_>, gs: &mut PyGameState, depth: Option<i32>, time_limit: Option<f64>, parallel: Option<bool>) -> PyResult<PyObject> {
    let d = depth.unwrap_or(6);
    let ranked = py.allow_threads(|| {
        let mut book = BOOK.lock().unwrap();
        let mut dirty = Vec::new();
        let res = search::find_best_move(&mut gs.inner, d, 1, &mut book, &mut dirty, time_limit, parallel);
        DIRTY_KEYS.lock().unwrap().append(&mut dirty);
        res
    });
    let (best, score) = ranked.first().copied().unwrap_or((Move::NULL, 0));
    let tup = PyTuple::new(py, &[
        rust_move_to_py(py, best),
        score.into_pyobject(py)?.into_any().unbind(),
    ])?;
    Ok(tup.into())
}

/// Turn the root-level parallel search on or off for this process.
/// Off by default, which keeps self-play training single-threaded.
#[pyfunction]
#[pyo3(signature = (enabled, min_depth=None))]
fn set_parallel_search(enabled: bool, min_depth: Option<i32>) {
    search::set_parallel_search(enabled, min_depth);
}

/// Returns (enabled, min_depth) for the root-level parallel search.
#[pyfunction]
fn get_parallel_search(py: Python<'_>) -> PyResult<PyObject> {
    let (enabled, min_depth) = search::parallel_search_config();
    let tup = PyTuple::new(py, &[
        enabled.into_pyobject(py)?.to_owned().into_any().unbind(),
        min_depth.into_pyobject(py)?.into_any().unbind(),
    ])?;
    Ok(tup.into())
}

#[pyfunction]
fn evaluate_position(_py: Python<'_>, gs: &PyGameState) -> f64 {
    eval::evaluate_position(&gs.inner) as f64
}

#[pyfunction]
fn get_position_hash(gs: &PyGameState) -> String {
    gs.inner.hash.to_string()
}

/// Loads the book from disk into this process. Names kept for API compatibility.
#[pyfunction]
fn load_move_cache_from_db() {
    let loaded = cache::load_book();
    *BOOK_HASHES.lock().unwrap() = loaded.keys().cloned().collect();
    let mut book = BOOK.lock().unwrap();
    *book = loaded;
    // Everything just loaded is already on disk.
    DIRTY_KEYS.lock().unwrap().clear();
}

/// Merges the analysis cache into the in-memory book, and returns how many entries it
/// added.
///
/// **The book wins every collision.** A curated row and a row from someone poking at a
/// position are not interchangeable, and a probe cannot tell them apart once they are in
/// the same map — so the repertoire's answer is the one that survives, whatever depth
/// the cached one reached.
///
/// Loading this is what makes the cache worth having: `probe_book` reads the one map, so
/// a position analysed in an earlier session comes back as a hit instead of a re-search.
/// It also means the engine will *play* from cached analysis, which is why nothing loads
/// it implicitly — `build_book.py` and the bot call `load_move_cache_from_db` alone and
/// see the book exactly as they always did.
#[pyfunction]
fn load_analysis_from_db() -> usize {
    let cached = cache::load_store(cache::Store::Analysis);
    let mut book = BOOK.lock().unwrap();
    let mut added = 0usize;
    for (hash, entry) in cached {
        if !book.contains_key(&hash) {
            book.insert(hash, entry);
            added += 1;
        }
    }
    // Loaded, therefore already on disk: not pending.
    DIRTY_KEYS.lock().unwrap().clear();
    added
}

/// Flushes every position searched since the last flush into the analysis tables.
///
/// This is the bulk write the book deliberately no longer gets. Everything a session
/// searched belongs in the cache — that is what a cache is — and nothing here can reach
/// `book_move`, which only `save_book_position` writes.
#[pyfunction]
fn save_analysis_to_db() -> PyResult<usize> {
    let book = BOOK.lock().unwrap();
    let mut dirty = DIRTY_KEYS.lock().unwrap();
    if dirty.is_empty() {
        return Ok(0);
    }
    cache::save_entries(&book, &dirty, cache::Store::Analysis)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = dirty.len();
    dirty.clear();
    Ok(n)
}

/// Drops the analysis tables and recreates them empty. The book is untouched.
#[pyfunction]
fn rebuild_analysis() -> PyResult<()> {
    cache::rebuild_analysis().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Whether this position already has a row in `book_move` on disk.
#[pyfunction]
fn book_has_row(gs: &PyGameState) -> bool {
    BOOK_HASHES.lock().unwrap().contains(&gs.inner.hash.to_string())
}

/// Writes the positions searched since the last save.
///
/// Raises rather than swallowing a failure: on a schema mismatch the rows are still in
/// memory and `dirty` is left intact, so a worker stops with its work recoverable
/// instead of quietly dropping every search it has done.
#[pyfunction]
#[pyo3(signature = (_cache_arg=None))]
fn save_move_cache_to_db(_py: Python<'_>, _cache_arg: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let book = BOOK.lock().unwrap();
    let mut dirty = DIRTY_KEYS.lock().unwrap();
    if dirty.is_empty() {
        return Ok(());
    }
    cache::save_book_entries(&book, &dirty)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    dirty.clear();
    Ok(())
}

/// Writes **one** position's book entry, named by the caller, and nothing else.
///
/// This is the deliberate counterpart to `save_move_cache_to_db`, which flushes every
/// hash queued since the last save. A GUI session searches whatever the user happens to
/// look at -- hints, engine replies, idle exploration -- and a bulk flush cannot tell
/// any of that apart from a line the user actually wants in the repertoire. The book is
/// a curated artefact, so filing a row into it is an explicit act: the caller names the
/// position and gets back whether there was anything to write.
///
/// Returns `false` when this process has no searched entry for the position -- nothing
/// was written, and that is not an error. Loading the book does not count: entries read
/// from disk are already there, and carry no FEN to write a `position` row from.
#[pyfunction]
fn save_book_position(gs: &PyGameState) -> PyResult<bool> {
    let hash = gs.inner.hash.to_string();
    let mut book = BOOK.lock().unwrap();
    let Some(entry) = book.get_mut(&hash) else {
        return Ok(false);
    };

    // Entries that came off disk carry no FEN -- neither load reads the positions table,
    // because the hot path has no use for one and 10k of them cost memory for nothing.
    // Saving such an entry to the book without a FEN would file a hash that can never be
    // turned back into a position, which is the exact defect that killed `move_cache`.
    // The caller is *looking at* the position, so the FEN is free and authoritative here.
    if entry.fen.is_none() {
        entry.fen = Some(fen::to_fen(&gs.inner));
    }
    if entry.ply.is_none() {
        entry.ply = Some(gs.inner.ply as i32);
    }

    let book = &*book;
    cache::save_book_entries(book, std::slice::from_ref(&hash))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // No longer pending: the analysis flush must not also file it as exploration.
    DIRTY_KEYS.lock().unwrap().retain(|k| k != &hash);
    BOOK_HASHES.lock().unwrap().insert(hash);
    Ok(true)
}

/// Whether this process holds a searched book entry for `gs` -- i.e. whether
/// `save_book_position` would write anything.
#[pyfunction]
fn book_has_position(gs: &PyGameState) -> bool {
    BOOK.lock().unwrap().contains_key(&gs.inner.hash.to_string())
}

/// How many searched positions are queued but not on disk.
///
/// The in-memory book is also the analysis cache, so this counts a session's exploration
/// as well as anything worth keeping. It exists so a front end can say what it is sitting
/// on rather than leaving the user to guess.
#[pyfunction]
fn pending_book_writes() -> usize {
    let dirty = DIRTY_KEYS.lock().unwrap();
    let unique: std::collections::HashSet<&str> = dirty.iter().map(|s| s.as_str()).collect();
    unique.len()
}

/// Drops the queue of unsaved positions without writing them.
///
/// The entries stay in the in-memory book, so they are still cache hits for the rest of
/// the session; they simply stop being candidates for a bulk flush.
#[pyfunction]
fn discard_pending_book_writes() -> usize {
    let mut dirty = DIRTY_KEYS.lock().unwrap();
    let n = dirty.len();
    dirty.clear();
    n
}

/// Creates the book schema if it is absent. Raises on a foreign schema; never drops.
#[pyfunction]
fn setup_db() -> PyResult<()> {
    cache::setup_db().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Drops both book tables and recreates them at the current SCHEMA_VERSION.
///
/// Destructive and deliberately unreachable by accident: nothing in the engine calls
/// this. `rebuild_book.py` is the front door, and it asks first.
#[pyfunction]
fn rebuild_book() -> PyResult<()> {
    cache::rebuild_db().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Number of positions the in-memory book holds.
#[pyfunction]
fn book_size() -> usize {
    BOOK.lock().unwrap().len()
}

/// Serialize a position to minihouse FEN. See `engine_rs/src/fen.rs` for the format.
#[pyfunction]
fn to_fen(gs: &PyGameState) -> String {
    fen::to_fen(&gs.inner)
}

/// Parse a minihouse FEN into a fresh GameState. Raises ValueError on malformed input.
#[pyfunction]
fn from_fen(py: Python<'_>, s: &str) -> PyResult<PyGameState> {
    let inner = fen::from_fen(s).map_err(PyValueError::new_err)?;
    let mut out = PyGameState::new(py);
    out.inner = inner;
    out.saved_states_count = 1;
    Ok(out)
}

#[pyfunction]
fn is_move_still_legal(py: Python<'_>, gs: &mut PyGameState, m: &Bound<'_, PyAny>) -> PyResult<bool> {
    let rm = py_move_to_rust(py, m)?;
    let legal = gs.inner.get_legal_moves_vec();
    Ok(legal.contains(&rm))
}

#[pyfunction]
fn parse_move_string(py: Python<'_>, s: &str) -> PyResult<PyObject> {
    match search::parse_move_repr(s) {
        Some(m) => Ok(rust_move_to_py(py, m)),
        None => Ok(py.None()),
    }
}

#[pymodule]
fn minichess_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_function(wrap_pyfunction!(find_best_move, m)?)?;
    m.add_function(wrap_pyfunction!(find_best_move_with_score, m)?)?;
    m.add_function(wrap_pyfunction!(set_parallel_search, m)?)?;
    m.add_function(wrap_pyfunction!(get_parallel_search, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_position, m)?)?;
    m.add_function(wrap_pyfunction!(get_position_hash, m)?)?;
    m.add_function(wrap_pyfunction!(load_move_cache_from_db, m)?)?;
    m.add_function(wrap_pyfunction!(save_move_cache_to_db, m)?)?;
    m.add_function(wrap_pyfunction!(save_book_position, m)?)?;
    m.add_function(wrap_pyfunction!(book_has_position, m)?)?;
    m.add_function(wrap_pyfunction!(pending_book_writes, m)?)?;
    m.add_function(wrap_pyfunction!(discard_pending_book_writes, m)?)?;
    m.add_function(wrap_pyfunction!(load_analysis_from_db, m)?)?;
    m.add_function(wrap_pyfunction!(save_analysis_to_db, m)?)?;
    m.add_function(wrap_pyfunction!(rebuild_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(book_has_row, m)?)?;
    m.add_function(wrap_pyfunction!(setup_db, m)?)?;
    m.add_function(wrap_pyfunction!(rebuild_book, m)?)?;
    m.add_function(wrap_pyfunction!(book_size, m)?)?;
    m.add_function(wrap_pyfunction!(to_fen, m)?)?;
    m.add_function(wrap_pyfunction!(from_fen, m)?)?;
    m.add_function(wrap_pyfunction!(is_move_still_legal, m)?)?;
    m.add_function(wrap_pyfunction!(parse_move_string, m)?)?;
    
    // Re-export constants needed by Python
    m.add("CHECKMATE_SCORE", eval::CHECKMATE_SCORE)?;
    m.add("STALEMATE_SCORE", eval::STALEMATE_SCORE)?;
    m.add("BOARD_SIZE", BOARD_SIZE)?;
    m.add("EVAL_VERSION", eval::EVAL_VERSION)?;
    m.add("SCHEMA_VERSION", cache::SCHEMA_VERSION)?;
    
    Ok(())
}
