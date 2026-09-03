mod types;
mod zobrist;
mod gamestate;
mod eval;
mod fen;
mod search;
mod cache;
mod encode;
mod mcts;

use pyo3::prelude::*;
use pyo3::buffer::PyBuffer;
use pyo3::types::{PyByteArray, PyList, PyTuple, PyDict};
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use std::collections::HashMap;
use std::sync::Mutex;

use types::*;
use gamestate::GameState as RustGameState;

lazy_static::lazy_static! {
    static ref BOOK: Mutex<cache::Book> = Mutex::new(HashMap::new());
    static ref DIRTY_KEYS: Mutex<Vec<String>> = Mutex::new(Vec::new());
    static ref BOOK_HASHES: Mutex<std::collections::HashSet<String>> =
        Mutex::new(std::collections::HashSet::new());
}

fn py_move_to_rust(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Move> {
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

    #[setter]
    fn set_ply(&mut self, v: u32) { self.inner.ply = v; }

    #[getter]
    fn ply_limit(&self) -> u32 { self.inner.ply_limit }

    #[setter]
    fn set_ply_limit(&mut self, v: u32) { self.inner.ply_limit = v; }

    fn repetition_count(&self) -> usize { self.inner.repetition_count() }

    fn is_terminal_draw(&self) -> bool { self.inner.is_terminal_draw() }

    #[getter]
    fn reversible_plies(&self) -> u32 { self.inner.reversible_plies }

    #[setter]
    fn set_reversible_plies(&mut self, v: u32) { self.inner.reversible_plies = v; }

    #[getter]
    fn position_history(&self) -> Vec<u64> { self.inner.position_history.clone() }

    #[setter]
    fn set_position_history(&mut self, v: Vec<u64>) {
        let reversible = v.len().saturating_sub(1).min(crate::gamestate::MAX_REPETITION_SCAN) as u32;
        self.inner.position_history = v;
        if self.inner.position_history.is_empty() {
            self.inner.position_history.push(self.inner.hash);
            self.inner.reversible_plies = 0;
        } else {
            self.inner.reversible_plies = reversible;
        }
        self.inner.history_root = self.inner.position_history.len();
    }

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
        self.inner.hash = self.inner.compute_hash();
        Ok(())
    }

    #[getter]
    fn move_log<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        Ok(PyList::empty(py))
    }

    #[getter]
    fn last_move_for_promotion<'py>(&self, py: Python<'py>) -> PyObject {
        py.None()
    }

    #[getter]
    fn saved_states<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<i32> = (0..self.saved_states_count as i32).collect();
        Ok(PyList::new(py, &items)?)
    }

    #[getter]
    fn ai_history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<i32> = (0..self.inner.ai_history.len() as i32).collect();
        Ok(PyList::new(py, &items)?)
    }

    #[getter]
    fn _all_legal_moves_cache(&self) -> Option<bool> {
        None
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

#[pyfunction]
#[pyo3(signature = (gs, depth=None, return_top_n=None, time_limit=None, parallel=None))]
fn find_best_move(py: Python<'_>, gs: &mut PyGameState, depth: Option<i32>, return_top_n: Option<i32>, time_limit: Option<f64>, parallel: Option<bool>) -> PyResult<PyObject> {
    let d = depth.unwrap_or(6);
    let top_n = return_top_n.unwrap_or(1);

    let ranked = py.allow_threads(|| {
        let mut dirty = Vec::new();
        let res = search::find_best_move(&mut gs.inner, d, top_n, &BOOK, &mut dirty, time_limit, parallel);
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

#[pyfunction]
#[pyo3(signature = (gs, depth=None, time_limit=None, parallel=None))]
fn find_best_move_with_score(py: Python<'_>, gs: &mut PyGameState, depth: Option<i32>, time_limit: Option<f64>, parallel: Option<bool>) -> PyResult<PyObject> {
    let d = depth.unwrap_or(6);
    let ranked = py.allow_threads(|| {
        let mut dirty = Vec::new();
        let res = search::find_best_move(&mut gs.inner, d, 1, &BOOK, &mut dirty, time_limit, parallel);
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

#[pyfunction]
#[pyo3(signature = (enabled, min_depth=None))]
fn set_parallel_search(enabled: bool, min_depth: Option<i32>) {
    search::set_parallel_search(enabled, min_depth);
}

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

#[pyfunction]
fn load_move_cache_from_db() {
    let loaded = cache::load_book();
    *BOOK_HASHES.lock().unwrap() = loaded.keys().cloned().collect();
    let mut book = BOOK.lock().unwrap();
    *book = loaded;
    DIRTY_KEYS.lock().unwrap().clear();
}

fn analysis_supersedes(cached: &cache::BookEntry, book: &cache::BookEntry) -> bool {
    let (Some(new1), Some(old1)) = (cached.moves.first(), book.moves.first()) else {
        return false;
    };
    new1.move_repr == old1.move_repr
        && new1.eval_version == old1.eval_version
        && new1.depth >= old1.depth
        && cached.moves.len() >= book.moves.len()
        && (new1.depth > old1.depth || cached.moves.len() > book.moves.len())
}

#[pyfunction]
fn load_analysis_from_db() -> usize {
    let cached = cache::load_store(cache::Store::Analysis);
    let mut book = BOOK.lock().unwrap();
    let mut added = 0usize;
    let mut deepened = 0usize;
    for (hash, entry) in cached {
        match book.get(&hash) {
            None => {
                book.insert(hash, entry);
                added += 1;
            }
            Some(existing) if analysis_supersedes(&entry, existing) => {
                book.insert(hash, entry);
                deepened += 1;
            }
            Some(_) => {}
        }
    }
    if deepened > 0 {
        eprintln!(
            "[ANALYSIS] {} book entr{} replaced by a cached search of the same move with \
             more ranks or more depth",
            deepened,
            if deepened == 1 { "y" } else { "ies" }
        );
    }
    DIRTY_KEYS.lock().unwrap().clear();
    added + deepened
}

#[pyfunction]
fn save_analysis_to_db(py: Python<'_>) -> PyResult<usize> {
    py.allow_threads(|| {
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
    })
}

#[pyfunction]
fn rebuild_analysis() -> PyResult<()> {
    cache::rebuild_analysis().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn book_has_row(gs: &PyGameState) -> bool {
    BOOK_HASHES.lock().unwrap().contains(&gs.inner.hash.to_string())
}

#[pyfunction]
#[pyo3(signature = (_cache_arg=None))]
fn save_move_cache_to_db(py: Python<'_>, _cache_arg: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    py.allow_threads(|| {
        let book = BOOK.lock().unwrap();
        let mut dirty = DIRTY_KEYS.lock().unwrap();
        if dirty.is_empty() {
            return Ok(());
        }
        cache::save_book_entries(&book, &dirty)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        dirty.clear();
        Ok(())
    })
}

#[pyfunction]
fn save_book_position(gs: &PyGameState) -> PyResult<bool> {
    let hash = gs.inner.hash.to_string();
    let mut book = BOOK.lock().unwrap();
    let Some(entry) = book.get_mut(&hash) else {
        return Ok(false);
    };

    if entry.fen.is_none() {
        entry.fen = Some(fen::to_fen(&gs.inner));
    }
    if entry.ply.is_none() {
        entry.ply = Some(gs.inner.ply as i32);
    }

    let book = &*book;
    cache::save_book_entries(book, std::slice::from_ref(&hash))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    DIRTY_KEYS.lock().unwrap().retain(|k| k != &hash);
    BOOK_HASHES.lock().unwrap().insert(hash);
    Ok(true)
}

#[pyfunction]
fn book_has_position(gs: &PyGameState) -> bool {
    BOOK.lock().unwrap().contains_key(&gs.inner.hash.to_string())
}

#[pyfunction]
fn pending_book_writes() -> usize {
    let dirty = DIRTY_KEYS.lock().unwrap();
    let unique: std::collections::HashSet<&str> = dirty.iter().map(|s| s.as_str()).collect();
    unique.len()
}

#[pyfunction]
fn discard_pending_book_writes() -> usize {
    let mut dirty = DIRTY_KEYS.lock().unwrap();
    let n = dirty.len();
    dirty.clear();
    n
}

#[pyfunction]
fn setup_db() -> PyResult<()> {
    cache::setup_db().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn rebuild_book() -> PyResult<()> {
    cache::rebuild_db().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn book_size() -> usize {
    BOOK.lock().unwrap().len()
}

#[pyfunction]
fn to_fen(gs: &PyGameState) -> String {
    fen::to_fen(&gs.inner)
}

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
fn set_search_knobs(kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
    let mut k = search::knobs();
    for (key, val) in kwargs.iter() {
        let name: String = key.extract()?;
        match name.as_str() {
            "null_move" => k.null_move = val.extract()?,
            "lmr" => k.lmr = val.extract()?,
            "use_tt" => k.use_tt = val.extract()?,
            "use_book" => k.use_book = val.extract()?,
            "delta_margin" => k.delta_margin = val.extract()?,
            "order_seed" => k.order_seed = val.extract()?,
            other => return Err(PyValueError::new_err(format!("unknown search knob {:?}", other))),
        }
    }
    search::set_knobs(k);
    Ok(())
}

#[pyfunction]
fn reset_search_knobs() {
    search::set_knobs(search::DEFAULT_KNOBS);
}

#[pyfunction]
fn get_search_knobs(py: Python<'_>) -> PyResult<PyObject> {
    let k = search::knobs();
    let d = PyDict::new(py);
    d.set_item("null_move", k.null_move)?;
    d.set_item("lmr", k.lmr)?;
    d.set_item("use_tt", k.use_tt)?;
    d.set_item("use_book", k.use_book)?;
    d.set_item("delta_margin", k.delta_margin)?;
    d.set_item("order_seed", k.order_seed)?;
    Ok(d.into())
}

#[pyfunction]
fn last_search_nodes(py: Python<'_>) -> PyResult<PyObject> {
    let (nodes, qnodes) = search::last_search_nodes();
    let tup = PyTuple::new(py, &[
        nodes.into_pyobject(py)?.into_any().unbind(),
        qnodes.into_pyobject(py)?.into_any().unbind(),
    ])?;
    Ok(tup.into())
}

#[pyfunction]
fn parse_move_string(py: Python<'_>, s: &str) -> PyResult<PyObject> {
    match search::parse_move_repr(s) {
        Some(m) => Ok(rust_move_to_py(py, m)),
        None => Ok(py.None()),
    }
}

#[pyfunction]
fn encode_position(gs: &PyGameState) -> Vec<f32> {
    encode::encode_position(&gs.inner)
}

#[pyfunction]
fn move_to_action_index(py: Python<'_>, gs: &PyGameState, m: &Bound<'_, PyAny>) -> PyResult<usize> {
    let mv = py_move_to_rust(py, m)?;
    encode::move_to_index(mv, gs.inner.current_turn)
        .ok_or_else(|| PyValueError::new_err("move has no action index"))
}

#[pyfunction]
fn action_index_to_move(py: Python<'_>, gs: &PyGameState, index: usize) -> PyObject {
    match encode::index_to_move(index, gs.inner.current_turn) {
        Some(m) => rust_move_to_py(py, m),
        None => py.None(),
    }
}

#[pyfunction]
fn legal_action_indices(gs: &mut PyGameState) -> Vec<usize> {
    encode::legal_action_indices(&mut gs.inner)
}

/// A `&[f32]` seen as the bytes behind it, for handing a batch to numpy in one copy.
/// Sound for any `f32`: every bit pattern is a valid `u8` and `u8`'s alignment is 1.
fn f32_as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// float32 out of anything that will give it: a contiguous buffer in one copy, else the
/// sequence protocol one element at a time.
fn extract_f32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(buf) = PyBuffer::<f32>::get(obj) {
        if let Ok(v) = buf.to_vec(obj.py()) {
            return Ok(v);
        }
    }
    obj.extract::<Vec<f32>>()
}

#[pyclass(name = "Mcts", unsendable)]
struct PyMcts {
    inner: mcts::Mcts,
}

#[pymethods]
impl PyMcts {
    /// A search rooted at `gs`. The position is copied, so the caller may keep playing.
    #[new]
    #[pyo3(signature = (gs, c_puct=None, fpu=None))]
    fn new(gs: &PyGameState, c_puct: Option<f32>, fpu: Option<f32>) -> Self {
        let mut cfg = mcts::Config::default();
        if let Some(c) = c_puct { cfg.c_puct = c; }
        if let Some(f) = fpu { cfg.fpu = f; }
        PyMcts { inner: mcts::Mcts::new(&gs.inner, cfg) }
    }

    /// Descend until `max_leaves` positions need the network, resolving terminals in
    /// place. Returns their planes as **raw little-endian f32 bytes** --
    /// `4 * ENCODE_INPUT_SIZE` per leaf, `max_leaves` leaves at most -- for
    /// `np.frombuffer(..., dtype=np.float32)`. Empty means the tree could not produce
    /// new work and the search is finished.
    ///
    /// Bytes rather than a list because this is the hot path: a list of 110k floats is
    /// 110k boxed, refcounted `PyObject`s built here and walked again by numpy, which
    /// measured as most of a self-play ply against ~3ms of GPU (docs/ZERO.md). A
    /// bytearray is one allocation and one memcpy, and numpy wraps it without copying.
    /// Mutable rather than `bytes` so `torch.from_numpy` does not warn about a
    /// read-only buffer on every batch.
    fn collect<'py>(&mut self, py: Python<'py>, max_leaves: usize) -> Bound<'py, PyByteArray> {
        let planes = py.allow_threads(|| self.inner.collect(max_leaves));
        PyByteArray::new(py, f32_as_bytes(&planes))
    }

    /// Answer the last `collect`. `priors` is one masked, normalised row of ACTION_SPACE
    /// per leaf; `values` one scalar per leaf, in that leaf's own frame.
    ///
    /// Both come in through the buffer protocol when the caller hands over a contiguous
    /// float32 array -- one memcpy for the whole batch -- and fall back to element-wise
    /// extraction for a plain Python sequence, which is what the tests use.
    fn expand(&mut self, py: Python<'_>, priors: &Bound<'_, PyAny>,
              values: &Bound<'_, PyAny>) -> PyResult<()> {
        let priors = extract_f32(priors)?;
        let values = extract_f32(values)?;
        py.allow_threads(|| self.inner.expand(&priors, &values))
            .map_err(PyValueError::new_err)
    }

    /// Re-root onto `gs` if the position is at most `max_depth` plies below this tree's
    /// root, keeping the statistics already gathered there; returns the number of plies
    /// skipped, or None when the tree cannot reach it and the caller needs a new one.
    ///
    /// Matching is by position, not by move: the caller does not have to report what was
    /// played, and two plies of lookahead cover an opponent's reply as well as our own
    /// move. `simulations` restarts at 0 -- it counts what this search does, and
    /// `root_total_visits` is what it inherited.
    #[pyo3(signature = (gs, max_depth=2))]
    fn advance_to(&mut self, gs: &PyGameState, max_depth: usize) -> Option<usize> {
        self.inner.advance_to(&gs.inner, max_depth)
    }

    /// The root's priors in `root_visits()` order, and their replacement. Self-play mixes
    /// Dirichlet noise here rather than into the evaluator's answer, because a reused
    /// tree arrives with its root already expanded.
    fn root_priors(&self) -> Vec<f32> { self.inner.root_priors() }

    fn set_root_priors(&mut self, priors: &Bound<'_, PyAny>) -> PyResult<()> {
        let p = extract_f32(priors)?;
        self.inner.set_root_priors(&p).map_err(PyValueError::new_err)
    }

    #[getter]
    fn root_expanded(&self) -> bool { self.inner.root_expanded() }

    /// Visits standing at the root, banked ones included. `simulations` is what this
    /// search paid for; the difference is what tree reuse saved.
    #[getter]
    fn root_total_visits(&self) -> u32 { self.inner.root_total_visits() }

    #[getter]
    fn simulations(&self) -> usize { self.inner.simulations() }

    #[getter]
    fn pending(&self) -> usize { self.inner.pending_count() }

    #[getter]
    fn nodes(&self) -> usize { self.inner.node_count() }

    /// The root's value in the frame of the side to move there.
    #[getter]
    fn value(&self) -> f32 { self.inner.root_value() }

    fn best_move(&self, py: Python<'_>) -> PyObject {
        match self.inner.best_move() {
            Some(m) => rust_move_to_py(py, m),
            None => py.None(),
        }
    }

    /// [(move, visits)] for every root move -- the visit distribution a policy target is
    /// built from, and what the caller samples for temperature play.
    fn root_visits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<PyObject> = self
            .inner
            .root_moves()
            .into_iter()
            .map(|(m, v)| {
                let mv = rust_move_to_py(py, m);
                PyTuple::new(py, &[mv, v.into_pyobject(py).unwrap().into_any().unbind()])
                    .unwrap()
                    .into()
            })
            .collect();
        PyList::new(py, items)
    }

    /// [(action_index, visits, mean_value)] at the root, most-visited first.
    fn root_stats(&self) -> Vec<(u32, u32, f32)> { self.inner.root_stats() }
}

#[pymodule]
fn minichess_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_class::<PyMcts>()?;
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
    m.add_function(wrap_pyfunction!(set_search_knobs, m)?)?;
    m.add_function(wrap_pyfunction!(reset_search_knobs, m)?)?;
    m.add_function(wrap_pyfunction!(get_search_knobs, m)?)?;
    m.add_function(wrap_pyfunction!(last_search_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(encode_position, m)?)?;
    m.add_function(wrap_pyfunction!(move_to_action_index, m)?)?;
    m.add_function(wrap_pyfunction!(action_index_to_move, m)?)?;
    m.add_function(wrap_pyfunction!(legal_action_indices, m)?)?;
    
    m.add("CHECKMATE_SCORE", eval::CHECKMATE_SCORE)?;
    m.add("STALEMATE_SCORE", eval::STALEMATE_SCORE)?;
    m.add("BOARD_SIZE", BOARD_SIZE)?;
    m.add("EVAL_VERSION", eval::EVAL_VERSION)?;
    m.add("SCHEMA_VERSION", cache::SCHEMA_VERSION)?;
    m.add("ENCODE_PLANES", encode::PLANES)?;
    m.add("ENCODE_INPUT_SIZE", encode::INPUT_SIZE)?;
    m.add("ACTION_PLANES", encode::ACTION_PLANES)?;
    m.add("ACTION_SPACE", encode::ACTION_SPACE)?;
    
    Ok(())
}
