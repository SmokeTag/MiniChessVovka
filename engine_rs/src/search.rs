use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::types::*;
use crate::gamestate::GameState;
use crate::eval::{evaluate_position, CHECKMATE_SCORE, STALEMATE_SCORE};
use crate::zobrist;

const MAX_QUIESCENCE_DEPTH: i32 = 4;

/// Deepest entry the persistent move cache is known to hold, which is where `probe_move_cache`
/// starts its walk. Set from the DB when it is loaded and raised by every insert, so a run at
/// depth 12 or 16 widens the probe by itself — no constant to keep in sync. 0 means nothing is
/// cached yet, which makes the probe a no-op.
static MAX_CACHED_DEPTH: AtomicI32 = AtomicI32::new(0);

/// Record that the cache now holds an entry at `depth`. Callers that fill `move_cache` from
/// outside a search (the DB load) must call this, or the probe will not look that deep.
pub fn note_cached_depth(depth: i32) {
    MAX_CACHED_DEPTH.fetch_max(depth, Ordering::Relaxed);
}

// Root-level parallel search is a runtime knob, not a compile-time constant.
// Default is OFF: self-play training runs many independent games side by side across
// cores, which scales better than parallelising a single game. Interactive analysis
// turns it on to throw every core at one position.
const DEFAULT_PARALLEL_MIN_DEPTH: i32 = 3;
static PARALLEL_ENABLED: AtomicBool = AtomicBool::new(false);
static PARALLEL_MIN_DEPTH: AtomicI32 = AtomicI32::new(DEFAULT_PARALLEL_MIN_DEPTH);

/// Enable or disable the root-level parallel search process-wide.
/// `min_depth` is the first iterative-deepening iteration that gets split across
/// threads; shallower iterations stay sequential because the fan-out costs more
/// than it saves. `None` keeps the current minimum depth.
pub fn set_parallel_search(enabled: bool, min_depth: Option<i32>) {
    if let Some(d) = min_depth {
        PARALLEL_MIN_DEPTH.store(d.max(1), Ordering::Relaxed);
    }
    PARALLEL_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Current `(enabled, min_depth)` of the root-level parallel search.
pub fn parallel_search_config() -> (bool, i32) {
    (
        PARALLEL_ENABLED.load(Ordering::Relaxed),
        PARALLEL_MIN_DEPTH.load(Ordering::Relaxed),
    )
}

// TT entry
#[derive(Clone)]
pub struct TTEntry {
    pub depth: i32,
    pub score: i32,
    pub flag: TTFlag,
    pub best_move: Move,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

pub struct SearchState {
    pub tt: HashMap<u64, TTEntry>,
    /// Read-only transposition entries shared with the root workers. Probed after the
    /// local table misses; never written to, so no locking is needed.
    pub base_tt: Option<Arc<HashMap<u64, TTEntry>>>,
    pub killer_moves: HashMap<i32, [Move; 2]>,
    pub history_scores: HashMap<u32, i32>, // move.data -> score
    pub deadline: Option<Instant>,
    pub stopped: bool,
    nodes_since_check: u32,
}

impl SearchState {
    pub fn new() -> Self {
        SearchState::with_tt_capacity(1 << 20)
    }

    /// Root workers are short-lived and there can be one per core, so they ask for a
    /// small table instead of pre-allocating the full-size one the root search uses.
    pub fn with_tt_capacity(cap: usize) -> Self {
        SearchState {
            tt: HashMap::with_capacity(cap),
            base_tt: None,
            killer_moves: HashMap::new(),
            history_scores: HashMap::new(),
            deadline: None,
            stopped: false,
            nodes_since_check: 0,
        }
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.base_tt = None;
        self.killer_moves.clear();
        self.history_scores.clear();
    }

    /// Probe the local table, falling back to the shared read-only snapshot.
    #[inline]
    pub fn tt_get(&self, hash: u64) -> Option<TTEntry> {
        if let Some(e) = self.tt.get(&hash) {
            return Some(e.clone());
        }
        self.base_tt.as_ref().and_then(|b| b.get(&hash).cloned())
    }

    /// Check deadline every 4096 nodes to avoid syscall overhead
    #[inline]
    pub fn check_deadline(&mut self) -> bool {
        if self.stopped { return true; }
        self.nodes_since_check += 1;
        if self.nodes_since_check >= 4096 {
            self.nodes_since_check = 0;
            if let Some(dl) = self.deadline {
                if Instant::now() >= dl {
                    self.stopped = true;
                    return true;
                }
            }
        }
        false
    }
}

// Move ordering score
fn mvv_lva_score(gs: &GameState, m: Move, ss: &SearchState) -> i32 {
    let mut score = 0i32;

    if !m.is_drop() {
        let from = m.from_sq();
        let to = m.to_sq();
        let aggressor = gs.board[from];
        let victim = gs.board[to];

        if aggressor.is_empty() {
            return 0;
        }
        let aggressor_value = PIECE_VALUES[aggressor.piece_type().unwrap().index()];

        // Captures
        if !victim.is_empty() {
            let mut victim_value = PIECE_VALUES[victim.piece_type().unwrap().index()];
            if gs.promoted_pieces & (1u64 << to) != 0 {
                victim_value = (victim_value + PIECE_VALUES[PieceType::Pawn.index()]) / 2;
            }
            score += victim_value * 10 - aggressor_value;
        }

        // Promotion
        if m.promotion().is_some() {
            score += 900;
        }

        // King zone attack
        let piece_color = aggressor.color().unwrap();
        let enemy_king_sq = gs.king_pos[piece_color.opposite().index()];
        let ek_r = sq_row(enemy_king_sq) as i32;
        let ek_c = sq_file(enemy_king_sq) as i32;
        let to_r = sq_row(to) as i32;
        let to_c = sq_file(to) as i32;
        let dist = (to_r - ek_r).abs().max((to_c - ek_c).abs());
        if dist <= 2 {
            score += 500;
        }

        // Center
        if is_center_sq(to) {
            score += 30;
        }

        // Development
        if piece_color == Color::White && sq_row(from) == 5 && sq_row(to) < 5 {
            score += 20;
        } else if piece_color == Color::Black && sq_row(from) == 0 && sq_row(to) > 0 {
            score += 20;
        }

        // History heuristic
        score += *ss.history_scores.get(&m.data).unwrap_or(&0);

    } else {
        // Drop move
        let to = m.to_sq();
        let pt = m.drop_piece_type();
        let color = m.drop_color();

        let mut base = HAND_PIECE_VALUES[pt.index()] / 10;

        // Pawn drop near promotion
        if pt == PieceType::Pawn {
            let to_r = sq_row(to);
            if color == Color::White && to_r == 1 { base += 800; }
            else if color == Color::White && to_r == 2 { base += 200; }
            else if color == Color::Black && to_r == BOARD_SIZE - 2 { base += 800; }
            else if color == Color::Black && to_r == BOARD_SIZE - 3 { base += 200; }
        }

        if is_center_sq(to) {
            base += 50;
        }

        // Near enemy king
        let enemy_king_sq = gs.king_pos[color.opposite().index()];
        let ek_r = sq_row(enemy_king_sq) as i32;
        let ek_c = sq_file(enemy_king_sq) as i32;
        let to_r = sq_row(to) as i32;
        let to_c = sq_file(to) as i32;
        let dist = (to_r - ek_r).abs().max((to_c - ek_c).abs());
        if dist <= 1 { base += 200; }
        else if dist <= 2 { base += 100; }

        // Knight fork detection
        if pt == PieceType::Knight {
            let mut attacks = 0;
            let enemy_color = color.opposite();
            for &(dr, df) in &KNIGHT_OFFSETS {
                let nr = to_r + dr;
                let nf = to_c + df;
                if is_on_board(nr, nf) {
                    let target = gs.board[sq(nr as usize, nf as usize)];
                    if !target.is_empty() && target.color() == Some(enemy_color) {
                        attacks += 1;
                        if matches!(target.piece_type(), Some(PieceType::King) | Some(PieceType::Rook)) {
                            base += 300;
                        }
                    }
                }
            }
            if attacks >= 2 { base += 200; }
        }

        base += *ss.history_scores.get(&m.data).unwrap_or(&0);
        score = base;
    }

    score
}

fn is_center_sq(s: usize) -> bool {
    let r = sq_row(s);
    let c = sq_file(s);
    (r == 2 || r == 3) && (c == 2 || c == 3)
}

fn is_noisy_move(gs: &GameState, m: Move) -> bool {
    if m.is_drop() {
        return true;
    }
    let to = m.to_sq();
    if gs.board[to] != Piece::Empty {
        return true;
    }
    m.promotion().is_some()
}

fn is_check_move(gs: &mut GameState, m: Move) -> bool {
    gs.make_ai_move(m);
    let opponent = gs.current_turn;
    let in_check = gs.is_in_check(opponent);
    gs.undo_ai_move();
    in_check
}

fn is_drop_near_king(gs: &GameState, m: Move) -> bool {
    let to = m.to_sq();
    let color = m.drop_color();
    let enemy_king_sq = gs.king_pos[color.opposite().index()];
    let kr = sq_row(enemy_king_sq) as i32;
    let kf = sq_file(enemy_king_sq) as i32;
    let dr = sq_row(to) as i32;
    let df = sq_file(to) as i32;

    let pt = m.drop_piece_type();
    if pt == PieceType::Knight {
        if KNIGHT_OFFSETS.contains(&(kr - dr, kf - df)) {
            return true;
        }
    }
    (dr - kr).abs().max((df - kf).abs()) <= 2
}

fn get_noisy_moves(gs: &mut GameState) -> Vec<Move> {
    let legal = gs.get_legal_moves_vec();
    let mut noisy = Vec::new();
    let mut check_candidates = Vec::new();

    for m in &legal {
        if m.is_drop() {
            if is_drop_near_king(gs, *m) {
                noisy.push(*m);
            }
            continue;
        }
        let to = m.to_sq();
        let is_capture = gs.board[to] != Piece::Empty;
        let is_promotion = m.promotion().is_some();
        if is_capture || is_promotion {
            noisy.push(*m);
        } else {
            check_candidates.push(*m);
        }
    }

    // Also include quiet moves that give check (limit to 12)
    let limit = check_candidates.len().min(12);
    for &m in &check_candidates[..limit] {
        if is_check_move(gs, m) {
            noisy.push(m);
        }
    }

    noisy
}

// Quiescence search
fn quiescence_search(gs: &mut GameState, mut alpha: i32, mut beta: i32, maximizing: bool, depth: i32) -> i32 {
    let stand_pat = evaluate_position(gs);

    let legal = gs.get_legal_moves_vec();
    if legal.is_empty() {
        if gs.is_in_check(gs.current_turn) {
            return if gs.current_turn == Color::White { -CHECKMATE_SCORE } else { CHECKMATE_SCORE };
        }
        return STALEMATE_SCORE;
    }

    if depth == 0 {
        return stand_pat;
    }

    let delta_margin = PIECE_VALUES[PieceType::Rook.index()];
    if maximizing && stand_pat < alpha.saturating_sub(delta_margin) {
        return alpha;
    }
    if !maximizing && stand_pat > beta.saturating_add(delta_margin) {
        return beta;
    }

    if maximizing {
        if stand_pat >= beta { return beta; }
        alpha = alpha.max(stand_pat);

        let noisy = get_noisy_moves(gs);
        if noisy.is_empty() { return stand_pat; }

        for m in &noisy {
            gs.make_ai_move(*m);
            let score = quiescence_search(gs, alpha, beta, false, depth - 1);
            gs.undo_ai_move();

            if score >= CHECKMATE_SCORE { return CHECKMATE_SCORE; }
            alpha = alpha.max(score);
            if alpha >= beta { break; }
        }
        alpha
    } else {
        if stand_pat <= alpha { return alpha; }
        beta = beta.min(stand_pat);

        let noisy = get_noisy_moves(gs);
        if noisy.is_empty() { return stand_pat; }

        for m in &noisy {
            gs.make_ai_move(*m);
            let score = quiescence_search(gs, alpha, beta, true, depth - 1);
            gs.undo_ai_move();

            if score <= -CHECKMATE_SCORE { return -CHECKMATE_SCORE; }
            beta = beta.min(score);
            if beta <= alpha { break; }
        }
        beta
    }
}

// Minimax with alpha-beta, PVS, LMR, null-move, check extensions, TT
fn minimax_ab(
    gs: &mut GameState,
    mut depth: i32,
    mut alpha: i32,
    mut beta: i32,
    maximizing: bool,
    allow_null: bool,
    ss: &mut SearchState,
) -> (i32, Move) {
    // Abort search if time limit reached
    if ss.check_deadline() {
        return (evaluate_position(gs), Move::NULL);
    }

    let current_color = if maximizing { Color::White } else { Color::Black };
    let in_check = gs.is_in_check(current_color);
    if in_check && depth > 0 && depth < 3 {
        depth += 1;
    }

    if depth <= 0 || gs.checkmate || gs.stalemate {
        let q = quiescence_search(gs, alpha, beta, maximizing, MAX_QUIESCENCE_DEPTH);
        return (q, Move::NULL);
    }

    let pos_hash = gs.hash;
    let tt_entry = ss.tt_get(pos_hash);

    let mut legal_moves = gs.get_legal_moves_vec();
    if legal_moves.is_empty() {
        if gs.is_in_check(gs.current_turn) {
            return (if gs.current_turn == Color::White { -CHECKMATE_SCORE } else { CHECKMATE_SCORE }, Move::NULL);
        }
        return (STALEMATE_SCORE, Move::NULL);
    }

    // TT probe
    if let Some(ref entry) = tt_entry {
        if entry.depth >= depth {
            match entry.flag {
                TTFlag::Exact => return (entry.score, entry.best_move),
                TTFlag::LowerBound => alpha = alpha.max(entry.score),
                TTFlag::UpperBound => beta = beta.min(entry.score),
            }
            if alpha >= beta {
                return (entry.score, entry.best_move);
            }
        }
    }

    // Null-move pruning
    let null_move_r = 2;
    let opponent_color = current_color.opposite();
    let opponent_hand: u8 = gs.hands[opponent_color.index()].iter().sum();
    if allow_null && depth >= null_move_r + 1 && !in_check && opponent_hand == 0 {
        gs.current_turn = gs.current_turn.opposite();
        gs.hash = gs.compute_hash();
        gs.invalidate_cache();
        let (null_score, _) = minimax_ab(gs, depth - 1 - null_move_r, alpha, beta, !maximizing, false, ss);
        gs.current_turn = gs.current_turn.opposite();
        gs.hash = gs.compute_hash();
        gs.invalidate_cache();

        if maximizing && null_score >= beta { return (beta, Move::NULL); }
        if !maximizing && null_score <= alpha { return (alpha, Move::NULL); }
    }

    // Move ordering
    let tt_best = tt_entry.as_ref().map(|e| e.best_move).unwrap_or(Move::NULL);
    let killers = ss.killer_moves.get(&depth).cloned().unwrap_or([Move::NULL; 2]);

    let mut scored: Vec<(Move, i32)> = legal_moves
        .iter()
        .map(|&m| {
            let s = if !tt_best.is_null() && m == tt_best {
                1_000_000
            } else if m == killers[0] || m == killers[1] {
                50_000
            } else {
                mvv_lva_score(gs, m, ss)
            };
            (m, s)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    legal_moves = scored.into_iter().map(|(m, _)| m).collect();

    let orig_alpha = alpha;
    let orig_beta = beta;
    let mut best_move = Move::NULL;

    let lmr_full_depth = 4;
    let lmr_reduction_limit = 3;

    if maximizing {
        let mut max_eval = i32::MIN;
        for (i, &m) in legal_moves.iter().enumerate() {
            let noisy = is_noisy_move(gs, m);
            gs.make_ai_move(m);
            let gives_check = gs.is_in_check(gs.current_turn);

            let eval_score;
            if i == 0 {
                let (s, _) = minimax_ab(gs, depth - 1, alpha, beta, false, true, ss);
                eval_score = s;
            } else {
                let reduced = i >= lmr_full_depth && depth >= lmr_reduction_limit && !noisy && !gives_check;
                let (s, _) = if reduced {
                    minimax_ab(gs, depth - 2, alpha, alpha + 1, false, true, ss)
                } else {
                    minimax_ab(gs, depth - 1, alpha, alpha + 1, false, true, ss)
                };
                eval_score = if alpha < s && s < beta {
                    let (re, _) = minimax_ab(gs, depth - 1, alpha, beta, false, true, ss);
                    re
                } else {
                    s
                };
            }

            gs.undo_ai_move();

            if eval_score > max_eval {
                max_eval = eval_score;
                best_move = m;
            }
            alpha = alpha.max(eval_score);
            if alpha >= beta {
                // Update killer moves & history
                let h = ss.history_scores.entry(m.data).or_insert(0);
                *h += depth * depth;
                let killers = ss.killer_moves.entry(depth).or_insert([Move::NULL; 2]);
                if m != killers[0] {
                    killers[1] = killers[0];
                    killers[0] = m;
                }
                break;
            }
        }

        let flag = if max_eval <= orig_alpha {
            TTFlag::UpperBound
        } else if max_eval >= orig_beta {
            TTFlag::LowerBound
        } else {
            TTFlag::Exact
        };
        ss.tt.insert(pos_hash, TTEntry { depth, score: max_eval, flag, best_move });
        (max_eval, best_move)
    } else {
        let mut min_eval = i32::MAX;
        for (i, &m) in legal_moves.iter().enumerate() {
            let noisy = is_noisy_move(gs, m);
            gs.make_ai_move(m);
            let gives_check = gs.is_in_check(gs.current_turn);

            let eval_score;
            if i == 0 {
                let (s, _) = minimax_ab(gs, depth - 1, alpha, beta, true, true, ss);
                eval_score = s;
            } else {
                let reduced = i >= lmr_full_depth && depth >= lmr_reduction_limit && !noisy && !gives_check;
                let (s, _) = if reduced {
                    minimax_ab(gs, depth - 2, beta - 1, beta, true, true, ss)
                } else {
                    minimax_ab(gs, depth - 1, beta - 1, beta, true, true, ss)
                };
                eval_score = if alpha < s && s < beta {
                    let (re, _) = minimax_ab(gs, depth - 1, alpha, beta, true, true, ss);
                    re
                } else {
                    s
                };
            }

            gs.undo_ai_move();

            if eval_score < min_eval {
                min_eval = eval_score;
                best_move = m;
            }
            beta = beta.min(eval_score);
            if beta <= alpha {
                let h = ss.history_scores.entry(m.data).or_insert(0);
                *h += depth * depth;
                let killers = ss.killer_moves.entry(depth).or_insert([Move::NULL; 2]);
                if m != killers[0] {
                    killers[1] = killers[0];
                    killers[0] = m;
                }
                break;
            }
        }

        let flag = if min_eval >= orig_beta {
            TTFlag::LowerBound
        } else if min_eval <= orig_alpha {
            TTFlag::UpperBound
        } else {
            TTFlag::Exact
        };
        ss.tt.insert(pos_hash, TTEntry { depth, score: min_eval, flag, best_move });
        (min_eval, best_move)
    }
}

// Read-only transposition snapshot shared by every root worker. Sharing it behind an
// Arc instead of cloning the map per worker keeps the fan-out cheap.
type SharedTT = Arc<HashMap<u64, TTEntry>>;

/// Move-ordering heuristics handed down from the root search to every worker.
struct RootHeuristics {
    killer_moves: HashMap<i32, [Move; 2]>,
    history_scores: HashMap<u32, i32>,
}

/// Searches one root move in its own SearchState, on its own copy of the board.
/// Returns `(move, score, aborted)`. When `aborted` is true the score comes from
/// truncated subtrees and must not be trusted.
///
/// The worker's own table dies with the worker, and that is deliberate. Ordinary bound
/// entries would be safe to share, but this engine's null-move pruning returns a hard
/// `beta` (see the null-move cutoff below), and the store site classifies flags by
/// comparing the result against the *original* alpha/beta -- so under the razor-thin
/// window a scout is given, a window-clamped value can be written with flag `Exact`.
/// Fed back to the root, a later full-window probe returns that value verbatim and the
/// search answers differently than the sequential one does. Measured: opening_start at
/// depth 7 returned d1c2 scored 24 where the sequential search returns c1e2 scored 74.
fn search_worker(
    gs: &GameState,
    m: Move,
    depth: i32,
    alpha: i32,
    beta: i32,
    maximizing: bool,
    base_tt: &SharedTT,
    heuristics: &RootHeuristics,
    deadline: Option<Instant>,
) -> (Move, i32, bool) {
    let mut gs_copy = gs.fast_copy();
    let mut ss = SearchState::with_tt_capacity(1 << 14);
    ss.base_tt = Some(Arc::clone(base_tt));
    // Seed the ordering heuristics from the root so workers order moves the way the
    // sequential search would; starting them empty makes the two paths drift.
    ss.killer_moves = heuristics.killer_moves.clone();
    ss.history_scores = heuristics.history_scores.clone();
    ss.deadline = deadline;

    gs_copy.make_ai_move(m);
    // Workers run the very same routine as the sequential search, so the two paths
    // cannot drift apart in what they score or how they prune.
    let (score, _) = minimax_ab(&mut gs_copy, depth - 1, alpha, beta, !maximizing, true, &mut ss);

    // An aborted worker returns a score built on cut-off subtrees.
    (m, score, ss.stopped)
}

// Parallel minimax at root level.
//
// Same PVS shape minimax_ab uses at every other node, spread across cores: order the
// root moves, search the best one sequentially for a real alpha, scout all the others
// in parallel against a null window, and re-search at full width only those that beat
// it. The returned score is therefore a full-window score of the returned move.
fn minimax_parallel(
    gs: &mut GameState,
    depth: i32,
    ss: &mut SearchState,
) -> (Move, i32, Vec<(Move, i32)>) {
    let mut legal_moves = gs.get_legal_moves_vec();
    if legal_moves.is_empty() {
        return (Move::NULL, 0, vec![]);
    }

    let maximizing = gs.current_turn == Color::White;
    let deadline = ss.deadline;
    let neg_inf = i32::MIN + 1;
    let inf = i32::MAX - 1;

    // Root move ordering, same keys as minimax_ab: TT move, then killers, then mvv/lva.
    // Without this the baseline move is arbitrary and every scout fails high.
    let tt_best = ss.tt_get(gs.hash).map(|e| e.best_move).unwrap_or(Move::NULL);
    let killers = ss.killer_moves.get(&depth).cloned().unwrap_or([Move::NULL; 2]);
    let mut scored: Vec<(Move, i32)> = legal_moves
        .iter()
        .map(|&m| {
            let s = if !tt_best.is_null() && m == tt_best {
                1_000_000
            } else if m == killers[0] || m == killers[1] {
                50_000
            } else {
                mvv_lva_score(gs, m, ss)
            };
            (m, s)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    legal_moves = scored.into_iter().map(|(m, _)| m).collect();

    // Baseline: full-window search of the best-ordered move, in the root's own state so
    // it sees the whole transposition table — byte for byte what the sequential root does.
    let first_move = legal_moves[0];
    let baseline_start = Instant::now();
    gs.make_ai_move(first_move);
    let (baseline, _) = minimax_ab(gs, depth - 1, neg_inf, inf, !maximizing, true, ss);
    gs.undo_ai_move();
    let baseline_secs = baseline_start.elapsed().as_secs_f64();
    if ss.stopped {
        // Without a trustworthy baseline every scout window below is meaningless.
        return (Move::NULL, 0, vec![]);
    }

    let mut best_move = first_move;
    let mut best_score = baseline;
    let mut all_results: Vec<(Move, i32)> = vec![(first_move, baseline)];

    // Snapshots for the workers, taken after the baseline so they inherit its work.
    let min_useful = (depth - 3).max(1);
    let base_tt: SharedTT = Arc::new(
        ss.tt.iter()
            .filter(|(_, e)| e.depth >= min_useful)
            .map(|(k, v)| (*k, v.clone()))
            .collect(),
    );
    let heuristics = RootHeuristics {
        killer_moves: ss.killer_moves.clone(),
        history_scores: ss.history_scores.clone(),
    };

    // Scout every remaining move in parallel against a null window at the baseline.
    let remaining: Vec<Move> = legal_moves[1..].to_vec();
    let gs_snapshot = gs.fast_copy();
    let (scout_alpha, scout_beta) = if maximizing {
        (best_score, best_score + 1)
    } else {
        (best_score - 1, best_score)
    };

    let scout_start = Instant::now();
    let scouted: Vec<(Move, i32, bool)> = {
        use rayon::prelude::*;
        remaining.par_iter().map(|&m| {
            search_worker(&gs_snapshot, m, depth, scout_alpha, scout_beta, maximizing, &base_tt, &heuristics, deadline)
        }).collect()
    };
    let scout_secs = scout_start.elapsed().as_secs_f64();

    let mut candidates: Vec<(Move, i32)> = Vec::new();
    for (m, score, worker_stopped) in scouted {
        if worker_stopped {
            // Some root moves were never scored, so the best of the rest is not the best.
            ss.stopped = true;
            return (Move::NULL, 0, vec![]);
        }

        let beats_baseline = if maximizing { score > scout_alpha } else { score < scout_beta };
        if beats_baseline {
            candidates.push((m, score));
        } else {
            all_results.push((m, score));
        }
    }

    // Re-search the moves that beat the baseline at full width -- in parallel, for the
    // same reason the scout is parallel. Serially this loop raises alpha as it goes, which
    // prunes a little harder but makes the wall time the SUM of the re-searches; on the
    // bench that sum was routinely larger than the whole scout phase it followed (5 moves
    // costing 0.41s against 45 scouts costing 0.29s). Fixing alpha at the baseline instead
    // makes the wall time the MAX, costs only the pruning a climbing alpha would have
    // bought, and drops the last scheduling-dependent step: results are folded in root-move
    // order with a strict comparison, so the best-ordered move still wins any tie.
    if maximizing {
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    } else {
        candidates.sort_unstable_by(|a, b| a.1.cmp(&b.1));
    }
    let n_candidates = candidates.len();
    let research_start = Instant::now();
    let (window_alpha, window_beta) = if maximizing { (best_score, inf) } else { (neg_inf, best_score) };
    let researched: Vec<(Move, i32, bool)> = if n_candidates > 1 {
        use rayon::prelude::*;
        candidates
            .par_iter()
            .map(|&(m, _)| {
                search_worker(&gs_snapshot, m, depth, window_alpha, window_beta, maximizing, &base_tt, &heuristics, deadline)
            })
            .collect()
    } else {
        // A single candidate has nothing to run beside it, so keep it in the root's state
        // where it sees the whole transposition table rather than the filtered snapshot.
        candidates
            .iter()
            .map(|&(m, _)| {
                gs.make_ai_move(m);
                let (score, _) = minimax_ab(gs, depth - 1, window_alpha, window_beta, !maximizing, true, ss);
                gs.undo_ai_move();
                (m, score, ss.stopped)
            })
            .collect()
    };

    for (m, score, stopped) in researched {
        if stopped {
            return (Move::NULL, 0, vec![]);
        }
        all_results.push((m, score));
        if (maximizing && score > best_score) || (!maximizing && score < best_score) {
            best_score = score;
            best_move = m;
        }
    }

    eprintln!(
        "  [PARALLEL] depth {}: baseline {:.2}s, scout {:.2}s ({} moves), re-search {:.2}s ({} moves)",
        depth, baseline_secs, scout_secs, remaining.len(), research_start.elapsed().as_secs_f64(), n_candidates
    );

    // Check for mate
    let mating_score = if maximizing { CHECKMATE_SCORE } else { -CHECKMATE_SCORE };
    for &(m, s) in &all_results {
        if s == mating_score {
            return (m, s, all_results);
        }
    }

    (best_move, best_score, all_results)
}

/// The one way to write the persistent cache. An entry that misses `dirty` never reaches the
/// DB; one that misses `note_cached_depth` is invisible to `probe_move_cache`. Going through
/// here keeps both true.
fn cache_store(
    move_cache: &mut HashMap<(String, i32), String>,
    dirty: &mut Vec<(String, i32)>,
    pos_hash_str: &str,
    depth: i32,
    m: Move,
) {
    let key = (pos_hash_str.to_string(), depth);
    move_cache.insert(key.clone(), format_move_repr(m));
    dirty.push(key);
    note_cached_depth(depth);
}

/// Deepest usable persistent-cache entry for `pos_hash_str`, at or above `min_depth`.
///
/// The cache is keyed by `(hash, depth)`, so finding "any entry at least this deep" means
/// walking the depths rather than the map — walking the map is O(cache), which is six figures
/// per search once a worker has been running a while. The walk starts at `MAX_CACHED_DEPTH`,
/// so it costs nothing beyond the depths that actually exist. Deeper entries win: they came
/// from a search that saw strictly more. A row whose move is not legal in this position is a
/// hash collision (or a stale schema) and is skipped rather than trusted.
fn probe_move_cache(
    gs: &mut GameState,
    move_cache: &HashMap<(String, i32), String>,
    pos_hash_str: &str,
    min_depth: i32,
) -> Option<(Move, i32)> {
    let max_depth = MAX_CACHED_DEPTH.load(Ordering::Relaxed);
    if min_depth > max_depth {
        return None;
    }
    let mut legal: Option<Vec<Move>> = None;
    for d in (min_depth..=max_depth).rev() {
        let Some(cached_repr) = move_cache.get(&(pos_hash_str.to_string(), d)) else { continue };
        let Some(m) = parse_move_repr(cached_repr) else { continue };
        // Only pay for move generation once we actually have a candidate to validate.
        let legal = legal.get_or_insert_with(|| gs.get_legal_moves_vec());
        if legal.contains(&m) {
            return Some((m, d));
        }
    }
    None
}

/// Main entry: iterative deepening + cache
/// `dirty` collects every cache key this call inserted, so the caller can persist
/// just those instead of rewriting the whole cache.
pub fn find_best_move(
    gs: &mut GameState,
    depth: i32,
    move_cache: &mut HashMap<(String, i32), String>,
    dirty: &mut Vec<(String, i32)>,
    time_limit: Option<f64>,
    parallel: Option<bool>,
) -> (Move, i32) {
    let start = Instant::now();
    let maximizing = gs.current_turn == Color::White;
    let pos_hash_str = gs.hash.to_string();

    // Check cache. An entry searched deeper than we were asked for answers the same
    // question with more evidence behind it, so probe from the deepest stored depth
    // down to the requested one and take the first legal hit. Never below `depth`:
    // a shallower row is a weaker answer than the search we are about to run.
    if let Some((m, hit_depth)) = probe_move_cache(gs, move_cache, &pos_hash_str, depth) {
        eprintln!(
            "[CACHE HIT] depth {} (requested {}) in {:.2}s",
            hit_depth, depth, start.elapsed().as_secs_f64()
        );
        return (m, 0);
    }

    let legal = gs.get_legal_moves_vec();
    if legal.is_empty() {
        return (Move::NULL, evaluate_position(gs));
    }
    if legal.len() == 1 {
        let m = legal[0];
        cache_store(move_cache, dirty, &pos_hash_str, depth, m);
        return (m, 0);
    }

    let mut ss = SearchState::new();
    ss.deadline = time_limit.map(|secs| start + std::time::Duration::from_secs_f64(secs));
    let mut best_move = Move::NULL;
    let mut best_score = 0i32;

    // Per-call `parallel` wins over the process-wide setting; None means "use the setting".
    let (global_parallel, parallel_min_depth) = parallel_search_config();
    let use_parallel = parallel.unwrap_or(global_parallel);
    // i32::MAX keeps the parallel branch unreachable when parallel search is off.
    // Only the last iteration is worth splitting: the shallow ones cost more in fan-out
    // than they save, and nothing downstream consumes their result any more now that the
    // workers' tables are discarded instead of merged into the root table. Measured on the
    // bench suite, splitting every iteration from depth 3 up left depth 7 at 1.02x while
    // splitting only the last gave 2.07x on the same positions.
    let parallel_threshold = if use_parallel { parallel_min_depth.max(depth) } else { i32::MAX };
    if use_parallel {
        eprintln!("  [PARALLEL] root split enabled from depth {}", parallel_threshold);
    }

    // Splitting a cheap iteration loses: the fan-out clones a heuristic set and a table
    // snapshot per worker, which costs more than the search it replaces. Gate on what the
    // previous iteration actually cost rather than on core count or move count -- it is the
    // one predictor already measured on the machine the search is running on. (A core-count
    // guard was tried and rejected: it scales the wrong way, disabling the split entirely on
    // a 64-thread box.)
    const MIN_SPLIT_SECS: f64 = 0.03;
    let mut last_iter_secs = 0.0f64;

    for current_depth in 1..=depth {
        eprintln!("  [ID] depth {}...", current_depth);
        let iter_start = Instant::now();

        let too_cheap_to_split = last_iter_secs < MIN_SPLIT_SECS;
        if current_depth >= parallel_threshold && too_cheap_to_split {
            eprintln!(
                "  [PARALLEL] depth {} not split: previous iteration took {:.3}s < {:.2}s",
                current_depth, last_iter_secs, MIN_SPLIT_SECS
            );
        }

        if current_depth < parallel_threshold || too_cheap_to_split {
            let (score, m) = minimax_ab(gs, current_depth, i32::MIN + 1, i32::MAX - 1, maximizing, true, &mut ss);
            // If stopped mid-search, only use result if we have a previous best
            if ss.stopped {
                if !m.is_null() && current_depth <= 2 {
                    // Only trust very shallow aborted results
                    best_move = m;
                    best_score = score;
                }
                eprintln!("  Time limit reached during depth {}, aborting", current_depth);
                break;
            }
            if !m.is_null() {
                best_move = m;
                best_score = score;
                cache_store(move_cache, dirty, &pos_hash_str, current_depth, m);
            }
        } else {
            let (m, score, _) = minimax_parallel(gs, current_depth, &mut ss);
            // Same guard as the sequential path: an aborted iteration is not a result.
            if ss.stopped {
                if !m.is_null() && current_depth <= 2 {
                    // Only trust very shallow aborted results
                    best_move = m;
                    best_score = score;
                }
                eprintln!("  Time limit reached during depth {}, aborting", current_depth);
                break;
            }
            if !m.is_null() {
                best_move = m;
                best_score = score;
                cache_store(move_cache, dirty, &pos_hash_str, current_depth, m);
            }
        }

        let elapsed = iter_start.elapsed().as_secs_f64();
        last_iter_secs = elapsed;
        eprintln!("  [ID] depth {} done in {:.2}s, score={}", current_depth, elapsed, best_score);

        if best_score.abs() >= CHECKMATE_SCORE * 9 / 10 {
            eprintln!("  Mate found at depth {}", current_depth);
            break;
        }

        if let Some(limit) = time_limit {
            if start.elapsed().as_secs_f64() >= limit {
                eprintln!("  Time limit reached after depth {}", current_depth);
                break;
            }
        }
    }

    // Store best
    if !best_move.is_null() {
        cache_store(move_cache, dirty, &pos_hash_str, depth, best_move);

        // Persist TT entries to cache
        let mut tt_saved = 0;
        for (h, e) in &ss.tt {
            if e.depth >= 4 && e.flag == TTFlag::Exact && !e.best_move.is_null() {
                let h_str = h.to_string();
                let key = (h_str.clone(), e.depth);
                if !move_cache.contains_key(&key) {
                    cache_store(move_cache, dirty, &h_str, e.depth, e.best_move);
                    tt_saved += 1;
                }
            }
        }
        if tt_saved > 0 {
            eprintln!("  [TT→CACHE] saved {} positions", tt_saved);
        }
    }

    eprintln!("AI done in {:.2}s", start.elapsed().as_secs_f64());
    (best_move, best_score)
}

// Move repr formatting for cache compatibility
pub fn format_move_repr(m: Move) -> String {
    if m.is_null() {
        return "None".to_string();
    }
    if m.is_drop() {
        let to = m.to_sq();
        let r = sq_row(to);
        let f = sq_file(to);
        let color_char = match m.drop_color() {
            Color::White => 'w',
            Color::Black => 'b',
        };
        let pt_char = match m.drop_piece_type() {
            PieceType::Pawn => 'P',
            PieceType::Knight => 'N',
            PieceType::Bishop => 'B',
            PieceType::Rook => 'R',
            PieceType::Queen => 'Q',
            _ => '?',
        };
        format!("('drop', '{}{}'  , ({}, {}))", color_char, pt_char, r, f)
    } else {
        let from = m.from_sq();
        let to = m.to_sq();
        let fr = sq_row(from);
        let ff = sq_file(from);
        let tr = sq_row(to);
        let tf = sq_file(to);
        match m.promotion() {
            Some(pt) => {
                let pc = match pt {
                    PieceType::Rook => if fr > tr { "R" } else { "r" },
                    PieceType::Knight => if fr > tr { "N" } else { "n" },
                    PieceType::Bishop => if fr > tr { "B" } else { "b" },
                    _ => "?",
                };
                format!("(({}, {}), ({}, {}), '{}')", fr, ff, tr, tf, pc)
            }
            None => format!("(({}, {}), ({}, {}), None)", fr, ff, tr, tf),
        }
    }
}

pub fn parse_move_repr(s: &str) -> Option<Move> {
    // Parse Python repr format: ((<r1>, <f1>), (<r2>, <f2>), None/'R'/etc) or ('drop', '<code>', (<r>, <f>))
    let s = s.trim();
    if s.starts_with("('drop'") {
        // Drop format: ('drop', 'wN', (3, 3))
        let parts: Vec<&str> = s.split('\'').collect();
        if parts.len() >= 4 {
            let code = parts[3];
            if code.len() >= 2 {
                let color = match code.chars().next()? {
                    'w' => Color::White,
                    'b' => Color::Black,
                    _ => return None,
                };
                let pt = match code.chars().nth(1)? {
                    'P' => PieceType::Pawn,
                    'N' => PieceType::Knight,
                    'B' => PieceType::Bishop,
                    'R' => PieceType::Rook,
                    'Q' => PieceType::Queen,
                    _ => return None,
                };
                // Parse (r, f) from the end
                let nums: Vec<usize> = s.chars()
                    .filter(|c| c.is_ascii_digit())
                    .map(|c| c.to_digit(10).unwrap() as usize)
                    .collect();
                if nums.len() >= 2 {
                    let r = nums[nums.len() - 2];
                    let f = nums[nums.len() - 1];
                    if r < BOARD_SIZE && f < BOARD_SIZE {
                        return Some(Move::new_drop(sq(r, f), pt, color));
                    }
                }
            }
        }
        None
    } else {
        // Normal: ((r1, f1), (r2, f2), None/'R')
        let nums: Vec<usize> = s.chars()
            .filter(|c| c.is_ascii_digit())
            .map(|c| c.to_digit(10).unwrap() as usize)
            .collect();
        if nums.len() >= 4 {
            let from = sq(nums[0], nums[1]);
            let to = sq(nums[2], nums[3]);
            // Check for promotion
            let promo = if s.contains("'R'") { Some(PieceType::Rook) }
                else if s.contains("'r'") { Some(PieceType::Rook) }
                else if s.contains("'N'") { Some(PieceType::Knight) }
                else if s.contains("'n'") { Some(PieceType::Knight) }
                else if s.contains("'B'") { Some(PieceType::Bishop) }
                else if s.contains("'b'") { Some(PieceType::Bishop) }
                else { None };
            Some(Move::new_normal(from, to, promo))
        } else {
            None
        }
    }
}
