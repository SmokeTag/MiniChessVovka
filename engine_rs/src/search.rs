use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::types::*;
use crate::gamestate::GameState;
use crate::cache::{Book, BookEntry, BookMove};
use crate::eval::{evaluate_position, CHECKMATE_SCORE, STALEMATE_SCORE};
use crate::zobrist;

const MAX_QUIESCENCE_DEPTH: i32 = 4;

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
    // Root move ordering, same keys as minimax_ab: TT move, then killers, then mvv/lva.
    // Without this the baseline move is arbitrary and every scout fails high.
    let legal_moves = ordered_root_moves(gs, depth, ss);
    if legal_moves.is_empty() {
        return (Move::NULL, 0, vec![]);
    }

    let maximizing = gs.current_turn == Color::White;
    let deadline = ss.deadline;
    let neg_inf = i32::MIN + 1;
    let inf = i32::MAX - 1;

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

/// The one way to write the book. An entry that misses `dirty` never reaches the DB, so
/// every insert goes through here.
///
/// Called once per search, with the finished ranked list. Nothing else writes: no row
/// per iterative-deepening iteration (a depth-10 probe would reject all nine of them)
/// and no transposition-table dump (those are interior nodes, with no FEN to store
/// beside them and no completed root depth behind their scores).
fn book_store(
    book: &mut Book,
    dirty: &mut Vec<String>,
    pos_hash_str: &str,
    fen: String,
    ply: Option<i32>,
    ranked: &[(Move, i32)],
    depth_completed: i32,
) {
    if ranked.is_empty() || depth_completed <= 0 {
        return;
    }
    // Never trade a deeper answer for a shallower one. `build_book.py`'s narrowing pass
    // runs a cheap shallow search to rank the opponent's replies, and a transposition can
    // land that result on a position some other line already searched deep -- an entry is
    // replaced wholesale, so without this the deep row would be silently overwritten by
    // the shallow one. A re-search at the same depth or deeper still stores, which is what
    // lets a later pass add ranks; an entry from a superseded `eval_version` is stale
    // whatever its depth, and is replaced.
    if let Some(prev) = book.get(pos_hash_str).and_then(|e| e.moves.first()) {
        if prev.eval_version == crate::eval::EVAL_VERSION && prev.depth > depth_completed {
            return;
        }
    }
    let moves = ranked
        .iter()
        .enumerate()
        .map(|(i, &(m, score))| BookMove {
            rank: i as i32 + 1,
            move_repr: format_move_repr(m),
            score,
            depth: depth_completed,
            eval_version: crate::eval::EVAL_VERSION,
        })
        .collect();
    book.insert(
        pos_hash_str.to_string(),
        BookEntry { moves, fen: Some(fen), ply },
    );
    dirty.push(pos_hash_str.to_string());
}

/// A mate score is conclusive at whatever depth found it.
///
/// The mate break in the iterative-deepening loop exits as soon as an iteration returns a
/// mate, so `depth` on such a row is the iteration that found it rather than the depth the
/// caller asked for -- about 27% of a ply-8 book. Rejecting those on depth alone made every
/// probe re-search them, and since the re-search breaks at the same iteration it stored the
/// same shallow depth right back, so they never converged: the same mates were re-derived on
/// every walk, forever. Measured on the ply-8 book that is ~54 CPU-minutes per full re-walk,
/// 43 of which sit above `--split-ply`, where all 20 build workers each pay it in parallel.
///
/// Accepting them is sound because the score carries no mate distance -- `CHECKMATE_SCORE` is
/// flat, so a deeper search has nothing to add. Verified over 48 rows spanning stored depths
/// 1-9: a fresh depth-10 search returns the identical move and score for every one. The
/// threshold matches the mate break's own, and is slack for the same reason -- scores near the
/// bound do not land exactly on `CHECKMATE_SCORE` (a -999999 is in the book).
fn is_mate_score(score: i32) -> bool {
    score.abs() >= CHECKMATE_SCORE * 9 / 10
}

/// The book entry for this position, if it answers the question being asked.
///
/// Rejects, in order: a shallower search than requested (a weaker answer than the one we
/// are about to run) *unless* it is a mate, which `is_mate_score` above explains, a score
/// written by a different evaluation (`eval_version`), fewer ranks than the caller wants,
/// and any move that is not legal here -- the last being a hash collision or a stale row,
/// which is skipped rather than trusted.
///
/// Deeper-than-requested entries are kept: they answer the same question with strictly
/// more evidence. This reads only `book_move`; the `position` table is never joined on
/// the hot path.
fn probe_book(
    gs: &mut GameState,
    book: &Book,
    pos_hash_str: &str,
    min_depth: i32,
    want_ranks: usize,
) -> Option<(Vec<(Move, i32)>, i32)> {
    let entry = book.get(pos_hash_str)?;
    if entry.moves.len() < want_ranks {
        return None;
    }
    let head = &entry.moves[..want_ranks];
    if head.iter().any(|bm| {
        (bm.depth < min_depth && !is_mate_score(bm.score))
            || bm.eval_version != crate::eval::EVAL_VERSION
    }) {
        return None;
    }

    let legal = gs.get_legal_moves_vec();
    let mut ranked = Vec::with_capacity(head.len());
    for bm in head {
        let m = parse_move_repr(&bm.move_repr)?;
        if !legal.contains(&m) {
            return None;
        }
        ranked.push((m, bm.score));
    }
    let depth = head.iter().map(|bm| bm.depth).min().unwrap_or(0);
    Some((ranked, depth))
}

/// Orders the root moves the way `minimax_ab` orders every other node: TT move first,
/// then killers, then MVV/LVA. Shared by the parallel root and the MultiPV root so all
/// three paths see the same move list in the same order.
fn ordered_root_moves(gs: &mut GameState, depth: i32, ss: &mut SearchState) -> Vec<Move> {
    let legal_moves = gs.get_legal_moves_vec();
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
    scored.into_iter().map(|(m, _)| m).collect()
}

/// Exact scores for the best `k` root moves -- the book's ranks 2..k.
///
/// Ranks below 1 cannot be read off an ordinary alpha-beta search: once alpha is raised
/// by the best move, every later root move returns a *bound* ("no better than alpha"),
/// not a value, and filing those as scores would fill the book with numbers that mean
/// nothing. So every score this returns comes from a full window -- for the first `k`
/// moves directly, and for anything that displaces them via the re-search below.
///
/// Measured at depth 9, against the same search asked for one move: K=2 costs 2.4x from
/// the initial position and 3.0x six plies in; K=3 costs 4.4x and 3.8x. (The estimate
/// this was specced against, 1.5-2.5x, holds only for K=2 in the opening.)
///
/// What "exact" means here is procedural: never a fail-low or fail-high bound. It is not
/// a claim that the number is the true depth-N minimax value, because this engine's
/// scores are path-dependent -- searching each root move independently, with its own
/// table, produces different numbers, and so does the ordinary single-PV root. That
/// drift predates the book (see the null-move/TT-flag note on `search_worker`) and
/// fixing it is an engine change, not a schema one.
///
/// The remaining moves still have to be checked in case one of them belongs in the top
/// `k`, but they do not need a full window to do it. Searching them against
/// `(cutoff, +inf)` (mirrored for Black) has a useful property: with the far side of the
/// window open, a returned value can never fail high, so anything that beats the cutoff
/// is already exact and goes straight into the ranking -- no re-search, and the raised
/// alpha prunes hard on everything that does not.
///
/// Returns `None` if the search was cut short, because a partial pass ranks by move
/// ordering rather than by score, and the caller has an honest single-PV result from the
/// previous iteration to fall back on.
///
/// Runs sequentially even when the root split is enabled: `minimax_parallel` reports one
/// move, and its workers' narrow scout windows are exactly the bounds this routine
/// exists to avoid.
fn multipv_root(
    gs: &mut GameState,
    depth: i32,
    k: usize,
    ss: &mut SearchState,
) -> Option<Vec<(Move, i32)>> {
    let maximizing = gs.current_turn == Color::White;
    let neg_inf = i32::MIN + 1;
    let inf = i32::MAX - 1;
    let legal_moves = ordered_root_moves(gs, depth, ss);
    if legal_moves.is_empty() {
        return Some(Vec::new());
    }
    let k = k.min(legal_moves.len());

    // Best-first for the side to move; scores stay white-relative throughout.
    let better = |a: i32, b: i32| if maximizing { a > b } else { a < b };
    let sort_ranked = |v: &mut Vec<(Move, i32)>| {
        v.sort_by(|x, y| if maximizing { y.1.cmp(&x.1) } else { x.1.cmp(&y.1) });
    };

    let mut ranked: Vec<(Move, i32)> = Vec::with_capacity(k);
    for &m in legal_moves.iter().take(k) {
        gs.make_ai_move(m);
        let (score, _) = minimax_ab(gs, depth - 1, neg_inf, inf, !maximizing, true, ss);
        gs.undo_ai_move();
        if ss.stopped {
            return None;
        }
        ranked.push((m, score));
    }
    sort_ranked(&mut ranked);

    for &m in legal_moves.iter().skip(k) {
        // The score to beat: the weakest move currently holding a rank.
        let cutoff = ranked[ranked.len() - 1].1;
        let (alpha, beta) = if maximizing { (cutoff, inf) } else { (neg_inf, cutoff) };
        gs.make_ai_move(m);
        let (score, _) = minimax_ab(gs, depth - 1, alpha, beta, !maximizing, true, ss);
        if !better(score, cutoff) {
            // Not good enough to hold a rank. What came back is a bound, and it is
            // discarded here rather than stored -- which is the whole reason ranks 2+
            // cannot be read off an ordinary alpha-beta search.
            gs.undo_ai_move();
            if ss.stopped {
                return None;
            }
            continue;
        }
        // It belongs in the ranking, so it needs a value rather than a bound. The
        // narrow search above cannot supply one: this engine's null-move pruning returns
        // a hard beta and the TT store site classifies flags against the original window,
        // so a window-clamped score can come back tagged Exact (measured: the same
        // position ranked 27 under a full window and 193 under a capped one). Every score
        // that reaches the book comes from a full window.
        let (exact, _) = minimax_ab(gs, depth - 1, neg_inf, inf, !maximizing, true, ss);
        gs.undo_ai_move();
        if ss.stopped {
            return None;
        }
        ranked.pop();
        ranked.push((m, exact));
        sort_ranked(&mut ranked);
    }

    eprintln!(
        "  [MULTIPV] {} ranked move(s) at depth {}: {}",
        ranked.len(),
        depth,
        ranked
            .iter()
            .map(|(m, s)| format!("{:?}={}", m, s))
            .collect::<Vec<_>>()
            .join(" ")
    );
    Some(ranked)
}

/// Main entry: book probe, then iterative deepening.
///
/// `top_n` is how many ranked moves the caller wants. Anything above 1 runs the MultiPV
/// root pass at the final depth so ranks 2.. carry exact scores -- see [`multipv_root`]
/// for what that costs. There is no ply or depth gate on it: the caller decides where to
/// spend, because the caller is the one that knows whether it is filling a book or
/// playing a move.
///
/// `dirty` collects the position hashes this call wrote, so the caller can persist just
/// those instead of rewriting the whole book.
///
/// `book` is taken as a `&Mutex` rather than a `&mut`, and that is load-bearing: the book
/// is touched exactly twice here -- the probe at the top and the single store at the
/// bottom -- so the lock is held for microseconds at each end and *not* for the seconds
/// or minutes in between. Holding it across the search serialises every caller onto one
/// search at a time, and because `lib.rs`'s book accessors (`book_has_position`,
/// `save_analysis_to_db`, ...) do not release the GIL while they wait for it, a GUI
/// thread that merely asked "is this position in the book?" froze for the whole duration
/// of a background hint -- 46s, measured, at depth 10 with four lines.
///
/// Returns the ranked moves, best-first **for the side to move**, with white-relative
/// scores: White's list runs non-increasing and Black's non-decreasing, so rank 1 is
/// always the move that side wants and the sign never depends on whose turn it is. The
/// list is empty only when the position has no legal moves.
pub fn find_best_move(
    gs: &mut GameState,
    depth: i32,
    top_n: i32,
    book: &Mutex<Book>,
    dirty: &mut Vec<String>,
    time_limit: Option<f64>,
    parallel: Option<bool>,
) -> Vec<(Move, i32)> {
    let start = Instant::now();
    let maximizing = gs.current_turn == Color::White;
    let pos_hash_str = gs.hash.to_string();
    let want_ranks = top_n.max(1) as usize;

    let probed = {
        let guard = book.lock().unwrap();
        probe_book(gs, &guard, &pos_hash_str, depth, want_ranks)
    };
    if let Some((ranked, hit_depth)) = probed {
        eprintln!(
            "[BOOK HIT] {} rank(s) at depth {} (requested {}) in {:.2}s",
            ranked.len(),
            hit_depth,
            depth,
            start.elapsed().as_secs_f64()
        );
        return ranked;
    }

    let legal = gs.get_legal_moves_vec();
    if legal.is_empty() {
        return Vec::new();
    }
    if legal.len() == 1 {
        // Forced. Returned with a placeholder score and deliberately not stored: there is
        // no completed depth behind it and no evaluation behind the 0, and the book's
        // contract is that every row it holds is a real search result. Re-deriving this
        // costs one move generation.
        return vec![(legal[0], 0)];
    }

    let mut ss = SearchState::new();
    ss.deadline = time_limit.map(|secs| start + std::time::Duration::from_secs_f64(secs));

    // The last iteration that ran to completion, and what it returned. Only a completed
    // iteration may set this: the book records the depth actually reached, so a result
    // salvaged from an aborted iteration (below) can be returned but must never be filed
    // as if that depth had finished.
    let mut completed: Option<(Vec<(Move, i32)>, i32)> = None;
    // Fallback for a search cut short before any iteration finished.
    let mut aborted_best: Option<(Move, i32)> = None;

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

        // MultiPV only at the final depth: the shallow iterations exist to warm the table
        // and the move ordering, and paying the full-window premium on them buys ranks
        // that the last iteration is about to overwrite.
        let multipv_here = want_ranks > 1 && current_depth == depth;

        let too_cheap_to_split = last_iter_secs < MIN_SPLIT_SECS;
        if current_depth >= parallel_threshold && too_cheap_to_split && !multipv_here {
            eprintln!(
                "  [PARALLEL] depth {} not split: previous iteration took {:.3}s < {:.2}s",
                current_depth, last_iter_secs, MIN_SPLIT_SECS
            );
        }

        if multipv_here {
            match multipv_root(gs, current_depth, want_ranks, &mut ss) {
                Some(ranked) if !ranked.is_empty() => {
                    completed = Some((ranked, current_depth));
                }
                Some(_) => {}
                None => {
                    eprintln!("  Time limit reached during MultiPV at depth {}, keeping depth {}",
                        current_depth,
                        completed.as_ref().map(|(_, d)| *d).unwrap_or(0));
                    break;
                }
            }
        } else if current_depth < parallel_threshold || too_cheap_to_split {
            let (score, m) = minimax_ab(gs, current_depth, i32::MIN + 1, i32::MAX - 1, maximizing, true, &mut ss);
            if ss.stopped {
                // If stopped mid-search, only trust very shallow aborted results
                if !m.is_null() && current_depth <= 2 {
                    aborted_best = Some((m, score));
                }
                eprintln!("  Time limit reached during depth {}, aborting", current_depth);
                break;
            }
            if !m.is_null() {
                completed = Some((vec![(m, score)], current_depth));
            }
        } else {
            let (m, score, _) = minimax_parallel(gs, current_depth, &mut ss);
            // Same guard as the sequential path: an aborted iteration is not a result.
            if ss.stopped {
                if !m.is_null() && current_depth <= 2 {
                    aborted_best = Some((m, score));
                }
                eprintln!("  Time limit reached during depth {}, aborting", current_depth);
                break;
            }
            if !m.is_null() {
                completed = Some((vec![(m, score)], current_depth));
            }
        }

        let elapsed = iter_start.elapsed().as_secs_f64();
        last_iter_secs = elapsed;
        let iter_score = completed.as_ref().and_then(|(r, _)| r.first().map(|&(_, s)| s)).unwrap_or(0);
        eprintln!("  [ID] depth {} done in {:.2}s, score={}", current_depth, elapsed, iter_score);

        if is_mate_score(iter_score) {
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

    eprintln!("AI done in {:.2}s", start.elapsed().as_secs_f64());

    match completed {
        Some((ranked, depth_completed)) => {
            // One book entry per search: the ranks this search produced, at the depth it
            // actually reached, with the FEN that makes the hash mean something.
            book_store(
                &mut book.lock().unwrap(),
                dirty,
                &pos_hash_str,
                crate::fen::to_fen(gs),
                Some(gs.ply as i32),
                &ranked,
                depth_completed,
            );
            if ranked.len() < want_ranks {
                eprintln!(
                    "  [BOOK] stored {} rank(s), {} requested (search ended early or ran out of moves)",
                    ranked.len(),
                    want_ranks
                );
            }
            ranked
        }
        // Nothing finished. Return whatever the aborted first iterations salvaged so the
        // caller still has a move to play, but file none of it: `depth_completed` is 0
        // and `book_store` refuses those.
        None => aborted_best.map(|ms| vec![ms]).unwrap_or_default(),
    }
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
