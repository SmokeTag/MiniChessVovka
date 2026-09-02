use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::types::*;
use crate::gamestate::GameState;
use crate::cache::{Book, BookEntry, BookMove};
use crate::eval::{evaluate_position, CHECKMATE_SCORE, STALEMATE_SCORE};
use crate::zobrist;

const MAX_QUIESCENCE_DEPTH: i32 = 4;
const DELTA_MARGIN: i32 = 900;
const LMR_MIN_MOVE: usize = 8;
const LMR_MIN_DEPTH: i32 = 3;

#[derive(Clone, Copy)]
pub struct Knobs {
    pub null_move: bool,
    pub lmr: bool,
    pub use_tt: bool,
    pub use_book: bool,
    pub delta_margin: i32,
    pub order_seed: u64,
}

pub const DEFAULT_KNOBS: Knobs = Knobs {
    null_move: true,
    lmr: true,
    use_tt: true,
    use_book: true,
    delta_margin: DELTA_MARGIN,
    order_seed: 0,
};

static KNOBS: Mutex<Knobs> = Mutex::new(DEFAULT_KNOBS);
static LAST_NODES: AtomicU64 = AtomicU64::new(0);
static LAST_QNODES: AtomicU64 = AtomicU64::new(0);

pub fn knobs() -> Knobs {
    *KNOBS.lock().unwrap()
}

pub fn set_knobs(k: Knobs) {
    *KNOBS.lock().unwrap() = k;
}

pub fn last_search_nodes() -> (u64, u64) {
    (LAST_NODES.load(Ordering::Relaxed), LAST_QNODES.load(Ordering::Relaxed))
}

#[inline]
fn order_tiebreak(seed: u64, m: Move) -> u64 {
    let mut x = seed ^ (m.data as u64).wrapping_mul(0x9E3779B97F4A7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 32;
    x
}

#[derive(Default, Clone, Copy)]
pub struct MixHasher(u64);

impl std::hash::Hasher for MixHasher {
    #[inline]
    fn finish(&self) -> u64 {
        let mut z = self.0;
        z ^= z >> 33;
        z = z.wrapping_mul(0xff51_afd7_ed55_8ccd);
        z ^= z >> 33;
        z
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    #[inline]
    fn write_u64(&mut self, n: u64) { self.0 = n; }
    #[inline]
    fn write_u32(&mut self, n: u32) { self.0 = n as u64; }
    #[inline]
    fn write_i32(&mut self, n: i32) { self.0 = n as u64; }
}

type MixBuild = std::hash::BuildHasherDefault<MixHasher>;
type FastMap<K, V> = HashMap<K, V, MixBuild>;

const DEFAULT_PARALLEL_MIN_DEPTH: i32 = 3;
static PARALLEL_ENABLED: AtomicBool = AtomicBool::new(false);
static PARALLEL_MIN_DEPTH: AtomicI32 = AtomicI32::new(DEFAULT_PARALLEL_MIN_DEPTH);

pub fn set_parallel_search(enabled: bool, min_depth: Option<i32>) {
    if let Some(d) = min_depth {
        PARALLEL_MIN_DEPTH.store(d.max(1), Ordering::Relaxed);
    }
    PARALLEL_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn parallel_search_config() -> (bool, i32) {
    (
        PARALLEL_ENABLED.load(Ordering::Relaxed),
        PARALLEL_MIN_DEPTH.load(Ordering::Relaxed),
    )
}

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

const TT_OCCUPIED: u64 = 1 << 58;

#[inline]
fn tt_pack(e: &TTEntry) -> u64 {
    let mv = if e.best_move.is_null() { 0xFFFF } else { (e.best_move.data as u64) & 0xFFFF };
    let d = e.depth.clamp(0, 255) as u64;
    let f = match e.flag { TTFlag::Exact => 0u64, TTFlag::LowerBound => 1, TTFlag::UpperBound => 2 };
    (e.score as u32 as u64) | (mv << 32) | (d << 48) | (f << 56) | TT_OCCUPIED
}

#[inline]
fn tt_unpack(w: u64) -> TTEntry {
    let mv = ((w >> 32) & 0xFFFF) as u32;
    TTEntry {
        score: (w & 0xFFFF_FFFF) as u32 as i32,
        best_move: if mv == 0xFFFF { Move::NULL } else { Move { data: mv } },
        depth: ((w >> 48) & 0xFF) as i32,
        flag: match (w >> 56) & 0x3 {
            0 => TTFlag::Exact,
            1 => TTFlag::LowerBound,
            _ => TTFlag::UpperBound,
        },
    }
}

#[inline]
fn tt_depth_of(w: u64) -> i32 { ((w >> 48) & 0xFF) as i32 }

const TT_WAYS: usize = 4;

const TT_DEFAULT_SLOTS: usize = 1 << 21;

pub struct TransTable {
    words: Vec<u64>,
    mask: usize,
    slots: usize,
}

impl TransTable {
    pub fn with_slots(slots: usize) -> Self {
        let n = (slots / TT_WAYS).next_power_of_two().max(1 << 8);
        TransTable {
            words: vec![0u64; n * TT_WAYS * 2],
            mask: n - 1,
            slots: n * TT_WAYS,
        }
    }

    #[inline]
    pub fn slots(&self) -> usize { self.slots }

    #[inline]
    pub fn get(&mut self, hash: u64) -> Option<TTEntry> {
        let b = (hash as usize & self.mask) * (TT_WAYS * 2);
        for w in 0..TT_WAYS {
            let d = self.words[b + w * 2 + 1];
            if d & TT_OCCUPIED != 0 && self.words[b + w * 2] == hash {
                return Some(tt_unpack(d));
            }
        }
        None
    }

    #[inline]
    pub fn insert(&mut self, hash: u64, e: &TTEntry) {
        let b = (hash as usize & self.mask) * (TT_WAYS * 2);
        for w in 0..TT_WAYS {
            let d = self.words[b + w * 2 + 1];
            if d & TT_OCCUPIED == 0 {
                self.words[b + w * 2] = hash;
                self.words[b + w * 2 + 1] = tt_pack(e);
                return;
            }
            if self.words[b + w * 2] == hash {
                self.words[b + w * 2 + 1] = tt_pack(e);
                return;
            }
        }
        let mut victim = 0usize;
        let mut vdepth = i32::MAX;
        for w in 0..TT_WAYS {
            let dd = tt_depth_of(self.words[b + w * 2 + 1]);
            if dd < vdepth {
                vdepth = dd;
                victim = w;
            }
        }
        if vdepth > e.depth {
            return;
        }
        self.words[b + victim * 2] = hash;
        self.words[b + victim * 2 + 1] = tt_pack(e);
    }

    pub fn clear(&mut self) {
        self.words.iter_mut().for_each(|w| *w = 0);
    }

    pub fn len(&self) -> usize {
        self.words.chunks_exact(2).filter(|c| c[1] & TT_OCCUPIED != 0).count()
    }

    pub fn iter_entries(&self) -> impl Iterator<Item = (u64, TTEntry)> + '_ {
        self.words
            .chunks_exact(2)
            .filter(|c| c[1] & TT_OCCUPIED != 0)
            .map(|c| (c[0], tt_unpack(c[1])))
    }
}

pub struct SearchState {
    pub tt: TransTable,
    pub base_tt: Option<Arc<FastMap<u64, TTEntry>>>,
    pub killer_moves: FastMap<i32, [Move; 2]>,
    pub history_scores: FastMap<u32, i32>,
    pub deadline: Option<Instant>,
    pub stopped: bool,
    pub knobs: Knobs,
    pub nodes: u64,
    pub qnodes: u64,
    nodes_since_check: u32,
}

impl SearchState {
    pub fn new() -> Self {
        SearchState::with_tt_capacity(TT_DEFAULT_SLOTS)
    }

    pub fn with_tt_capacity(cap: usize) -> Self {
        SearchState {
            tt: TransTable::with_slots(cap),
            base_tt: None,
            killer_moves: FastMap::default(),
            history_scores: FastMap::default(),
            deadline: None,
            stopped: false,
            knobs: knobs(),
            nodes: 0,
            qnodes: 0,
            nodes_since_check: 0,
        }
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.base_tt = None;
        self.killer_moves.clear();
        self.history_scores.clear();
    }

    #[inline]
    pub fn tt_get(&mut self, hash: u64) -> Option<TTEntry> {
        if let Some(e) = self.tt.get(hash) {
            return Some(e);
        }
        self.base_tt.as_ref().and_then(|b| b.get(&hash).cloned())
    }

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

        if !victim.is_empty() {
            let mut victim_value = PIECE_VALUES[victim.piece_type().unwrap().index()];
            if gs.promoted_pieces & (1u64 << to) != 0 {
                victim_value = (victim_value + PIECE_VALUES[PieceType::Pawn.index()]) / 2;
            }
            score += victim_value * 10 - aggressor_value;
        }

        if m.promotion().is_some() {
            score += 900;
        }

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

        if is_center_sq(to) {
            score += 30;
        }

        if piece_color == Color::White && sq_row(from) == 5 && sq_row(to) < 5 {
            score += 20;
        } else if piece_color == Color::Black && sq_row(from) == 0 && sq_row(to) > 0 {
            score += 20;
        }

        score += *ss.history_scores.get(&m.data).unwrap_or(&0);

    } else {
        let to = m.to_sq();
        let pt = m.drop_piece_type();
        let color = m.drop_color();

        let mut base = HAND_PIECE_VALUES[pt.index()] / 10;

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

        let enemy_king_sq = gs.king_pos[color.opposite().index()];
        let ek_r = sq_row(enemy_king_sq) as i32;
        let ek_c = sq_file(enemy_king_sq) as i32;
        let to_r = sq_row(to) as i32;
        let to_c = sq_file(to) as i32;
        let dist = (to_r - ek_r).abs().max((to_c - ek_c).abs());
        if dist <= 1 { base += 200; }
        else if dist <= 2 { base += 100; }

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
        return is_drop_near_king(gs, m);
    }
    let to = m.to_sq();
    if gs.board[to] != Piece::Empty {
        return true;
    }
    m.promotion().is_some()
}

fn is_check_move(gs: &mut GameState, m: Move) -> bool {
    gs.make_ai_move(m);
    let in_check = gs.side_to_move_in_check();
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

    let limit = check_candidates.len().min(12);
    for &m in &check_candidates[..limit] {
        if is_check_move(gs, m) {
            noisy.push(m);
        }
    }

    noisy
}

fn quiescence_search(gs: &mut GameState, mut alpha: i32, mut beta: i32, maximizing: bool, depth: i32, ss: &mut SearchState) -> i32 {
    ss.qnodes += 1;
    let stand_pat = evaluate_position(gs);

    let legal = gs.get_legal_moves_vec();
    if legal.is_empty() {
        if gs.side_to_move_in_check() {
            return if gs.current_turn == Color::White { -CHECKMATE_SCORE } else { CHECKMATE_SCORE };
        }
        return STALEMATE_SCORE;
    }

    if depth == 0 {
        return stand_pat;
    }

    let delta_margin = ss.knobs.delta_margin;
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
            let score = quiescence_search(gs, alpha, beta, false, depth - 1, ss);
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
            let score = quiescence_search(gs, alpha, beta, true, depth - 1, ss);
            gs.undo_ai_move();

            if score <= -CHECKMATE_SCORE { return -CHECKMATE_SCORE; }
            beta = beta.min(score);
            if beta <= alpha { break; }
        }
        beta
    }
}

fn sort_scored(scored: &mut Vec<(Move, i32)>, seed: u64) {
    if seed == 0 {
        scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    } else {
        scored.sort_unstable_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| order_tiebreak(seed, a.0).cmp(&order_tiebreak(seed, b.0)))
        });
    }
}

fn minimax_ab(
    gs: &mut GameState,
    mut depth: i32,
    mut alpha: i32,
    mut beta: i32,
    maximizing: bool,
    allow_null: bool,
    ss: &mut SearchState,
) -> (i32, Move) {
    if ss.check_deadline() {
        return (evaluate_position(gs), Move::NULL);
    }
    ss.nodes += 1;

    let current_color = if maximizing { Color::White } else { Color::Black };
    let in_check = if current_color == gs.current_turn {
        gs.side_to_move_in_check()
    } else {
        gs.is_in_check(current_color)
    };
    if in_check && depth > 0 && depth < 3 {
        depth += 1;
    }

    if depth <= 0 || gs.checkmate || gs.stalemate {
        let q = quiescence_search(gs, alpha, beta, maximizing, MAX_QUIESCENCE_DEPTH, ss);
        return (q, Move::NULL);
    }

    let pos_hash = gs.hash;
    let tt_entry = if ss.knobs.use_tt { ss.tt_get(pos_hash) } else { None };

    let mut legal_moves = gs.get_legal_moves_vec();
    if legal_moves.is_empty() {
        if gs.side_to_move_in_check() {
            return (if gs.current_turn == Color::White { -CHECKMATE_SCORE } else { CHECKMATE_SCORE }, Move::NULL);
        }
        return (STALEMATE_SCORE, Move::NULL);
    }

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

    let null_move_r = 2;
    let is_pv = beta.saturating_sub(alpha) > 1;
    let opponent_color = current_color.opposite();
    let opponent_hand: u8 = gs.hands[opponent_color.index()].iter().sum();
    if ss.knobs.null_move && allow_null && !is_pv && depth >= null_move_r + 1 && !in_check && opponent_hand == 0 {
        gs.current_turn = gs.current_turn.opposite();
        gs.hash = gs.compute_hash();
        gs.invalidate_cache();
        let (null_score, _) = minimax_ab(gs, depth - 1 - null_move_r, alpha, beta, !maximizing, false, ss);
        gs.current_turn = gs.current_turn.opposite();
        gs.hash = gs.compute_hash();
        gs.invalidate_cache();

        let fails_high = if maximizing { null_score >= beta } else { null_score <= alpha };
        if fails_high && !ss.stopped {
            let (verified, _) = minimax_ab(gs, depth - null_move_r, alpha, beta, maximizing, false, ss);
            let confirmed = if maximizing { verified >= beta } else { verified <= alpha };
            if confirmed {
                return (if maximizing { beta } else { alpha }, Move::NULL);
            }
        }
    }

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
    sort_scored(&mut scored, ss.knobs.order_seed);
    for (slot, &(m, _)) in legal_moves.iter_mut().zip(scored.iter()) {
        *slot = m;
    }

    let orig_alpha = alpha;
    let orig_beta = beta;
    let mut best_move = Move::NULL;

    if maximizing {
        let mut max_eval = i32::MIN;
        for (i, &m) in legal_moves.iter().enumerate() {
            let noisy = is_noisy_move(gs, m);
            gs.make_ai_move(m);
            let gives_check = gs.side_to_move_in_check();

            let eval_score;
            if i == 0 {
                let (s, _) = minimax_ab(gs, depth - 1, alpha, beta, false, true, ss);
                eval_score = s;
            } else {
                let reduced = ss.knobs.lmr && i >= LMR_MIN_MOVE && depth >= LMR_MIN_DEPTH
                    && !noisy && !gives_check;
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
        if ss.knobs.use_tt {
            ss.tt.insert(pos_hash, &TTEntry { depth, score: max_eval, flag, best_move });
        }
        (max_eval, best_move)
    } else {
        let mut min_eval = i32::MAX;
        for (i, &m) in legal_moves.iter().enumerate() {
            let noisy = is_noisy_move(gs, m);
            gs.make_ai_move(m);
            let gives_check = gs.side_to_move_in_check();

            let eval_score;
            if i == 0 {
                let (s, _) = minimax_ab(gs, depth - 1, alpha, beta, true, true, ss);
                eval_score = s;
            } else {
                let reduced = ss.knobs.lmr && i >= LMR_MIN_MOVE && depth >= LMR_MIN_DEPTH
                    && !noisy && !gives_check;
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
        if ss.knobs.use_tt {
            ss.tt.insert(pos_hash, &TTEntry { depth, score: min_eval, flag, best_move });
        }
        (min_eval, best_move)
    }
}

type SharedTT = Arc<FastMap<u64, TTEntry>>;

struct RootHeuristics {
    killer_moves: FastMap<i32, [Move; 2]>,
    history_scores: FastMap<u32, i32>,
}

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
    let mut ss = SearchState::with_tt_capacity(1 << 18);
    ss.base_tt = Some(Arc::clone(base_tt));
    ss.killer_moves = heuristics.killer_moves.clone();
    ss.history_scores = heuristics.history_scores.clone();
    ss.deadline = deadline;

    gs_copy.make_ai_move(m);
    let (score, _) = minimax_ab(&mut gs_copy, depth - 1, alpha, beta, !maximizing, true, &mut ss);

    (m, score, ss.stopped)
}

fn minimax_parallel(
    gs: &mut GameState,
    depth: i32,
    ss: &mut SearchState,
) -> (Move, i32, Vec<(Move, i32)>) {
    let legal_moves = ordered_root_moves(gs, depth, ss);
    if legal_moves.is_empty() {
        return (Move::NULL, 0, vec![]);
    }

    let maximizing = gs.current_turn == Color::White;
    let deadline = ss.deadline;
    let neg_inf = i32::MIN + 1;
    let inf = i32::MAX - 1;

    let first_move = legal_moves[0];
    let baseline_start = Instant::now();
    gs.make_ai_move(first_move);
    let (baseline, _) = minimax_ab(gs, depth - 1, neg_inf, inf, !maximizing, true, ss);
    gs.undo_ai_move();
    let baseline_secs = baseline_start.elapsed().as_secs_f64();
    if ss.stopped {
        return (Move::NULL, 0, vec![]);
    }

    let mut best_move = first_move;
    let mut best_score = baseline;
    let mut all_results: Vec<(Move, i32)> = vec![(first_move, baseline)];

    let min_useful = (depth - 3).max(1);
    let base_tt: SharedTT = Arc::new(
        ss.tt.iter_entries()
            .filter(|(_, e)| e.depth >= min_useful)
            .collect(),
    );
    let heuristics = RootHeuristics {
        killer_moves: ss.killer_moves.clone(),
        history_scores: ss.history_scores.clone(),
    };

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

    let mating_score = if maximizing { CHECKMATE_SCORE } else { -CHECKMATE_SCORE };
    for &(m, s) in &all_results {
        if s == mating_score {
            return (m, s, all_results);
        }
    }

    (best_move, best_score, all_results)
}

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

fn is_mate_score(score: i32) -> bool {
    score.abs() >= CHECKMATE_SCORE * 9 / 10
}

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
    sort_scored(&mut scored, ss.knobs.order_seed);
    scored.into_iter().map(|(m, _)| m).collect()
}

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
        let cutoff = ranked[ranked.len() - 1].1;
        let (alpha, beta) = if maximizing { (cutoff, inf) } else { (neg_inf, cutoff) };
        gs.make_ai_move(m);
        let (score, _) = minimax_ab(gs, depth - 1, alpha, beta, !maximizing, true, ss);
        if !better(score, cutoff) {
            gs.undo_ai_move();
            if ss.stopped {
                return None;
            }
            continue;
        }
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

    let run_knobs = knobs();
    let probed = if run_knobs.use_book {
        let guard = book.lock().unwrap();
        probe_book(gs, &guard, &pos_hash_str, depth, want_ranks)
    } else {
        None
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
        return vec![(legal[0], 0)];
    }

    let mut ss = SearchState::new();
    ss.deadline = time_limit.map(|secs| start + std::time::Duration::from_secs_f64(secs));

    let mut completed: Option<(Vec<(Move, i32)>, i32)> = None;
    let mut aborted_best: Option<(Move, i32)> = None;

    let (global_parallel, parallel_min_depth) = parallel_search_config();
    let use_parallel = parallel.unwrap_or(global_parallel);
    let parallel_threshold = if use_parallel { parallel_min_depth.max(depth) } else { i32::MAX };
    if use_parallel {
        eprintln!("  [PARALLEL] root split enabled from depth {}", parallel_threshold);
    }

    const MIN_SPLIT_SECS: f64 = 0.03;
    let mut last_iter_secs = 0.0f64;

    for current_depth in 1..=depth {
        eprintln!("  [ID] depth {}...", current_depth);
        let iter_start = Instant::now();

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
    LAST_NODES.store(ss.nodes, Ordering::Relaxed);
    LAST_QNODES.store(ss.qnodes, Ordering::Relaxed);

    match completed {
        Some((ranked, depth_completed)) => {
            if run_knobs.use_book {
                book_store(
                    &mut book.lock().unwrap(),
                    dirty,
                    &pos_hash_str,
                    crate::fen::to_fen(gs),
                    Some(gs.ply as i32),
                    &ranked,
                    depth_completed,
                );
            }
            if ranked.len() < want_ranks {
                eprintln!(
                    "  [BOOK] stored {} rank(s), {} requested (search ended early or ran out of moves)",
                    ranked.len(),
                    want_ranks
                );
            }
            ranked
        }
        None => aborted_best.map(|ms| vec![ms]).unwrap_or_default(),
    }
}

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
    let s = s.trim();
    if s.starts_with("('drop'") {
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
        let nums: Vec<usize> = s.chars()
            .filter(|c| c.is_ascii_digit())
            .map(|c| c.to_digit(10).unwrap() as usize)
            .collect();
        if nums.len() >= 4 {
            let from = sq(nums[0], nums[1]);
            let to = sq(nums[2], nums[3]);
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
