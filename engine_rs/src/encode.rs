use crate::gamestate::GameState;
use crate::types::*;

pub const PLANES: usize = 24;
pub const PLANE_SIZE: usize = NUM_SQUARES;
pub const INPUT_SIZE: usize = PLANES * PLANE_SIZE;

pub const RAY_PLANES: usize = 40;
pub const KNIGHT_PLANES: usize = 8;
pub const PROMO_PLANES: usize = 9;
pub const DROP_PLANES: usize = 4;
pub const ACTION_PLANES: usize = RAY_PLANES + KNIGHT_PLANES + PROMO_PLANES + DROP_PLANES;
pub const ACTION_SPACE: usize = ACTION_PLANES * PLANE_SIZE;

const RAY_BASE: usize = 0;
const KNIGHT_BASE: usize = RAY_BASE + RAY_PLANES;
const PROMO_BASE: usize = KNIGHT_BASE + KNIGHT_PLANES;
const DROP_BASE: usize = PROMO_BASE + PROMO_PLANES;

const MAX_RAY: usize = BOARD_SIZE - 1;

pub const RAY_DIRS: [(i32, i32); 8] = [
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
];

pub const PROMO_DIRS: [(i32, i32); 3] = [(-1, 0), (-1, -1), (-1, 1)];

pub const HAND_TYPES: [PieceType; 4] = [
    PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook,
];

const OWN_PIECES: usize = 0;
const OPP_PIECES: usize = 5;
const OWN_PROMOTED: usize = 10;
const OPP_PROMOTED: usize = 11;
const OWN_HANDS: usize = 12;
const OPP_HANDS: usize = 16;
const SIDE_TO_MOVE: usize = 20;
const REPEAT_ONCE: usize = 21;
const REPEAT_TWICE: usize = 22;
const PROGRESS: usize = 23;

pub const MAX_HAND_COUNT: f32 = 8.0;

#[inline]
fn canon(s: usize, flip: bool) -> usize {
    if flip { NUM_SQUARES - 1 - s } else { s }
}

#[inline]
fn board_type_index(pt: PieceType) -> Option<usize> {
    match pt {
        PieceType::Pawn => Some(0),
        PieceType::Knight => Some(1),
        PieceType::Bishop => Some(2),
        PieceType::Rook => Some(3),
        PieceType::King => Some(4),
        PieceType::Queen => None,
    }
}

#[inline]
fn hand_type_index(pt: PieceType) -> Option<usize> {
    match pt {
        PieceType::Pawn => Some(0),
        PieceType::Knight => Some(1),
        PieceType::Bishop => Some(2),
        PieceType::Rook => Some(3),
        _ => None,
    }
}

#[inline]
fn fill(out: &mut [f32], plane: usize, v: f32) {
    if v == 0.0 {
        return;
    }
    let base = plane * PLANE_SIZE;
    for x in &mut out[base..base + PLANE_SIZE] {
        *x = v;
    }
}

pub fn encode_position(gs: &GameState) -> Vec<f32> {
    let mut out = vec![0.0f32; INPUT_SIZE];
    encode_position_into(gs, &mut out);
    out
}

pub fn encode_position_into(gs: &GameState, out: &mut [f32]) {
    debug_assert_eq!(out.len(), INPUT_SIZE);
    for x in out.iter_mut() {
        *x = 0.0;
    }

    let us = gs.current_turn;
    let flip = us == Color::Black;

    for s in 0..NUM_SQUARES {
        let piece = gs.board[s];
        if piece.is_empty() {
            continue;
        }
        let color = piece.color().unwrap();
        let ti = match board_type_index(piece.piece_type().unwrap()) {
            Some(i) => i,
            None => continue,
        };
        let cs = canon(s, flip);
        let base = if color == us { OWN_PIECES } else { OPP_PIECES };
        out[(base + ti) * PLANE_SIZE + cs] = 1.0;

        if gs.promoted_pieces & (1u64 << s) != 0 {
            let p = if color == us { OWN_PROMOTED } else { OPP_PROMOTED };
            out[p * PLANE_SIZE + cs] = 1.0;
        }
    }

    let them = us.opposite();
    for (i, &pt) in HAND_TYPES.iter().enumerate() {
        let own = (gs.hand_count(us, pt) as f32 / MAX_HAND_COUNT).min(1.0);
        let opp = (gs.hand_count(them, pt) as f32 / MAX_HAND_COUNT).min(1.0);
        fill(out, OWN_HANDS + i, own);
        fill(out, OPP_HANDS + i, opp);
    }

    if us == Color::White {
        fill(out, SIDE_TO_MOVE, 1.0);
    }

    let reps = gs.repetition_count();
    if reps >= 2 {
        fill(out, REPEAT_ONCE, 1.0);
    }
    if reps >= 3 {
        fill(out, REPEAT_TWICE, 1.0);
    }

    let progress = if gs.ply_limit == 0 {
        0.0
    } else {
        (gs.ply as f32 / gs.ply_limit as f32).min(1.0)
    };
    fill(out, PROGRESS, progress);
}

#[inline]
fn ray_of(dr: i32, df: i32) -> Option<(usize, usize)> {
    if dr == 0 && df == 0 {
        return None;
    }
    if dr != 0 && df != 0 && dr.abs() != df.abs() {
        return None;
    }
    let dist = dr.abs().max(df.abs());
    if dist as usize > MAX_RAY {
        return None;
    }
    let unit = (dr / dist, df / dist);
    RAY_DIRS
        .iter()
        .position(|&d| d == unit)
        .map(|i| (i, dist as usize))
}

pub fn move_to_index(m: Move, side_to_move: Color) -> Option<usize> {
    if m.is_null() {
        return None;
    }
    let flip = side_to_move == Color::Black;

    if m.is_drop() {
        let hi = hand_type_index(m.drop_piece_type())?;
        return Some((DROP_BASE + hi) * PLANE_SIZE + canon(m.to_sq(), flip));
    }

    let from = canon(m.from_sq(), flip);
    let to = canon(m.to_sq(), flip);
    let dr = sq_row(to) as i32 - sq_row(from) as i32;
    let df = sq_file(to) as i32 - sq_file(from) as i32;

    if let Some(pt) = m.promotion() {
        let di = PROMO_DIRS.iter().position(|&d| d == (dr, df))?;
        let pi = PROMOTION_PIECES.iter().position(|&p| p == pt)?;
        return Some((PROMO_BASE + di * PROMOTION_PIECES.len() + pi) * PLANE_SIZE + from);
    }

    if let Some(ki) = KNIGHT_OFFSETS.iter().position(|&d| d == (dr, df)) {
        return Some((KNIGHT_BASE + ki) * PLANE_SIZE + from);
    }

    let (dir, dist) = ray_of(dr, df)?;
    Some((RAY_BASE + dir * MAX_RAY + (dist - 1)) * PLANE_SIZE + from)
}

pub fn index_to_move(idx: usize, side_to_move: Color) -> Option<Move> {
    if idx >= ACTION_SPACE {
        return None;
    }
    let plane = idx / PLANE_SIZE;
    let csq = idx % PLANE_SIZE;
    let flip = side_to_move == Color::Black;

    if plane >= DROP_BASE {
        let pt = HAND_TYPES[plane - DROP_BASE];
        return Some(Move::new_drop(canon(csq, flip), pt, side_to_move));
    }

    let (dr, df, promo) = if plane >= PROMO_BASE {
        let p = plane - PROMO_BASE;
        let (dr, df) = PROMO_DIRS[p / PROMOTION_PIECES.len()];
        (dr, df, Some(PROMOTION_PIECES[p % PROMOTION_PIECES.len()]))
    } else if plane >= KNIGHT_BASE {
        let (dr, df) = KNIGHT_OFFSETS[plane - KNIGHT_BASE];
        (dr, df, None)
    } else {
        let (ur, uf) = RAY_DIRS[plane / MAX_RAY];
        let dist = (plane % MAX_RAY) as i32 + 1;
        (ur * dist, uf * dist, None)
    };

    let r = sq_row(csq) as i32 + dr;
    let f = sq_file(csq) as i32 + df;
    if !is_on_board(r, f) {
        return None;
    }
    let cto = sq(r as usize, f as usize);
    Some(Move::new_normal(canon(csq, flip), canon(cto, flip), promo))
}

pub fn legal_action_indices(gs: &mut GameState) -> Vec<usize> {
    let side = gs.current_turn;
    gs.get_legal_moves_vec()
        .iter()
        .filter_map(|&m| move_to_index(m, side))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashMap;

    #[test]
    fn action_space_is_2196() {
        assert_eq!(ACTION_PLANES, 61);
        assert_eq!(ACTION_SPACE, 2196);
        assert_eq!(INPUT_SIZE, 864);
        assert_eq!(DROP_BASE + DROP_PLANES, ACTION_PLANES);
    }

    #[test]
    fn plane_groups_do_not_overlap() {
        assert!(RAY_BASE < KNIGHT_BASE);
        assert!(KNIGHT_BASE < PROMO_BASE);
        assert!(PROMO_BASE < DROP_BASE);
        assert_eq!(KNIGHT_BASE, 40);
        assert_eq!(PROMO_BASE, 48);
        assert_eq!(DROP_BASE, 57);
    }

    #[test]
    fn ray_dirs_are_distinct_units() {
        for (i, &(dr, df)) in RAY_DIRS.iter().enumerate() {
            assert!(dr.abs() <= 1 && df.abs() <= 1 && (dr != 0 || df != 0));
            for (j, &other) in RAY_DIRS.iter().enumerate() {
                assert!(i == j || other != (dr, df), "duplicate ray direction");
            }
        }
    }

    #[test]
    fn initial_position_planes_are_sane() {
        let mut gs = GameState::new();
        gs.setup_initial_board();
        let planes = encode_position(&gs);
        assert_eq!(planes.len(), INPUT_SIZE);

        // White to move, no flip: own king on (5,0), opponent king on (0,5).
        assert_eq!(planes[(OWN_PIECES + 4) * PLANE_SIZE + sq(5, 0)], 1.0);
        assert_eq!(planes[(OPP_PIECES + 4) * PLANE_SIZE + sq(0, 5)], 1.0);
        assert_eq!(planes[SIDE_TO_MOVE * PLANE_SIZE], 1.0);
        // Ten men on the board, one plane each, nothing promoted, hands empty.
        let occupied: f32 = planes[..OWN_PROMOTED * PLANE_SIZE].iter().sum();
        assert_eq!(occupied, 10.0);
        let promoted: f32 = planes[OWN_PROMOTED * PLANE_SIZE..OWN_HANDS * PLANE_SIZE]
            .iter()
            .sum();
        assert_eq!(promoted, 0.0);
        let hands: f32 = planes[OWN_HANDS * PLANE_SIZE..SIDE_TO_MOVE * PLANE_SIZE]
            .iter()
            .sum();
        assert_eq!(hands, 0.0);
    }

    #[test]
    fn black_to_move_is_the_mirror_of_white_to_move() {
        let mut white = GameState::new();
        white.setup_initial_board();

        // The same material, colours swapped and the board rotated 180 degrees.
        let mut black = GameState::new();
        black.setup_initial_board();
        black.board = [Piece::Empty; NUM_SQUARES];
        for s in 0..NUM_SQUARES {
            let p = white.board[s];
            if p.is_empty() {
                continue;
            }
            let flipped = Piece::from_color_type(
                p.color().unwrap().opposite(),
                p.piece_type().unwrap(),
            );
            black.board[NUM_SQUARES - 1 - s] = flipped;
        }
        black.current_turn = Color::Black;
        black.find_kings();
        black.hash = black.compute_hash();
        black.position_history = vec![black.hash];

        let a = encode_position(&white);
        let b = encode_position(&black);
        // Everything but the side-to-move plane must land identically.
        for plane in 0..PLANES {
            if plane == SIDE_TO_MOVE {
                continue;
            }
            let lo = plane * PLANE_SIZE;
            assert_eq!(
                a[lo..lo + PLANE_SIZE],
                b[lo..lo + PLANE_SIZE],
                "plane {} differs under canonicalisation",
                plane
            );
        }
        assert_eq!(a[SIDE_TO_MOVE * PLANE_SIZE], 1.0);
        assert_eq!(b[SIDE_TO_MOVE * PLANE_SIZE], 0.0);
    }

    #[test]
    fn every_legal_move_round_trips_without_collisions() {
        let mut rng = StdRng::seed_from_u64(0x5EED);
        let mut positions = 0usize;
        let mut actions = 0usize;

        for game in 0..400 {
            let mut gs = GameState::new();
            gs.setup_initial_board();
            if game % 2 == 1 {
                gs.ply_limit = 60;
            }

            loop {
                let side = gs.current_turn;
                let moves = gs.get_legal_moves_vec();
                if moves.is_empty() || gs.is_terminal_draw() {
                    break;
                }

                // The planes must be well formed at every position we visit.
                let planes = encode_position(&gs);
                assert_eq!(planes.len(), INPUT_SIZE);
                assert!(planes.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));

                let mut seen: HashMap<usize, Move> = HashMap::new();
                for &m in &moves {
                    let idx = move_to_index(m, side)
                        .unwrap_or_else(|| panic!("no action index for {:?}", m));
                    assert!(idx < ACTION_SPACE);
                    if let Some(&prev) = seen.get(&idx) {
                        panic!("index {} collides: {:?} and {:?}", idx, prev, m);
                    }
                    seen.insert(idx, m);

                    let back = index_to_move(idx, side)
                        .unwrap_or_else(|| panic!("index {} decoded to nothing", idx));
                    assert_eq!(back, m, "round trip changed {:?} (index {})", m, idx);
                }
                positions += 1;
                actions += moves.len();

                let pick = moves[rng.gen_range(0..moves.len())];
                gs.make_ai_move(pick);
            }
        }

        assert!(positions > 10_000, "only sampled {} positions", positions);
        assert!(actions > 100_000, "only sampled {} actions", actions);
    }

    #[test]
    fn decoding_stays_on_the_board() {
        for &side in &[Color::White, Color::Black] {
            for idx in 0..ACTION_SPACE {
                if let Some(m) = index_to_move(idx, side) {
                    assert!(m.to_sq() < NUM_SQUARES);
                    if !m.is_drop() {
                        assert!(m.from_sq() < NUM_SQUARES);
                        assert_ne!(m.from_sq(), m.to_sq());
                    }
                    // Whatever decodes must encode back to the index it came from.
                    assert_eq!(move_to_index(m, side), Some(idx));
                }
            }
        }
    }
}
