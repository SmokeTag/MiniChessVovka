
use crate::gamestate::GameState;
use crate::types::*;

const HAND_ORDER: [(PieceType, char); 5] = [
    (PieceType::Pawn, 'P'),
    (PieceType::Knight, 'N'),
    (PieceType::Bishop, 'B'),
    (PieceType::Rook, 'R'),
    (PieceType::Queen, 'Q'),
];

const MAX_HAND_COUNT: u8 = 7;

pub fn to_fen(gs: &GameState) -> String {
    let mut out = String::with_capacity(48);

    for r in 0..BOARD_SIZE {
        if r > 0 {
            out.push('/');
        }
        let mut empty = 0;
        for f in 0..BOARD_SIZE {
            let s = sq(r, f);
            let piece = gs.board[s];
            if piece.is_empty() {
                empty += 1;
                continue;
            }
            if empty > 0 {
                out.push_str(&empty.to_string());
                empty = 0;
            }
            out.push(piece.to_char());
            if gs.promoted_pieces & (1u64 << s) != 0 {
                out.push('~');
            }
        }
        if empty > 0 {
            out.push_str(&empty.to_string());
        }
    }

    out.push('[');
    for (ci, upper) in [(0usize, true), (1usize, false)] {
        for (pt, ch) in HAND_ORDER {
            let count = gs.hands[ci][pt.index()];
            let ch = if upper { ch } else { ch.to_ascii_lowercase() };
            for _ in 0..count {
                out.push(ch);
            }
        }
    }
    out.push(']');

    out.push(' ');
    out.push(match gs.current_turn {
        Color::White => 'w',
        Color::Black => 'b',
    });

    out
}

pub fn from_fen(fen: &str) -> Result<GameState, String> {
    let mut fields = fen.split_whitespace();
    let placement = fields
        .next()
        .ok_or_else(|| "empty FEN".to_string())?;
    let turn_field = fields
        .next()
        .ok_or_else(|| "FEN is missing the side-to-move field".to_string())?;
    if let Some(extra) = fields.next() {
        return Err(format!("unexpected trailing field in FEN: {:?}", extra));
    }

    let (board_part, hand_part) = match placement.find('[') {
        Some(i) => {
            if !placement.ends_with(']') {
                return Err("unterminated hand: '[' with no closing ']'".to_string());
            }
            (&placement[..i], &placement[i + 1..placement.len() - 1])
        }
        None => {
            if placement.contains(']') {
                return Err("closing ']' with no opening '['".to_string());
            }
            (placement, "")
        }
    };

    let mut gs = GameState::new();

    let rows: Vec<&str> = board_part.split('/').collect();
    if rows.len() != BOARD_SIZE {
        return Err(format!(
            "FEN has {} ranks, expected {}",
            rows.len(),
            BOARD_SIZE
        ));
    }

    for (r, row) in rows.iter().enumerate() {
        let mut f = 0usize;
        let mut last_sq: Option<usize> = None;
        for c in row.chars() {
            match c {
                '1'..='9' => {
                    let n = c.to_digit(10).unwrap() as usize;
                    f += n;
                    if f > BOARD_SIZE {
                        return Err(format!("rank {} overflows the board", BOARD_SIZE - r));
                    }
                    last_sq = None;
                }
                '~' => {
                    let s = last_sq.ok_or_else(|| {
                        format!("'~' on rank {} does not follow a piece", BOARD_SIZE - r)
                    })?;
                    gs.promoted_pieces |= 1u64 << s;
                }
                _ => {
                    let piece = Piece::from_char(c);
                    if piece.is_empty() {
                        return Err(format!("unknown piece character {:?} in FEN", c));
                    }
                    if f >= BOARD_SIZE {
                        return Err(format!("rank {} overflows the board", BOARD_SIZE - r));
                    }
                    let s = sq(r, f);
                    gs.board[s] = piece;
                    last_sq = Some(s);
                    f += 1;
                }
            }
        }
        if f != BOARD_SIZE {
            return Err(format!(
                "rank {} covers {} files, expected {}",
                BOARD_SIZE - r,
                f,
                BOARD_SIZE
            ));
        }
    }

    for c in hand_part.chars() {
        let piece = Piece::from_char(c);
        let (Some(color), Some(pt)) = (piece.color(), piece.piece_type()) else {
            return Err(format!("unknown piece character {:?} in hand", c));
        };
        if pt == PieceType::King {
            return Err("a king cannot be in hand".to_string());
        }
        let slot = &mut gs.hands[color.index()][pt.index()];
        if *slot >= MAX_HAND_COUNT {
            return Err(format!(
                "more than {} {:?} in hand: the Zobrist hand table saturates there",
                MAX_HAND_COUNT, pt
            ));
        }
        *slot += 1;
    }

    gs.current_turn = match turn_field {
        "w" => Color::White,
        "b" => Color::Black,
        other => return Err(format!("side to move must be 'w' or 'b', got {:?}", other)),
    };

    for (color, piece) in [
        (Color::White, Piece::WhiteKing),
        (Color::Black, Piece::BlackKing),
    ] {
        let n = gs.board.iter().filter(|&&p| p == piece).count();
        if n != 1 {
            return Err(format!("FEN has {} {:?} kings, expected exactly 1", n, color));
        }
    }

    for s in 0..NUM_SQUARES {
        if gs.promoted_pieces & (1u64 << s) != 0 && gs.board[s].is_empty() {
            return Err(format!("'~' marks empty square {}", s));
        }
    }

    gs.find_kings();
    gs.hash = gs.compute_hash();
    gs.position_history.clear();
    gs.position_history.push(gs.hash);
    Ok(gs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_position_round_trips() {
        let mut gs = GameState::new();
        gs.setup_initial_board();
        let fen = to_fen(&gs);
        let back = from_fen(&fen).expect("initial position must parse");
        assert_eq!(back.hash, gs.hash);
        assert_eq!(to_fen(&back), fen);
    }

    #[test]
    fn hands_and_promotions_survive() {
        let mut gs = GameState::new();
        gs.setup_initial_board();
        gs.hands[0][PieceType::Pawn.index()] = 2;
        gs.hands[1][PieceType::Rook.index()] = 1;
        gs.promoted_pieces |= 1u64 << sq(5, 1);
        gs.current_turn = Color::Black;
        gs.hash = gs.compute_hash();

        let fen = to_fen(&gs);
        assert!(fen.contains("[PPr]"), "unexpected hand encoding: {}", fen);
        let back = from_fen(&fen).unwrap();
        assert_eq!(back.hash, gs.hash);
        assert_eq!(back.hands, gs.hands);
        assert_eq!(back.promoted_pieces, gs.promoted_pieces);
    }

    #[test]
    fn malformed_input_is_rejected() {
        for bad in [
            "",
            "6/6/6/6/6/6 w",
            "bnrk2/5p/6/6/P5/KRNB2 x",
            "bnrk2/5p/6/6/P5 w",
            "bnrk2/5p/6/6/P5/KRNB3 w",
            "bnrk2/5p/6/6/P5/KRNB2[Pk] w",
            "~bnrk2/5p/6/6/P5/KRNB2 w",
        ] {
            assert!(from_fen(bad).is_err(), "should have rejected {:?}", bad);
        }
    }
}
