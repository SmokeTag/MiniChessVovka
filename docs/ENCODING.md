# The network encoding: 24 planes in, 2,196 actions out

`engine_rs/src/encode.rs` is phase 1 of Minihouse Zero — the translation layer between a
`GameState` and the tensors a policy–value network reads and writes. It has no opinions about
chess; it is pure bookkeeping, and it exists as its own module because a bug in it is close to
undebuggable. A scrambled board or a permuted policy still trains to a plausible-looking loss
curve, and the failure surfaces a week later as a network that never got stronger. Everything
here is therefore held by a round-trip fuzz rather than by inspection.

Nothing in this module allocates a network, imports torch, or depends on a new crate. It is
reachable from the existing extension module and costs the GUI and the bot nothing.

## Canonicalisation

**Every tensor is in the frame of the side to move.** When Black is to move the board is rotated
180° and the colours are swapped, so the network only ever sees a position from the mover's side,
always moving *toward row 0*.

The rotation is `(r, f) → (5 - r, 5 - f)`, which on the flat square index is just `35 - s`. It is
exact for this variant: there is no castling and no en passant, so no asymmetric state survives
the flip. Pawn direction is the only thing that would break it, and rotating fixes precisely
that — Black's `+1` row step becomes White's `-1`.

Canonicalisation is applied to the *squares*, before any direction arithmetic. That is what makes
the action map fall out for free: flip both endpoints and the direction negates itself.

## Input: 24 planes at 6×6

Row-major within a plane, plane-major overall — element `plane * 36 + r * 6 + f`, 864 floats,
every one in `[0, 1]`.

| planes | # | contents |
| --- | --- | --- |
| 0–4 | 5 | own pieces, one-hot per square, in `PieceType` order **P, N, B, R, K** |
| 5–9 | 5 | opponent pieces, same order |
| 10 | 1 | own promoted pieces |
| 11 | 1 | opponent promoted pieces |
| 12–15 | 4 | own hand counts **P, N, B, R**, broadcast over the plane, `count / 8` |
| 16–19 | 4 | opponent hand counts, same |
| 20 | 1 | side to move — `1.0` if White is really to move |
| 21 | 1 | this position has occurred once before |
| 22 | 1 | …twice before |
| 23 | 1 | `ply / ply_limit` |

**There is no queen plane, and there never will be.** The initial array holds none
(`2bnrk/5p/6/6/P5/KRNB2`) and `PROMOTION_PIECES` is `[R, N, B]`, so a queen cannot arise; a queen
on the board or in a hand is skipped by the encoder rather than encoded. `HAND_PIECE_TYPES` still
carries a queen slot because the hands array is shaped by `PieceType`, which is why the hand
planes are 4 wide and not 5.

**The promoted planes are not decoration.** In Crazyhouse an ex-pawn reverts to a pawn when
captured, so a promoted rook is worth materially less than a real one. That fact is invisible in
the piece planes and is exactly the sort of thing a value head will otherwise learn wrong.

**Hands are normalised by 8**, which is the total non-king material in the game (R, N, B, P per
side), and clamped. No single hand count can exceed it.

**The side-to-move plane is redundant** under canonicalisation and is kept anyway: it costs one
plane and it is the assertion that says which flip was applied. `test_black_to_move_mirrors_white_to_move`
builds the mirror of the opening by hand and requires all 23 other planes to land identically —
a scrambled flip cannot hide behind a symmetric position.

**The repetition planes use `repetition_count()`**, the same counter `is_terminal_draw()` reads,
and `position_history` includes the current position — so "seen once before" is `count >= 2` and
"seen twice" is `count >= 3`, matching the threefold rule exactly. See `docs/SEARCH.md` for what
that history holds. There is no history stack of past positions: AlphaZero carries eight largely
to handle repetition and en passant, and here the repetition planes cover it for 1/13th of the
input volume. It can be added if the value head plateaus.

## Output: 61 action planes, 2,196 logits

A fully convolutional policy head — a 1×1 convolution from the trunk to 61 planes over the 6×6
board. The action index is `plane * 36 + square`.

| planes | # | encodes | indexed by |
| --- | --- | --- | --- |
| 0–39 | 40 | rays: 8 directions × 5 distances | origin |
| 40–47 | 8 | the eight knight jumps | origin |
| 48–56 | 9 | promotions: 3 directions × 3 pieces (R, N, B) | origin |
| 57–60 | 4 | drops: P, N, B, R | **target** |

- **Rays** cover rook, bishop, king and ordinary pawn moves in one group — everything that is a
  unit direction times a distance. `RAY_DIRS` starts at "forward" `(-1, 0)` and goes clockwise;
  the plane is `dir * 5 + (dist - 1)`. Five distances because 5 is the longest move on a 6×6 board.
- **Knights** use `KNIGHT_OFFSETS` from `types.rs` in its declared order. No knight offset is a
  ray, so the two groups cannot be confused.
- **Promotions** get their own planes rather than a flag on a ray, because with no queen the
  choice among R, N and B is a real one the policy has to express. The three directions are
  forward, capture-toward-file-0, capture-toward-file-5, all in the canonical frame where the
  mover advances toward row 0.
- **Drops are indexed by where the piece lands**, not where it came from — a piece in hand has no
  origin square. That is the whole reason they fit the same convolutional head.

The alternative encoding — a flat from × to × promotion table plus drops — is 5,328 outputs and
needs a ~12M-parameter dense layer, larger than the entire trunk. The plane form costs about 4,000
weights, in exchange for the index arithmetic above.

Illegal actions are masked to −∞ before the softmax, at search time and at training time.
`legal_action_indices()` produces the mask.

## The two invariants

Both are properties of a *position*, not of the encoder in isolation:

1. **Injective within a position.** No two legal moves of the same position share an index. Two
   moves sharing one means the search can never tell them apart and the policy target is a lie.
   It holds because origin square plus direction plus distance determines a unique piece and
   destination, promotions live in disjoint planes, and drops are keyed on target plus type.
2. **Exactly invertible.** `index_to_move(move_to_index(m)) == m`, so a visit distribution over
   indices reads back as moves.

`index_to_move` decodes *structurally* and does not know the position — it returns `None` only
when the plane and square imply a destination off the board. It can name a move that is not legal;
that is what the mask is for.

## Testing it

The exit criterion for phase 1 is a 10,000-position round-trip fuzz with zero collisions. It runs
on both sides of the boundary, and both are fast enough to leave at their defaults.

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --release   # inside engine_rs/
./venv/bin/python -m pytest tests/test_encoding.py -q
ENCODING_FUZZ_GAMES=1000 ./venv/bin/python -m pytest tests/test_encoding.py -q
```

`tests/test_encoding.py` walks ~22,000 positions of random games at its default 200 games and
round-trips every legal move of every one — ~480,000 actions, including ~111,000 drops and ~2,900
promotions. The counters are asserted too: a sweep that never reaches a drop, a promotion, a
promoted piece on the board or a Black-to-move position proves nothing, so
`test_the_fuzz_reaches_the_hard_cases` fails if coverage collapses.

`test_the_python_gamestate_encodes_the_same_position` runs the smaller sweep that matters for the
front ends: through `gamestate.py` and `ai._sync_to_rust`, checking that the mask the encoder
builds agrees with the Python move list. The Rust suite carries the same fuzz over 400 games plus
a full sweep of all 2,196 indices in both frames.

## The Python surface

```python
import minichess_engine as rs

rs.encode_position(gs)              # 864 floats, plane-major
rs.move_to_action_index(gs, move)   # ValueError if the move has no index
rs.action_index_to_move(gs, index)  # move tuple, or None if it leaves the board
rs.legal_action_indices(gs)         # the mask, aligned with get_all_legal_moves()

rs.ENCODE_PLANES      # 24
rs.ENCODE_INPUT_SIZE  # 864
rs.ACTION_PLANES      # 61
rs.ACTION_SPACE       # 2196
```

All four take the **Rust** `GameState`; from a Python one, go through `ai._sync_to_rust`. Each
reads `current_turn` to decide the flip, so the same move gets a different index depending on
whose turn it is — that is the point, and it means an index is only meaningful alongside the
position it came from.

A flat `Vec<f32>` rather than a numpy array is deliberate: it keeps `numpy` out of the extension's
build and out of the GUI's and the bot's import path. Phase 3 batches these across K games, where
one copy of 512 × 864 floats is nothing against a forward pass; if that stops being true, the
`numpy` crate pairs with PyO3 0.24 and the change is local to `lib.rs`.
