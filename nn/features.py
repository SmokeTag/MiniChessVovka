"""numpy views of what engine_rs/src/encode.rs produces.

The Rust side deliberately hands back plain Python lists rather than numpy arrays, so
that `minichess_engine` never pulls numpy into the GUI's or the bot's import path (see
docs/ENCODING.md). This module is where that choice is paid for -- one reshape, inside
the only package allowed to depend on the scientific stack.

Nothing here decides anything about the encoding. The plane layout, the action map and
the canonical flip all live in Rust and are held by the fuzz in tests/test_encoding.py.
"""
import numpy as np
import minichess_engine as rs

PLANES = rs.ENCODE_PLANES
BOARD = rs.BOARD_SIZE
ACTION_SPACE = rs.ACTION_SPACE
INPUT_SHAPE = (PLANES, BOARD, BOARD)

def encode(gs):
    """A Rust GameState -> (24, 6, 6) float32.

    `encode_position` returns `plane * 36 + r * 6 + f`, so the reshape is the identity
    on the layout and a policy conv's `.flatten(1)` lands on the same action index.
    """
    return np.asarray(rs.encode_position(gs), dtype=np.float32).reshape(INPUT_SHAPE)

def encode_batch(states):
    out = np.empty((len(states),) + INPUT_SHAPE, dtype=np.float32)
    for i, gs in enumerate(states):
        out[i] = encode(gs)
    return out

def legal_indices(gs):
    return np.asarray(rs.legal_action_indices(gs), dtype=np.int64)

def legal_mask(gs):
    mask = np.zeros(ACTION_SPACE, dtype=bool)
    mask[legal_indices(gs)] = True
    return mask

def masks_from_ragged(index_lists, out=None):
    """Ragged per-position legal indices -> a dense (N, 2196) bool mask.

    Storing masks densely for a training set costs 2,196 bytes a position against ~22
    legal moves; the ragged form is two orders of magnitude smaller and this rebuilds a
    batch's slab in microseconds.
    """
    n = len(index_lists)
    if out is None:
        out = np.zeros((n, ACTION_SPACE), dtype=bool)
    else:
        out[:n].fill(False)
    for i, idx in enumerate(index_lists):
        out[i, idx] = True
    return out[:n]
