"""Teacher records -> the arrays a training step consumes.

A record stores a position, not its encoding (see nn/teacher.py), so this is where the
encoder is actually applied. That indirection is the point: re-encoding 150k records is
seconds, regenerating them is an hour and a half of 20 cores.

Three targets come out of it:

    x       (N, 24, 6, 6) float32   the position, canonicalised to the side to move
    policy  (N,)          int64     the action index of the depth-8 best move
    value   (N,)          float32   tanh(score / 400), in the mover's own frame

plus the legal action indices, kept ragged. A dense mask would be 2,196 bytes a
position against a mean of ~22 legal moves; `masks_from_ragged` rebuilds a batch's slab
on demand.

The split is by FEN hash, not by position order or a shuffle seed. Shards are generated
by independent workers walking independent games, so order carries no structure -- but a
hash split is reproducible across regenerations and across machines, and it keeps a
position in the same half when the set grows.
"""
import hashlib
import os
import time

import numpy as np

from nn import features, paths, symmetry, teacher

VAL_FRACTION = 0.05

def _is_val(fen, fraction=VAL_FRACTION):
    digest = hashlib.blake2b(fen.encode(), digest_size=8).digest()
    return (int.from_bytes(digest, "big") % 10_000) < fraction * 10_000

def encode_records(records, progress_every=25_000):
    """Records -> (x, policy, value, ragged legal indices).

    Every record is rebuilt through `teacher.restore`, which puts back the `ply` and
    repetition state a FEN cannot carry. The generator already asserted that this
    reproduces the live encoding; here it is simply the only way to get the tensor.
    """
    import minichess_engine as rs

    n = len(records)
    x = np.empty((n,) + features.INPUT_SHAPE, dtype=np.float32)
    policy = np.empty(n, dtype=np.int64)
    value = np.empty(n, dtype=np.float32)
    legal = []

    started = time.time()
    for i, rec in enumerate(records):
        gs = teacher.restore(rec)
        x[i] = features.encode(gs)
        policy[i] = rs.move_to_action_index(gs, teacher.decode_move(rec["move"]))
        value[i] = teacher.value_target(rec["score"], rec["turn"])
        legal.append(np.asarray(rs.legal_action_indices(gs), dtype=np.int32))

        if progress_every and (i + 1) % progress_every == 0:
            print("  encoded %d/%d (%.0fs)" % (i + 1, n, time.time() - started), flush=True)

    return x, policy, value, legal

class Split:
    def __init__(self, x, policy, value, legal):
        self.x = x
        self.policy = policy
        self.value = value
        self.legal = legal

    def __len__(self):
        return len(self.policy)

    def batches(self, batch_size, rng=None, shuffle=True, augment=False):
        """Iterate the split. `augment` mirrors a random half of each batch.

        File mirroring is an exact symmetry here (nn/symmetry.py), so a mirrored sample
        is a real second example rather than noise. Applied per batch instead of by
        materialising a doubled array: the flip is a view and the index remap is a
        gather, both far cheaper than another 500MB of planes.
        """
        rng = rng or np.random
        order = np.arange(len(self))
        if shuffle:
            rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            x = self.x[idx]
            policy = self.policy[idx]
            legal = [self.legal[j] for j in idx]

            if augment:
                flip = rng.random(len(idx)) < 0.5
                if flip.any():
                    x = x.copy()
                    x[flip] = symmetry.mirror_planes(x[flip])
                    policy = policy.copy()
                    policy[flip] = symmetry.mirror_actions(policy[flip])
                    legal = [symmetry.mirror_actions(l) if f else l
                             for l, f in zip(legal, flip)]

            yield x, policy, self.value[idx], features.masks_from_ragged(legal)

def load(name="depth8", limit=None, val_fraction=VAL_FRACTION):
    records = teacher.load(name, limit=limit)
    if not records:
        raise SystemExit("teacher/%s is empty" % name)

    print("dataset: %d unique positions from teacher/%s" % (len(records), name))
    x, policy, value, legal = encode_records(records)

    val = np.array([_is_val(r["fen"], val_fraction) for r in records])
    train_idx = np.flatnonzero(~val)
    val_idx = np.flatnonzero(val)

    def take(idx):
        return Split(x[idx], policy[idx], value[idx], [legal[j] for j in idx])

    train, valid = take(train_idx), take(val_idx)
    print("         train %d, val %d (%.1f%%)"
          % (len(train), len(valid), 100.0 * len(valid) / len(records)))
    return train, valid

def describe(split, name):
    """The two sanity checks worth printing before a run consumes an hour of GPU.

    A policy target outside its own legal set means the encoder and the generator
    disagree, and every gradient after that is noise.
    """
    legal_sizes = np.array([len(l) for l in split.legal])
    in_legal = sum(1 for i in range(len(split)) if split.policy[i] in split.legal[i])
    print("  %-5s n=%d  legal mean %.1f  value mean %+.3f sd %.3f  target-in-legal %d/%d"
          % (name, len(split), legal_sizes.mean(), split.value.mean(), split.value.std(),
             in_legal, len(split)))
    if in_legal != len(split):
        raise SystemExit("policy targets outside their legal set -- encoder/generator disagree")
