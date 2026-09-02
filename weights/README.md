# Milestone network weights

Checkpoints worth being able to return to, tracked in git so they survive a retrain, a
cleaned-up data directory, or a lost machine.

| tag | what it is | val top-1 | vs random | vs depth-2 |
| --- | --- | --- | --- | --- |
| `v0.1-first-playable-net` | the first net a human played | 0.465 | 0.970 | ~0.08 |
| `v0.2-bootstrap-150k` | end of phase 2 | 0.508 | 0.985 | 0.185 |

Each directory carries `best.pt`, the training `history.json`, a `MANIFEST.md` generated
from the checkpoint's own metadata (so no number here is hand-typed), and `SHA256SUMS`.
The weight files are mode 444, so an accidental overwrite fails loudly instead of quietly
replacing an artefact nobody can regenerate.

## Restoring one

```bash
export MINIZERO_CHECKPOINT=$PWD/weights/v0.2-bootstrap-150k/best.pt
./play.sh
```

`nn/backend.py` otherwise plays the newest `best.pt` under
`$MINIZERO_DATA/checkpoints/*/`, which is convenient day to day — training a new network
is enough to make the GUI play it — and is exactly why milestones need somewhere else to
live. Every training run silently takes over the GUI, and a rerun under the same run name
overwrites the file outright.

## What goes here, and what does not

**Milestones only.** A checkpoint earns a directory here when someone wants to come back
to it: the first playable network, the end of a phase, a checkpoint that beat a gate.
Ordinary training output stays in `$MINIZERO_DATA/checkpoints/` and is expected to be
overwritten.

At ~1.9 MB a checkpoint this is plain git rather than LFS — a handful of milestones is
noise against the repository, and LFS would add a setup step to every clone for no gain
at this size. If this directory ever holds dozens of checkpoints, that trade has changed
and the answer is to prune it, not to grow it.

**A checkpoint is only loadable against the encoding it was trained with.** Every one
stamps the plane count and action space, and `nn/model.load_checkpoint` refuses a
mismatch rather than loading weights that would play nonsense. A change to
`engine_rs/src/encode.rs` invalidates everything here; that is the point of the stamp.

## A note on the eval readouts

Both networks carry a side-to-move bias inherited from their teacher set and neither
eval is a trustworthy position assessment — v0.1 reads +395cp at the opening position and
v0.2 +178cp, where a depth-8 search says +6. Read the moves, not the number.
`docs/ZERO.md` explains where the bias comes from.
