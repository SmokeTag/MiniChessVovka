# Minihouse Zero: the neural engine

An AlphaZero-style policy–value network and MCTS for 6×6 crazyhouse, built on the Rust
core that already exists. Three decisions were settled at scoping and everything here
follows from them:

- **Success is beating the alpha-beta engine** at equal time control. Every checkpoint
  gates against depth-6 and depth-8 alpha-beta, not only against its own predecessor —
  measuring against your own past best is how a plateau disguises itself as progress.
  This repo is unusual in *having* an absolute yardstick; most reimplementations don't.
- **Bootstrap, then zero.** Pretrain on alpha-beta labels to prove the pipeline, then
  discard the weights and run pure zero on infrastructure that has already been trusted
  once. The bootstrap checkpoint inherits the hand-written eval's blind spots and is
  explicitly *not* "zero".
- **MCTS lives in Rust, the batch loop in Python.** Rust threads calling back into torch
  would serialise on the GIL and give up the parallelism they were spawned for.
  Inverting it — Rust hands out leaves, Python feeds them to the GPU — keeps one GIL
  holder doing useful work. Batching is across K concurrent games, not within one search.

| phase | | exit criterion |
| --- | --- | --- |
| 0 | draw rules in both `GameState`s | no random game fails to terminate |
| 1 | encoding (`encode.rs`) | 10k-position round-trip fuzz, zero collisions |
| 2 | network + teacher data | raw policy argmax beats random ≥95%, matches depth-2 |
| 3 | MCTS in Rust | 800 sims/move beats depth-6 alpha-beta over 200 games |
| 4 | self-play loop | three consecutive gate passes against the previous best |
| 5 | integration behind `ai.find_best_move` | GUI and bot play through the network |

Phases 0 and 1 are done; `docs/ENCODING.md` owns phase 1. This page owns phase 2 on.

## Dependencies

`requirements-nn.txt`, installed into **the same venv as everything else**:

```bash
./venv/bin/pip install -r requirements-nn.txt
```

A separate `venv-nn` was the obvious alternative and was rejected: it would mean running
`maturin develop` into two interpreters, and a stale `.so` importing silently is this
project's worst failure mode (CLAUDE.md opens with it). Doubling that hazard to save
disk is a bad trade. The isolation that actually matters — the GUI and the bot not
carrying 3 GB of CUDA — is a matter of **import discipline**, and is enforced rather than
hoped for: nothing outside `nn/` imports torch or numpy, `tests/test_nn.py` skips itself
when torch is absent, and the Rust extension returns plain lists precisely so that
`minichess_engine` never pulls numpy into a front end.

The pin is `torch==2.11.0+cu128`. cu128 is not optional on this machine: the RTX 5070 Ti
is Blackwell (sm_120) and will not run kernels from an older CUDA build. The `+cu128`
local version exists only on the PyTorch index, which makes the resolution unambiguous.

## Where things live

**Never the repo root.** `book.db` is there and `test_nightly.py` drops its tables;
CLAUDE.md generalises the rule to anything that persists. `nn/paths.py` is the single
answer: `$MINIZERO_DATA`, else `~/.local/share/minihouse-zero`, holding `teacher/<name>/`
and `checkpoints/<run>/`. `tests/test_nn.py` asserts the data root is not inside the
repository.

## Phase 2, part one: the teacher

```bash
./venv/bin/python -m nn.teacher generate --positions 50000 --jobs 20 --depth 8
./venv/bin/python -m nn.teacher stats
```

Each record is a position plus the depth-8 search of it: the best move is the policy
target, `tanh(score / 400)` in the mover's own frame is the value target.

**A record stores the position, not its encoding** — a FEN plus `ply` and `reps`, which
is exactly what `encode_position` reads. So a change to the plane layout costs a
re-encode (seconds) rather than a regeneration (hours). `teacher.restore()` is the
inverse, and the generator asserts on *every* position that it reproduces the live
encoding bit for bit. A FEN carries nothing path-dependent, so without that restoration
the progress and repetition planes would read one way in training and another in play —
a systematic skew that no loss curve would reveal.

**No search here touches `book.db`.** Every one runs under `use_book=False`, which gates
the probe *and* the store in `search.rs`. Labels must all come from one depth, and a book
row is whatever depth it happened to be searched at.

Three things about the cost, all measured, none of them what was assumed:

- **The walk is on the Rust `GameState`, not the Python one.** `gamestate.py` owns the
  live game, but walking it costs milliseconds a ply; the Rust one does ~126k plies/s.
  Driving the walk from Python made the *walk* the bottleneck and the generator ~15x
  slower than the labels it was feeding. The two implementations agree by construction
  (`tests/test_rules_parity.py`).
- **A depth-8 label is not a fixed cost.** ~0.35s in an opening, ~1.6s median in a deep
  full-hand position, with a tail past 9s — crazyhouse branching lives in the hands.
  `--time-limit` (default 2.0s) trims that tail; a capped search still returns the
  deepest iteration it completed.
- **Parallel efficiency is poor and that is the hardware.** 20 workers yield ~6
  positions/s, against ~1.1/s single-threaded — this is a 8 P-core + 16 E-core part, and
  the E-cores are much slower. Budget from the measured rate, not from core count.

Shards are JSONL, one per task, flushed **every record**: Python's text buffer holds tens
of kilobytes, so an unflushed multi-hour run is both unobservable and entirely lost if it
dies. Runs are resumable — a complete shard is skipped, a partial one is appended to with
its FENs preloaded, and the task rng replays the same walks so the dedup drops them
before the expensive search.

Positions come from fresh random walks of 2–80 plies, each walk drawing its own mix of
random and depth-3 engine moves so the sample spans noise and engine-like play rather
than sitting at either extreme. Positions with fewer than two legal moves are dropped: a
forced reply is answered without a search, so it carries no score, and a one-hot policy
over a single legal move teaches nothing.

## Phase 2, part two: the network

`nn/model.py`. 24→64 channels, 6 residual blocks, two heads, **480,910 parameters**.

The policy head is a 1×1 convolution to 61 planes **and nothing else**. `flatten(1)` on
its `(N, 61, 6, 6)` output lands on `plane * 36 + r * 6 + f`, which *is* the action index
— so the head cannot get the mapping wrong, because it does not contain the mapping. The
dense alternative is 5,328 outputs and ~12M parameters, larger than the whole trunk.
`test_policy_flatten_lands_on_the_action_index` pins the one assumption this rests on.

The value head is a single `tanh` in the frame of the side to move: +1 means "the player
to move wins". The input is canonicalised the same way, so the network never has to
represent which colour it is.

**Checkpoints carry the encoding they were trained against** and refuse to load against a
different one. A plane-layout change otherwise loads cleanly, runs cleanly, and plays
nonsense.

```bash
./venv/bin/python -m nn.train --epochs 40
```

Masked cross-entropy on the policy, MSE on the value, L2 from AdamW. The headline
metric is **policy top-1 on held-out positions** — how often the raw network, with no
search at all, names the move a depth-8 search does. Value MAE is watched alongside it:
a good policy with an uncalibrated value makes a bad MCTS, and the value head is the half
a bootstrap run is most likely to get quietly wrong. The train/val split is by FEN hash,
so it is reproducible across regenerations and keeps a position in the same half when the
set grows.

## Free data: file mirroring

Reflecting the board across the file axis, `(r, f) -> (r, 5 - f)`, is an **exact
symmetry** of this variant. It holds because there is no castling and no en passant --
the two things that break left-right symmetry in ordinary chess -- and because every
offset set in `types.rs` is closed under `df -> -df`. Pawns move along ranks, so pushes
are untouched and the two captures swap; drops mirror trivially and their one
restriction is on the rank; promotions keep their rank and swap capture-left with
capture-right.

**One permutation serves both colours.** The action index lives in the canonical frame,
which for Black is the board rotated 180 degrees. Mirroring the board and then
canonicalising gives the file flip of the canonical square, so the rotation and the
reflection commute. `nn/symmetry.py` builds the 2,196-entry permutation once from the
engine's own `action_index_to_move`/`move_to_action_index`, and `tests/test_nn.py`
fuzzes it against the move generator rather than trusting the argument -- including a
Black-to-move case, which is where the commuting claim would break.

Augmentation is applied per batch (a view plus a gather) rather than by materialising a
doubled array. It is worth having: on the same 48k positions it moved val top-1 from
**0.439 to 0.473**, top-5 from 0.791 to 0.826, and cut val policy loss from 4.10 to 3.01
-- most of that is reduced overfitting, which is the binding constraint at this data size.

## Calibrating the value target

The obvious `tanh(score / 400)` was wrong here, and wrong in a way that is invisible in
the loss curve. **Positions reached by walking with weak moves are wildly unbalanced**:
the median non-mate `|score|` in the teacher set is ~1100cp, over ten pawns, and 27% of
positions are already decided. At a scale of 400 that put **79% of all value targets past
|0.9|**, so the head learned to answer +-1 almost everywhere and could not rank two
candidate moves at all.

The symptom was concrete: one ply of value lookahead scored the *same* as no lookahead.
`VALUE_SCALE = 2000` brings non-mate saturation to 6.7%, leaving the mass at +-1 to the
positions that really are decided, and moved value MAE to 0.198 and sign agreement to
0.921.

**This cost nothing to fix**, because a record stores the score rather than the tensor --
the whole reason for that indirection. `nn.teacher stats` reports the saturated fraction;
measure before changing the constant. The scale is stamped into every checkpoint, since
the value head's output is only interpretable against the scale its targets were built
with, and phase 3 reads values back out of it.

Note that engine-weighting the walks did **not** fix the imbalance -- the engine-guided
half of the set has a median non-mate `|score|` of 1064 against the random half's 1038.
Depth-3 walk moves are not strong enough to keep a crazyhouse game level.

## Phase 2, part three: the bar

```bash
./venv/bin/python -m nn.arena --opponent random    --games 200
./venv/bin/python -m nn.arena --opponent alphabeta --depth 2 --games 200
```

The network plays with **no search at all** — one forward pass, mask the illegal actions,
argmax. That is the point of the phase: it isolates whether the encoding and the policy
head learned anything, with no tree to cover for them.

Both players are deterministic, so every game would otherwise be the same game. Openings
are short random legal walks, and **every opening is played twice, once with each
colour** — without the swap the result would measure the opening's bias as much as the
players'. The score is wins plus half draws with a binomial standard error.

An illegal or rejected move forfeits the game and is reported as an anomaly rather than
being quietly replaced. A masked argmax cannot pick an illegal action, so an anomaly is a
bug in the move/index map, not a weak player.

### What the bar actually costs

**Against random the criterion is met: 0.970 (+188 =12 -0)** at 47% val top-1. All 12
draws are conversion failures -- 10 stalemates, 2 repetitions -- a searchless policy
winning material and then failing to finish. Depth-2, by contrast, beats random 100-0
with no draws at all.

**Against depth-2 it is not, and the reason is not a bug.** Feed a synthetic player the
depth-8 move with probability p and a uniform random legal move otherwise:

| depth-8 move played | score vs depth-2 |
| --- | --- |
| 31% | 0.000 |
| 47% | 0.050 |
| 70% | 0.237 |
| 100% | 0.912 |

The network scores 0.077-0.125, which is *at or slightly above* the synthetic curve for
its own agreement rate (47% on held-out teacher positions, 31% on the positions it
reaches in its own games) -- so its mistakes are already better than random substitutions
and it is performing exactly as its accuracy predicts.

What the table says is that **top-1 agreement converts to playing strength brutally
non-linearly in this variant**: roughly 90%+ agreement with depth-8 is needed to match a
2-ply search. Crazyhouse punishes a single hanging piece immediately, because the
opponent captures it and drops it straight back, so one blunder in a 28-ply game is
usually the game. Reaching that from a searchless 0.5M-parameter policy is not a data
problem, and more teacher data will not get there: the phase-2 exit criterion as written
was optimistic about what a raw policy can do here.

That is an argument for phase 3, not against the network. Search is what converts a
policy that is right half the time into a player that does not hang pieces, and MCTS is
the designed remedy. Note also that **pure value-greedy is not that remedy** -- one ply
of value lookahead scored *worse* than the raw policy (0.060 vs 0.125), because it throws
the policy away entirely. PUCT uses the policy as a prior and the value only at leaves,
which is why it needs both heads and neither alone.

## Playing it: the phase-5 seam, early

`nn/backend.py` gives the network the same contract as `ai.find_best_move_with_score` --
`(move, white_relative_score)` -- and the GUI's **Engine** button selects it. That is
phase 5 in its smallest form, landed early because being able to sit down and play the
thing is worth more than its ordering in the plan. What it is not yet: the chess.com bot
still always uses the search, and there is no MCTS behind it.

See `docs/GUI.md` for the control and the invariants it has to respect -- chiefly that
importing the GUI must not import torch, which is checked in a subprocess.

## Risks still live

| risk | mitigation |
| --- | --- |
| draw-heavy self-play starves the value head | random play already draws 60%, and a network that has learnt not to lose will draw more. Watch the rate from the first self-play run; if it dominates, raise the ply cap or adjudicate on material rather than scoring 0 |
| a silent encoding bug | the phase-1 fuzz, the per-record restore assertion, and the phase-2 exit criterion — a network that cannot beat random says so immediately |
| teacher labels inherit the eval's blind spots | by design, and the reason the weights are discarded before the zero run |
