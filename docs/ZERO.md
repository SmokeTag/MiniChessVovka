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

**Size the schedule to the overfit knee; do not early-stop past it.** At 139k positions
val top-1 peaks around epoch 13-14 and declines after -- by epoch 60 the val policy loss
has climbed from 1.57 to 3.68 while the train loss is still falling. Running a 60-epoch
cosine and keeping the best checkpoint gives 0.497 top-1 at a value MAE of 0.199, because
the peak lands while the learning rate is still near its maximum and the value head has
not settled. A **20**-epoch cosine anneals into the same knee and beats it on both heads
at once: **0.508 top-1, 0.170 value MAE**. It also removes a real hazard in checkpoint
selection -- under the longer schedule the best-policy epoch was *not* the best-value
epoch, so `best.pt` handed the GUI a good policy with a weak eval readout. The knee moves
with the data; re-measure if the teacher set grows.

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

**The value head inherits a side-to-move bias, and the GUI's eval readout shows it.**
The teacher set's mean value target is **+0.194**: a walk that reaches an interesting
position usually got there because somebody just blundered, so the side to move is
genuinely better on average. The head learns that prior. At the opening position the
shipped bootstrap net reads **+178cp** for White where depth-8 says +6 (v0.1, trained
before the scale fix, read +395). The moves are worth more than the score -- do not
present this eval as a position assessment.

It is not worth chasing with the teacher: the bias is a property of walked positions, and
phase 4 self-play visits balanced ones by construction. Worth re-measuring there rather
than correcting here.

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

Measured on the shipped bootstrap checkpoint: 139,301 unique teacher positions, val
top-1 **0.508**, top-5 0.876, value MAE 0.170, value sign agreement 0.940.

**Against random the criterion is met: 0.985 (+194 =6 -0) over 200 games.** It never
loses. All six draws are conversion failures -- stalemates, a searchless policy winning
material and then failing to finish. Depth-2, by contrast, beats random 100-0 with no
draws at all.

**Against depth-2 it is not, and the reason is not a bug.** Feed a synthetic player the
depth-8 move with probability p and a uniform random legal move otherwise:

| depth-8 move played | score vs depth-2 |
| --- | --- |
| 31% | 0.000 |
| 47% | 0.050 |
| 70% | 0.237 |
| 100% | 0.912 |

**The network scores 0.185** (+34 =6 -160) at 51% top-1, which is well *above* the
synthetic curve there: interpolating the table gives ~0.07 for a player that picks the
depth-8 move half the time and a uniform random one otherwise. So its mistakes are much
better than random substitutions -- when it is not playing the best move it is usually
playing a reasonable one -- and it performs better than its raw agreement rate predicts,
not worse.

For scale, tripling the teacher set from 48k to 139k moved top-1 from 0.473 to 0.508 and
the depth-2 score from 0.077 to 0.185. That gain is real, and it is also the shape of the
problem: the remaining distance to 0.5 is not another tripling away.

What the table says is that **top-1 agreement converts to playing strength brutally
non-linearly in this variant**: roughly 90%+ agreement with depth-8 is needed to match a
2-ply search. Crazyhouse punishes a single hanging piece immediately, because the
opponent captures it and drops it straight back, so one blunder in a 28-ply game is
usually the game. Reaching that from a searchless 0.5M-parameter policy is not a data
problem, and more teacher data will not get there: the phase-2 exit criterion as written
was optimistic about what a raw policy can do here.

**Where phase 2 ended up: the random half of the criterion is met with room to spare
(0.985), the depth-2 half is not (0.185), and the gap is understood rather than
mysterious.**

That is an argument for phase 3, not against the network. Search is what converts a
policy that is right half the time into a player that does not hang pieces, and MCTS is
the designed remedy. Note also that **pure value-greedy is not that remedy** -- one ply
of value lookahead scored *worse* than the raw policy (0.060 vs 0.125), because it throws
the policy away entirely. PUCT uses the policy as a prior and the value only at leaves,
which is why it needs both heads and neither alone.

## Phase 3: MCTS in Rust

`engine_rs/src/mcts.rs` owns the tree; `nn/mcts.py` owns the GPU and nothing else. Rust
descends until it has a batch of leaves that need evaluating, hands their planes across
one call, and takes back priors and values. That inversion is the phase-4 design decision
made early: Rust threads calling *into* torch would serialise on the GIL and give up
exactly the parallelism they were spawned for.

Nodes and edges are flat `Vec`s addressed by index -- no `Rc`, no `RefCell` -- and a
descent walks a single scratch board with `make_ai_move`/`undo_ai_move` rather than
cloning a `GameState` per node. A simulation therefore costs two cheap move applications
per ply.

**The batch exists because of virtual loss.** A descent that reaches an unevaluated leaf
marks its path as if it had lost, so the next descent in the same `collect` goes
elsewhere. Without it every descent in a batch returns the same leaf and the batch is
worth one simulation. It is worth 8x: 800 simulations take 1.08s one leaf at a time and
0.13s at sixteen, which is ~6,000 simulations a second.

**Masking is implicit and cannot be forgotten.** Rust reads priors only at the legal
actions of the leaf it is expanding and renormalises them, so a plain softmax over all
2,196 logits in Python *is* a masked softmax -- `p_i / sum_{j in legal} p_j` cancels the
partition function. There is no masking step to get wrong.

**Values are in the frame of the side to move at each node**, matching the network's own
convention, so `backup` flips sign at every step up the path.

### Two bugs worth keeping in mind

Both were found by `tests/test_mcts.py` before any strength was measured, and both are
the kind that produce a plausible-looking search rather than a crash.

- **The root needs a terminal verdict like any other node.** Only child nodes got one
  during a descent, so a search started from a finished game expanded a node with no
  edges and then selected out of an empty range.
- **An empty `collect` is not the stop condition.** A descent ending on a terminal
  position backs up *without* adding to the batch, so `pending.len() < max_leaves` never
  advances when every reachable leaf is terminal -- an infinite loop. `collect` is now
  bounded per call, and the driver stops only when a collect yields no leaves **and** the
  simulation count did not move.

### Testing a search rather than a network

Every test in `tests/test_mcts.py` uses a **uniform** evaluator: flat priors, zero
values. With a trained network it is impossible to tell a working search from a working
policy -- the network finds the mate on its own and every result is evidence about the
weights. Given an evaluator that knows nothing, any preference the search shows is the
search's own, and "finds mate in one" becomes a real test of selection, backup and
terminal scoring together.

That framing also caught a wrong expectation of mine. Given a uniform evaluator, an
*opening* position carries no information at all, so PUCT spreading visits evenly across
the legal moves is correct behaviour -- a sharpening distribution there would mean
something was leaking a preference it had not earned. The test now asserts both halves:
flat where there is nothing to find, concentrated near a mate that is.

### What the search is worth

Same weights, same question, with and without a tree:

| | vs depth-2 | vs depth-6 |
| --- | --- | --- |
| raw policy argmax | 0.185 | — |
| MCTS, 800 simulations | **0.833** | **0.420** (+83 =2 -115, 200 games) |

The phase-3 gate is 800 simulations beating depth-6, and 0.420 does not.

**But the gate and the project's success bar are different claims.** The bar is beating
alpha-beta *at equal time control*, and that run was not it: depth-6 spent roughly three
and a half times the wall clock per move that MCTS-800 did. Measuring the gate is
therefore measuring a handicap match. The arena now reports ms/move for both sides, so a
result can never again be read without knowing who was given more thinking time.

Given the same budget the picture changes:

| | score vs depth-6 | ms/move, subject vs opponent |
| --- | --- | --- |
| MCTS, 800 sims | 0.420 +/- 0.035 (200 games) | ~100 vs ~360 |
| MCTS, 2400 sims | **0.525 +/- 0.050** (100 games) | 343 vs 404 (0.85x) |

At 2,400 simulations the network is **level with depth-6 alpha-beta while using slightly
less time per move than it**. Level, not ahead: +-0.050 puts 0.5 comfortably inside the
interval, and the honest reading of +52 =1 -47 is a dead heat. It is still the first point
in this project at which the network is competitive with the search on the search's own
terms, and it was reached with a network that is itself a bootstrap -- trained on the very
engine it is now matching, and due to be thrown away before the zero run.

The gate should be restated in time rather than simulations. "800 sims" was a guess at a
budget made before anything could be measured; the thing worth gating on is what the
scoping doc actually asked for.

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
