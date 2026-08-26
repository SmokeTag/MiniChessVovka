# MiniChess 6×6 Crazyhouse — Root-Parallel Search Evaluation

> **Subject**: `minimax_parallel()` in `engine_rs/src/search.rs` — the rayon root split used for interactive analysis
> **Question asked**: is it worth having, and can it be made worth having, without touching the sequential path?
> **Short answer**: as it ships today it buys almost nothing (geomean 1.00–1.27× on 24 cores) and it returns a different answer from the sequential search in 8 of 28 measurable cells. Three candidates were implemented and adversarially re-measured. One is a real 2× win on heavy positions with a caveat that has to be fixed first; one is a small accuracy gain; one was refuted.
> **Date**: 2026-08-25 / 26. Machine: 24 cores.
> **Status**: **superseded by §0 below.** A fix has since been applied to `engine_rs/src/search.rs` in the working tree. Sections 1–5 are kept as the record of how the problem was found; §0 is what the code now does.

---

## 0. What was applied (2026-08-26)

Sections 1–5 below diagnosed the problem and proposed three candidates. Two of those were confirmed,
one was refuted, and none was applied as written. Further measurement produced the change that is now
in `engine_rs/src/search.rs`. It is four parts, all of them parallel-only:

1. **The worker TT merge is gone.** `merge_worker_tt()` and the entry harvesting in `search_worker()`
   are deleted; a worker's table now dies with the worker. This is the correctness fix — it is what
   stopped the parallel search returning answers *worse* than the sequential one.
2. **Only the final iterative-deepening iteration is split** (`parallel_min_depth.max(depth)`). The
   shallow iterations cost more in fan-out than they save.
3. **A cheap iteration is not split at all** — if the previous ID iteration took under `MIN_SPLIT_SECS`
   (30 ms), the final one runs sequentially. This is what removes the losses on small positions. The
   gate is on *measured* cost, deliberately not on core count: a core-count guard was tried and
   refuted (see §4.3) because it scales the wrong way and disables the split entirely on a 64-thread box.
4. **The re-search runs in parallel** when more than one root move beats the baseline. Serially it was
   routinely more expensive than the whole scout phase that preceded it (5 moves costing 0.41 s against
   45 scouts costing 0.29 s). A single candidate still re-searches in the root's state, where it sees
   the full transposition table.

### Measured result (24 cores, quiet box, min-of-3, fresh process per measurement)

| position | depth | sequential | parallel before | parallel after | speedup |
|---|---|---|---|---|---|
| middlegame_hand_a | 7 | 6.88 s | 5.61 s (1.34×) | **3.11 s** | **2.21×** |
| endgame_thin_a | 7 | 5.97 s | 6.33 s (0.95×) | **2.85 s** | **2.09×** |
| endgame_thin_b | 7 | 2.05 s | 1.96 s (1.01×) | 1.89 s | 1.08× |
| tactical_check_a | 7 | 1.81 s | 1.74 s (1.03×) | 1.79 s | 1.01× |
| endgame_thin_a | 6 | 1.29 s | 1.40 s (1.01×) | 0.96 s | 1.34× |
| endgame_thin_b | 6 | 0.86 s | 1.21 s (0.72×) | 0.85 s | 1.02× |
| middlegame_hand_a | 6 | 1.60 s | 1.29 s (1.22×) | 1.91 s | 0.84× |
| tactical_check_a | 6 | 0.63 s | 0.66 s (0.99×) | 0.63 s | 1.00× |

Geomean **1.017× → 1.245×**, median 1.010 → 1.084, worst cell 0.715× → 0.838×.

**Correctness.** Stock, the parallel search differed from the sequential one in 8 of 32 paired cells,
two of them *worse* (opening_start d7 returned a move scored 50 lower). After the change, one cell
differs and it is **better** (middlegame_hand_a d6: 1382 against the sequential 1380). Exact parity is
not achievable while null-move pruning and LMR make results window- and order-dependent; the
achievable and now-met bar is that the parallel search never returns a worse move than sequential.

**The one remaining loss** is middlegame_hand_a d6 at 0.84×: one candidate move beats the baseline and
its serial full-window re-search costs 0.69 s against 0.42 s for all 72 parallel scouts. A single
candidate has nothing to run beside it. Fixing it needs splitting one ply deeper, which is a larger
change than this one.

### Sequential path: unchanged, verified two ways

- **Machine code.** Normalised `objdump` of the before/after builds: all 18 sequential-path symbols —
  `minimax_ab`, `quiescence_search`, `evaluate_position`, `mvv_lva_score`, `SearchState::tt_get`,
  `get_legal_moves_vec`, `get_noisy_moves`, `make_ai_move`, `undo_ai_move` — are **instruction-identical**.
  The only differing symbols are rayon plumbing, reached only from the parallel branch.
- **Measurement.** The sequential column of the full bench matrix returned **identical moves and scores
  in every cell**, with a geomean timing ratio of 0.987 (noise; the code is the same instructions).

`set_parallel_search` still defaults to `(False, 3)`, and `src/self_play.py` / `src/scheduled_self_play.py`
still pin `parallel=False`. Overnight self-play remains single-threaded, which is the right call: many
independent games across cores scale better than one parallelised search.

### Reproducing

```bash
./venv/bin/python bench/run_bench.py --depths 6,7 --repeats 3 --modes seq,par --out bench/results/after.json
./venv/bin/python bench/compare.py bench/results/baseline.json bench/results/after.json
```

Run it on an idle box: the parallel mode uses all 24 cores, so a concurrent self-play job makes the
scaling numbers meaningless (the harness records load average per cell and flags it).

---

## 1. What the parallel root search does today

`find_best_move()` (search.rs ~line 796) runs iterative deepening. At each iteration it picks the
parallel branch when `current_depth >= parallel_threshold`, where the threshold is
`PARALLEL_MIN_DEPTH` (default **3**) when parallel search is on and `i32::MAX` when it is off — which
is what keeps `minimax_parallel` structurally unreachable on the sequential path.

`minimax_parallel()` (~line 649) is a root PVS split:

1. **Baseline (serial)** — order the root moves with the same keys `minimax_ab` uses (TT move,
   killers, mvv/lva), then search the best-ordered move sequentially at full width to get a real alpha.
2. **Scout (parallel)** — every remaining root move, all of them, every iteration, handed to
   `search_worker()` (~line 592) via `rayon::par_iter`. Each worker gets its own `GameState` copy,
   its own `SearchState` with a `1<<14` TT, cloned killers/history, a read-only `Arc<HashMap>`
   snapshot as `base_tt`, and a **null window** `(baseline, baseline+1)`.
3. **Re-search (serial)** — only the moves that beat the baseline are re-searched at full width.
4. **Merge** — `merge_worker_tt()` (~line 631) folds each worker's harvested TT entries back into the
   root's `ss.tt`, keeping the deeper entry, **ignoring `TTFlag` and ignoring the window the entry was
   derived under**.

Timings for phases 1–3 are printed to stderr as `[PARALLEL] depth N: baseline …s, scout …s (N moves),
re-search …s (N moves)`. That line is the single most useful diagnostic in this whole exercise.

### 1.1 Baseline measurement

8 positions × depths 4–7 × {seq, par} × 3 repeats = 192 searches. Every measurement ran in a **fresh**
`./venv/bin/python` (the process-global `MOVE_CACHE` is keyed by `(position hash, depth)`, so a repeat
in the same process is a ~0 s cache hit), `load_move_cache_from_db()` was never called, and the
minimum of 3 is reported. Results: `bench/results/baseline.json`. Ran 23:21 → 23:24 on 2026-08-25.

**Machine load**: 1.48 at start, 4.11 at the end, 24 cores. `/proc/loadavg` was sampled every 15 s and
the runnable-process count never exceeded 4 of 24. The only genuine external load was two
`main.py` processes at ~0.9 cores total, ~4 % of machine capacity. The harness flagged 41/64 cells
`load_suspect`, but that flag is over-firing: the 1-minute average it reads is driven up by our *own*
24-thread rayon bursts and lags into the sequential cells that follow. Sequential cells were
uncontended; parallel cells lost ~4 % of capacity, which makes the reported speedups **conservative by
about 4 %**. Repeat-to-repeat spread was 1–3 % on every cell above the measurement floor.

| Position | Root moves | d4 | d5 | d6 | d7 |
|---|---:|---|---|---|---|
| `endgame_thin_a` | 75 | 2.54× | 2.60× | 1.00× | 0.95× |
| `middlegame_hand_a` | 73 | 1.74× ✗ | 2.78× | 1.33× ✗ | 1.29× |
| `endgame_thin_b` | 46 | 0.90× ✗ | **0.59×** ✗ | 0.73× | 0.93× |
| `opening_early` | 26 | 0.79× | 0.95× | 0.97× | 1.11× |
| `opening_start` | 15 | 0.86× ✗ | 1.54× | 1.05× ✗ | 1.44× ✗ |
| `tactical_check_b` | 9 | 0.84× ✗ | 0.84× | 0.95× | 0.79× |
| `tactical_check_a` | 8 | 1.00× | 1.01× | 1.04× | 1.05× |
| `middlegame_hand_b` | — | *degenerate, excluded* | | | |

✗ = the parallel search returned a different move and/or a different score from the sequential search.

| | d4 | d5 | d6 | d7 |
|---|---|---|---|---|
| **Geomean speedup** (7 measurable positions) | 1.13× | 1.27× | 1.00× | 1.06× |
| **Median speedup** | 0.90× | 1.01× | 1.00× | 1.05× |
| **Exact (move + score) agreement with sequential** | 3/7 | 6/7 | 5/7 | 6/7 |
| Moves re-searched at the deepest iteration, summed | 5 | 8 | 9 | 8 |
| Scout share of parallel wall time | 34 % | 40 % | 45 % | 41 % |

**12 of the 28 measurable cells are slower in parallel than sequential**, worst `endgame_thin_b` d5 at
0.59×. The typical position sees no gain at all; the geometric mean is carried entirely by the two
wide-root positions.

`middlegame_hand_b` is excluded from every aggregate: it finds mate (−1000000) at depth 1, the ID loop
breaks immediately, the parallel branch is never entered, and the whole cell runs in 0.001–0.002 s.
It measures nothing. Note that `bench/compare.py`'s per-depth table *does* include it, which is why its
d4 figure reads 1.50× against the 1.13× above — it sums minima across positions and is dominated by
whichever position happens to be slowest. **Use the geomean and the median.**

Two more positions, `opening_start` and `opening_early`, run in 0.01–0.12 s even at d7. Their sub-1.0×
numbers are fixed split overhead, not scaling. Only four positions exceed ~0.7 s at d7.

---

## 2. Where the time goes, and what the ceiling is

Summed over every ID iteration, as a fraction of parallel wall time:

| Depth | Baseline (serial) | Scout (parallel) | Re-search (serial) | Other |
|---|---:|---:|---:|---:|
| 4 | 21 % | 34 % | 21 % | 25 % |
| 5 | 18 % | 40 % | 35 % | 6 % |
| 6 | 26 % | 45 % | 25 % | 3 % |
| 7 | **44 %** | 41 % | 14 % | 1 % |

**Two serial phases bracket the one parallel phase.** At depth 7 the serial baseline alone is 44 % of
wall time and the serial re-search another 14 %, so **58 % of the parallel run is single-threaded**.
Amdahl caps the achievable speedup at about 1.7× no matter how many cores are added — and the trend
runs the wrong way, because the baseline share grew 21 % → 44 % from d4 to d7 as deeper searches
concentrate more time in the PV move. `endgame_thin_a` d7 is the clean illustration: baseline 4.15 s of
a 7.20 s total, cell finishes at 0.95×, i.e. *slower* than sequential.

Three further reasons the speedup is far below the core count:

- **The scout is redundant, not wrong.** `scout_moves` is the entire remaining root list on every
  iteration (14/72/74/45/25/7/8 moves depending on position) with no shared cutoff between workers, so
  the workers redo work the sequential root would have pruned after the first few moves. Failed scouts
  are *rare* — 5/8/9/8 re-searched moves out of 245 scouted — so the null window is doing its job. The
  cost is the redundant scouting itself.
- **The split runs at every iteration.** With `PARALLEL_MIN_DEPTH = 3`, a depth-7 search splits at
  3, 4, 5, 6 *and* 7 and pays the fixed fan-out cost five times, while only the last iteration's result
  is returned. This is also why d4's "other" bucket is 25 %: pure fixed overhead on a few-millisecond
  search.
- **Narrow roots cannot fill the cores.** Four of the seven measurable positions have 8–26 root moves
  on a 24-thread box. All four regress at d7.

### 2.1 The correctness problem, which matters more than any speedup

**8 of 32 paired cells disagree with the sequential search**, and every single one of them involves a
*score* change, not just a move change. Two paths that agreed would produce equal-score alternatives;
producing only score changes says the parallel path is searching a different — corrupted — tree.

| Class | Cell | Sequential | Parallel | Δ |
|---|---|---|---|---|
| MATERIAL | `opening_start` d4 | d1c2 (40) | b1b6 (160) | +120 |
| MATERIAL | `opening_start` d6 | b1b6 (26) | b1b6 (8) | −18 |
| MATERIAL | `opening_start` d7 | c1e2 (74) | **d1c2** (24) | −50 |
| MATERIAL | `endgame_thin_b` d4 | a4d4 (−389) | a4b4 (−347) | +42 |
| MATERIAL | `endgame_thin_b` d5 | R@c5 (−379) | a4b4 (−345) | +34 |
| trivial | `middlegame_hand_a` d4 | c5c6 (1359) | c1d3 (1361) | +2 |
| trivial | `middlegame_hand_a` d6 | B@b3 (1380) | c5c2 (1381) | +1 |
| trivial | `tactical_check_b` d4 | f6e6 (−1807) | c5d4 (−1808) | −1 |

`opening_start` d7 is the headline defect: the parallel path returns a *different move* that it scores
50 lower, in 1.44× the speed. **A fast wrong answer is a regression, not a win.**

**The reproducer** is `opening_start` d6: same move, sequential 26, parallel 8, and the depth-6
iteration re-searched **zero** moves. With `research_moves = 0` the parallel result *is* just
`minimax_ab` on the top-ordered root move in the root's own `SearchState` — the identical call the
sequential path makes. The split logic did not produce the wrong number; the **transposition table**
did. The only thing that differs is that `merge_worker_tt()` has already folded worker entries from the
depth-3/4/5 iterations into `ss.tt`. Those entries were produced under the null window
`(baseline, baseline+1)`; width-1 windows make `TTFlag::Exact` unreachable through the normal path, so
the harvest is all window-relative bounds, merged depth-preferred with no regard for flag or window,
and the root's later full-width searches clamp alpha/beta on them.

That hypothesis was tested directly by the candidates below. **It is confirmed as a contributor but is
not the whole story** — see §4.1.

---

## 3. Candidates considered

Three were implemented. The rest were rejected before implementation:

| Candidate | Why rejected |
|---|---|
| `quarantine-worker-tt-merge`, `quarantine-worker-tt-entries`, `merge-worker-tt-flag-gate` | Three restatements of the same TT-quarantine fix that was implemented as §4.2. The flag-gated form is kept in reserve as the documented fallback; do not implement two merge policies at once. |
| `overlap-baseline-with-scout` | Its whole projection rests on the serial baseline being 44 % of d7 wall time — but that 4.51 s baseline was TT pollution, and §4.1 cuts the same phase to 0.13 s with one line. It would parallelise a phase that no longer costs anything. |
| `split-the-pv-move-one-ply-deeper` | High risk (one-ply Young Brothers Wait inside `minimax_parallel`), and its motivating number is the same pollution artefact §4.1 removes. Re-derive the case afterwards. |
| `pvs-the-research-loop` | Projected, not measured. Its target cell (`endgame_thin_b` d5 at 0.59×) is a 46-move root that §4.1/§4.3 already address, and its shared `AtomicI32` alpha makes pruning scheduling-dependent exactly when we are trying to prove answer parity. |
| `parallel-root-tt-store` | Inert once the split runs only on the final iteration — there is no later iteration left to read the root entry it would store. |
| `staged-scout-waves` | Measured to lose: the wave-barrier experiment cost **+29 %** on `endgame_thin_a` scout (1.90 s → 2.45 s). Chunking with inter-wave merges is a known regression on this engine. |
| `share-live-move-ordering-heuristics-across-workers` | Rejected on the standing constraint: it needs `minimax_ab`'s history reads/writes (search.rs:175, 228, 502, 556) made generic over the heuristic backend, i.e. it touches the sequential path. Its own estimate caps the upside at ~15 % of wall time. |
| `scout-lmr-parity`, `parallel-root-lmr-parity` | Changes the tree the scout searches while we are mid-way through proving seq/par answer parity. Land only after the TT work is settled. |
| `seq-par-differential-harness` | The capability already exists (`run_bench.py --par-min-depth`) and has been used; the acceptance criteria below are built on it. |
| `parallel-abort-and-mate-divergence` | Not exercised by any current caller, and the one mate position breaks the ID loop at depth 1 before the parallel branch engages. It would ship untested against zero measured impact. |
| `keep-partial-parallel-iteration-on-timeout` | Four files including `main.py` and `thread_utils.py`, and it first requires a hint clock that does not exist. A feature, not a fix. |
| `parallel-multipv-api` | A new feature across five files that surfaces scores the fixes below are about to change. Build it on a parallel path that already agrees with sequential. |

---

## 4. Implemented candidates

Each was built in its own git worktree, into a throwaway venv created there. **`maturin develop` was
never run against the repo's root `venv/`** — overnight self-play may have that module loaded.
Each was then re-measured from scratch by an independent verifier who tried to break it.

### 4.1 Split only the final iterative-deepening iteration — **CONFIRMED, with a caveat that must be fixed first**

*Worktree*: `/home/andre/Development/chess/.claude/worktrees/wf_0cf53a6a-713-8` (search.rs, +13 −1)

**Change.** In `find_best_move`, when there is no deadline, set
`parallel_threshold = depth.max(parallel_min_depth)` instead of `parallel_min_depth`. With a time limit
the final iteration is unknowable, so today's behaviour is kept. `parallel=False` still leaves the
threshold at `i32::MAX`. Effect: `minimax_parallel` runs once instead of 3–5 times, so
`merge_worker_tt()` never pollutes the table that the *next* iteration's serial baseline reads, and the
fixed split cost is paid once.

**Measured.** Full matrix before and after, back-to-back, fresh process per search.

| d7 cell | seq | par before | par after | after speedup |
|---|---:|---:|---:|---|
| `endgame_thin_a` | 6.10 s | 6.31 s | **2.92 s** | 2.09× (par gain 2.16×) |
| `middlegame_hand_a` | 6.62 s | 5.17 s | **3.07 s** | 2.16× (par gain 1.69×) |
| `endgame_thin_b` | 1.95 s | 2.59 s | 1.75 s | 1.12× (par gain 1.48×) |
| `opening_start` | 0.104 s | 0.076 s | 0.106 s | 0.99× (par **loss** 0.72×) |

Geomean over all 7 measurable positions: d7 1.01× → **1.23×**, d6 0.97× → 0.98×. Restricted to the four
positions above ~1 s at d7: d7 0.98× → **1.50×**, d6 0.98× → 1.08×.

**Mechanism confirmed in the stderr**, which is the valuable part: on `endgame_thin_a` d7 the serial
baseline collapses from **3.53 s to 0.10 s** and the re-search from 0.10 s to 0.00 s, because iterations
1–6 now run sequentially and hand the depth-7 baseline a clean TT.

**Correctness not fixed.** Disagreements go 8/28 → 7/28. One cell is fixed (`endgame_thin_b` d4 now
matches sequential exactly), one improves (`opening_start` d4, Δ120 → Δ1), and one gets **worse**
(`opening_start` d6: was the same move at Δ18, now a different move at Δ27). Since `merge_worker_tt()`
now runs exactly once, cross-iteration pollution is structurally impossible — so **the surviving
disagreements prove the corruption also happens within a single split.** That is the most useful thing
this candidate produced: it localises the remaining bug to `merge_worker_tt()` / the scout during one
iteration, and nowhere else.

**Sequential path unchanged**: 0/32 cells differ in move or score against the before-build or against
`baseline.json`. An interleaved A/B test (depth 7, `parallel=False`, 5 reps each, the `.so` swapped
between *every* run, md5-checked) gives a mean after/before ratio of 1.0098 with mixed signs, inside a
within-build noise floor of 1.08–1.36×. Tests 16/16. `get_parallel_search()` still `(False, 3)`.

**Verifier verdict: CONFIRMED.** Independently rebuilt and re-ran the full matrix under load 5–6
(3–14 % slower absolutes, gains reproduce): `endgame_thin_a` d7 3.508 s min-of-5, `middlegame_hand_a`
d7 3.758 s min-of-5, geomean d7 1.22× vs the claimed 1.23×, correctness cell-for-cell identical. A
`diff -rq` against the candidate's worktree showed the published diff is complete.

**Verifier's serious caveats — read before applying:**

1. **The new branch only fires when `time_limit is None`, and if the clock expires before the final
   iteration is reached, the split never runs at all.** Verified: `middlegame_hand_a`, depth 8,
   `time_limit=3.0` — the stock build parallelises depths 3–6 and returns 1381; the patched build
   prints "root split enabled from depth 8", runs 100 % sequentially on 24 idle cores, and returns a
   different move at 1380. `play_online.py:1195/1200` always passes a time limit (90 s, 30 s in time
   trouble), as does `precalc_openings.py`. Only `thread_utils.py` (GUI move/hint threads) and the
   benchmark call with `time_limit=None`. **The whole bench matrix is structurally blind to this.**
2. `.max(depth)` voids the `PARALLEL_MIN_DEPTH` knob: `set_parallel_search(True, n)` has no effect for
   any `n <= depth`, and disables parallel search entirely for `n > depth`, while the engine still
   prints a threshold that will never be reached. `run_bench.py --par-min-depth` is inert against this
   build.
3. Both stated geomean acceptance targets **fail** (1.22× vs ≥1.5× at d7; 1.00× vs ≥1.3× at d6), and
   the "seq column within 3 % of `baseline.json`" guard also fails — but that guard is unsatisfiable
   across sessions: the same source built twice came out 5–15 % *faster* in one session and 3–14 %
   *slower* in another. It is box drift, not the build.
4. Positions under ~1 s regress: `opening_start` d7 par 0.087 s → 0.147 s. The single deep split is
   pure fixed overhead there.
5. The win is largely a **move-ordering repair**, not a parallelism improvement: the depth-7 root TT
   entry is now written by a sequential iteration 6 instead of being left stale by five successive
   `minimax_parallel` iterations, which never store a root TT entry.

**Recommendation: the strongest of the three, but do not apply as written.** Rework the condition so
the time-limited path (the actual interactive call site) benefits too, and add a size guard so short
searches do not pay the split. The 2× on heavy positions is real.

### 4.2 Stop merging scout TT entries into the root table — **CONFIRMED, partial fix**

*Worktree*: `/home/andre/Development/chess/.claude/worktrees/wf_0cf53a6a-713-9` (search.rs, +17 −31)

**Change.** Delete `merge_worker_tt()` entirely and stop harvesting entries from workers;
`search_worker` returns `(move, score, aborted)`. `tt_get` is left byte-identical. Landed **paired with
§4.1**, as required — with a single split there is no next iteration to feed, and the pairing is what
makes the removal free.

**Measured (clean order-balanced protocol, two matrix passes per build, min of 6).**

- **Acceptance (a) PASS.** `opening_start` d7 parallel now returns c1e2 with score 74, *exactly*
  matching sequential — the reproducer the candidate was written for.
- **Acceptance (b) FAIL.** Disagreements drop 8/32 → 6/32 (bar was ≤2/32), and two MATERIAL cells
  survive: `opening_start` d6 (Δ+27) and `endgame_thin_b` d5 (Δ+34, entirely unchanged).
- **Acceptance (c) VIOLATED by one trivial cell.** `endgame_thin_b` d4, White to move: sequential
  a4d4 (−389), parallel a4d4 (−390). Same move, one centipawn lower. The rule was written to catch
  exactly this shape, and the pick-§4.1-only build agrees with sequential there, so **this change
  introduced it.**
- **Speed guard PASS.** No re-search blow-up: `middlegame_hand_a` d7 re-searches 0 moves for 0.00 s.
  d7 geomean 1.06× → 1.21×; d5 geomean **regresses** 1.25× → 1.10×.

**Attribution, measured with a third build (§4.1 only):** 4–5 of the 6 surviving disagreements are
already present with §4.1 alone. Removing the merge, on top of §4.1, **fixes 2 cells** (`opening_start`
d7, `tactical_check_b` d4) and **breaks 1** (`endgame_thin_b` d4). Its own contribution to speed is
zero — §4.1 alone captures the entire d7 win.

**Verifier verdict: CONFIRMED**, reproduced cell for cell and delta for delta. The verifier proved
constraint 2 *harder than the candidate did*: normalised `objdump` of `minimax_ab` (1104 insns),
`quiescence_search`, `mvv_lva_score`, `evaluate_position`, `tt_get`, `with_tt_capacity` and
`get_noisy_moves` are **zero instructions different** between the two builds. The sequential path is not
"unchanged within noise" — it is the same machine code.

**Verifier's caveats:**

1. **Measurement conditions were misreported.** The run is labelled "clean" but 211 of 256 cells were
   `load_suspect` (peak 7.64). The verifier independently measured this box's floor: the sequential
   column — proven byte-identical machine code across builds — still varied by 7–9 % between builds.
   **Nothing under ~10 % is resolvable here**, which invalidates the small per-cell deltas in both
   directions and makes the "within 3 %" speed guard unmeasurable as specified.
2. **The stated mechanism in the added comment is wrong**, and it is baked into the source. The comment
   says worker bounds "are window-relative and only sound for that window". That is false in general:
   `LowerBound`/`UpperBound` TT entries are bounds on the true value and are sound under any window,
   which is why every engine shares them. The real reason merged scout entries corrupt *this* engine is
   narrower: null-move pruning returns a hard `beta`/`alpha` (search.rs:435-436), and the store site
   classifies flags against `orig_alpha`/`orig_beta`, so a window-clamped value can be written with flag
   `Exact` and later returned verbatim by a probe. Under a razor-thin scout window that path fires
   constantly. **Fix the comment before merging — as written it will teach the next reader that
   ordinary TT sharing is unsound.**
3. **The acceptance bar was never reachable by this change.** The residual MATERIAL cells are scout
   *false negatives*, not table corruption: `search_worker` scouts with a zero-width window, so a
   null-move cutoff at the scouted child returns exactly `scout_alpha`, `beats_baseline` is false, and a
   genuinely better root move is dropped and never re-searched. `endgame_thin_b` is thin-material
   crazyhouse — precisely where null-move pruning is least sound — and its d5 delta is +34 in *every*
   build. No merge policy can fix that class.
4. Bookkeeping: the MATERIAL count 5 → 2 applies only the "Δ > 5" half of the stated criterion; under
   the full criterion (which also counts a differing move at a non-tied score) it is 8 → 5.

**Recommendation: apply only together with §4.1, and only after fixing the source comment and the
`endgame_thin_b` d4 directional regression.** It is worth having — it clears the single worst material
error in the matrix — but it is a partial fix, not the fix, and it contributes no speed of its own.

### 4.3 Skip the split on narrow roots — **REFUTED**

*Worktree*: `/home/andre/Development/chess/.claude/worktrees/wf_0cf53a6a-713-10` (search.rs, +16)

**Change.** At the top of `minimax_parallel`, if `legal_moves.len() - 1 < 2 * rayon::current_num_threads()`
(i.e. < 48 on this box), fall back to `minimax_ab` with the root window and return its result unchanged.

**What was measured.** The guard fires on 5 of 8 positions (root widths 8, 9, 15, 26, 46) and leaves the
split running on the two wide ones (73, 75). Disagreements drop **8/28 → 2/28**, every MATERIAL
disagreement disappears including `opening_start` d7, and the worst cell goes 0.56× → 0.94×. Both
survivors are trivial (Δ2, Δ1) and both sit on the wide root where the split still runs. Machine-code
comparison confirms the sequential path is untouched. Tests 16/16.

On its face the best correctness result of the three. **It does not survive contact.**

**Verifier verdict: PLAUSIBLE — does not hold up.** The verifier reproduced the before matrix exactly
and the after matrix exactly, then ran the experiment the candidate did not:

- **`RAYON_NUM_THREADS=4` on the same guard build brings all six "eliminated" disagreements back
  byte-for-byte**, including the flagship `opening_start` d7 (seq c1e2/74 vs par d1c2/24) and both
  material `endgame_thin_b` cells. Nothing in `minimax_parallel` was repaired; the buggy code is simply
  not entered at this core count.
- **The margin is two cores.** `endgame_thin_b` has 46 root moves, so the guard needs `2 × threads > 45`.
  At 23 threads it is clean; at **22 threads the Δ42 and Δ34 material bugs return.**
- **The threshold scales the wrong way.** At `RAYON_NUM_THREADS=64` the guard fires on *every* bench
  position, no `[PARALLEL]` line appears anywhere, and parallel analysis becomes a complete no-op. Any
  box with ≥37 hardware threads loses the widest position; a 64-core box loses the feature entirely.
  A width guard for a work-splitting decision should key off **work per move** (depth, subtree size),
  not off core count.
- **Conditional on the split actually running, the disagreement rate is unchanged**: 2 of the 8
  wide-root cells disagreed before, and the same 2 disagree after. "8/28 → 2/28" is achieved entirely by
  shrinking the denominator.
- The candidate's own defence for waiving its speed criterion — "par and seq execute identical code
  there, so the deviation is pure noise" — is exactly the admission that the criterion existed to detect
  whether the split still runs, waived on the grounds that it does not. The verifier also showed the
  ±3 % band *is* resolvable on this box at load 2.2–3.6 (every cell above 0.1 s landed in 0.97–1.03);
  the candidate's 0.866–1.218 envelope came from measuring under load 6.6–10.4 with two sibling rayon
  benchmarks live.

**Recommendation: do not apply.** The experiment is still valuable — it is the cleanest confirmation
that `merge_worker_tt` and the scout, not the split scheduling, are the defect — but as a change it
disables the feature rather than fixing it, and its correctness gain is a property of this machine's
core count.

### 4.4 Where that leaves the parallel search

| | Speed | Correctness | Verdict |
|---|---|---|---|
| Stock | geomean 1.00–1.27× | 8/28 disagree, 5 MATERIAL | baseline |
| §4.1 split-last | **1.50× on heavy positions**, regresses <1 s ones | 7/28 disagree | CONFIRMED, needs the time-limit rework |
| §4.1 + §4.2 no-merge | same as §4.1 (§4.2 adds no speed) | 6/32, 2 MATERIAL, 1 new trivial regression | CONFIRMED partial |
| §4.3 narrow guard | ~1.0× by construction on 5 of 7 positions | 2/28 **on 24 cores only** | REFUTED |

**The remaining bug is now well localised.** Cross-iteration TT pollution is confirmed and removable.
What survives is (a) within-split corruption through the depth-preferred, flag-ignoring merge combined
with the null-move hard-`beta` return at search.rs:435-436, and (b) scout false negatives, where a
zero-width window plus a null-move cutoff returns exactly `scout_alpha` and a genuinely better root move
is silently dropped. (b) is not fixable by any merge policy and is the next thing to work on.

---

## 5. Standing constraint — read this before changing anything here

1. **Root-parallel search stays OFF by default.** `PARALLEL_ENABLED` starts `false`;
   `get_parallel_search()` must return `(False, 3)` in a fresh process. Parallel search exists for
   interactive analysis of a single position and nothing else.
2. **Overnight self-play runs single-threaded, deliberately.** Many independent games spread across
   cores scale far better than one parallelised search. `src/self_play.py` and
   `src/scheduled_self_play.py` pass `parallel=False` explicitly; leave those call sites alone.
3. **No change may alter the behaviour or the speed of the sequential path** — `minimax_ab`,
   `quiescence_search`, move ordering, evaluation, and the TT as used when `parallel=False`.
   Parallel-only code paths only. If a candidate must touch shared code it has to prove the sequential
   path is unchanged **by measurement, not by argument**. The strongest available proof is the one used
   in §4.2: normalised `objdump` of the sequential symbols, showing zero instructions different.
4. **Never run `maturin develop` against the repo's root `venv/`.** Overnight self-play may have that
   module loaded, and replacing the `.so` underneath it is destructive. Build into a throwaway venv
   inside a worktree.

---

## 6. How to re-run

Every measurement runs in its own fresh interpreter, because `find_best_move` consults a process-global
`MOVE_CACHE` keyed by `(position hash, depth)` and a repeat inside one process is a ~0 s cache hit. The
child never calls `load_move_cache_from_db()`, so `move_cache.db` is neither read nor written.

```bash
# Full baseline matrix: 8 positions x depths 4-7 x {seq,par} x 3 repeats = 192 fresh processes, ~3 min
./venv/bin/python bench/run_bench.py \
    --depths 4,5,6,7 --repeats 3 --modes seq,par \
    --out bench/results/baseline.json

# One position, one depth, both modes - the fastest way to check a single cell
./venv/bin/python bench/run_bench.py --positions endgame_thin_a --depths 7 --modes seq,par --repeats 5

# Keep the [PARALLEL] baseline/scout/re-search lines in the results file. Do this:
# it is the most useful signal available for where the time goes.
./venv/bin/python bench/run_bench.py --depths 7 --modes par --keep-stderr --out bench/results/par7.json

# Override PARALLEL_MIN_DEPTH for the parallel children only (the seq child never
# calls set_parallel_search, so its path stays the default one)
./venv/bin/python bench/run_bench.py --modes par --par-min-depth 7 --out bench/results/par-lastonly.json

# Pair two result files and report speedup, move agreement and score deltas.
# Disagreeing cells are printed loudly and set a non-zero exit status.
./venv/bin/python bench/compare.py bench/results/baseline.json bench/results/after.json
```

Reading the output:

- **`min` is reported, not mean.** Contention only ever adds time, so the minimum is the least
  contaminated estimate. The per-cell sample spread is kept in the results file — check it.
- **Record `uptime` with every run.** The harness stores it in the header and marks cells above load
  4.0 with `(!)`. Those markers over-fire on our own rayon bursts (see §1.1), but a run taken under
  genuine external load above 4 is unreliable and must be **reported as such, not averaged away**.
  The measured floor on this box under real contention is ~10 %; on a quiet box (load < 4) it is ~3 %.
- **Judge agreement before speed.** A different move with a different score means the two paths
  disagree about the position. A fast wrong answer is a regression.
- **`middlegame_hand_b` measures nothing** — mate at depth 1, parallel branch never entered. Exclude it
  from every aggregate; `compare.py` does not.
- **Positions are built by replaying a fixed move list** from a fresh `GameState()` — there is no FEN
  parser. See `bench/positions.py` / `bench/positions.json`.

---

*Document created: 2026-08-26, from a measured evaluation of `minimax_parallel` on a 24-core box.*
*Nothing described in §4 has been merged into the working tree.*
