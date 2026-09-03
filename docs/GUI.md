# The Pygame front end

> Extracted from CLAUDE.md. Load this before touching `main.py`, `gui.py`, `layout.py`,
> `settings.py` or `thread_utils.py`.


`main.py` + `gui.py` + `layout.py` + `settings.py`. `layout.Layout` recomputes every rectangle from
the current window size; nothing else may assume a pixel dimension. Four invariants are load-bearing:

- **The search never blocks the UI.** `main.Game` polls worker threads and keeps drawing; Undo / New
  game / Flip / Hint stay live during a search. The engine's own move carries a `generation` tag and
  is discarded if the position moved on, so taking back a move mid-search cannot apply a stale answer
  to the wrong board. Anything that changes the position must call `Game.invalidate()`. Hints do not
  use the counter — see below.

  This invariant is breakable from Rust with no sign of it in Python: `lib.rs`'s book accessors take
  the `BOOK` mutex **without releasing the GIL**, so a main-thread `ai.book_has_position` blocks on
  any lock a background search holds. The book is touched exactly twice in a search (probe at the
  top, one store at the bottom), so `find_best_move` takes `&Mutex<Book>` and locks at those two
  points only. **Do not put the lock back around the search**, and do not add a book accessor to
  `lib.rs` that holds it across a DB write without `py.allow_threads`.
- **Panel zones have fixed heights** (`layout.py`): header, controls and the bottom status band are
  reserved, and only the move list flexes. Hands live in strips beside the board and show a `×N`
  count rather than one sprite per copy, so a filling hand can never push a button out from under the
  cursor. The hint-lines list is sized from `Layout(..., analysis_rows=n)` — a *setting*, never live
  search state, so it cannot resize mid-search. Changing it calls `Game.relayout()`. When the
  reserved bands want more panel than a short window has, the two flexing zones give way in order
  rather than painting over each other: the move list takes what is left and collapses to nothing,
  and the analysis band then shows as many of the requested rows as still fit (`Layout` clamps
  `analysis_rows`, and `gui._draw_analysis` bounds its row loop by that same count).
  `tests/test_gui_controls.py::test_panel_bands_never_paint_over_each_other` sweeps the window
  sizes and hint-line counts that used to overlap.
- **Redraw is dirty-flagged.** `Game.dirty` gates rendering; the loop idles at `IDLE_FPS` and rises
  to `FPS` only while something animates. Fonts and scaled sprites are memoised in `gui.py`. Any new
  per-frame `smoothscale` undoes this.
- **Flip is a view transform only.** It must never change what is true about the position.

Drawing takes a `BoardView` snapshot rather than reading the live `GameState`, which is what lets the
arrow keys render any past ply from `gamestate.saved_states`. UI state lives in `main.UIState`; the
`selected_square` / `highlighted_moves` fields on `GameState` are legacy. Hit regions are produced by
the code that draws each control, so a control can never be clickable where it is not visible — and
the promotion picker clears every other region, which stops clicks reaching the buttons underneath.

`gui_settings.json` (gitignored) remembers window size, depth, AI toggles, hints, hint-line count,
`hint_workers`, and orientation. Both AI toggles default to **off**.

## Search settings the user can see

**The depth control governs every search, including the hint.** There is no private cap on any search
path; a setting the user can see must be the one that runs. Wanting a faster hint is what lowering
the depth is for, and the header shows the hint's elapsed time so a deep one is visible rather than
mistaken for a hang.

**Controls that cannot change anything still answer.** The depth and hint-line steppers keep their end
buttons in `hits` at the top and bottom of their ladders, and `set_depth` / `set_hint_lines` toast
"Depth 10 is the deepest setting" rather than returning silently. The same rects are registered under
`hits['wheel']` so the mouse wheel nudges whichever stepper it is over.

**Scores are white-relative everywhere.** `search::find_best_move`, `eval::evaluate_position` and
`ai.find_best_move_with_score` all return positive-favours-White whichever side is to move;
`utils.format_score` / `score_advantage` are the only formatters. The readout falls back to the static
evaluation on every position change and is overwritten by a completed search labelling itself
`depth N` — "static" and "depth 10" are different claims about the same number, so the source is
always shown. `eval.rs` encodes a mate as flat ±`CHECKMATE_SCORE`, so the display says "White mates",
never "M3".

**Hints run several at a time, one core each.** `thread_utils.HintPool` keys every hint search by the
question it answers — `(fen, depth, lines)` — and starts up to `hint_workers` at once. Nothing is
cancelled when the board moves on, so playing three moves quickly searches all three positions side
by side. Answers are cached (`KEEP_RESULTS`), so stepping back into a position shows its lines with
no search.

- **A Python thread cannot be stopped**, so `hint_workers` is a hard budget. `submit` declines when
  every slot is taken and the caller retries next frame; lowering the setting does not free a core
  that is already busy.
- **`hint_key` folds depth and line count in.** That is what makes `drop_hint` cheap — a search
  started under the old setting files its answer under the old key and stays out of the way. It is
  also why `set_depth` bumps `generation` itself: `drop_hint` no longer does, and the engine's own
  move still has to be invalidated.
- **`poll_hint` gates the display on `show_hint` and whose turn it is.** The pool keeps answering
  regardless; without the gate a cached hint would reappear the moment hints were switched off.
- **The ladder is capped to the machine** (`settings.hint_worker_choices`, `cpu_count - 1`). A hint
  search is single-threaded, so N workers means N cores busy. Turning `set_parallel_search` on as
  well would oversubscribe.
- **What a high setting costs is memory, not stability.** `SearchState::new()` pre-allocates a
  transposition table *per search*, so every concurrent worker is tens of MB of RSS — the one
  consequence the window cannot otherwise show, which is why `set_hint_workers` names it in its toast.

## Choosing the engine

The **Engine** button (below the steppers, above Save position to book) switches the AI's moves between the
alpha-beta search and the network searched with MCTS (`nn/backend.py`, `docs/ZERO.md`). It
defaults to the search, and the setting persists in `gui_settings.json`.

- **Selecting the network is what loads torch.** `thread_utils.AIThread` imports
  `nn.backend` inside `run()`, never at module scope, so a user who never touches the
  button never pays for the CUDA stack — and the several seconds of first import happen
  on the worker thread, where the "search never blocks the UI" invariant already covers
  them. `tests/test_nn.py::test_the_front_ends_do_not_import_torch` checks this in a
  subprocess, because in-process it would prove nothing.
- **The toggle refuses rather than breaking.** `backend.available()` answers from the
  filesystem without importing torch, and `unavailable_reason()` says which of "torch is
  not installed" and "no trained checkpoint" applies. Both must agree on every branch or
  the GUI formats `None` into a toast.
- **The checkpoint is the newest `best.pt`** under `$MINIZERO_DATA/checkpoints/*/`, or
  `$MINIZERO_CHECKPOINT`. Training a new one is enough to make the GUI play it; there is
  no second place to update and so no way for the two to disagree.
- **The readout says `network`, not `depth N`.** `set_search_score` takes the source
  because "static", "depth 10" and "network" are different claims about the same number.
  The network has no depth, so `player_label` says "network · 0.5s" and the thinking row
  (`ui.think_label`) names the budget rather than a search that did not happen. The score
  is the MCTS root's mean value put back through `VALUE_SCALE * atanh(v)` and flipped
  into the white-relative convention everything else in the app speaks — **but it is not
  a trustworthy position assessment**: the value head carries a side-to-move bias from
  its teacher set and reads about +178cp at the opening position, where the search says
  +6. The tree averages that bias rather than removing it. Read the moves, not the number
  (`docs/ZERO.md`).
- **Hints always use the search.** The network has no depth to vary and no ranked lines
  to show, so the hint path is untouched.
- **Every control needs a grid row of its own.** The controls grid is nine rows: 0-2
  buttons, 3-5 the hint/depth steppers, 6 the Engine button, 7 the network's budget, 8
  Save position to book. `layout.controls_h` counts them, so adding a control means
  bumping that count as well as picking a free row. Reusing an occupied row makes the two
  controls fail in opposite directions — drawing is sequential so the later one paints
  over the earlier, while hit-testing returns on the first match in insertion order, so
  the *invisible* control takes the clicks. `tests/test_gui_controls.py` asserts no two
  hit regions overlap, which is the machine-checkable form of the "a control can never be
  clickable where it is not visible" invariant above.

### The network's budget is a time, not a depth

The **Network** stepper directly under the Engine button sets how long MCTS searches for
each of the engine's moves: `settings.NET_TIME_CHOICES`, 0.1s to 5s, default 0.5s,
persisted as `net_seconds`. It reads like the depth stepper and obeys the same rule — the
setting the user can see is the one that runs — but it counts seconds rather than plies.

- **Why time and not simulations.** A simulation costs whatever the position makes it
  cost: a full-hand middlegame descends further and branches wider than an opening, so
  the same count is not the same wait twice running. The wait is what an interactive user
  is actually spending, and it is the only budget the GUI can promise. It is also the
  unit the project's own success bar is stated in (`docs/ZERO.md`): equal *time control*
  against alpha-beta, which is why "800 sims" was retired as a gate. The arena and the
  self-play loop still count simulations, because a measurement wants reproducibility
  rather than a clock; `nn.mcts.search` takes either, or both, and stops at the first.
- **0.5s is the default because it is roughly what depth 6 costs.** ~4,000 simulations at
  the opening on this machine, against the 2,400 that drew level with depth-6 alpha-beta
  at 343ms a move. The two engines therefore feel like each other out of the box.
- **The first move pays no warm-up out of its budget.** The first forward pass on a
  fresh CUDA device builds the context and picks kernels — ~250ms here, a whole default
  think time — so before `nn.backend._Network._warm_up` the first network move came back
  after *one* simulation, which is a move nothing chose. The warm-up runs at load, on the
  background thread where a slow start is already covered. `nn.mcts.search` guards the
  same failure from the other side: the deadline cannot stop a search that has not yet
  visited a child.
- **The deadline is read between batches**, so a move can overrun it by one network call
  — a few milliseconds at the batch of 16 `nn.backend.DEFAULT_BATCH` fixes. The batch is
  not user-facing: it is a throughput knob whose value (~8x, from virtual loss) is
  measured in `docs/ZERO.md`.
- **The tree survives between moves.** `nn.backend._Network` holds one
  `nn.mcts.Searcher`, so a move starts from the subtree under the move actually played
  rather than from an empty root — the same second of thinking buys a root standing at
  two to three times the visits. It needs no signal from the GUI: reuse is keyed on the
  **position**, so a new game, an undo, or any position the tree cannot reach in two
  plies simply starts a fresh tree. The thinking row still reports the simulations *this*
  move paid for, not the ones it inherited, because that is what the second actually
  bought.
- **It is muted, not hidden, while the alpha-beta engine is selected**, the same way the
  hint steppers mute when hints are off — a ladder that appears and disappears moves the
  buttons under the cursor. Changing it while the search is alpha-beta toasts that it
  will not take effect yet.
- **No key binding.** The wheel over the stepper nudges it (`hits['wheel']['nettime']`),
  which is what the status band already advertises; the letter keys are full.
- **Changing it bumps `generation`**, like `set_depth`: a search already in flight was
  started under the old budget and is not the answer the new setting promises.

**Tree reuse across moves is not implemented.** The engine builds a fresh tree every move,
so the subtree under the move actually played — worth roughly 2x — is discarded. Doing it
needs a re-root on the Rust side (`rs.Mcts` only takes a starting position) plus a tree
kept alive across `AIThread`s, which the GUI's deep-copy-per-move path does not currently
allow.

**It is now competitive with the search**, which it was not as a raw policy argmax:
`docs/ZERO.md` measures the same weights at 0.185 against depth-2 with no tree and 0.833
with 800 simulations, and level with depth-6 at 2,400. Playing it against depth 6 is a
fair fight.

**More than one hint line means MultiPV.** `HintThread(lines=n)` with `n > 1` calls
`find_best_move(..., return_top_n=n)`, several times the cost of a single-PV search at the same depth
— that cost is why the count is a user-facing setting and why the label names the multiplier. `n == 1`
routes through `find_best_move_with_score` instead. A forced move carries no search behind it, so its
placeholder `0` is turned back into `None` rather than shown as `+0.00`.

