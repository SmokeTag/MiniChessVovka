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
  search state, so it cannot resize mid-search. Changing it calls `Game.relayout()`.
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

**More than one hint line means MultiPV.** `HintThread(lines=n)` with `n > 1` calls
`find_best_move(..., return_top_n=n)`, several times the cost of a single-PV search at the same depth
— that cost is why the count is a user-facing setting and why the label names the multiplier. `n == 1`
routes through `find_best_move_with_score` instead. A forced move carries no search behind it, so its
placeholder `0` is turned back into `None` rather than shown as `+0.00`.

