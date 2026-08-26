export const meta = {
  name: 'parallel-search-eval',
  description: "Measure and improve the Rust engine's root-parallel search for interactive analysis, without touching the single-threaded self-play path",
  whenToUse: 'When you want a measured pass over minimax_parallel in engine_rs/src/search.rs: seq-vs-parallel baseline, candidate improvements from several lenses, each one implemented and benchmarked in its own worktree, adversarially verified, then written up. Self-play stays single-threaded throughout.',
  phases: [
    { title: 'Harness', detail: 'build/refresh the bench suite, record machine load' },
    { title: 'Baseline', detail: 'sequential vs parallel across depths, fresh process per run' },
    { title: 'Analyze', detail: 'four lenses over the root-parallel search' },
    { title: 'Rank', detail: 'dedup + judge; drop anything that changes the sequential path' },
    { title: 'Implement', detail: 'one worktree per candidate: patch, build, benchmark' },
    { title: 'Verify', detail: 'adversarial re-check of each claimed win' },
    { title: 'Report', detail: 'write docs/PARALLEL_SEARCH.md' },
  ],
}

// ---------------------------------------------------------------- config
const cfg = {
  depths: (args && args.depths) || [4, 5, 6, 7],
  positions: (args && args.positions) || 8,
  candidates: (args && args.candidates) || 3,
  repeats: (args && args.repeats) || 3,
  lenses: (args && args.lenses) || null,
}

// ------------------------------------------------------- shared context
const CONTEXT = `
Repo: /home/andre/Development/chess — 6x6 crazyhouse mini-chess. Python front end, Rust engine via pyo3.

Where the parallel search lives:
- engine_rs/src/search.rs
    minimax_parallel()      ~line 649 — root PVS split with rayon: order root moves, search the
                            best-ordered one sequentially for a real alpha, scout every other root
                            move in parallel against a null window, re-search at full width only
                            those that beat the baseline.
    search_worker()         ~line 592 — one root move, own GameState copy, own SearchState with a
                            1<<14 TT, seeded killers/history, Arc<HashMap> read-only base_tt.
    merge_worker_tt()       ~line 631 — folds worker TT entries back, deeper wins.
    SearchState / base_tt   ~line 56  — tt_get() probes the local table, then the shared snapshot.
    PARALLEL_ENABLED / PARALLEL_MIN_DEPTH atomics ~line 18 — default (false, 3).
    find_best_move()        ~line 796 — iterative deepening; picks the parallel branch when
                            current_depth >= parallel_threshold.
- engine_rs/src/lib.rs — pyo3 bindings. find_best_move(gs, depth, return_top_n, time_limit, parallel),
    set_parallel_search(enabled, min_depth), get_parallel_search(). Global MOVE_CACHE behind a Mutex.
- ai.py — thin Python wrapper over the above.
- src/self_play.py:221 and src/scheduled_self_play.py:430 — training pins parallel search OFF;
    both call find_best_move(..., parallel=False) explicitly.
- gamestate.py — Python GameState. There is NO FEN parser: build positions by replaying a fixed
    move list from a fresh GameState() via make_move / make_ai_move.

Machine: 24 cores. The interpreter with the engine installed is ./venv/bin/python.

HARD CONSTRAINTS — these are the point of the exercise, not boilerplate:
1. The user runs single-threaded self-play overnight, many independent games across cores, and that
   scales better than parallelising one search. Root-parallel search MUST stay off by default
   (PARALLEL_ENABLED starts false) and self-play must keep passing parallel=False. Parallel search
   is for interactive analysis of one position only.
2. No change may alter the behaviour or the speed of the sequential path — minimax_ab, quiescence,
   move ordering, evaluation, and the TT as used when parallel=False. Parallel-only code paths only.
   If a candidate must touch shared code, it has to prove the sequential path is unchanged by
   measurement, not by argument.
3. NEVER run 'maturin develop' against the repo's root venv (/home/andre/Development/chess/venv).
   Overnight self-play may have that module loaded right now; replacing the .so under it is
   destructive. Inside a worktree, build into a throwaway venv you create there.
4. Benchmarking pitfalls that will silently produce nonsense numbers:
   - find_best_move consults a process-global MOVE_CACHE keyed by (position hash, depth) and writes
     to it as it searches. Re-running the same (position, depth) in the same process returns a cache
     hit in ~0s. Run every (position, depth, mode) measurement in a FRESH process, and never call
     load_move_cache_from_db() from a benchmark.
   - The box may be busy with self-play. Record the 'uptime' load average alongside every
     measurement. Scaling numbers taken under load average > 4 are unreliable — report that fact
     rather than quietly averaging it away.
   - The parallel path prints [PARALLEL] timing lines to stderr (baseline / scout / re-search
     seconds, move counts). That split is the most useful signal you have for where the time goes;
     capture stderr, do not discard it.
`

// ------------------------------------------------------------- schemas
const HARNESS_SCHEMA = {
  type: 'object',
  required: ['ready', 'runCommand', 'positionsFile', 'loadAverage', 'notes'],
  properties: {
    ready: { type: 'boolean' },
    runCommand: { type: 'string', description: 'exact shell command that produces a results JSON' },
    positionsFile: { type: 'string' },
    resultsFile: { type: 'string' },
    loadAverage: { type: 'string' },
    notes: { type: 'string' },
  },
}

const BASELINE_SCHEMA = {
  type: 'object',
  required: ['rows', 'summary', 'loadAverage'],
  properties: {
    rows: {
      type: 'array',
      items: {
        type: 'object',
        required: ['position', 'depth', 'mode', 'seconds'],
        properties: {
          position: { type: 'string' },
          depth: { type: 'integer' },
          mode: { type: 'string', enum: ['sequential', 'parallel'] },
          seconds: { type: 'number' },
          bestMove: { type: 'string' },
          score: { type: 'number' },
        },
      },
    },
    speedupByDepth: {
      type: 'array',
      items: {
        type: 'object',
        required: ['depth', 'speedup'],
        properties: {
          depth: { type: 'integer' },
          speedup: { type: 'number' },
          moveAgreement: { type: 'string' },
          scoutFraction: { type: 'number' },
          researchMoves: { type: 'number' },
        },
      },
    },
    disagreements: { type: 'array', items: { type: 'string' } },
    resultsFile: { type: 'string' },
    loadAverage: { type: 'string' },
    summary: { type: 'string' },
  },
}

const CANDIDATES_SCHEMA = {
  type: 'object',
  required: ['candidates'],
  properties: {
    candidates: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'title', 'mechanism', 'expectedGain', 'risk', 'touchesSequentialPath', 'files', 'evidence'],
        properties: {
          id: { type: 'string', description: 'kebab-case, stable' },
          title: { type: 'string' },
          mechanism: { type: 'string', description: 'what changes, concretely, and why it is faster or more correct' },
          expectedGain: { type: 'string', description: 'quantified where possible, tied to a baseline number' },
          risk: { type: 'string', enum: ['low', 'medium', 'high'] },
          touchesSequentialPath: { type: 'boolean' },
          files: { type: 'array', items: { type: 'string' } },
          evidence: { type: 'string', description: 'the file:line or measurement this rests on' },
        },
      },
    },
  },
}

const RANK_SCHEMA = {
  type: 'object',
  required: ['picked', 'rejected'],
  properties: {
    picked: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'title', 'mechanism', 'why', 'acceptance'],
        properties: {
          id: { type: 'string' },
          title: { type: 'string' },
          mechanism: { type: 'string' },
          why: { type: 'string' },
          acceptance: { type: 'string', description: 'the measurement that decides pass/fail' },
          files: { type: 'array', items: { type: 'string' } },
        },
      },
    },
    rejected: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'reason'],
        properties: { id: { type: 'string' }, reason: { type: 'string' } },
      },
    },
  },
}

const IMPL_SCHEMA = {
  type: 'object',
  required: ['id', 'implemented', 'summary', 'diff', 'before', 'after', 'sequentialUnchanged'],
  properties: {
    id: { type: 'string' },
    implemented: { type: 'boolean' },
    summary: { type: 'string' },
    diff: { type: 'string', description: 'unified diff of the change, git diff output' },
    before: { type: 'string', description: 'baseline timings this worktree measured itself' },
    after: { type: 'string' },
    speedup: { type: 'number' },
    moveAgreement: { type: 'string' },
    sequentialUnchanged: { type: 'boolean' },
    sequentialEvidence: { type: 'string' },
    worktreePath: { type: 'string' },
    notes: { type: 'string' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  required: ['id', 'holdsUp', 'verdict', 'concerns'],
  properties: {
    id: { type: 'string' },
    holdsUp: { type: 'boolean' },
    verdict: { type: 'string', description: 'CONFIRMED | PLAUSIBLE | REFUTED' },
    concerns: { type: 'array', items: { type: 'string' } },
    remeasured: { type: 'string' },
  },
}

// ---------------------------------------------------------------- lenses
const LENSES = cfg.lenses || [
  {
    key: 'scaling',
    prompt: `LENS: parallel scaling and where the wall-clock actually goes.

Read minimax_parallel and the [PARALLEL] stderr lines in the baseline. Work out how much of each
iteration is the sequential baseline move, how much is the parallel scout, and how much is the
serial full-width re-search loop — then say what the Amdahl ceiling is at 24 cores given that split.
Look hard at: the baseline move being searched alone before any fan-out; the re-search loop running
serially in the root state; load imbalance across scout workers (one root move can dominate);
PARALLEL_MIN_DEPTH=3 being the right or wrong threshold; the fact that every iterative-deepening
iteration re-pays the fan-out; whether rayon's default pool size is right when 24 cores may already
be busy; and whether young-brothers-wait / lazy-SMP style alternatives would fit this engine better
than the current scout-then-re-search shape.`,
  },
  {
    key: 'correctness',
    prompt: `LENS: correctness and determinism of the parallel result.

The parallel path must return what the sequential path would return, or the analysis is worthless.
Scrutinise: the null-window (scout_alpha, scout_beta) construction for both colours; the re-search
window (best_score, inf) / (neg_inf, best_score) and whether it can miss a better move when several
candidates beat the baseline; the abort path — a stopped worker poisons the whole iteration and
returns Move::NULL, and find_best_move only trusts aborted results at depth <= 2; TT entries
harvested from workers that raced (merge_worker_tt keeps the deeper entry — is depth the right key
when entries came from different windows/bounds?); mate-score handling in all_results; and whether
repeated runs of the same position at the same depth in parallel mode give byte-identical results.
Propose a differential test that would catch any seq/par divergence, and name concrete divergences
you can already see in the code.`,
  },
  {
    key: 'memory',
    prompt: `LENS: transposition sharing, allocation, and per-worker cost.

Each root worker builds a fresh SearchState with a 1<<14 HashMap, clones the killer and history maps,
clones an Arc to a filtered base_tt snapshot, and fast_copy()s the board. The root then rebuilds that
base_tt snapshot from scratch at every depth. Quantify that cost against the measured iteration times.
Consider: filtering the snapshot every iteration vs. keeping it incrementally; std HashMap with the
default SipHash hasher on u64 Zobrist keys; a fixed-size array-indexed TT with atomic entries shared
read-write by all workers instead of the snapshot-and-merge scheme; the merge pass cost at the end of
each iteration; and whether workers should share a global TT at all given that this same TT code is
used by the sequential path the user runs overnight (constraint 2 applies — a shared-TT redesign that
changes sequential behaviour is out of bounds unless it is strictly parallel-only).`,
  },
  {
    key: 'analysis-ux',
    prompt: `LENS: what interactive analysis actually needs from this search.

The parallel path exists to analyse one position with every core. Check what the API gives an analyst
today: minimax_parallel computes all_results (every root move with a score) and then throws it away —
lib.rs find_best_move with return_top_n > 1 returns a single-element list regardless. Look at
thread_utils.py HintThread and gui.py to see how analysis is driven. Consider: real multi-PV / top-N
output from the parallel root, which is nearly free since the scores already exist; time-limited
analysis quality (the current abort discards a whole iteration — an analyst would rather keep the
previous depth's full result and the partial root scores); progress/PV reporting during a long think;
and whether set_parallel_search should be turned on automatically in the GUI/hint path while staying
off for self-play. Every proposal must leave src/self_play.py and src/scheduled_self_play.py behaviour
byte-identical.`,
  },
]

// ------------------------------------------------------------------ run
phase('Harness')
log('Building the seq-vs-parallel benchmark harness')

const harness = await agent(
  `${CONTEXT}

TASK: build (or refresh, if bench/ already exists) a benchmark harness for the root-parallel search,
committed nowhere — just written to the working tree at /home/andre/Development/chess.

Deliverables:
1. bench/positions.json — ${cfg.positions} positions spanning opening, middlegame with pieces in hand,
   sharp tactical, and endgame. Since there is no FEN parser, store each position as a name plus the
   exact move list to replay from a fresh GameState(), and include a loader that reconstructs it.
   Generate the move lists by playing the engine against itself briefly at low depth, or by hand —
   whichever you can make deterministic. Verify every position replays cleanly and has >= 8 legal moves
   (a position with one legal move short-circuits find_best_move and measures nothing).
2. bench/run_bench.py — for each position x depth x mode, spawn a FRESH ./venv/bin/python subprocess
   that searches exactly once and prints one JSON line: position, depth, mode, seconds, best move,
   score, plus the [PARALLEL] baseline/scout/re-search seconds parsed out of stderr. Flags:
   --depths, --repeats, --modes, --positions, --out. Take the minimum across repeats, not the mean —
   minimum is the least noise-contaminated estimator when a background job is stealing cores.
   Record 'uptime' load average and nproc into the results file header.
3. bench/compare.py — read two results files and print a table: per-depth speedup (sequential seconds
   / parallel seconds), best-move agreement, score deltas, and the scout/re-search time split.
   It must call out disagreements loudly; a fast wrong answer is a regression.

Rules: use the ALREADY-BUILT engine in ./venv — do NOT rebuild it, do not run maturin here.
Do not modify any engine, game, or training file. bench/ is new code only.
Confirm the harness runs end to end on ONE position at depth 4 in both modes before returning,
and report the load average you saw.`,
  { schema: HARNESS_SCHEMA, label: 'bench-harness' }
)

if (!harness || !harness.ready) {
  return { error: 'harness not ready', harness }
}
log(`Harness ready: ${harness.runCommand} (load ${harness.loadAverage})`)

phase('Baseline')
const baseline = await agent(
  `${CONTEXT}

The benchmark harness is built:
  run command: ${harness.runCommand}
  positions:   ${harness.positionsFile}
  notes:       ${harness.notes}

TASK: produce the baseline. Run the full matrix — depths ${cfg.depths.join(', ')}, both modes,
${cfg.repeats} repeats, all positions — and write the results to bench/results/baseline.json.

Then report, per depth: speedup (sequential / parallel), best-move agreement between the two modes,
the fraction of parallel time spent in the sequential baseline move vs. the scout vs. the serial
re-search, and how many moves needed re-searching. List every position where the two modes returned
different moves or materially different scores; those are correctness findings, not noise.

State the load average at the start and end. If the box is loaded, say the numbers are contaminated
and by roughly how much — do not present contaminated speedups as clean ones.`,
  { schema: BASELINE_SCHEMA, label: 'baseline' }
)

const baselineDigest = baseline
  ? `Baseline summary: ${baseline.summary}
Per-depth: ${JSON.stringify(baseline.speedupByDepth || [])}
Seq/par disagreements: ${JSON.stringify(baseline.disagreements || [])}
Load average: ${baseline.loadAverage}
Results file: ${baseline.resultsFile || 'bench/results/baseline.json'}`
  : 'Baseline measurement failed — reason about the code, and say plainly that no measured baseline exists.'

log('Baseline measured; fanning out four analysis lenses')

phase('Analyze')
const lensResults = await parallel(
  LENSES.map((l) => () =>
    agent(
      `${CONTEXT}

${baselineDigest}

${l.prompt}

Return concrete, implementable candidates — each one a change someone could make to this repo today,
with the file and the mechanism, tied to a baseline number or a specific line of code. No generic
engine-programming advice. Mark touchesSequentialPath honestly; a true value there is usually fatal
under constraint 2, so only set it false when the change really is parallel-only.
Between 2 and 5 candidates. Quality over count.`,
      { schema: CANDIDATES_SCHEMA, label: `lens:${l.key}`, phase: 'Analyze' }
    )
  )
)

const allCandidates = lensResults.filter(Boolean).flatMap((r) => r.candidates || [])
const seen = new Set()
const deduped = allCandidates.filter((c) => {
  const k = (c.id || c.title || '').toLowerCase().replace(/[^a-z0-9]+/g, '-')
  if (!k || seen.has(k)) return false
  seen.add(k)
  return true
})
log(`${allCandidates.length} candidates, ${deduped.length} after dedup`)

if (deduped.length === 0) {
  return { baseline, candidates: [], note: 'no candidates produced' }
}

phase('Rank')
const ranked = await agent(
  `${CONTEXT}

${baselineDigest}

Candidates from four independent lenses:
${JSON.stringify(deduped, null, 2)}

TASK: pick the ${cfg.candidates} worth implementing now, and reject the rest with a one-line reason.

Ranking rules, in order:
1. Reject anything that changes the sequential path's behaviour or speed, unless it is trivially
   provable as parallel-only. The overnight single-threaded self-play is the workload that matters.
2. Prefer candidates whose gain is tied to a measured number in the baseline over ones resting on
   general theory.
3. Prefer a correctness fix over a speed win of similar size — a fast wrong analysis is worse than a
   slow right one.
4. Prefer changes that fit in one file and can be measured in one benchmark run.

For each pick, write an acceptance criterion: the exact measurement that decides whether it worked,
expressed against the baseline numbers above.`,
  { schema: RANK_SCHEMA, label: 'rank' }
)

const picks = (ranked && ranked.picked ? ranked.picked : deduped.slice(0, cfg.candidates)).slice(0, cfg.candidates)
log(`Implementing ${picks.length}: ${picks.map((p) => p.id).join(', ')}`)

phase('Implement')
const results = await pipeline(
  picks,
  (pick) =>
    agent(
      `${CONTEXT}

${baselineDigest}

TASK: implement and measure exactly one candidate.

  id:          ${pick.id}
  title:       ${pick.title}
  mechanism:   ${pick.mechanism}
  acceptance:  ${pick.acceptance}
  files:       ${(pick.files || []).join(', ')}

You are in your own git worktree — a private copy of the repo. Work only here.

Build setup, in this order, and do not deviate:
  1. python3 -m venv .venv-bench          (inside YOUR worktree)
  2. .venv-bench/bin/pip install maturin
  3. cd engine_rs && ../.venv-bench/bin/maturin develop --release
  4. .venv-bench/bin/pip install -r requirements.txt
Never touch /home/andre/Development/chess/venv. Overnight self-play may be running against it.

Then:
  a. Measure YOUR OWN before-numbers with the bench harness (copy bench/ from the main worktree if
     your worktree predates it) at depths ${cfg.depths.join(', ')} — the main baseline was taken on a
     different build and a possibly different machine load, so it is context, not your control group.
  b. Implement the change. Keep it minimal and idiomatic to the surrounding Rust.
  c. Rebuild, re-measure the same matrix.
  d. Prove the sequential path is unaffected: run the same positions with parallel=False before and
     after, and compare both the timings and the moves returned. If they differ at all, say so — that
     is a failure of constraint 2, and reporting it honestly is worth more than a speedup.
  e. Run the existing tests: .venv-bench/bin/python -m pytest tests/ -x -q (note which pass/fail).

Return the unified diff (git diff), the before/after tables, and the sequential-path evidence.
If the change turns out not to help, return implemented:true with honest numbers showing it did not.
Do not report a win you did not measure.`,
      { schema: IMPL_SCHEMA, label: `impl:${pick.id}`, phase: 'Implement', isolation: 'worktree' }
    ),
  (impl, pick) =>
    impl && impl.implemented
      ? agent(
          `${CONTEXT}

A candidate was implemented and claims a result. Your job is to REFUTE it. Default to holdsUp:false
unless the evidence survives your attack.

  id:          ${impl.id}
  claim:       ${impl.summary}
  acceptance:  ${pick.acceptance}
  before:      ${impl.before}
  after:       ${impl.after}
  seq path unchanged: ${impl.sequentialUnchanged} — ${impl.sequentialEvidence || 'no evidence given'}

Diff:
${impl.diff}

Attack it on four fronts:
1. Measurement. Was every run in a fresh process (MOVE_CACHE!)? Repeats taken as minimum? Was the box
   loaded during one half of the comparison and idle during the other? Is the speedup inside the noise?
2. Correctness. Does the diff let the parallel search return a different move or score than the
   sequential search would? Walk the null-window and re-search windows for both colours, the abort
   path, and the TT merge. A speedup bought with a wrong move is a regression.
3. Constraint 2. Does anything in this diff execute when parallel=False? Trace it — do not take the
   claim on faith. minimax_ab, quiescence, eval, move ordering, and the TT as used sequentially must
   be untouched in behaviour and in speed.
4. Constraint 1. Does anything here flip the default on, or change what src/self_play.py and
   src/scheduled_self_play.py do?

You have your own worktree. Where the numbers are the crux, re-measure them yourself: set up
.venv-bench exactly as described (never the repo's root venv), apply the diff, and run it. Say what
you re-measured and what you only reasoned about.`,
          { schema: VERDICT_SCHEMA, label: `verify:${impl.id}`, phase: 'Verify', isolation: 'worktree' }
        ).then((v) => ({ ...impl, pick, verdict: v }))
      : { ...(impl || { id: pick.id, implemented: false }), pick, verdict: null }
)

const finished = results.filter(Boolean)
const confirmed = finished.filter((r) => r.verdict && r.verdict.holdsUp)
log(`${confirmed.length}/${finished.length} candidates survived verification`)

phase('Report')
const report = await agent(
  `${CONTEXT}

TASK: write /home/andre/Development/chess/docs/PARALLEL_SEARCH.md — the record of this evaluation.
Match the tone and structure of the existing files in docs/ (read one first).

Material:

BASELINE
${JSON.stringify(baseline, null, 2)}

RANKING
${JSON.stringify(ranked, null, 2)}

IMPLEMENTED AND VERIFIED (diffs elided where huge; keep the file:line references)
${JSON.stringify(
  finished.map((r) => ({
    id: r.id,
    summary: r.summary,
    before: r.before,
    after: r.after,
    speedup: r.speedup,
    moveAgreement: r.moveAgreement,
    sequentialUnchanged: r.sequentialUnchanged,
    worktreePath: r.worktreePath,
    verdict: r.verdict,
  })),
  null,
  2
)}

Sections:
1. What the parallel root search does today, and the measured baseline table (speedup per depth,
   scout vs re-search split, move agreement). State the machine load the numbers were taken under.
2. Where the time goes and what the ceiling is — the honest version, including where the speedup is
   worse than the core count would suggest and why.
3. Candidates considered, and why the rejected ones were rejected.
4. Each implemented candidate: what changed, measured before/after, the verifier's verdict, and
   whether it is recommended. Confirmed wins and refuted claims both get written down.
5. Standing constraint, stated plainly for the next reader: root-parallel search stays OFF by default;
   overnight self-play runs single-threaded because many independent games across cores scale better
   than one parallelised search; every change here is parallel-path-only.
6. How to re-run: the bench harness commands.

Then, in your returned text, give the user a short verdict: which changes are worth applying, where
they live (worktree paths — nothing has been merged into the main working tree), and what to run.
Do NOT apply any candidate's diff to the main working tree. That is the user's call.`,
  { schema: { type: 'object', required: ['reportPath', 'verdict'], properties: { reportPath: { type: 'string' }, verdict: { type: 'string' }, recommended: { type: 'array', items: { type: 'string' } } } }, label: 'report' }
)

return {
  baseline,
  ranked,
  implemented: finished.map((r) => ({ id: r.id, speedup: r.speedup, verdict: r.verdict, worktree: r.worktreePath })),
  confirmed: confirmed.map((r) => r.id),
  report,
}
