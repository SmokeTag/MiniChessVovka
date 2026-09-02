#!/usr/bin/env python3
"""Teacher data: positions labelled by the alpha-beta engine at a fixed depth.

Phase 2 of Minihouse Zero bootstraps from the engine that already exists rather than
from noise. Each record is a position plus the depth-N search of it: the best move
becomes the policy target, the score becomes the value target. Training on this proves
the encoding, the network and the training loop end to end in hours instead of days --
and then the weights are thrown away and zero runs from random on infrastructure that
has already been trusted once. See docs/ZERO.md.

    ./venv/bin/python -m nn.teacher generate --positions 150000 --jobs 20
    ./venv/bin/python -m nn.teacher stats

**No search here ever touches book.db.** Every one runs under `use_book=False`, which
gates the probe *and* the store (search.rs), so nothing is read from the process-global
book and nothing is queued for writing. Labels must all come from the same depth; a book
row is whatever depth it happened to be searched at, and the frozen book's rows predate
the 2026-09-01 quiescence fix besides.

**Records store the position, not its encoding.** A record is a FEN plus `ply` and
`reps`, which is exactly what `encode_position` reads -- so a change to the plane layout
costs a re-encode, not a regeneration, and regeneration is the expensive half by four
orders of magnitude. `restore()` is the inverse and the generator asserts on every
position that it reproduces the live encoding bit for bit; a systematic train/play skew
in the progress or repetition planes is otherwise invisible.

Output is JSONL shards under $MINIZERO_DATA/teacher/<name>/, one per task, and the run
is resumable: a shard that already holds its full quota is skipped.
"""
import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import paths

# Walk shape. A pure random walk reaches positions no game ever plays; a pure engine
# walk reaches one line. Each walk draws its own mix, so the sample spans both.
MIN_PLY = 2
MAX_PLY = 80
WALK_DEPTH = 3
# Weighted toward engine play. The first 50k of teacher/depth8 used
# [0.0, 0.25, 0.5, 0.75] and the network trained on it agreed with depth-8 on 44% of
# held-out *teacher* positions but only 31% of the positions it reached in its own
# games -- the walks were too random to look like the games the arena actually plays.
# Keeping both mixes in one set is deliberate: breadth from the random walks, relevance
# from the engine-guided ones.
ENGINE_MIX = [0.25, 0.5, 0.75, 1.0]

# tanh(score / VALUE_SCALE) is the value target, and the scale is set by the *teacher
# distribution*, not by taste. Positions reached by walking with weak moves are wildly
# unbalanced: the median non-mate |score| here is ~1100cp, over ten pawns. At the
# obvious-looking 400 that put 79% of all targets past |0.9|, and a value head trained on
# it saturates -- it cannot rank two candidate moves, which showed up as one ply of value
# lookahead scoring exactly the same as no lookahead at all. At 2000, 6.7% of non-mate
# targets saturate and the remaining mass past 0.9 is the 27% that really are decided.
#
# Retunable without regenerating anything, because a record stores the score and not the
# tensor. Measure before changing it: `nn.teacher stats` reports the saturated fraction.
VALUE_SCALE = 2000.0

# eval.rs encodes a mate as flat +/-CHECKMATE_SCORE with no distance, so anything past
# the cutoff is "decided" and clamps to +/-1.0 rather than passing through tanh.
MATE_CUTOFF = 900_000

def encode_move(m):
    if m[0] == "drop":
        return ["drop", m[1], [m[2][0], m[2][1]]]
    return [[m[0][0], m[0][1]], [m[1][0], m[1][1]], m[2]]

def decode_move(m):
    if isinstance(m, (list, tuple)) and len(m) == 3 and m[0] == "drop":
        return ("drop", m[1], (m[2][0], m[2][1]))
    return ((m[0][0], m[0][1]), (m[1][0], m[1][1]), m[2])

def value_target(score_white, turn):
    """White-relative centipawns -> the mover's own value in [-1, 1].

    Scores are white-relative everywhere in this project (CLAUDE.md); the network is
    canonicalised to the side to move, so the sign has to follow the same flip the
    board does.
    """
    import math
    s = score_white if turn == "w" else -score_white
    if s >= MATE_CUTOFF:
        return 1.0
    if s <= -MATE_CUTOFF:
        return -1.0
    return math.tanh(s / VALUE_SCALE)

def restore(record):
    """Record -> the Rust GameState it was taken from, exactly as the encoder sees it.

    A FEN carries what the Zobrist hash reads and nothing path-dependent, so `ply` and
    the repetition count have to be put back by hand or the progress and repetition
    planes would read 0 at training time and something else at play time.
    """
    import minichess_engine as rs
    gs = rs.from_fen(record["fen"])
    gs.ply = record["ply"]
    gs.ply_limit = record.get("ply_limit", 200)
    # Assigning an empty list makes the setter seed the history with the live hash,
    # which is the only way to read that hash back out of Python.
    gs.position_history = []
    reps = record.get("reps", 1)
    if reps > 1:
        gs.position_history = [int(gs.position_history[-1])] * reps
    return gs

def _silence_engine():
    """The engine narrates every iterative-deepening step -- `[ID] depth N...` on
    stderr, from search.rs. Twenty workers doing that at three searches a second is
    gigabytes of noise, so both descriptors go to /dev/null.

    Nothing is lost: a worker that raises has its exception re-raised in the parent by
    Pool, and the parent's own reporting is untouched.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

def _sample_position(rng, rs):
    """One fresh walk. Returns (GameState, legal moves) or None if the walk died.

    **The walk stays entirely inside the Rust GameState.** gamestate.py is the pure
    Python implementation and owns the live game, but walking it costs milliseconds a
    ply; the Rust one does ~126k plies/s, which puts the walk three orders of magnitude
    under the depth-8 label it exists to feed. Driving the walk from Python instead made
    the *walk* the bottleneck and the whole generator ~15x slower. The two
    implementations agree by construction (tests/test_rules_parity.py).

    A fresh walk per position, rather than several labels along one game, buys full
    independence between samples -- and now costs nothing.
    """
    gs = rs.GameState()
    gs.setup_initial_board()
    engine_prob = rng.choice(ENGINE_MIX)
    target = rng.randint(MIN_PLY, MAX_PLY)

    for _ in range(target):
        moves = gs.get_all_legal_moves()
        if not moves or gs.is_terminal_draw():
            return None
        if engine_prob and rng.random() < engine_prob:
            move = rs.find_best_move(gs, WALK_DEPTH) or rng.choice(moves)
        else:
            move = rng.choice(moves)
        gs.make_ai_move(move)

    moves = gs.get_all_legal_moves()
    # Two legal moves is the minimum a label can mean anything at. A forced reply is
    # answered without a search, so the search returns no score for it, and a one-hot
    # policy over a single legal move teaches nothing.
    if len(moves) < 2 or gs.is_terminal_draw():
        return None
    return gs, moves

def _run_task(task):
    """One shard, in its own process. Returns a stats dict."""
    task_id, seed, quota, depth, out_dir, verify, time_limit = task
    shard = os.path.join(out_dir, "shard_%04d.jsonl" % task_id)

    # Resume. A shard that already holds its quota is done; a partial one is appended
    # to, with its FENs preloaded into `seen`. The task's rng is seeded from the task
    # id, so a resumed run replays the same walks and the dedup drops them before the
    # search -- re-walking is ~200k plies/s, re-searching would be 0.3s a position.
    done_fens = []
    if os.path.exists(shard):
        with open(shard) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done_fens.append(json.loads(line)["fen"])
        if len(done_fens) >= quota:
            return {"task": task_id, "written": 0, "skipped": len(done_fens), "walks": 0,
                    "dead_walks": 0, "duplicates": 0, "seconds": 0.0}

    _silence_engine()
    sys.path.insert(0, paths.repo_root())
    import minichess_engine as rs

    rs.set_search_knobs({"use_book": False})

    rng = random.Random(seed)
    seen = set(done_fens)
    written = len(done_fens)
    walks = dead = dupes = 0
    started = time.time()

    with open(shard, "a") as fh:
        while written < quota:
            walks += 1
            sampled = _sample_position(rng, rs)
            if sampled is None:
                dead += 1
                continue
            gs, legal = sampled

            fen = rs.to_fen(gs)
            if fen in seen:
                dupes += 1
                continue
            seen.add(fen)

            move, score = rs.find_best_move_with_score(gs, depth, time_limit)
            if move is None or score is None:
                dead += 1
                continue

            record = {
                "fen": fen,
                "move": encode_move(move),
                "score": int(score),
                "depth": depth,
                "time_limit": time_limit,
                "turn": gs.current_turn,
                "ply": int(gs.ply),
                "ply_limit": int(gs.ply_limit),
                "reps": int(gs.repetition_count()),
                "n_legal": len(legal),
            }

            if verify:
                _verify(record, gs, rs, move)

            fh.write(json.dumps(record) + "\n")
            # Flushed every record, deliberately. Python's text buffer holds tens of
            # kilobytes, so an unflushed 90-minute run is both unobservable while it
            # runs and entirely lost if it dies. One write syscall per ~0.5s of search
            # is free.
            fh.flush()
            written += 1

    return {"task": task_id, "written": written - len(done_fens), "skipped": len(done_fens),
            "walks": walks,
            "dead_walks": dead, "duplicates": dupes, "seconds": time.time() - started}

def _verify(record, live, rs, move):
    """The guard that makes a systematic train/play skew impossible to sit on.

    Checks the two things that would otherwise corrupt every label in the run: that the
    record round-trips to the same input tensor, and that the labelled move has an
    action index in the position it came from.
    """
    rebuilt = restore(record)
    if rs.encode_position(rebuilt) != rs.encode_position(live):
        raise AssertionError("record does not reproduce its own encoding: %s" % record["fen"])
    idx = rs.move_to_action_index(live, move)
    if not 0 <= idx < rs.ACTION_SPACE:
        raise AssertionError("labelled move has no action index: %s" % record["fen"])

def generate(args):
    out_dir = paths.teacher_dir(args.name)
    tasks_wanted = max(args.jobs * 4, 1)
    per_task = max(1, -(-args.positions // tasks_wanted))
    tasks = [
        (i, args.seed + i * 7919, per_task, args.depth, out_dir, not args.no_verify,
         args.time_limit)
        for i in range(tasks_wanted)
    ]

    print("teacher: %d positions at depth %d -> %s" % (per_task * tasks_wanted, args.depth, out_dir))
    print("         %d tasks over %d workers, seed %d, verify=%s, cap %.1fs/search"
          % (len(tasks), args.jobs, args.seed, not args.no_verify, args.time_limit))

    totals = {"written": 0, "skipped": 0, "walks": 0, "dead_walks": 0, "duplicates": 0}
    started = time.time()
    done = 0

    with Pool(args.jobs) as pool:
        for stats in pool.imap_unordered(_run_task, tasks):
            done += 1
            for k in totals:
                totals[k] += stats[k]
            have = totals["written"] + totals["skipped"]
            elapsed = time.time() - started
            rate = totals["written"] / elapsed if elapsed else 0
            eta = (per_task * tasks_wanted - have) / rate if rate else 0
            print("  [%3d/%3d] %6d positions  %5.1f/s  elapsed %5.0fs  eta %5.0fs"
                  % (done, len(tasks), have, rate, elapsed, eta), flush=True)

    print("\ndone in %.0fs" % (time.time() - started))
    print("  written    %d" % totals["written"])
    print("  skipped    %d (shards already complete)" % totals["skipped"])
    print("  walks      %d, of which %d ended before the target ply"
          % (totals["walks"], totals["dead_walks"]))
    print("  duplicates %d dropped within their shard" % totals["duplicates"])

def load(name="depth8", limit=None):
    """Every record in a teacher directory, deduplicated by FEN across shards."""
    out_dir = paths.teacher_dir(name, create=False)
    if not os.path.isdir(out_dir):
        raise SystemExit("no teacher data at %s -- run `nn.teacher generate` first" % out_dir)

    seen = set()
    records = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(out_dir, fname)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["fen"] in seen:
                    continue
                seen.add(rec["fen"])
                records.append(rec)
                if limit and len(records) >= limit:
                    return records
    return records

def stats(args):
    records = load(args.name)
    if not records:
        raise SystemExit("no records")

    import collections
    plies = [r["ply"] for r in records]
    legal = [r["n_legal"] for r in records]
    values = [value_target(r["score"], r["turn"]) for r in records]
    decided = sum(1 for r in records if abs(r["score"]) >= MATE_CUTOFF)
    forced = sum(1 for r in records if r["n_legal"] == 1)  # generator excludes these
    drops = sum(1 for r in records if r["move"][0] == "drop")
    promos = sum(1 for r in records if r["move"][0] != "drop" and r["move"][2])
    turns = collections.Counter(r["turn"] for r in records)

    def pct(n):
        return 100.0 * n / len(records)

    print("teacher/%s: %d unique positions, depth %d requested, %.1fs cap"
          % (args.name, len(records), records[0]["depth"],
             records[0].get("time_limit", 0.0)))
    print("  ply         min %d  median %d  max %d"
          % (min(plies), sorted(plies)[len(plies) // 2], max(plies)))
    print("  legal moves min %d  mean %.1f  max %d"
          % (min(legal), sum(legal) / len(legal), max(legal)))
    print("  value       mean %+.3f  |v|>0.9 %.1f%%"
          % (sum(values) / len(values), pct(sum(1 for v in values if abs(v) > 0.9))))
    print("  decided     %.1f%% at |score| >= %d" % (pct(decided), MATE_CUTOFF))
    print("  forced      %.1f%% have one legal move" % pct(forced))
    print("  best move   %.1f%% drops, %.1f%% promotions" % (pct(drops), pct(promos)))
    print("  to move     w %.1f%%  b %.1f%%" % (pct(turns["w"]), pct(turns["b"])))

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="label positions with a depth-N search")
    g.add_argument("--positions", type=int, default=150_000)
    g.add_argument("--depth", type=int, default=8)
    g.add_argument("--time-limit", type=float, default=2.0,
                   help="per-search wall cap. Depth-8 costs ~0.35s in an opening and "
                        "~1.6s in a deep full-hand position, with a long tail past 9s; "
                        "the cap trims that tail, and a capped search still returns the "
                        "deepest iteration it completed")
    g.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    g.add_argument("--seed", type=int, default=20260902)
    g.add_argument("--name", default="depth8")
    g.add_argument("--no-verify", action="store_true",
                   help="skip the per-record encoding round trip (do not)")
    g.set_defaults(func=generate)

    s = sub.add_parser("stats", help="describe a generated set")
    s.add_argument("--name", default="depth8")
    s.set_defaults(func=stats)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
