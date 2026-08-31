#!/usr/bin/env python3
"""Fill book.db with an opening *repertoire*, not with whole games.

The book is only ever read at the root of `find_best_move`, which means it is only ever
read in positions where the engine is the side to move. That makes the tree asymmetric,
and the asymmetry is the whole reason this is affordable:

    our turn        one search, one entry, and exactly ONE child -- the move we will
                    actually play. We never play rank 2, so rank 2's subtree is dead
                    weight.
    opponent turn   no search and no entry, but every legal reply expands, because the
                    opponent is the one choosing.

So the tree grows like B^(plies/2) instead of B^plies, and half of what is left is never
searched at all. Measured branching here is ~15-17, and the searched-node counts come out:

    ply   0     2      4       6        8
    W     1    15    240   ~4,100   ~74,000        (Black is the same, shifted one ply)

Two bounds keep it finite:

  * `--max-ply`, the deepest ply the book will hold an answer for. Capped at HARD_PLY_CAP.
  * `--resign`, a score past which the line is already decided. This costs nothing extra:
    the score comes from the search we just ran, and cutting there means a bad opponent
    reply costs exactly one search instead of a whole subtree.

Past the point where full breadth stops being affordable, `--opponent-breadth` narrows the
opponent's replies to the best K, ranked by a shallow search (`--scan-depth`).

Everything here is idempotent. A node that has already been searched at this depth is a
`probe_book` hit that returns in ~0s, so re-running is how you resume, and how a worker
walks the prefix another worker already computed.

    ./venv/bin/python build_book.py --max-ply 6 --depth 10
    ./build_book_parallel.sh 20 6 10        # the same, sharded across 20 workers
"""

import argparse
import os
import signal
import sys
import time
from collections import deque

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import minichess_engine as engine
import ai

HARD_PLY_CAP = 20

shutdown_requested = False

def _on_signal(signum, frame):
    global shutdown_requested
    if shutdown_requested:
        return
    shutdown_requested = True
    print("\nInterrupt received — finishing the current search, then flushing to book.db.",
          flush=True)

def state_from(fen, ply):
    gs = engine.from_fen(fen)
    gs.ply = ply
    return gs

def child_after(gs, move):
    """The position after `move`, or None if it could not be played."""
    nxt = gs.copy()
    if not nxt.make_move(move):
        return None
    if nxt.needs_promotion_choice:
        return None
    return nxt

def parse_breadth(spec):
    """`"all"`, `"3"`, or `"0-8:all,9-16:3"` -> [(lo_ply, hi_ply, keep_or_None)].

    `keep_or_None` is None for "keep every reply". Ranges are inclusive and are matched
    in order, so the first rule covering a ply wins.
    """
    rules = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rng, sep, keep = part.partition(":")
        if not sep:
            rng, keep = "0-%d" % HARD_PLY_CAP, rng
        lo, dash, hi = rng.partition("-")
        try:
            lo = int(lo)
            hi = int(hi) if dash else lo
            k = None if keep.strip() == "all" else int(keep)
        except ValueError:
            raise argparse.ArgumentTypeError("bad --opponent-breadth term: %r" % part)
        if k is not None and k < 1:
            raise argparse.ArgumentTypeError("--opponent-breadth keep must be >= 1: %r" % part)
        rules.append((lo, hi, k))
    return rules

def breadth_at(rules, ply):
    for lo, hi, k in rules:
        if lo <= ply <= hi:
            return k
    return None

def rank_replies(opp_gs, keep, scan_depth):
    """The `keep` best replies for the side to move in `opp_gs`, ranked by one shallow
    MultiPV search *at the opponent node itself*.

    The obvious alternative -- shallow-search each reply separately and sort -- costs one
    search per reply and, worse, files a row at every one of them: on a 6x6 board with
    ~17 replies that is ~16 dead rows for every live one, which is the shape of mistake
    the pre-book `move_cache` made. One MultiPV call is cheaper (measured 4.4x a single-PV
    search for K=3, against 17x for seventeen of them) and writes a single entry, at an
    opponent-turn hash that the runtime never probes, because the engine is never the side
    to move there.

    MultiPV already returns the list best-first for the side to move, so no re-sorting:
    scores are white-relative and the ranking direction is the engine's problem, not ours.
    """
    ranked = engine.find_best_move(opp_gs, scan_depth, keep, None, False)
    kept = []
    for m, _score in ranked[:keep]:
        nxt = child_after(opp_gs, m)
        if nxt is not None and nxt.get_all_legal_moves():
            kept.append((m, nxt))
    return kept

def seeds(colors):
    """Roots of each repertoire, as our-turn nodes.

    White's is the initial position. Black's is fifteen roots, one per White first move --
    all of them, because White chooses and we have to answer whatever comes.
    """
    start = engine.GameState()
    start.setup_initial_board()
    out = []
    if "white" in colors:
        out.append((engine.to_fen(start), 0))
    if "black" in colors:
        for m in start.get_all_legal_moves():
            nxt = child_after(start, m)
            if nxt is not None:
                out.append((engine.to_fen(nxt), 1))
    return out

def build(args):
    rules = parse_breadth(args.opponent_breadth)
    shard_i, shard_n = args.shard
    tag = "" if shard_n == 1 else "[shard %d/%d] " % (shard_i, shard_n)

    ai.setup_db()
    ai.load_move_cache_from_db()
    size_before = ai.book_size()
    print("%sbook loaded: %d positions" % (tag, size_before), flush=True)

    queue = deque()
    seen = set()
    counters = {"next_subtree": 0, "skipped": 0}

    def enqueue(fen, ply, subtree):
        """Add a node, unless another shard owns it.

        The filter is here rather than at pop so a worker never holds a frontier it will
        not search: at ply 5 that frontier is already thousands of nodes, and at the
        depths --opponent-breadth is meant for it would be the dominant cost of the
        process. The id is still allocated for a node this shard drops, because the
        allocation order is the thing all the workers have to agree on.
        """
        if subtree is None and ply >= args.split_ply:
            subtree = counters["next_subtree"]
            counters["next_subtree"] += 1
        if subtree is not None and subtree % shard_n != shard_i:
            counters["skipped"] += 1
            return
        queue.append((fen, ply, subtree))

    for fen, ply in seeds(args.color):
        h = engine.get_position_hash(state_from(fen, ply))
        if h not in seen:
            seen.add(h)
            enqueue(fen, ply, None)

    visited = 0
    cut_resign = 0
    per_ply = {}
    started = time.time()
    last_save = 0

    while queue and not shutdown_requested:
        fen, ply, subtree = queue.popleft()

        gs = state_from(fen, ply)
        if not gs.get_all_legal_moves():
            continue

        move, score = engine.find_best_move_with_score(gs, args.depth, None, False)
        if move is None:
            continue
        visited += 1
        per_ply[ply] = per_ply.get(ply, 0) + 1

        if visited - last_save >= args.save_every:
            ai.save_move_cache_to_db()
            last_save = visited
            print("%s  saved — %d visited (+%d new), %d queued, %.0fs elapsed"
                  % (tag, visited, ai.book_size() - size_before, len(queue),
                     time.time() - started), flush=True)

        if abs(score) >= args.resign:
            cut_resign += 1
            continue

        if ply + 2 > args.max_ply:
            continue

        opp = child_after(gs, move)
        if opp is None:
            print("%s  !! could not replay engine move %r at ply %d (%s)"
                  % (tag, move, ply, fen), flush=True)
            continue

        replies = opp.get_all_legal_moves()
        keep = breadth_at(rules, opp.ply)
        if keep is not None and len(replies) > keep:
            kept = rank_replies(opp, keep, args.scan_depth)
        else:
            kept = []
            for m in replies:
                nxt = child_after(opp, m)
                if nxt is not None and nxt.get_all_legal_moves():
                    kept.append((m, nxt))

        for _m, nxt in kept:
            h = engine.get_position_hash(nxt)
            if h in seen:
                continue
            seen.add(h)
            enqueue(engine.to_fen(nxt), ply + 2, subtree)

    ai.save_move_cache_to_db()

    elapsed = time.time() - started
    print("\n%s%s after %.0fs" % (tag, "stopped" if shutdown_requested else "done", elapsed),
          flush=True)
    print("%s  visited:         %d" % (tag, visited))
    for p in sorted(per_ply):
        print("%s    ply %-2d %d" % (tag, p, per_ply[p]))
    print("%s  new entries:     %d" % (tag, ai.book_size() - size_before))
    print("%s  cut (decided):   %d" % (tag, cut_resign))
    if shard_n > 1:
        print("%s  other shards:    %d" % (tag, counters["skipped"]))
    print("%s  left in queue:   %d" % (tag, len(queue)))
    print("%s  book.db:         %d positions" % (tag, ai.book_size()))
    return 0 if not shutdown_requested else 130

def shard_arg(value):
    try:
        i, n = value.split("/")
        i, n = int(i), int(n)
    except ValueError:
        raise argparse.ArgumentTypeError("--shard takes I/N, e.g. 0/20")
    if n < 1 or not (0 <= i < n):
        raise argparse.ArgumentTypeError("--shard I/N needs N >= 1 and 0 <= I < N")
    return (i, n)

def main():
    p = argparse.ArgumentParser(
        description="Build book.db as an opening repertoire.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    p.add_argument("--max-ply", type=int, default=6,
                   help="deepest ply the book will answer for (default 6, hard cap %d)"
                        % HARD_PLY_CAP)
    p.add_argument("--depth", type=int, default=10,
                   help="search depth per entry (default 10). Must be >= the depth you "
                        "play at, or the probe rejects every row as too shallow.")
    p.add_argument("--color", choices=("white", "black", "both"), default="both",
                   help="which repertoire(s) to build (default both)")
    p.add_argument("--resign", type=int, default=1200,
                   help="stop expanding once |score| reaches this (default 1200, about a "
                        "queen on the 100=pawn scale)")
    p.add_argument("--opponent-breadth", default="all",
                   help='how many opponent replies to expand, by ply: "all" (default), '
                        '"3", or "0-8:all,9-16:3"')
    p.add_argument("--scan-depth", type=int, default=4,
                   help="depth used to rank opponent replies when --opponent-breadth "
                        "narrows them (default 4)")
    p.add_argument("--split-ply", type=int, default=None,
                   help="ply at which subtrees are handed to shards (default max-ply - 2). "
                        "Nodes above it are walked by every shard, which is free once "
                        "they are in the book.")
    p.add_argument("--shard", type=shard_arg, default=(0, 1), metavar="I/N",
                   help="build only subtree I of N (default 0/1, i.e. everything)")
    p.add_argument("--save-every", type=int, default=25,
                   help="flush to book.db every N searches (default 25)")
    args = p.parse_args()

    if args.max_ply < 0:
        p.error("--max-ply must be >= 0")
    if args.max_ply > HARD_PLY_CAP:
        p.error("--max-ply %d is past the %d-ply cap: a book that deep is no longer a "
                "repertoire and the tree is not affordable." % (args.max_ply, HARD_PLY_CAP))
    if args.depth < 1:
        p.error("--depth must be >= 1")
    if args.split_ply is None:
        args.split_ply = max(0, args.max_ply - 2)
    colors = ("white", "black") if args.color == "both" else (args.color,)
    args.color = colors

    os.chdir(_REPO_ROOT)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    ai.set_parallel_search(False)

    print("repertoire: colors=%s max_ply=%d depth=%d resign=%d breadth=%s split_ply=%d"
          % ("+".join(colors), args.max_ply, args.depth, args.resign,
             args.opponent_breadth, args.split_ply), flush=True)
    return build(args)

if __name__ == "__main__":
    sys.exit(main())
