"""
Accuracy harness: measure a search configuration against exact minimax.

bench/run_bench.py answers "how fast", and on its own it cannot judge any
heuristic that trades accuracy for speed - LMR, null move, delta pruning are all
faster *because* they are wrong more often. This script answers "how wrong", by
searching every position twice: once with the configuration under test, and once
with an exact reference (null move, LMR, delta pruning and the transposition
table all off), which is plain alpha-beta over the same depth-N + quiescence
tree and therefore returns that tree's true minimax value.

Three numbers per configuration, all against that reference:

    mean |err|    how far the reported score is from the true value, in cp
    regret        how much the *move* it played gives away, in cp: the true
                  value of the position minus the true value of the move chosen
                  (>= 0 by construction, 0 when the move is optimal)
    best moves    the count of positions where regret == 0, out of the suite

Regret is the number that matters. A search that misreports a score but keeps
playing the best move has lost nothing; the previous work in this repo only had
score error and best-move agreement, and agreement alone cannot tell a
half-pawn mistake from a lost rook.

METHOD - the part that is easy to get wrong, and was got wrong here before:

  * Pair over MOVE ORDERINGS and over a WIDE POSITION SAMPLE. --seeds re-runs
    every position under N value-neutral move orderings (a seeded tiebreak
    between moves the ordering scores equally - the move set and its ordering
    keys are untouched, so an exact search is unaffected and only the heuristics
    see a different order). One ordering cannot resolve an accuracy change: the
    same ablation has read -13.5 +/- 9.4 on 16 positions and -82.9 +/- 4.3 on
    48. The suite is 64 positions; do not shrink it to save time.
  * Error bars are the standard error over POSITIONS of the per-position mean
    across seeds, so n = the suite size, not the cell count - cells sharing a
    position are not independent samples. Deltas against the baseline are paired
    per position, which removes the (large) position-to-position variance.
  * Speed here is a side channel, not the headline: many searches share one
    process. Node counts (rs.last_search_nodes) are load-immune and are the
    honest speed number; wall time belongs to bench/run_bench.py, which forks a
    fresh interpreter per measurement.

The reference is expensive (~30s per position at depth 8, vs ~0.5s for the
shipped search) so it is cached in bench/results/reference_d{N}.json, keyed by
FEN, and reused by every later run at that depth. Delete that file if anything
below the search changes: the eval, the quiescence shape, the move generator.
It carries the eval version and refuses to load against a different one.

The book is disabled in every search (use_book=False), so no run here reads or
writes book.db, and no search is ever answered from the process-global book that
would make a repeat measurement report ~0s.

Usage:
    .venv-bench/bin/python bench/accuracy.py reference --depth 8 --jobs 4
    .venv-bench/bin/python bench/accuracy.py run --depth 8 --seeds 8 \
        --configs shipped,lmr_off --out bench/results/lmr.json
    .venv-bench/bin/python bench/accuracy.py report bench/results/lmr.json
    .venv-bench/bin/python bench/accuracy.py run --depth 8 --seeds 8 \
        --configs shipped,mine --knobs 'mine={"null_move":false}'

Keep --jobs at 1 whenever the wall-clock numbers matter, and low in general:
another agent may be benchmarking on the same machine.
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BENCH_DIR)
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
SUITE = os.path.join(BENCH_DIR, "accuracy_positions.json")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, BENCH_DIR)

MATE_CUTOFF = 5000

# Per-cell cap on |err| and regret. A search that walks into a forced mate is
# wrong by CHECKMATE_SCORE, and one such cell would otherwise be the entire
# average - the mean would report which sample contained a mate, not which
# configuration plays better. Capped cells are counted and printed.
CAP = 2000

# The exact reference: everything that trades accuracy for speed, off. A margin
# this wide can never fire, which is how delta pruning is disabled without a
# switch of its own.
REFERENCE_KNOBS = {
    "null_move": False,
    "lmr": False,
    "use_tt": False,
    "delta_margin": 1 << 28,
}

# Named configurations. {} is whatever the engine ships with today. These are
# ablations, not tuning: the shape of LMR (LMR_MIN_MOVE, LMR_MIN_DEPTH) and the
# quiescence margin are consts at the top of search.rs, so sweeping one means
# editing the const and rebuilding, then reporting the new run against a results
# file saved from the old build. bench/results/lmr_confirm.json is such a record
# - the run that set LMR_MIN_MOVE, measured against a temporarily parameterised
# build, its config names naming knobs this module no longer sets.
CONFIGS = {
    "shipped": {},
    "lmr_off": {"lmr": False},
    "null_off": {"null_move": False},
    "tt_off": {"use_tt": False},
    "delta_off": {"delta_margin": 1 << 28},
}


def knobs_for(name, extra):
    if name in extra:
        return dict(extra[name])
    if name not in CONFIGS:
        raise SystemExit("unknown config %r (known: %s)"
                         % (name, ", ".join(sorted(CONFIGS))))
    return dict(CONFIGS[name])


def move_key(m):
    """Compact, unique text form of a move tuple: e2e4, e6e7R, N@c3."""
    if m is None:
        return None
    files = "abcdef"

    def sq(rf):
        return "%s%d" % (files[rf[1]], 6 - rf[0])

    if m[0] == "drop":
        return "%s@%s" % (m[1], sq(m[2]))
    return "%s%s%s" % (sq(m[0]), sq(m[1]), m[2] or "")


_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        import minichess_engine as rs
        _ENGINE = rs
    return _ENGINE


def _init_worker(quiet):
    if quiet:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
    _engine()


def _search(fen, depth, knobs, seed, moves=()):
    """One search: apply `moves` to `fen`, then search `depth` under `knobs`."""
    rs = _engine()
    rs.reset_search_knobs()
    k = dict(knobs)
    k["use_book"] = False
    k["order_seed"] = seed
    rs.set_search_knobs(k)

    gs = rs.from_fen(fen)
    for m in moves:
        gs.make_ai_move(m)
    legal = gs.get_all_legal_moves()
    if not legal:
        white = gs.current_turn == "w"
        if gs.is_in_check(gs.current_turn):
            return {"terminal": True, "score": -rs.CHECKMATE_SCORE if white else rs.CHECKMATE_SCORE}
        return {"terminal": True, "score": 0}
    if len(legal) == 1 and moves:
        # find_best_move short-circuits a forced move with a placeholder 0, so
        # its value cannot be read this way. Reported, never guessed at.
        return {"forced": True, "score": None}

    start = time.perf_counter()
    move, score = rs.find_best_move_with_score(gs, depth, None, False)
    seconds = time.perf_counter() - start
    nodes, qnodes = rs.last_search_nodes()
    return {"move": move, "move_key": move_key(move), "score": score,
            "seconds": seconds, "nodes": nodes, "qnodes": qnodes}


def _task_root_ref(job):
    fen, depth = job
    out = _search(fen, depth, REFERENCE_KNOBS, 0)
    return fen, out


def _task_move_ref(job):
    fen, depth, move, key = job
    out = _search(fen, depth - 1, REFERENCE_KNOBS, 0, moves=[move])
    return fen, key, out


def _task_run(job):
    fen, depth, knobs, seed, config = job
    out = _search(fen, depth, knobs, seed)
    out.update({"fen": fen, "seed": seed, "config": config})
    return out


def ref_path(depth):
    return os.path.join(RESULTS_DIR, "reference_d%d.json" % depth)


def load_reference(depth):
    import minichess_engine as rs
    path = ref_path(depth)
    if not os.path.exists(path):
        return {"depth": depth, "eval_version": rs.EVAL_VERSION, "positions": {}}
    with open(path) as fh:
        data = json.load(fh)
    if data.get("eval_version") != rs.EVAL_VERSION:
        raise SystemExit("%s was built under eval_version %s, engine is at %s; "
                         "delete it and rebuild" % (path, data.get("eval_version"),
                                                    rs.EVAL_VERSION))
    if data.get("depth") != depth:
        raise SystemExit("%s is for depth %s" % (path, data.get("depth")))
    return data


def save_reference(data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = ref_path(data["depth"])
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def load_suite(path, limit=None):
    with open(path) as fh:
        positions = json.load(fh)["positions"]
    return positions[:limit] if limit else positions


def imap(jobs, fn, workers, quiet, label):
    """Run `fn` over `jobs`, printing progress. workers=1 stays in-process."""
    started = time.time()
    done = 0
    if workers <= 1:
        _init_worker(quiet)
        for job in jobs:
            yield fn(job)
            done += 1
            _progress(label, done, len(jobs), started)
    else:
        with Pool(workers, initializer=_init_worker, initargs=(quiet,)) as pool:
            for res in pool.imap_unordered(fn, jobs, chunksize=1):
                yield res
                done += 1
                _progress(label, done, len(jobs), started)
    sys.stderr.write("\n")


def _progress(label, done, total, started):
    elapsed = time.time() - started
    eta = elapsed / done * (total - done) if done else 0
    sys.stderr.write("\r  %s %d/%d  %.0fs elapsed, ~%.0fs left    "
                     % (label, done, total, elapsed, eta))
    sys.stderr.flush()


def ensure_root_reference(suite, depth, workers, quiet):
    ref = load_reference(depth)
    todo = [p["fen"] for p in suite if p["fen"] not in ref["positions"]]
    if todo:
        print("reference: %d position(s) to search at depth %d (exact, no TT)"
              % (len(todo), depth))
        jobs = [(fen, depth) for fen in todo]
        for i, (fen, out) in enumerate(imap(jobs, _task_root_ref, workers, quiet, "reference")):
            ref["positions"][fen] = {"score": out["score"], "move": out.get("move_key"),
                                     "seconds": out.get("seconds"), "moves": {}}
            if i % 8 == 7:
                save_reference(ref)
        save_reference(ref)
    return ref


def ensure_move_reference(ref, depth, wanted, workers, quiet):
    """wanted: {fen: {move_key: move_tuple}} - exact value of each chosen move."""
    jobs = []
    for fen, moves in wanted.items():
        have = ref["positions"][fen]["moves"]
        for key, move in moves.items():
            if key not in have:
                jobs.append((fen, depth, move, key))
    if not jobs:
        return ref
    print("reference: %d chosen move(s) to value exactly at depth %d"
          % (len(jobs), depth - 1))
    for i, (fen, key, out) in enumerate(imap(jobs, _task_move_ref, workers, quiet, "move refs")):
        ref["positions"][fen]["moves"][key] = out["score"]
        if i % 16 == 15:
            save_reference(ref)
    save_reference(ref)
    return ref


def mean_se(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def summarize(cells, suite_fens, seeds):
    """cells: list of per-(position, seed) records for ONE config."""
    by_fen = {}
    for c in cells:
        by_fen.setdefault(c["fen"], []).append(c)
    per_pos = {"err": {}, "regret": {}, "best": {}}
    totals = {"nodes": 0, "qnodes": 0, "seconds": 0.0, "unscored": 0, "capped": 0}
    for fen in suite_fens:
        rows = by_fen.get(fen, [])
        errs, regs, bests = [], [], []
        for c in rows:
            totals["nodes"] += c.get("nodes", 0)
            totals["qnodes"] += c.get("qnodes", 0)
            totals["seconds"] += c.get("seconds", 0.0)
            if c["err"] > CAP or (c["regret"] or 0) > CAP:
                totals["capped"] += 1
            errs.append(min(c["err"], CAP))
            if c["regret"] is None:
                totals["unscored"] += 1
            else:
                regs.append(min(c["regret"], CAP))
                bests.append(1.0 if c["regret"] == 0 else 0.0)
        if errs:
            per_pos["err"][fen] = sum(errs) / len(errs)
        if regs:
            per_pos["regret"][fen] = sum(regs) / len(regs)
            per_pos["best"][fen] = sum(bests) / len(bests)
    out = {"totals": totals, "per_position": per_pos,
           "n_positions": len(per_pos["err"]), "n_scored": len(per_pos["best"]),
           "cell_nodes": {(c["seed"], c["fen"]): c.get("nodes", 0) + c.get("qnodes", 0)
                          for c in cells}}
    for metric in ("err", "regret", "best"):
        m, se = mean_se(list(per_pos[metric].values()))
        out[metric] = {"mean": m, "se": se}
    # per-seed suite means, for the "better in k/N orderings" line
    out["by_seed"] = {}
    for seed in seeds:
        rows = [c for c in cells if c["seed"] == seed]
        regs = [min(c["regret"], CAP) for c in rows if c["regret"] is not None]
        out["by_seed"][str(seed)] = {
            "err": sum(min(c["err"], CAP) for c in rows) / max(1, len(rows)),
            "regret": sum(regs) / max(1, len(regs)),
            "best": sum(1 for r in regs if r == 0),
            "nodes": sum(c.get("nodes", 0) for c in rows),
            "seconds": sum(c.get("seconds", 0.0) for c in rows),
        }
    return out


def paired_delta(base, other, metric):
    """Mean +/- SE of the per-position difference (other - base)."""
    a, b = base["per_position"][metric], other["per_position"][metric]
    common = [f for f in a if f in b]
    diffs = [b[f] - a[f] for f in common]
    return mean_se(diffs)


def checkpoint(out, args, rs, config_names, extra, seeds, fens, cells):
    """Write the results file. Called during the run too, so a sweep that dies
    resumes instead of starting over."""
    from positions import encode_move
    payload = {
        "depth": args.depth,
        "jobs": args.jobs,
        "seeds": seeds,
        "suite": os.path.relpath(args.suite, REPO_ROOT),
        "n_positions": len(fens),
        "eval_version": rs.EVAL_VERSION,
        "configs": {n: knobs_for(n, extra) for n in config_names},
        "cells": [dict(c, move=encode_move(c["move"])) if c.get("move") else c
                  for c in cells],
    }
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        # One line: a sweep is tens of thousands of cells and this file is meant
        # to be committed as evidence, not read by eye. `report` re-prints it.
        json.dump(payload, fh, separators=(",", ":"))
        fh.write("\n")
    os.replace(tmp, out)
    return payload


def cmd_run(args):
    import minichess_engine as rs
    os.chdir(REPO_ROOT)
    extra = {}
    for spec in args.knobs or []:
        name, _, blob = spec.partition("=")
        extra[name] = json.loads(blob)
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    for name in config_names:
        knobs_for(name, extra)

    suite = load_suite(args.suite, args.limit)
    seeds = list(range(1 + args.seed_offset, args.seeds + 1 + args.seed_offset))

    ref = ensure_root_reference(suite, args.depth, args.jobs, not args.verbose)

    mates = [p["fen"] for p in suite if abs(ref["positions"][p["fen"]]["score"]) > MATE_CUTOFF]
    if mates and not args.include_mates:
        suite = [p for p in suite if p["fen"] not in mates]
        print("dropped %d position(s) whose exact value is a mate score "
              "(--include-mates keeps them)" % len(mates))
    fens = [p["fen"] for p in suite]

    out = args.out or os.path.join(RESULTS_DIR, "accuracy_d%d.json" % args.depth)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # A full sweep is hours long, so it resumes: any (config, seed, position)
    # already in --out is not searched again. Nothing about a cell depends on
    # the run it was measured in.
    cells, done = [], set()
    if os.path.exists(out):
        from positions import decode_move
        with open(out) as fh:
            old = json.load(fh)
        if old.get("depth") == args.depth and old.get("eval_version") == rs.EVAL_VERSION:
            for c in old.get("cells", []):
                if c.get("move"):
                    c["move"] = decode_move(c["move"])
                cells.append(c)
                done.add((c["config"], c["seed"], c["fen"]))
            print("resuming %s: %d cell(s) already measured" % (out, len(cells)))

    jobs = []
    for name in config_names:
        k = knobs_for(name, extra)
        for seed in seeds:
            for fen in fens:
                if (name, seed, fen) not in done:
                    jobs.append((fen, args.depth, k, seed, name))
    print("running %d searches: %d config(s) x %d seed(s) x %d positions at depth %d"
          % (len(jobs), len(config_names), len(seeds), len(fens), args.depth))
    for c in imap(jobs, _task_run, args.jobs, not args.verbose, "searches"):
        cells.append(c)
        if len(cells) % 250 == 0:
            checkpoint(out, args, rs, config_names, extra, seeds, fens, cells)
    checkpoint(out, args, rs, config_names, extra, seeds, fens, cells)

    wanted = {}
    for c in cells:
        if c.get("move_key"):
            wanted.setdefault(c["fen"], {})[c["move_key"]] = c["move"]
    ref = ensure_move_reference(ref, args.depth, wanted, args.jobs, not args.verbose)

    for c in cells:
        entry = ref["positions"][c["fen"]]
        white = c["fen"].rsplit(" ", 1)[-1] == "w"
        c["ref_score"] = entry["score"]
        c["err"] = abs(c["score"] - entry["score"])
        val = entry["moves"].get(c.get("move_key"))
        if val is None:
            c["regret"] = None
        else:
            # >= 0 by construction: no child of an exactly searched position can
            # beat that position's own exact value. A negative one is a bug in
            # the harness or a non-determinism in the engine, so it is reported
            # rather than clamped away.
            c["regret"] = (entry["score"] - val) if white else (val - entry["score"])
    bad = [c for c in cells if c["regret"] is not None and c["regret"] < 0]
    if bad:
        print("WARNING: %d cell(s) scored better than the exact value of their "
              "own position (worst %d cp). The reference is inconsistent."
              % (len(bad), min(c["regret"] for c in bad)))

    payload = checkpoint(out, args, rs, config_names, extra, seeds, fens, cells)
    print("wrote %s" % out)
    report(payload, args.baseline or config_names[0])


def cmd_selfcheck(args):
    """The reference must not depend on the move ordering. If it does, either
    the ordering seed is not value-neutral or something in the 'exact' config
    still prunes, and every number this harness prints is unfounded."""
    os.chdir(REPO_ROOT)
    _init_worker(True)
    suite = load_suite(args.suite, args.limit)
    bad = 0
    for p in suite:
        scores = [_search(p["fen"], args.depth, REFERENCE_KNOBS, seed)["score"]
                  for seed in range(args.seeds + 1)]
        ok = len(set(scores)) == 1
        bad += 0 if ok else 1
        print("%-4s %-52s %s" % ("ok" if ok else "BAD", p["fen"][:52],
                                 sorted(set(scores))))
    print("%d of %d positions disagreed across %d orderings at depth %d"
          % (bad, len(suite), args.seeds + 1, args.depth))
    return 1 if bad else 0


def cmd_report(args):
    """More than one file is merged: cells keyed by (config, seed, position) are
    disjoint across runs that used different --seed-offset values, so a sweep and
    its out-of-sample confirmation can be pooled into one table."""
    payload = None
    for path in args.file:
        with open(path) as fh:
            part = json.load(fh)
        if payload is None:
            payload = part
            continue
        if part["depth"] != payload["depth"]:
            raise SystemExit("%s is depth %s, not %s"
                             % (path, part["depth"], payload["depth"]))
        payload["seeds"] = sorted(set(payload["seeds"]) | set(part["seeds"]))
        payload["configs"].update(part["configs"])
        seen = {(c["config"], c["seed"], c["fen"]) for c in payload["cells"]}
        payload["cells"] += [c for c in part["cells"]
                             if (c["config"], c["seed"], c["fen"]) not in seen]
    report(payload, args.baseline or list(payload["configs"])[0])


def report(payload, baseline):
    seeds = payload["seeds"]
    fens = sorted({c["fen"] for c in payload["cells"]})
    names = list(payload["configs"])
    summaries = {}
    for name in names:
        cells = [c for c in payload["cells"] if c["config"] == name]
        summaries[name] = summarize(cells, fens, seeds)

    n = len(fens)
    print("\n%d positions x %d move orderings at depth %d, +/- is the standard "
          "error over positions" % (n, len(seeds), payload["depth"]))
    print("\n%-18s %14s %14s %14s %12s %10s" %
          ("config", "mean |err|", "regret", "best/%d" % n, "Mnodes", "sec"))
    for name in names:
        s = summaries[name]
        t = s["totals"]
        ns = s["n_scored"]
        print("%-18s %7.1f +/-%-5.1f %7.1f +/-%-5.1f %7.2f +/-%-5.2f %12.1f %10.1f"
              % (name, s["err"]["mean"], s["err"]["se"],
                 s["regret"]["mean"], s["regret"]["se"],
                 s["best"]["mean"] * ns, s["best"]["se"] * ns,
                 (t["nodes"] + t["qnodes"]) / 1e6 / max(1, len(seeds)),
                 t["seconds"] / max(1, len(seeds))))
        if t["unscored"]:
            print("%-18s   (best/regret over %d of %d positions: %d cell(s) "
                  "chose a move whose reply was forced, so it has no exact value)"
                  % ("", ns, n, t["unscored"]))
        if t["capped"]:
            print("%-18s   (%d cell(s) capped at %d cp)" % ("", t["capped"], CAP))

    base = summaries[baseline]
    print("\npaired against %r (per position, negative is better for err/regret):"
          % baseline)
    print("%-18s %16s %16s %16s %10s %10s" %
          ("config", "d mean |err|", "d regret", "d best moves", "nodes", "orderings"))
    for name in names:
        if name == baseline:
            continue
        s = summaries[name]
        de, dee = paired_delta(base, s, "err")
        dr, dre = paired_delta(base, s, "regret")
        db, dbe = paired_delta(base, s, "best")
        wins = sum(1 for seed in seeds
                   if s["by_seed"][str(seed)]["regret"] < base["by_seed"][str(seed)]["regret"])
        nb = len(set(base["per_position"]["best"]) & set(s["per_position"]["best"]))
        # node ratio over the cells both configurations actually measured: a
        # merged file can hold 16 orderings of one and 8 of another.
        shared = set(base["cell_nodes"]) & set(s["cell_nodes"])
        bn = sum(base["cell_nodes"][c] for c in shared)
        sn = sum(s["cell_nodes"][c] for c in shared)
        print("%-18s %8.1f +/-%-6.1f %8.1f +/-%-6.1f %8.2f +/-%-6.2f %9.3fx %6d/%d"
              % (name, de, dee, dr, dre, db * nb, dbe * nb,
                 sn / bn if bn else float("nan"), wins, len(seeds)))
    print("\nnodes are minimax + quiescence nodes per ordering, load-immune; "
          "seconds are in-process and only indicative - use bench/run_bench.py "
          "for wall clock.")
    if payload.get("jobs", 1) > 1:
        print("this run used --jobs %d: the seconds column is meaningless, the "
              "node counts are not." % payload["jobs"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="search the suite under one or more configs")
    run.add_argument("--depth", type=int, default=8)
    run.add_argument("--seeds", type=int, default=8, help="move orderings per position")
    run.add_argument("--seed-offset", type=int, default=0,
                     help="start the ordering seeds past this many, so a variant "
                          "picked as the best of a sweep can be confirmed on "
                          "orderings it was not chosen on")
    run.add_argument("--configs", default="shipped")
    run.add_argument("--knobs", action="append",
                     help='ad-hoc config, e.g. \'mine={"null_move":false}\'')
    run.add_argument("--suite", default=SUITE)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--out", default=None)
    run.add_argument("--baseline", default=None)
    run.add_argument("--include-mates", action="store_true")
    run.add_argument("--verbose", action="store_true", help="keep engine stderr")
    run.set_defaults(fn=cmd_run)

    ref = sub.add_parser("reference", help="fill the exact reference cache only")
    ref.add_argument("--depth", type=int, default=8)
    ref.add_argument("--suite", default=SUITE)
    ref.add_argument("--limit", type=int, default=None)
    ref.add_argument("--jobs", type=int, default=1)
    ref.add_argument("--verbose", action="store_true")
    ref.set_defaults(fn=lambda a: (os.chdir(REPO_ROOT),
                                   ensure_root_reference(load_suite(a.suite, a.limit),
                                                         a.depth, a.jobs, not a.verbose),
                                   print("reference cache: %s" % ref_path(a.depth))))

    chk = sub.add_parser("selfcheck",
                         help="assert the exact reference is ordering-invariant")
    chk.add_argument("--depth", type=int, default=6)
    chk.add_argument("--seeds", type=int, default=3)
    chk.add_argument("--suite", default=SUITE)
    chk.add_argument("--limit", type=int, default=8)
    chk.set_defaults(fn=cmd_selfcheck)

    rep = sub.add_parser("report", help="re-print the table from a results file")
    rep.add_argument("file", nargs="+")
    rep.add_argument("--baseline", default=None)
    rep.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
