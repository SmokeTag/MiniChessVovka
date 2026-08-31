"""
Benchmark harness for the engine's root-parallel search.

Every single measurement runs in its OWN fresh ./venv/bin/python subprocess.
That is not paranoia: find_best_move consults a process-global MOVE_CACHE keyed
by (position hash, depth) and fills it as it searches, so a second search of the
same (position, depth) inside one process is a cache hit and reports ~0s. One
process = one search = one honest number. The child never calls
load_move_cache_from_db(), so the on-disk book.db is never consulted or
written.

Modes:
    seq  -> find_best_move(..., parallel=False)   (the training / default path)
    par  -> find_best_move(..., parallel=True)    (interactive analysis path)
The flag is passed per call, so the process-wide default (off) is never relied on.

Repeats are reduced with MIN, not mean. Under a background self-play job the
noise is strictly additive - stolen cores only ever make a run slower - so the
minimum is the least contaminated estimate of the true cost.

Examples:
    ./venv/bin/python bench/run_bench.py --depths 6,7,8 --repeats 3 --out bench/seq.json --modes seq
    ./venv/bin/python bench/run_bench.py --positions opening_start --depths 4 --modes seq,par
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BENCH_DIR)
DEFAULT_PYTHON = os.path.join(REPO_ROOT, "venv", "bin", "python")

sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, REPO_ROOT)

RESULT_MARKER = "@@BENCH@@"

PARALLEL_RE = re.compile(
    r"\[PARALLEL\] depth (\d+): baseline ([0-9.]+)s, scout ([0-9.]+)s \((\d+) moves\), "
    r"re-search ([0-9.]+)s \((\d+) moves\)"
)
PARALLEL_ON_RE = re.compile(r"\[PARALLEL\] root split enabled from depth (\d+)")

LOAD_WARN_THRESHOLD = 4.0

def run_child(args):
    """Search exactly one (position, depth, mode) and print one JSON line."""
    from positions import load_positions, encode_move
    import ai
    import minichess_engine as rs

    position = load_positions()[args.position]
    from positions import build_gamestate
    gs = build_gamestate(position)

    legal = gs.get_all_legal_moves()
    if len(legal) < 2:
        emit({"position": args.position, "depth": args.depth, "mode": args.mode,
              "error": "position has %d legal move(s)" % len(legal)})
        return 1

    parallel = args.mode == "par"
    if parallel and args.par_min_depth:
        rs.set_parallel_search(True, args.par_min_depth)

    rust_gs = ai._sync_to_rust(gs)

    start = time.perf_counter()
    best_move, score = rs.find_best_move_with_score(rust_gs, args.depth, None, parallel)
    seconds = time.perf_counter() - start
    emit({
        "position": args.position,
        "depth": args.depth,
        "mode": args.mode,
        "seconds": seconds,
        "best_move": encode_move(best_move) if best_move else None,
        "best_move_str": move_str(best_move),
        "score": score,
        "legal_moves": len(legal),
        "side_to_move": gs.current_turn,
        "pid": os.getpid(),
    })
    return 0

def emit(payload):
    sys.stdout.write("%s %s\n" % (RESULT_MARKER, json.dumps(payload)))
    sys.stdout.flush()

def move_str(move):
    """Human-readable move, e.g. 'e2e4', 'e7e8R', 'N@c3'."""
    if not move:
        return None
    files = "abcdef"
    size = 6

    def sq(rf):
        r, f = rf
        return "%s%d" % (files[f], size - r)

    if move[0] == "drop":
        return "%s@%s" % (move[1][1], sq(move[2]))
    promo = move[2] or ""
    return "%s%s%s" % (sq(move[0]), sq(move[1]), promo)

def load_average():
    try:
        return list(os.getloadavg())
    except (OSError, AttributeError):
        return None

def uptime_line():
    try:
        return subprocess.run(["uptime"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return None

def nproc():
    try:
        return int(subprocess.run(["nproc"], capture_output=True, text=True,
                                  timeout=10).stdout.strip())
    except Exception:
        return os.cpu_count()

def parse_parallel_stderr(stderr):
    """Pull the [PARALLEL] split timings out of the engine's stderr."""
    splits = []
    for m in PARALLEL_RE.finditer(stderr):
        splits.append({
            "depth": int(m.group(1)),
            "baseline_s": float(m.group(2)),
            "scout_s": float(m.group(3)),
            "scout_moves": int(m.group(4)),
            "research_s": float(m.group(5)),
            "research_moves": int(m.group(6)),
        })
    enabled_from = PARALLEL_ON_RE.search(stderr)
    return splits, (int(enabled_from.group(1)) if enabled_from else None)

def run_one(python, position, depth, mode, par_min_depth, timeout, keep_stderr):
    """Spawn one fresh interpreter for exactly one search."""
    cmd = [python, os.path.abspath(__file__), "--child",
           "--position", position, "--depth", str(depth), "--mode", mode]
    if par_min_depth:
        cmd += ["--par-min-depth", str(par_min_depth)]

    load_before = load_average()
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=REPO_ROOT)
    except subprocess.TimeoutExpired:
        return {"position": position, "depth": depth, "mode": mode,
                "error": "timeout after %ss" % timeout,
                "load_before": load_before}
    wall = time.time() - started

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            payload = json.loads(line[len(RESULT_MARKER):])
    if payload is None:
        return {"position": position, "depth": depth, "mode": mode,
                "error": "child produced no result line (exit %d)" % proc.returncode,
                "stderr_tail": proc.stderr[-2000:],
                "load_before": load_before}

    splits, enabled_from = parse_parallel_stderr(proc.stderr)
    payload["parallel_splits"] = splits
    payload["parallel_enabled_from_depth"] = enabled_from
    payload["subprocess_wall_s"] = wall
    payload["load_before"] = load_before
    payload["load_after"] = load_average()
    if keep_stderr:
        payload["stderr"] = proc.stderr
    return payload

def reduce_repeats(runs):
    """Minimum across repeats; keep every sample so the spread stays visible."""
    ok = [r for r in runs if "error" not in r and r.get("seconds") is not None]
    if not ok:
        return {"error": runs[0].get("error", "all repeats failed"),
                "repeats": runs}
    best = min(ok, key=lambda r: r["seconds"])
    samples = [r["seconds"] for r in ok]
    loads = [max(r["load_before"][0], r["load_after"][0])
             for r in ok if r.get("load_before") and r.get("load_after")]
    out = dict(best)
    out["seconds_samples"] = samples
    out["seconds_min"] = min(samples)
    out["seconds_max"] = max(samples)
    out["repeats"] = len(samples)
    out["failed_repeats"] = len(runs) - len(ok)
    out["max_load_seen"] = max(loads) if loads else None
    out["load_suspect"] = bool(loads and max(loads) > LOAD_WARN_THRESHOLD)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depths", default="6,7,8",
                    help="comma-separated search depths (default 6,7,8)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="runs per cell; the MINIMUM is reported (default 3)")
    ap.add_argument("--modes", default="seq,par",
                    help="comma-separated: seq,par (default seq,par)")
    ap.add_argument("--positions", default="all",
                    help="comma-separated position names, or 'all' (default all)")
    ap.add_argument("--out", default=None,
                    help="results JSON path (default bench/results-<timestamp>.json)")
    ap.add_argument("--python", default=DEFAULT_PYTHON,
                    help="interpreter for the child processes (default ./venv/bin/python)")
    ap.add_argument("--par-min-depth", type=int, default=None,
                    help="PARALLEL_MIN_DEPTH for par children (default: engine default, 3)")
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-search subprocess timeout in seconds (default 900)")
    ap.add_argument("--keep-stderr", action="store_true",
                    help="store the full child stderr in the results file")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--position", help=argparse.SUPPRESS)
    ap.add_argument("--depth", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--mode", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        return run_child(args)

    from positions import load_positions_list
    all_positions = load_positions_list()
    by_name = {p["name"]: p for p in all_positions}
    if args.positions == "all":
        names = [p["name"] for p in all_positions]
    else:
        names = [n.strip() for n in args.positions.split(",") if n.strip()]
        unknown = [n for n in names if n not in by_name]
        if unknown:
            ap.error("unknown position(s): %s (have: %s)"
                     % (", ".join(unknown), ", ".join(by_name)))

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("seq", "par"):
            ap.error("unknown mode %r (use seq and/or par)" % m)

    if not os.path.exists(args.python):
        ap.error("interpreter not found: %s" % args.python)

    out_path = args.out or os.path.join(
        BENCH_DIR, "results-%s.json" % time.strftime("%Y%m%d-%H%M%S"))

    header = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "uptime": uptime_line(),
        "load_average_at_start": load_average(),
        "nproc": nproc(),
        "python": args.python,
        "platform": platform.platform(),
        "depths": depths,
        "modes": modes,
        "repeats": args.repeats,
        "par_min_depth": args.par_min_depth,
        "positions": names,
        "load_warn_threshold": LOAD_WARN_THRESHOLD,
    }
    print("host: %d cores | %s" % (header["nproc"], header["uptime"]))
    if header["load_average_at_start"] and header["load_average_at_start"][0] > LOAD_WARN_THRESHOLD:
        print("WARNING: load average %.2f > %.1f at start - scaling numbers will be "
              "unreliable and are flagged in the results."
              % (header["load_average_at_start"][0], LOAD_WARN_THRESHOLD))
    print("plan: %d positions x %d depths x %d modes x %d repeats = %d searches"
          % (len(names), len(depths), len(modes), args.repeats,
             len(names) * len(depths) * len(modes) * args.repeats))

    results = []
    total = len(names) * len(depths) * len(modes)
    done = 0
    for name in names:
        for depth in depths:
            for mode in modes:
                runs = [run_one(args.python, name, depth, mode,
                                args.par_min_depth, args.timeout, args.keep_stderr)
                        for _ in range(args.repeats)]
                cell = reduce_repeats(runs)
                cell.setdefault("position", name)
                cell.setdefault("depth", depth)
                cell.setdefault("mode", mode)
                cell["phase"] = by_name[name]["phase"]
                results.append(cell)
                done += 1
                if "error" in cell:
                    print("[%d/%d] %-20s d%-2d %-3s  FAILED: %s"
                          % (done, total, name, depth, mode, cell["error"]))
                else:
                    split = cell["parallel_splits"][-1] if cell.get("parallel_splits") else None
                    extra = ""
                    if split:
                        extra = "  base %.2fs scout %.2fs(%d) re %.2fs(%d)" % (
                            split["baseline_s"], split["scout_s"], split["scout_moves"],
                            split["research_s"], split["research_moves"])
                    print("[%d/%d] %-20s d%-2d %-3s  %7.3fs  %-7s score=%-7s%s%s"
                          % (done, total, name, depth, mode, cell["seconds_min"],
                             cell["best_move_str"], cell["score"], extra,
                             "  [LOAD SUSPECT]" if cell["load_suspect"] else ""))

    header["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    header["uptime_at_end"] = uptime_line()
    header["load_average_at_end"] = load_average()
    with open(out_path, "w") as fh:
        json.dump({"header": header, "results": results}, fh, indent=2)
    print("\nwrote %s" % out_path)
    suspect = [r for r in results if r.get("load_suspect")]
    if suspect:
        print("NOTE: %d/%d cells ran with load average > %.1f - treat their "
              "scaling numbers as unreliable." % (len(suspect), len(results), LOAD_WARN_THRESHOLD))
    return 0

if __name__ == "__main__":
    sys.exit(main())
