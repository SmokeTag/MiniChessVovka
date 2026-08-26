# -*- coding: utf-8 -*-
"""
Compare two run_bench.py results files: sequential vs root-parallel.

    ./venv/bin/python bench/compare.py bench/seq.json bench/par.json

Reads every cell from both files, pairs them by (position, depth), and reports
speedup, best-move agreement, score deltas and where the parallel time went
(baseline / scout / re-search).

Agreement is the headline, not speed. The parallel root split searches the same
tree with a different alpha history and a different transposition table, so it
can legitimately return a different move of equal score - but a different move
with a DIFFERENT score means the two paths disagree about the position, and a
fast wrong answer is a regression, not a win. Those rows are printed loudly and
set a non-zero exit status.

A cell whose repeats ran with load average above the threshold is marked (!);
its speedup is noise, not measurement.
"""
import argparse
import json
import sys

BAR = "=" * 100


def load(path):
    with open(path) as fh:
        data = json.load(fh)
    return data.get("header", {}), data.get("results", [])


def collect(files):
    """
    (position, depth, mode) -> best cell.

    If the same cell appears in both files, keep the faster one: consistent with
    taking the minimum across repeats, since contention only adds time.
    """
    cells = {}
    duplicates = 0
    for path, (_, results) in files.items():
        for cell in results:
            if "error" in cell or cell.get("seconds_min") is None:
                continue
            key = (cell["position"], cell["depth"], cell["mode"])
            cell = dict(cell, _source=path)
            if key in cells:
                duplicates += 1
                if cell["seconds_min"] < cells[key]["seconds_min"]:
                    cells[key] = cell
            else:
                cells[key] = cell
    return cells, duplicates


def fmt_split(cell):
    splits = cell.get("parallel_splits") or []
    if not splits:
        return "-"
    s = splits[-1]
    return "b%.2f/s%.2f(%d)/r%.2f(%d)" % (
        s["baseline_s"], s["scout_s"], s["scout_moves"],
        s["research_s"], s["research_moves"])


def classify(seq, par):
    """-> (status, note). status in ok / alt / DISAGREE."""
    if seq["best_move_str"] == par["best_move_str"]:
        if seq["score"] != par["score"]:
            return "DISAGREE", "same move, score %s -> %s" % (seq["score"], par["score"])
        return "ok", ""
    if seq["score"] == par["score"]:
        return "alt", "different move, equal score %s" % seq["score"]
    return "DISAGREE", "%s(%s) -> %s(%s)" % (
        seq["best_move_str"], seq["score"], par["best_move_str"], par["score"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--quiet-alt", action="store_true",
                    help="do not list equal-score move changes in the summary")
    args = ap.parse_args()

    files = {p: load(p) for p in (args.file_a, args.file_b)}
    cells, duplicates = collect(files)

    print(BAR)
    for path, (header, _) in files.items():
        print("%s" % path)
        print("    run %s -> %s | %s cores | %s"
              % (header.get("started", "?"), header.get("finished", "?"),
                 header.get("nproc", "?"), header.get("uptime", "no uptime recorded")))
        if header.get("uptime_at_end"):
            print("    at end: %s" % header["uptime_at_end"])
    if duplicates:
        print("note: %d cell(s) appeared in both files; kept the faster one." % duplicates)
    print(BAR)

    pairs = []
    seq_only, par_only = [], []
    for (pos, depth, mode), cell in sorted(cells.items()):
        if mode != "seq":
            continue
        par = cells.get((pos, depth, "par"))
        if par is None:
            seq_only.append((pos, depth))
        else:
            pairs.append((pos, depth, cell, par))
    for (pos, depth, mode) in cells:
        if mode == "par" and (pos, depth, "seq") not in cells:
            par_only.append((pos, depth))

    if not pairs:
        print("No (position, depth) has both a seq and a par measurement.")
        if seq_only or par_only:
            print("  seq-only cells: %s" % sorted(set(seq_only)))
            print("  par-only cells: %s" % sorted(set(par_only)))
        return 2

    hdr = ("%-20s %2s %9s %9s %7s  %-8s %-8s %-6s %-9s %-4s %s"
           % ("position", "d", "seq s", "par s", "speedup",
              "seq move", "par move", "dscore", "agree", "", "par split b/s/r"))
    print(hdr)
    print("-" * len(hdr))

    disagreements, alternatives, suspect = [], [], []
    for pos, depth, seq, par in sorted(pairs, key=lambda t: (t[1], t[0])):
        speedup = seq["seconds_min"] / par["seconds_min"] if par["seconds_min"] else float("nan")
        status, note = classify(seq, par)
        dscore = (par["score"] - seq["score"]
                  if seq["score"] is not None and par["score"] is not None else None)
        flag = ""
        if seq.get("load_suspect") or par.get("load_suspect"):
            flag = " (!)"
            suspect.append((pos, depth))
        if status == "DISAGREE":
            disagreements.append((pos, depth, note, speedup))
        elif status == "alt":
            alternatives.append((pos, depth, note, speedup))
        print("%-20s %2d %9.3f %9.3f %6.2fx  %-8s %-8s %-6s %-9s %-4s %s"
              % (pos, depth, seq["seconds_min"], par["seconds_min"], speedup,
                 seq["best_move_str"], par["best_move_str"],
                 "-" if dscore is None else str(dscore),
                 status, flag, fmt_split(par)))

    # ---- per-depth aggregate ------------------------------------------------
    print()
    print("per-depth totals (sum of minimums across positions)")
    print("%2s %6s %10s %10s %8s %10s %s"
          % ("d", "cells", "seq s", "par s", "speedup", "agreement", "avg par split b/s/r"))
    depths = sorted({d for _, d, _, _ in pairs})
    for depth in depths:
        rows = [(s, p) for _, d, s, p in pairs if d == depth]
        seq_total = sum(s["seconds_min"] for s, _ in rows)
        par_total = sum(p["seconds_min"] for _, p in rows)
        agree = sum(1 for s, p in rows if classify(s, p)[0] == "ok")
        splits = [p["parallel_splits"][-1] for _, p in rows if p.get("parallel_splits")]
        if splits:
            avg = "b%.2f/s%.2f/r%.2f" % (
                sum(x["baseline_s"] for x in splits) / len(splits),
                sum(x["scout_s"] for x in splits) / len(splits),
                sum(x["research_s"] for x in splits) / len(splits))
            share = sum(x["research_s"] for x in splits) / max(
                1e-9, sum(x["baseline_s"] + x["scout_s"] + x["research_s"] for x in splits))
            avg += "  re-search=%.0f%% of split time" % (100 * share)
        else:
            avg = "-"
        print("%2d %6d %10.3f %10.3f %7.2fx %6d/%-3d %s"
              % (depth, len(rows), seq_total, par_total,
                 seq_total / par_total if par_total else float("nan"),
                 agree, len(rows), avg))

    # ---- verdict ------------------------------------------------------------
    print()
    if disagreements:
        print("!" * len(BAR))
        print("!! %d BEST-MOVE / SCORE DISAGREEMENT(S) - a fast wrong answer is a regression."
              % len(disagreements))
        for pos, depth, note, speedup in disagreements:
            print("!!   %-20s depth %-2d  %-42s (par was %.2fx)" % (pos, depth, note, speedup))
        print("!! The parallel path must return the same score as the sequential one.")
        print("!" * len(BAR))
    else:
        print("No score disagreements: every paired cell returned the same score.")

    if alternatives and not args.quiet_alt:
        print()
        print("%d equal-score move change(s) (allowed, but check they are really equal):"
              % len(alternatives))
        for pos, depth, note, speedup in alternatives:
            print("     %-20s depth %-2d  %s" % (pos, depth, note))

    if suspect:
        print()
        print("(!) %d row(s) measured under load average above the harness threshold; "
              "their speedups are unreliable: %s"
              % (len(suspect), ", ".join("%s d%d" % (p, d) for p, d in sorted(set(suspect)))))

    if seq_only or par_only:
        print()
        print("unpaired cells - seq-only: %s | par-only: %s"
              % (sorted(set(seq_only)) or "none", sorted(set(par_only)) or "none"))

    return 1 if disagreements else 0


if __name__ == "__main__":
    sys.exit(main())
