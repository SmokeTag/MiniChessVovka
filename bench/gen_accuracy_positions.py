"""
Regenerate bench/accuracy_positions.json, the position suite bench/accuracy.py
measures over.

Why a second suite, when bench/positions.json already exists: run_bench.py needs
8 positions it can time to the millisecond, and accuracy needs a *sample*. The
merge that landed the quiescence margin says why in one line - the same ablation
read -13.5 +/- 9.4 over 16 positions and -82.9 +/- 4.3 over 48. A search change
is not resolvable at n=8.

Positions are stored as FENs (engine_rs/src/fen.rs carries exactly what the hash
reads), not as move lists, because the harness has to hand the same position to
many engine configurations and a FEN round-trips through ai.from_fen in one
call. Every FEN written here is verified to satisfy to_fen(from_fen(f)) == f.

The walk mixes random legal moves with shallow engine moves so the sample is
neither pure noise nor a single engine line, and positions are stratified by how
full the hands are: crazyhouse cost and crazyhouse tactics both live in the
hands, and a suite of empty-hand openings measures a game nobody plays.

Run:  .venv-bench/bin/python bench/gen_accuracy_positions.py
Never loads or writes book.db.
"""
import argparse
import json
import os
import random
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, BENCH_DIR)

OUT = os.path.join(BENCH_DIR, "accuracy_positions.json")

WALK_DEPTH = 3
MIN_LEGAL = 6
MIN_PLY = 4
MAX_PLY = 44

# hand-size bucket -> share of the suite
BUCKETS = [("empty", 0, 0, 20), ("light", 1, 2, 24), ("heavy", 3, 99, 20)]

# Roughly a third of a random crazyhouse walk is already decided by depth 8, and
# bench/accuracy.py drops those (a score error against +/-CHECKMATE_SCORE is not
# a number anyone can average). Generate enough that what survives is still a
# wide sample.


def fresh_state():
    from gamestate import GameState
    gs = GameState()
    gs.setup_initial_board()
    return gs


def hand_count(gs):
    return sum(sum(h.values()) for h in gs.hands.values())


def snapshot(gs, ai):
    import minichess_engine as rs
    rust_gs = ai._sync_to_rust(gs)
    return rs.to_fen(rust_gs)


def walk(seed, rng, ai, out, seen):
    import minichess_engine as rs
    gs = fresh_state()
    for ply in range(MAX_PLY + 1):
        if gs.checkmate or gs.stalemate:
            return
        legal = gs.get_all_legal_moves()
        if not legal:
            return
        if MIN_PLY <= ply and len(legal) >= MIN_LEGAL:
            fen = snapshot(gs, ai)
            if fen not in seen:
                seen.add(fen)
                out.append({
                    "fen": fen,
                    "ply": ply,
                    "side_to_move": gs.current_turn,
                    "legal_moves": len(legal),
                    "in_hand": hand_count(gs),
                    "in_check": gs.is_in_check(gs.current_turn),
                    "seed": seed,
                })
        if rng.random() < 0.5:
            move = rng.choice(legal)
        else:
            rust_gs = ai._sync_to_rust(gs)
            move, _ = rs.find_best_move_with_score(rust_gs, WALK_DEPTH, None, False)
            if move is None or move not in legal:
                move = rng.choice(legal)
        if not gs.make_move(move):
            return
        if gs.needs_promotion_choice:
            gs.complete_promotion("R" if gs.current_turn == "w" else "r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every bucket's size (default 1.0 -> 64 positions)")
    ap.add_argument("--append", action="store_true",
                    help="keep the positions already in --out and add to them; "
                         "the reference cache is keyed by FEN, so nothing already "
                         "measured has to be recomputed")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    import ai
    import minichess_engine as rs
    rs.set_search_knobs({"use_book": False})

    kept = []
    if args.append and os.path.exists(args.out):
        with open(args.out) as fh:
            kept = json.load(fh)["positions"]

    rng = random.Random(args.seed)
    pool, seen = [], {p["fen"] for p in kept}
    for g in range(args.games):
        walk(g, rng, ai, pool, seen)

    picked = []
    for name, lo, hi, want in BUCKETS:
        want = max(0, int(round(want * args.scale))
                   - len([p for p in kept if p.get("bucket") == name]))
        cands = [p for p in pool if lo <= p["in_hand"] <= hi]
        rng.shuffle(cands)
        cands.sort(key=lambda p: p["ply"])
        stride = max(1, len(cands) // want)
        chosen = cands[::stride][:want]
        for p in chosen:
            p["bucket"] = name
        picked.extend(chosen)
        if len(chosen) < want:
            print("bucket %s: only %d of %d wanted" % (name, len(chosen), want))

    picked.extend(kept)
    for p in picked:
        gs = rs.from_fen(p["fen"])
        assert rs.to_fen(gs) == p["fen"], p["fen"]

    picked.sort(key=lambda p: (p["bucket"], p["ply"], p["fen"]))
    payload = {
        "note": "Accuracy suite for bench/accuracy.py. FENs only; see the module "
                "docstring in bench/gen_accuracy_positions.py.",
        "walk_depth": WALK_DEPTH,
        "seed": args.seed,
        "games": args.games,
        "positions": picked,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print("wrote %d positions to %s" % (len(picked), args.out))
    for name, _, _, _ in BUCKETS:
        n = [p for p in picked if p["bucket"] == name]
        if n:
            print("  %-6s %2d positions, %.1f legal moves avg, plies %d-%d"
                  % (name, len(n), sum(p["legal_moves"] for p in n) / len(n),
                     min(p["ply"] for p in n), max(p["ply"] for p in n)))


if __name__ == "__main__":
    main()
