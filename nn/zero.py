"""The zero loop: self-play, train, repeat, from random weights.

**No teacher, and no evaluation gate.** Dropping the teacher is the point of phase 3 --
`nn.teacher` was scaffolding that got a network playing before MCTS existed, and nothing
here reads it. Dropping the gate is the AlphaZero paper's own simplification over AlphaGo
Zero: there is no candidate-versus-incumbent match and no promotion rule, just one network
that is continuously updated and always the one generating games. A gate costs a few
hundred games an iteration to answer a question ("is this better?") that the next
iteration answers for free, and a gate that stalls leaves the run generating games from a
frozen network all night.

The alpha-beta match played every few iterations is therefore a **diagnostic, not a gate**
-- nothing branches on it. It exists so that the morning's question ("did it learn
anything?") has an answer that does not depend on the network grading itself.

Resumable: an iteration whose self-play directory and checkpoint both exist is skipped, so
a killed run continues rather than restarting the night.
"""
import argparse
import json
import os
import time

import numpy as np

from nn import paths, selfplay

STATUS = "status.json"


def _log_path(run):
    return os.path.join(paths.subdir("zero", run), "log.jsonl")


def _append(run, row):
    with open(_log_path(run), "a") as fh:
        fh.write(json.dumps(row) + "\n")


def _ckpt(run, i):
    return os.path.join(paths.subdir("zero", run, "checkpoints"), "iter%03d.pt" % i)


def _window(run, i, window):
    return ["%s/iter%03d" % (run, j) for j in range(max(0, i - window + 1), i + 1)]


def sample_records(names, cap, rng, log=print):
    """The replay window, subsampled to `cap` positions so RAM stays bounded.

    Encoded positions are 3.4kB each (24 planes of 6x6 float32), so an uncapped window of
    four 2,000-game iterations is over 2GB before torch copies it. Subsampling uniformly
    across the window is the cheap fix and costs nothing statistically -- the window is
    already far larger than a single epoch needs.
    """
    recs = []
    for name in names:
        try:
            recs.extend(selfplay.load_records(name))
        except FileNotFoundError:
            continue
    if cap and len(recs) > cap:
        sel = rng.choice(len(recs), size=cap, replace=False)
        recs = [recs[i] for i in sel]
        log("  replay window subsampled to %d positions" % cap)
    return recs


def _eval_worker(task):
    """One slice of the diagnostic match, in its own process.

    Silenced with the same dup2 trick `nn.teacher` uses -- `search.rs` narrates every
    iterative-deepening step, and the loop's own reporting shares fd 1. Doing it in a
    worker is what keeps the parent's log readable.
    """
    (wid, game_ids, ckpt, depth, sims, device, seed, opening_plies) = task
    import os as _os
    devnull = _os.open(_os.devnull, _os.O_WRONLY)
    _os.dup2(devnull, 1)
    _os.dup2(devnull, 2)
    _os.close(devnull)

    import minichess_engine as rs
    from nn import mcts
    from nn.model import load_checkpoint

    net, _ = load_checkpoint(ckpt, device=device)
    ev = mcts.Evaluator(net, device)
    # One tree per ladder game, re-rooted past our move and the opponent's reply.
    searcher = mcts.Searcher(ev, simulations=sims, batch=64)
    wins = draws = losses = 0

    for g in game_ids:
        rng = np.random.default_rng(seed + g)
        # Both players are deterministic -- argmax visits, and a fixed-depth search --
        # so without this every game with the same colour assignment is the *same game*.
        # A few temperature-sampled opening plies from the network, plus a distinct
        # move-ordering seed for the opponent, is what makes N games worth N games.
        rs.set_search_knobs({"use_book": False, "order_seed": int(seed + g)})
        searcher.reset()
        gs = rs.GameState()
        gs.setup_initial_board()
        net_is_white = (g % 2 == 0)
        while True:
            if gs.check_game_over() or gs.is_terminal_draw():
                break
            moves = gs.get_all_legal_moves()
            if not moves:
                break
            if (gs.current_turn == "w") == net_is_white:
                tree = searcher.search(gs)
                if gs.ply < opening_plies:
                    entries = tree.root_visits()
                    if not entries:
                        break
                    counts = np.array([c for _, c in entries], dtype=np.float64)
                    if counts.sum() == 0:
                        counts = np.ones_like(counts)
                    probs = counts / counts.sum()
                    mv = entries[int(rng.choice(len(entries), p=probs))][0]
                else:
                    mv = tree.best_move()
            else:
                mv = rs.find_best_move(gs, depth=depth)
            if mv is None:
                break
            gs.make_move(mv)

        if gs.checkmate:
            if (gs.current_turn == "w") == net_is_white:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1
    return {"wins": wins, "draws": draws, "losses": losses}


def evaluate_vs_alphabeta(ckpt, games, depth, sims, device, seed, workers=8,
                          opening_plies=6, log=print):
    """A diagnostic match against the alpha-beta engine. Score in [0, 1], draws a half.

    Nothing branches on this -- see the module docstring. It answers "did it learn
    anything" in the morning without the network grading itself.
    """
    import multiprocessing as mp

    workers = max(1, min(workers, games))
    slices = [list(range(g, games, workers)) for g in range(workers)]
    tasks = [(w, s, ckpt, depth, sims, device, seed, opening_plies)
             for w, s in enumerate(slices) if s]
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(tasks)) as pool:
        parts = pool.map(_eval_worker, tasks)

    wins = sum(p["wins"] for p in parts)
    draws = sum(p["draws"] for p in parts)
    losses = sum(p["losses"] for p in parts)
    score = (wins + 0.5 * draws) / max(1, games)
    out = {"games": games, "depth": depth, "sims": sims,
           "wins": wins, "draws": draws, "losses": losses, "score": score}
    log("  vs alphabeta d%d: score %.3f (%dW %dD %dL over %d)"
        % (depth, score, wins, draws, losses, games))
    return out


def main():
    ap = argparse.ArgumentParser(description="Tabula rasa AlphaZero loop.")
    ap.add_argument("--run", default="zero1")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--window", type=int, default=4, help="replay window in iterations")
    ap.add_argument("--max-positions", type=int, default=400_000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--train-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--eval-depth", type=int, default=2)
    ap.add_argument("--eval-sims", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deadline-hours", type=float, default=0.0,
                    help="stop starting new iterations after this many hours")
    args = ap.parse_args()

    from nn import train_zero

    rng = np.random.default_rng(args.seed)
    root = paths.subdir("zero", args.run)
    started = time.time()
    print("zero run %r -> %s" % (args.run, root), flush=True)
    print("  %d iterations x %d games @ %d sims, %d workers"
          % (args.iterations, args.games, args.sims, args.workers), flush=True)

    for i in range(args.iterations):
        if args.deadline_hours and (time.time() - started) / 3600.0 > args.deadline_hours:
            print("deadline reached; stopping before iteration %d" % i, flush=True)
            break

        name = "%s/iter%03d" % (args.run, i)
        ckpt = _ckpt(args.run, i)
        prev = _ckpt(args.run, i - 1) if i > 0 else None
        it_started = time.time()
        print("\n=== iteration %d/%d ===" % (i, args.iterations - 1), flush=True)

        if os.path.exists(ckpt):
            print("  already done, skipping", flush=True)
            continue

        sp_dir = paths.subdir("selfplay", name, create=False)
        if os.path.isdir(sp_dir) and os.listdir(sp_dir):
            print("  self-play already present, reusing", flush=True)
            sp = {"games": None, "records": None, "seconds": 0.0, "wins": None}
        else:
            sp = selfplay.generate(
                name=name, games=args.games, workers=args.workers,
                ckpt=prev, sims=args.sims, batch=args.batch,
                alpha=selfplay.DIRICHLET_ALPHA, eps=selfplay.DIRICHLET_EPS,
                temp_plies=selfplay.TEMP_PLIES, seed=args.seed + i * 104729,
                device=args.device)

        records = sample_records(_window(args.run, i, args.window), args.max_positions, rng)
        print("  training on window %s" % ", ".join(_window(args.run, i, args.window)), flush=True)
        _net, history = train_zero.run(
            records, ckpt, epochs=args.epochs, batch=args.train_batch, lr=args.lr,
            weight_decay=1e-4, value_weight=train_zero.VALUE_WEIGHT,
            device=args.device, seed=args.seed + i, channels=64, blocks=6,
            init_from=prev, log=lambda s: print(s, flush=True))

        row = {"iteration": i, "seconds": time.time() - it_started,
               "selfplay": {k: sp.get(k) for k in ("games", "records", "seconds", "wins")},
               "train": history[-1] if history else None,
               "checkpoint": ckpt}

        if args.eval_every and (i % args.eval_every == 0 or i == args.iterations - 1):
            row["eval"] = evaluate_vs_alphabeta(
                ckpt, args.eval_games, args.eval_depth, args.eval_sims,
                args.device, args.seed + 1000 + i, log=lambda s: print(s, flush=True))

        _append(args.run, row)
        with open(os.path.join(root, STATUS), "w") as fh:
            json.dump({"run": args.run, "last_iteration": i, "checkpoint": ckpt,
                       "elapsed_hours": (time.time() - started) / 3600.0,
                       "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)
        print("  iteration %d done in %.1f min" % (i, (time.time() - it_started) / 60.0), flush=True)

    print("\nrun complete: %.2f hours" % ((time.time() - started) / 3600.0), flush=True)


if __name__ == "__main__":
    main()
