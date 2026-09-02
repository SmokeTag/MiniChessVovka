"""Tabula rasa self-play: MCTS games against itself, written as training records.

This is the generator phase 3 actually needs, and it replaces `nn.teacher` rather than
extending it. A teacher record carries the *alpha-beta* verdict on a position -- one best
move and a centipawn score. A self-play record carries what the search itself believed:
the full root visit distribution as the policy target, and the eventual game result as the
value target. No alpha-beta anywhere, which is the whole point of going zero.

**Root Dirichlet noise is applied here, in Python, not in the Rust tree.** The first
`collect()` on a fresh tree returns exactly one leaf -- the root -- so the first evaluator
call of a search is the only place the root's priors are ever set. Rust reads priors only
at the legal actions and renormalises them, so mixing the noise over the legal set (and
leaving it summing to 1 there) makes that renormalisation a no-op and reproduces
AlphaZero's rule exactly:

    P(s,a) = (1 - eps) * p_a + eps * eta_a,   eta ~ Dir(alpha) over legal actions only

Spreading Dir over all 2,196 logits and letting Rust renormalise would *not* be the same
rule -- the noise mass landing on illegal actions is discarded, so the effective epsilon
shrinks by a random amount that depends on the branching factor. That difference is silent
and would only show up as self-play that never explores.

Records reuse `teacher.restore`'s field names (fen/ply/ply_limit/reps) so the same
position-not-encoding indirection applies: a plane-layout change costs a re-encode, not a
regenerated night of games.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from nn import paths

# AlphaZero's chess numbers, scaled to this game's branching factor. Their alpha is
# 0.3 for chess (~35 legal moves); minihouse averages ~22 with drops, so a slightly
# flatter draw is right. eps is the paper's 0.25 unchanged.
DIRICHLET_ALPHA = 0.5
DIRICHLET_EPS = 0.25

# Temperature 1 for the opening, then argmax. AlphaZero uses 30 plies of full chess;
# minihouse games are far shorter (a random net mates in 13-47 plies), so 12 keeps the
# sampled window a comparable *fraction* of the game rather than most of it.
TEMP_PLIES = 12

DEFAULT_SIMS = 400
DEFAULT_BATCH = 128
PLY_LIMIT = 200


def _mix_root_noise(priors, legal, rng, alpha, eps):
    """Return priors with Dirichlet noise mixed in over `legal`, summing to 1 there."""
    p = priors[legal].astype(np.float64)
    total = p.sum()
    p = p / total if total > 0 else np.full(len(legal), 1.0 / len(legal))
    eta = rng.dirichlet([alpha] * len(legal))
    out = priors.copy()
    out[legal] = (1.0 - eps) * p + eps * eta
    return out


class FastEvaluator:
    """`nn.mcts.Evaluator`, with the Python-list marshalling taken off the hot path.

    Profiling one self-play worker at 400 sims found the network forward pass was ~3ms of
    an 80ms ply; nearly everything else was converting Python lists at the PyO3 boundary.
    `collect` hands back a list of 110k floats and `torch.tensor(list)` parses it at
    4.6ms a call, where `np.fromiter` does the same job in 1.3ms.

    Going the other way the result is counter-intuitive and was measured, not assumed:
    `expand` accepts a numpy array, but PyO3 extracts `Vec<f32>` from it through the
    sequence protocol one element at a time, which is *slower* than materialising a
    Python list first. So the fast path is `fromiter` in and `.tolist()` out.

    The remaining marshalling (~50ms/ply) is only removable at the Rust boundary itself,
    by moving `collect`/`expand` to the buffer protocol. That is the next real lever.
    """

    def __init__(self, net, device):
        import torch
        self.torch = torch
        self.net = net.eval()
        self.device = device
        self.calls = 0
        self.positions = 0

    def __call__(self, flat_planes):
        import minichess_engine as rs
        torch = self.torch
        n = len(flat_planes) // rs.ENCODE_INPUT_SIZE
        a = np.fromiter(flat_planes, dtype=np.float32, count=len(flat_planes))
        with torch.inference_mode():
            x = torch.from_numpy(a).view(n, rs.ENCODE_PLANES, rs.BOARD_SIZE,
                                         rs.BOARD_SIZE).to(self.device, non_blocking=True)
            logits, values = self.net(x)
            priors = logits.float().softmax(-1).reshape(-1).cpu().numpy()
            vals = values.float().reshape(-1).cpu().numpy()
        self.calls += 1
        self.positions += n
        return priors, vals


def search_with_root_noise(gs, evaluator, rs, rng, sims, batch, alpha, eps):
    """`nn.mcts.search`, with the root's priors perturbed on the one call that sets them.

    Deliberately a copy of that loop rather than a flag on it: the noise belongs to
    self-play, and an arena or a GUI move must never get it.
    """
    from nn import mcts

    tree = rs.Mcts(gs, mcts.DEFAULT_C_PUCT)
    first = True
    while tree.simulations < sims:
        want = min(batch, max(1, sims - tree.simulations))
        before = tree.simulations
        planes = tree.collect(want)
        if planes:
            priors, values = evaluator(planes)
            if first:
                # The first batch of a fresh tree is the root alone.
                n = len(planes) // rs.ENCODE_INPUT_SIZE
                assert n == 1, "expected the root alone in the first batch, got %d" % n
                legal = np.asarray(rs.legal_action_indices(gs), dtype=np.int64)
                priors = _mix_root_noise(np.asarray(priors), legal, rng, alpha, eps)
                first = False
            tree.expand(priors.tolist(), values.tolist())
        elif tree.simulations == before:
            break
    return tree


def play_game(evaluator, rs, rng, sims, batch, alpha, eps, temp_plies):
    """One self-play game. Returns the records, with `z` filled in from the result."""
    from nn import mcts

    gs = rs.GameState()
    gs.setup_initial_board()
    gs.ply_limit = PLY_LIMIT

    pending = []
    result_white = 0.0

    while True:
        if gs.check_game_over() or gs.is_terminal_draw():
            if gs.checkmate:
                # `current_turn` is the side to move, i.e. the side that was mated.
                result_white = -1.0 if gs.current_turn == "w" else 1.0
            else:
                result_white = 0.0
            break
        legal_moves = gs.get_all_legal_moves()
        if not legal_moves:
            result_white = 0.0
            break

        fen = rs.to_fen(gs)
        turn = gs.current_turn
        tree = search_with_root_noise(gs, evaluator, rs, rng, sims, batch, alpha, eps)

        entries = tree.root_visits() if hasattr(tree, "root_visits") else tree.root_moves()
        if not entries:
            result_white = 0.0
            break

        moves = [m for m, _ in entries]
        counts = np.array([c for _, c in entries], dtype=np.float64)
        if counts.sum() == 0:
            counts = np.ones_like(counts)

        # The policy target is the visit distribution at temperature 1, always --
        # temperature only ever affects which move gets *played*.
        target = counts / counts.sum()
        pending.append({
            "fen": fen,
            "turn": turn,
            "ply": int(gs.ply),
            "ply_limit": int(gs.ply_limit),
            "reps": int(gs.repetition_count()),
            "visits": [[int(rs.move_to_action_index(gs, m)), float(p)]
                       for m, p in zip(moves, target) if p > 0],
            "n_legal": len(legal_moves),
        })

        if gs.ply < temp_plies:
            pick = rng.choice(len(moves), p=target)
        else:
            pick = int(counts.argmax())
        gs.make_move(moves[pick])

    for rec in pending:
        # Value is in the mover's own frame, like the teacher's -- the network is
        # canonicalised to the side to move, so the sign follows the same flip.
        rec["z"] = result_white if rec["turn"] == "w" else -result_white
    return pending, result_white


def _worker(task):
    (wid, games, out_dir, ckpt, sims, batch, alpha, eps, temp_plies, seed, device) = task
    import torch
    import minichess_engine as rs
    from nn import mcts
    from nn.model import MinihouseNet, load_checkpoint

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if ckpt:
        net, _meta = load_checkpoint(ckpt, device=device)
    else:
        # Iteration 0 of a zero run: the weights are random and that is the point.
        net = MinihouseNet().to(device)
    evaluator = FastEvaluator(net, device)

    shard = os.path.join(out_dir, "shard_%04d.jsonl" % wid)
    written = plies = 0
    wins = {"w": 0, "b": 0, "d": 0}
    started = time.time()
    with open(shard, "w") as fh:
        for g in range(games):
            recs, res = play_game(evaluator, rs, rng, sims, batch, alpha, eps, temp_plies)
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            written += len(recs)
            plies += len(recs)
            wins["w" if res > 0 else "b" if res < 0 else "d"] += 1
    return {"worker": wid, "games": games, "records": written, "plies": plies,
            "wins": wins, "seconds": time.time() - started}


def generate(name, games, workers, ckpt, sims, batch, alpha, eps, temp_plies, seed, device):
    out_dir = paths.subdir("selfplay", name)
    per = [games // workers + (1 if i < games % workers else 0) for i in range(workers)]
    tasks = [(i, per[i], out_dir, ckpt, sims, batch, alpha, eps, temp_plies, seed + i * 7919, device)
             for i in range(workers) if per[i] > 0]

    started = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(tasks)) as pool:
        stats = pool.map(_worker, tasks)

    elapsed = time.time() - started
    tot_games = sum(s["games"] for s in stats)
    tot_recs = sum(s["records"] for s in stats)
    wins = {"w": 0, "b": 0, "d": 0}
    for s in stats:
        for k in wins:
            wins[k] += s["wins"][k]
    print("self-play %s: %d games, %d records in %.1fs (%.1f games/s, %.0f plies/s)"
          % (name, tot_games, tot_recs, elapsed, tot_games / elapsed, tot_recs / elapsed))
    print("  W %d  B %d  draw %d   mean game %.1f plies"
          % (wins["w"], wins["b"], wins["d"], tot_recs / max(1, tot_games)))
    return {"dir": out_dir, "games": tot_games, "records": tot_recs,
            "seconds": elapsed, "wins": wins}


def load_records(name):
    """Every self-play record under a run directory."""
    out = []
    d = paths.subdir("selfplay", name, create=False)
    for fname in sorted(os.listdir(d)):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(d, fname)) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser(description="Tabula rasa MCTS self-play.")
    ap.add_argument("--name", default="iter000")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--checkpoint", default=None, help="omit for random init (iteration 0)")
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--alpha", type=float, default=DIRICHLET_ALPHA)
    ap.add_argument("--eps", type=float, default=DIRICHLET_EPS)
    ap.add_argument("--temp-plies", type=int, default=TEMP_PLIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    generate(args.name, args.games, args.workers, args.checkpoint, args.sims,
             args.batch, args.alpha, args.eps, args.temp_plies, args.seed, args.device)


if __name__ == "__main__":
    main()
