#!/usr/bin/env python3
"""Play the network against a reference opponent and report a score.

    ./venv/bin/python -m nn.arena --opponent random --games 200
    ./venv/bin/python -m nn.arena --opponent alphabeta --depth 2 --games 200
    ./venv/bin/python -m nn.arena --opponent alphabeta --depth 6 --games 200 --run bootstrap

This is the phase-2 exit criterion and, later, the gate every checkpoint has to pass.
The success bar for the whole project is stated against this harness and not against a
previous checkpoint: **beat the alpha-beta engine**. Measuring only against your own
past best is how a plateau disguises itself as progress.

The network here plays with **no search at all** -- one forward pass, mask the illegal
actions, take the argmax. That is deliberate for phase 2: it isolates whether the
encoding and the policy head learned anything, with no tree to cover for them. Phase 3
puts MCTS on top.

Pairing. Openings are random legal walks of a few plies, and **every opening is played
twice, once with each colour**. Both players are deterministic, so without an opening
book every game would be the same game; without the colour swap the result would
measure the opening's bias as much as the players'. The reported score is over all
games, wins + half draws, with a binomial standard error.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai
import minichess_engine as rs
from gamestate import GameState
from nn import paths

WIN, DRAW, LOSS = "win", "draw", "loss"

class RandomPlayer:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def move(self, gs, legal):
        return self.rng.choice(legal)

class AlphaBetaPlayer:
    def __init__(self, depth):
        self.depth = depth
        self.name = "alphabeta-d%d" % depth

    def move(self, gs, legal):
        move = ai.find_best_move(gs, depth=self.depth)
        return move if move in legal else None

class PolicyPlayer:
    """Raw policy argmax over the legal actions. No search, no temperature."""

    def __init__(self, net, device, name="policy"):
        import torch
        self.torch = torch
        self.net = net.eval()
        self.device = device
        self.name = name

    def move(self, gs, legal):
        from nn import features
        torch = self.torch

        synced = ai._sync_to_rust(gs)
        x = torch.from_numpy(features.encode(synced)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits, _ = self.net(x)
        logits = logits.float().squeeze(0)

        indices = rs.legal_action_indices(synced)
        if not indices:
            return None
        best = max(indices, key=lambda i: logits[i].item())
        move = rs.action_index_to_move(synced, best)
        move = ai._normalize_promotion(move, gs.current_turn)
        return move if move in legal else None

class ValueLookaheadPlayer:
    """One ply of search: play the move whose resulting position the value head likes
    least *for the opponent*, resolving terminals exactly rather than evaluating them.

    **This is not the phase-2 exit criterion**, which is raw policy argmax. It exists to
    separate two explanations of a bad arena result: a network that has learnt nothing,
    versus a network that has learnt plenty and is being destroyed by having no tree. A
    searchless policy cannot see a stalemate coming or a mate available, and in a variant
    this tactical that is most of the gap. If one ply moves the number a long way, the
    representation and the heads are sound and phase 3 is the fix.
    """

    def __init__(self, net, device, name="value-1ply"):
        import torch
        self.torch = torch
        self.net = net.eval()
        self.device = device
        self.name = name

    def move(self, gs, legal):
        import numpy as np
        from nn import features
        torch = self.torch

        parent = ai._sync_to_rust(gs)
        mover = parent.current_turn
        planes, settled = [], []

        for m in legal:
            parent.make_ai_move(m)
            replies = parent.get_all_legal_moves()
            if not replies:
                # No reply: mate if the opponent is in check, otherwise stalemate.
                settled.append(1.0 if parent.is_in_check(parent.current_turn) else 0.0)
            elif parent.is_terminal_draw():
                settled.append(0.0)
            else:
                settled.append(None)
            planes.append(features.encode(parent))
            parent.undo_ai_move()

        x = torch.from_numpy(np.stack(planes)).to(self.device)
        with torch.inference_mode():
            _, child_value = self.net(x)
        child_value = child_value.float().cpu().numpy()

        # The child's value is in the *opponent's* frame; negate it into ours.
        scores = [s if s is not None else -float(v) for s, v in zip(settled, child_value)]
        best = max(range(len(legal)), key=lambda i: scores[i])
        assert mover == gs.current_turn
        return legal[best]

def random_opening(rng, plies):
    """A short legal walk whose end position is playable by both sides."""
    for _ in range(50):
        gs = GameState()
        gs.setup_initial_board()
        moves = []
        ok = True
        for _ in range(plies):
            legal = gs.get_all_legal_moves()
            if not legal:
                ok = False
                break
            move = rng.choice(legal)
            if not gs.make_move(move, False):
                ok = False
                break
            if gs.needs_promotion_choice:
                piece = rng.choice(["R", "N", "B"])
                gs.complete_promotion(piece if gs.current_turn == "b" else piece.lower())
                move = (move[0], move[1], piece if gs.current_turn == "w" else piece.lower())
            moves.append(move)
            gs.check_game_over()
            if gs.checkmate or gs.stalemate or gs.is_draw:
                ok = False
                break
        if ok and len(gs.get_all_legal_moves()) >= 2:
            return moves
    raise RuntimeError("could not build an opening of %d plies" % plies)

def play_game(white, black, opening, ply_cap=200):
    """Returns (result_for_white, plies, reason). Illegal choices forfeit, loudly."""
    gs = GameState()
    gs.setup_initial_board()
    gs.ply_limit = ply_cap

    for move in opening:
        gs.make_move(move, False)
        if gs.needs_promotion_choice:
            gs.complete_promotion("R" if gs.current_turn == "w" else "r")
        gs.check_game_over()

    while True:
        if gs.check_game_over():
            break
        legal = gs.get_all_legal_moves()
        if not legal:
            break

        player = white if gs.current_turn == "w" else black
        move = player.move(gs, legal)
        if move is None:
            # A masked argmax cannot pick an illegal action, so this is a real bug in
            # the move/index map, not a weak player. Never silently substitute.
            return (LOSS if gs.current_turn == "w" else WIN), gs.ply_count, \
                   "illegal move by %s" % player.name
        if not gs.make_move(move, False):
            return (LOSS if gs.current_turn == "w" else WIN), gs.ply_count, \
                   "rejected move by %s" % player.name
        if gs.needs_promotion_choice:
            gs.complete_promotion("R" if gs.current_turn == "w" else "r")

    if gs.checkmate:
        # check_game_over sets checkmate for the side *to move*, which has lost.
        return (LOSS if gs.current_turn == "w" else WIN), gs.ply_count, gs.game_over_message
    return DRAW, gs.ply_count, gs.game_over_message or "no legal moves"

def run_match(subject, opponent, games, seed, opening_plies, ply_cap, quiet=True):
    rng = random.Random(seed)
    openings = [random_opening(rng, rng.randint(*opening_plies)) for _ in range(games // 2)]

    tally = {WIN: 0, DRAW: 0, LOSS: 0}
    plies = []
    anomalies = []
    reasons = {}
    started = time.time()

    for i, opening in enumerate(openings):
        for subject_is_white in (True, False):
            white, black = ((subject, opponent) if subject_is_white
                            else (opponent, subject))
            result, n, reason = play_game(white, black, opening, ply_cap)
            if not subject_is_white:
                result = {WIN: LOSS, LOSS: WIN, DRAW: DRAW}[result]
            tally[result] += 1
            plies.append(n)
            if result == DRAW:
                # How a draw happened is the whole diagnosis. A searchless policy cannot
                # see a stalemate coming or a repetition arriving, so a won game thrown
                # away that way is a conversion failure, not a strength failure -- and
                # it is exactly what a tree on top is expected to fix.
                reasons[reason] = reasons.get(reason, 0) + 1
            if "illegal" in reason or "rejected" in reason:
                anomalies.append(reason)

        if not quiet and (i + 1) % 10 == 0:
            done = (i + 1) * 2
            print("  %4d/%4d games  score %.3f  (%.0fs)"
                  % (done, games, score_of(tally), time.time() - started), flush=True)

    return {
        "subject": subject.name,
        "opponent": opponent.name,
        "games": sum(tally.values()),
        "wins": tally[WIN], "draws": tally[DRAW], "losses": tally[LOSS],
        "score": score_of(tally),
        "stderr": stderr_of(tally),
        "mean_plies": sum(plies) / len(plies) if plies else 0.0,
        "draw_reasons": reasons,
        "anomalies": anomalies,
        "seconds": time.time() - started,
    }

def score_of(tally):
    n = sum(tally.values())
    return (tally[WIN] + 0.5 * tally[DRAW]) / n if n else 0.0

def stderr_of(tally):
    n = sum(tally.values())
    if n < 2:
        return 0.0
    p = score_of(tally)
    return (p * (1 - p) / n) ** 0.5

def build_subject(args):
    import torch
    from nn.model import load_checkpoint

    path = args.checkpoint or os.path.join(paths.checkpoint_dir(args.run, create=False), "best.pt")
    if not os.path.exists(path):
        raise SystemExit("no checkpoint at %s -- run nn.train first" % path)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    net, meta = load_checkpoint(path, device=device)
    print("subject: %s (epoch %s, val top1 %.4f)"
          % (path, meta.get("epoch"), meta.get("metrics", {}).get("top1", float("nan"))))
    if args.lookahead:
        return ValueLookaheadPlayer(net, device, name="value1ply@%s" % os.path.basename(path))
    return PolicyPlayer(net, device, name="policy@%s" % os.path.basename(path))

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--opponent", choices=["random", "alphabeta"], default="random")
    ap.add_argument("--depth", type=int, default=2, help="alpha-beta opponent depth")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--opening-plies", type=int, nargs=2, default=[2, 8])
    ap.add_argument("--ply-cap", type=int, default=200)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--run", default="bootstrap")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lookahead", action="store_true",
                    help="one ply of value-head search instead of raw policy argmax. "
                         "Diagnostic only -- the phase-2 criterion is the raw policy")
    args = ap.parse_args()

    if args.games % 2:
        raise SystemExit("--games must be even: every opening is played with both colours")

    rs.set_search_knobs({"use_book": False})

    subject = build_subject(args)
    opponent = (RandomPlayer(seed=args.seed) if args.opponent == "random"
                else AlphaBetaPlayer(args.depth))

    print("arena: %s vs %s, %d games (%d openings x 2 colours)"
          % (subject.name, opponent.name, args.games, args.games // 2))
    result = run_match(subject, opponent, args.games, args.seed,
                       tuple(args.opening_plies), args.ply_cap, quiet=False)

    print("\n%s vs %s" % (result["subject"], result["opponent"]))
    print("  %d games: +%d =%d -%d" % (result["games"], result["wins"],
                                       result["draws"], result["losses"]))
    print("  score  %.3f +/- %.3f" % (result["score"], result["stderr"]))
    print("  mean game %.0f plies, %.0fs" % (result["mean_plies"], result["seconds"]))
    for reason, count in sorted(result["draw_reasons"].items(), key=lambda kv: -kv[1]):
        print("  draw   %3d  %s" % (count, reason))
    if result["anomalies"]:
        print("  !! %d anomalies: %s" % (len(result["anomalies"]), result["anomalies"][:3]))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
        print("  written to %s" % args.out)

if __name__ == "__main__":
    main()
