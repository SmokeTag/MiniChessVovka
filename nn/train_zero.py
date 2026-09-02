"""Fit the network to its own self-play, over a replay buffer of recent iterations.

Separate from `nn.train` on purpose. That module fits the *alpha-beta teacher*: a hard
class label per position, and a headline metric -- held-out policy top-1 -- that only
means something because a stronger oracle produced the label. Here there is no oracle.
The policy target is a distribution (MCTS visit counts), the value target is the game's
own result, and "top-1 against the label" would just measure agreement with a search that
used this very network. Sharing one module would mean a flag on every line that differs;
the losses, the metrics and the data window are all different, so they are two modules.

The metrics that do mean something without an oracle:

    policy entropy   collapse detector. A zero run that stops exploring drives this to
                     ~0 and then trains on its own certainty forever.
    value MAE        against the game result, so ~1.0 at random init (predicting 0 for a
                     decisive game) falling as the value head learns who is winning.
    draw rate        tracked by the loop, not here, but it is the other collapse signal.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from nn import features, paths, selfplay
from nn.model import MinihouseNet, masked_log_softmax, save_checkpoint, load_checkpoint

VALUE_WEIGHT = 1.0


def encode_records(records, progress_every=50_000):
    """Self-play records -> (x, ragged policy indices, ragged policy probs, z, ragged legal).

    Like `nn.dataset.encode_records`, the record stores the position and not its encoding,
    so this is where the encoder is applied and a plane-layout change costs only a
    re-encode. Unlike it, the policy target is ragged in *two* arrays -- the visited
    action indices and their probabilities -- because a dense 2,196-wide target per
    position would be 8.8kB against a mean support of ~22.
    """
    from nn import teacher

    n = len(records)
    x = np.empty((n,) + features.INPUT_SHAPE, dtype=np.float32)
    value = np.empty(n, dtype=np.float32)
    pol_idx, pol_p, legal = [], [], []

    started = time.time()
    for i, rec in enumerate(records):
        gs = teacher.restore(rec)
        x[i] = features.encode(gs)
        value[i] = rec["z"]
        v = rec["visits"]
        pol_idx.append(np.asarray([a for a, _ in v], dtype=np.int32))
        pol_p.append(np.asarray([p for _, p in v], dtype=np.float32))
        legal.append(np.asarray(features.legal_indices(gs), dtype=np.int32))
        if progress_every and (i + 1) % progress_every == 0:
            print("  encoded %d/%d (%.0fs)" % (i + 1, n, time.time() - started), flush=True)
    return x, pol_idx, pol_p, value, legal


def _batches(n, batch, rng, shuffle=True):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, batch):
        yield order[s:s + batch]


def _make_targets(sel, pol_idx, pol_p, legal, action_space, device):
    """Dense soft policy target and legal mask for one batch, scattered on the device.

    The dense form is 2.8MB a batch (256 x 2,196, float target plus bool mask) against
    ~90kB of ragged data, so it is built *on the GPU* from the small arrays rather than
    built on the host and copied over. Measured: 12.0ms a batch host-side, 1-2ms this
    way, against a 5ms training step -- host-side scatter was two thirds of training.

    The targets stay ragged in memory for the same reason: a stored dense target would
    be 8.8kB a position against a mean support of ~22 actions.
    """
    b = len(sel)
    target = torch.zeros((b, action_space), dtype=torch.float32, device=device)
    mask = torch.zeros((b, action_space), dtype=torch.bool, device=device)

    plens = np.fromiter((len(pol_idx[i]) for i in sel), dtype=np.int64, count=b)
    if plens.sum():
        rows = torch.from_numpy(np.repeat(np.arange(b, dtype=np.int64), plens)).to(device)
        cols = torch.from_numpy(np.concatenate([pol_idx[i] for i in sel]).astype(np.int64)).to(device)
        vals = torch.from_numpy(np.concatenate([pol_p[i] for i in sel]).astype(np.float32)).to(device)
        target[rows, cols] = vals

    llens = np.fromiter((len(legal[i]) for i in sel), dtype=np.int64, count=b)
    if llens.sum():
        lrows = torch.from_numpy(np.repeat(np.arange(b, dtype=np.int64), llens)).to(device)
        lcols = torch.from_numpy(np.concatenate([legal[i] for i in sel]).astype(np.int64)).to(device)
        mask[lrows, lcols] = True

    return target, mask


def run(records, out_path, epochs, batch, lr, weight_decay, value_weight,
        device, seed, channels, blocks, init_from=None, log=print):
    import minichess_engine as rs
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    x, pol_idx, pol_p, value, legal = encode_records(records)
    n = len(records)
    log("  training on %d positions" % n)

    if init_from:
        net, _ = load_checkpoint(init_from, device=device)
    else:
        net = MinihouseNet(channels=channels, blocks=blocks).to(device)
    net.train()

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    # A gather out of host memory costs 3.5ms a batch against a 5ms step, so the
    # inputs live on the device when they fit. 400k positions is 1.38GB of the 12GB
    # card and training never overlaps self-play, but the guard keeps a larger replay
    # window from turning into an OOM at 3am rather than a slower epoch.
    xt = torch.from_numpy(x)
    vt = torch.from_numpy(value)
    resident = False
    if device.startswith("cuda") and torch.cuda.is_available():
        need = x.nbytes + value.nbytes
        free, _total = torch.cuda.mem_get_info()
        if need < 0.5 * free:
            xt = xt.to(device)
            vt = vt.to(device)
            resident = True
    log("  inputs %s (%.2f GB)" % ("on device" if resident else "on host", x.nbytes / 1e9))

    history = []
    for epoch in range(1, epochs + 1):
        tot = {"p": 0.0, "v": 0.0, "ent": 0.0, "mae": 0.0, "n": 0}
        for sel in _batches(n, batch, rng):
            idx = torch.from_numpy(sel).to(xt.device)
            xb = xt[idx] if resident else xt[idx].to(device, non_blocking=True)
            vb = vt[idx] if resident else vt[idx].to(device, non_blocking=True)
            target, mask = _make_targets(sel, pol_idx, pol_p, legal, rs.ACTION_SPACE, device)

            logits, pred = net(xb)
            logp = masked_log_softmax(logits, mask)
            # Soft cross-entropy: the visit distribution is the label, so this is a
            # sum over the support rather than a lookup at one index.
            policy_loss = -(target * logp).sum(-1).mean()
            pred = pred.float().reshape(-1)
            value_loss = F.mse_loss(pred, vb)
            loss = policy_loss + value_weight * value_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                p = logp.exp()
                ent = -(p * logp.clamp_min(-30)).sum(-1).mean()
            k = len(sel)
            tot["p"] += policy_loss.item() * k
            tot["v"] += value_loss.item() * k
            tot["ent"] += ent.item() * k
            tot["mae"] += (pred - vb).abs().sum().item()
            tot["n"] += k

        m = tot["n"]
        row = {"epoch": epoch, "policy_loss": tot["p"] / m, "value_loss": tot["v"] / m,
               "entropy": tot["ent"] / m, "value_mae": tot["mae"] / m}
        history.append(row)
        log("  epoch %2d  policy %.4f  value %.4f  entropy %.3f  value_mae %.3f"
            % (epoch, row["policy_loss"], row["value_loss"], row["entropy"], row["value_mae"]))

    save_checkpoint(out_path, net, meta={
        "trained_on": "selfplay",
        "positions": n,
        # Self-play values are game results, already in [-1, 1]: there is no centipawn
        # scale to record, and anything reading `value_scale` must treat 1.0 as "the
        # value head already speaks in outcomes".
        "value_scale": 1.0,
        "epochs": epochs,
        "history": history,
    })
    return net, history


def main():
    ap = argparse.ArgumentParser(description="Fit the network to its own self-play.")
    ap.add_argument("--data", nargs="+", required=True, help="self-play run names (replay window)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=VALUE_WEIGHT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--init-from", default=None)
    args = ap.parse_args()

    records = []
    for name in args.data:
        records.extend(selfplay.load_records(name))
    print("replay window: %d records from %s" % (len(records), ", ".join(args.data)))
    run(records, args.out, args.epochs, args.batch, args.lr, args.weight_decay,
        args.value_weight, args.device, args.seed, args.channels, args.blocks, args.init_from)


if __name__ == "__main__":
    main()
