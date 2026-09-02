#!/usr/bin/env python3
"""Fit the policy-value network to the alpha-beta teacher.

    ./venv/bin/python -m nn.train --epochs 40
    ./venv/bin/python -m nn.train --limit 20000 --epochs 5 --run smoke

Two losses, both standard: masked cross-entropy against the depth-8 best move, and MSE
against tanh(score / 400) in the mover's own frame. L2 comes from AdamW's weight decay.

**The headline metric is policy top-1 on held-out positions** -- how often the raw
network, with no search at all, names the same move a depth-8 search does. That is the
quantity the phase-2 exit criterion is really about; `nn.arena` then converts it into
games. Value MAE is watched alongside it because a policy that is right and a value that
is uncalibrated makes a bad MCTS, and the value head is the half a bootstrap run is
most likely to get quietly wrong (see the draw-rate risk in docs/ZERO.md).

Checkpoints go under $MINIZERO_DATA/checkpoints/<run>/, never the repo root, and carry
the encoding they were trained against so a plane-layout change fails at load rather
than playing nonsense.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import dataset, paths, teacher
from nn.model import MinihouseNet, masked_log_softmax, parameter_count, save_checkpoint

def device_of(arg):
    if arg != "auto":
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_torch(batch, device):
    x, policy, value, mask = batch
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(policy).to(device),
            torch.from_numpy(value).to(device),
            torch.from_numpy(mask).to(device))

def lr_at(step, total, peak, warmup, floor_ratio=0.01):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return peak * (floor_ratio + (1.0 - floor_ratio) * cosine)

@torch.no_grad()
def evaluate(net, split, batch_size, device, amp_dtype):
    net.eval()
    totals = {"n": 0, "policy_loss": 0.0, "value_loss": 0.0,
              "top1": 0, "top5": 0, "value_abs": 0.0, "value_sign": 0}

    for batch in split.batches(batch_size, shuffle=False):
        x, target, value, mask = to_torch(batch, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, pred = net(x)
        logits, pred = logits.float(), pred.float()
        logp = masked_log_softmax(logits, mask)

        n = x.shape[0]
        totals["n"] += n
        totals["policy_loss"] += F.nll_loss(logp, target, reduction="sum").item()
        totals["value_loss"] += F.mse_loss(pred, value, reduction="sum").item()
        totals["top1"] += (logp.argmax(-1) == target).sum().item()
        totals["top5"] += (logp.topk(5, dim=-1).indices == target[:, None]).any(-1).sum().item()
        totals["value_abs"] += (pred - value).abs().sum().item()
        # Sign agreement is the coarse question a search actually asks of a value head:
        # does it know who is better? Positions scored near zero are excluded because
        # their sign is noise in the teacher too.
        decided = value.abs() > 0.1
        totals["value_sign"] += ((pred.sign() == value.sign()) & decided).sum().item()
        totals.setdefault("decided", 0)
        totals["decided"] += decided.sum().item()

    n = totals["n"]
    return {
        "policy_loss": totals["policy_loss"] / n,
        "value_loss": totals["value_loss"] / n,
        "top1": totals["top1"] / n,
        "top5": totals["top5"] / n,
        "value_mae": totals["value_abs"] / n,
        "value_sign": totals["value_sign"] / max(1, totals["decided"]),
    }

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = device_of(args.device)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and not args.no_amp) else None
    print("device %s, amp %s" % (device, amp_dtype))

    train_split, val_split = dataset.load(args.data, limit=args.limit)
    dataset.describe(train_split, "train")
    dataset.describe(val_split, "val")

    net = MinihouseNet(channels=args.channels, blocks=args.blocks).to(device)
    print("network: %d channels x %d blocks, %d parameters"
          % (args.channels, args.blocks, parameter_count(net)))

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, -(-len(train_split) // args.batch))
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup, total_steps // 10)

    out_dir = paths.checkpoint_dir(args.run)
    history = []
    best = {"top1": -1.0}
    step = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        net.train()
        running = {"policy": 0.0, "value": 0.0, "n": 0}

        for batch in train_split.batches(args.batch, rng=rng, augment=not args.no_mirror):
            lr = lr_at(step, total_steps, args.lr, max(1, warmup))
            for group in opt.param_groups:
                group["lr"] = lr

            x, target, value, mask = to_torch(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits, pred = net(x)
            logp = masked_log_softmax(logits.float(), mask)
            policy_loss = F.nll_loss(logp, target)
            value_loss = F.mse_loss(pred.float(), value)
            loss = policy_loss + args.value_weight * value_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()

            running["policy"] += policy_loss.item() * x.shape[0]
            running["value"] += value_loss.item() * x.shape[0]
            running["n"] += x.shape[0]
            step += 1

        metrics = evaluate(net, val_split, args.batch, device, amp_dtype)
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_policy_loss": running["policy"] / running["n"],
            "train_value_loss": running["value"] / running["n"],
            "seconds": time.time() - started,
            **{"val_" + k: v for k, v in metrics.items()},
        }
        history.append(row)
        print("epoch %3d  lr %.2e  train p %.4f v %.4f | val p %.4f v %.4f  "
              "top1 %.3f  top5 %.3f  vMAE %.3f  vSign %.3f  (%.0fs)"
              % (epoch, lr, row["train_policy_loss"], row["train_value_loss"],
                 metrics["policy_loss"], metrics["value_loss"], metrics["top1"],
                 metrics["top5"], metrics["value_mae"], metrics["value_sign"],
                 row["seconds"]), flush=True)

        meta = {"run": args.run, "epoch": epoch, "data": args.data,
                "positions": len(train_split), "metrics": metrics, "args": vars(args),
                # The value head's output is only interpretable against the scale its
                # targets were built with; phase 3 reads values back out of it.
                "value_scale": teacher.VALUE_SCALE,
                "mirror": not args.no_mirror}
        save_checkpoint(os.path.join(out_dir, "last.pt"), net, meta)
        if metrics["top1"] > best["top1"]:
            best = dict(metrics, epoch=epoch)
            save_checkpoint(os.path.join(out_dir, "best.pt"), net, meta)

    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump({"history": history, "best": best, "args": vars(args)}, fh, indent=1)

    print("\nbest val top1 %.4f at epoch %d (value MAE %.3f)"
          % (best["top1"], best["epoch"], best["value_mae"]))
    print("checkpoints in %s" % out_dir)

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="depth8", help="teacher set name")
    ap.add_argument("--run", default="bootstrap", help="checkpoint subdirectory")
    ap.add_argument("--limit", type=int, default=None)
    # 20, not 40: at 139k positions val top-1 peaks around epoch 13-14 and the network
    # overfits after it (val policy loss climbs from 1.57 to 3.68 by epoch 60 while train
    # loss keeps falling). Sizing the cosine schedule to the knee anneals the learning
    # rate *into* the peak instead of past it, and beats early-stopping a longer run on
    # both heads at once: 0.508 top-1 / 0.170 value MAE against 0.497 / 0.199. Re-measure
    # this if the teacher set grows -- the knee moves with the data.
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-mirror", action="store_true",
                    help="disable file-mirror augmentation (an exact symmetry here)")
    train(ap.parse_args())

if __name__ == "__main__":
    main()
