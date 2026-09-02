"""The policy-value network: a small ResNet over the 24-plane board.

Shape is set by the variant, not by taste. 36 squares and a 2,196-action space are tiny
next to full chess, so ~0.5M parameters is already generous -- 6 residual blocks of 64
filters. The GPU at this spatial size is latency-bound rather than compute-bound, which
is why phase 3 batches across concurrent games instead of growing the trunk.

The policy head is a 1x1 convolution to 61 planes and nothing else. `flatten(1)` on its
(N, 61, 6, 6) output lands on `plane * 36 + r * 6 + f`, which *is* the action index --
so the head has no idea it is producing a move distribution and cannot get the mapping
wrong. The alternative, a dense layer over from x to x promotion, is 5,328 outputs and
~12M parameters, larger than the whole trunk. See docs/ENCODING.md.

Value is a single tanh, in the frame of the side to move: +1 is "the player to move
wins". The input is canonicalised the same way, so the network never has to represent
"which colour am I".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import minichess_engine as rs

INPUT_PLANES = rs.ENCODE_PLANES
ACTION_PLANES = rs.ACTION_PLANES
ACTION_SPACE = rs.ACTION_SPACE
BOARD = rs.BOARD_SIZE

DEFAULT_CHANNELS = 64
DEFAULT_BLOCKS = 6
VALUE_HEAD_CHANNELS = 8
VALUE_HIDDEN = 64

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)

class MinihouseNet(nn.Module):
    def __init__(self, channels=DEFAULT_CHANNELS, blocks=DEFAULT_BLOCKS):
        super().__init__()
        self.channels = channels
        self.blocks = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy = nn.Conv2d(channels, ACTION_PLANES, 1)
        self.value = nn.Sequential(
            nn.Conv2d(channels, VALUE_HEAD_CHANNELS, 1, bias=False),
            nn.BatchNorm2d(VALUE_HEAD_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(VALUE_HEAD_CHANNELS * BOARD * BOARD, VALUE_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(VALUE_HIDDEN, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """(N, 24, 6, 6) -> (policy logits (N, 2196), value (N,)).

        Logits are unmasked: masking is the caller's, because the mask is a property of
        the position and training and search apply it at different moments.
        """
        h = self.trunk(self.stem(x))
        return self.policy(h).flatten(1), self.value(h).squeeze(-1)

    def config(self):
        return {"channels": self.channels, "blocks": self.blocks}

def parameter_count(net):
    return sum(p.numel() for p in net.parameters())

def masked_log_softmax(logits, mask):
    """log-softmax over legal actions only.

    `-inf` on an all-illegal row would produce NaN, so the fill is a large finite
    number. No position reaching this has zero legal moves -- the generator drops
    terminals and the arena stops before them -- but a NaN here poisons every gradient
    in the batch, which is not a failure worth risking to save a constant.
    """
    return F.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)

CHECKPOINT_FORMAT = 1

def save_checkpoint(path, net, meta=None):
    """Weights plus the encoding they were trained against.

    A change to the plane layout or the action map silently invalidates every trained
    checkpoint -- the weights still load, the network still runs, and it plays
    nonsense. Stamping the shapes in makes that a load-time error instead.
    """
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": net.config(),
        "state_dict": net.state_dict(),
        "encoding": {
            "planes": INPUT_PLANES,
            "action_planes": ACTION_PLANES,
            "action_space": ACTION_SPACE,
            "board": BOARD,
        },
        "meta": meta or {},
    }
    torch.save(payload, path)

def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint %s is format %r, expected %d"
                         % (path, payload.get("format"), CHECKPOINT_FORMAT))
    enc = payload["encoding"]
    live = {"planes": INPUT_PLANES, "action_planes": ACTION_PLANES,
            "action_space": ACTION_SPACE, "board": BOARD}
    if enc != live:
        raise ValueError(
            "checkpoint %s was trained against a different encoding: %r, engine has %r. "
            "The weights are not transferable; regenerate or retrain." % (path, enc, live))
    net = MinihouseNet(**payload["config"])
    net.load_state_dict(payload["state_dict"])
    net.to(device)
    return net, payload.get("meta", {})
