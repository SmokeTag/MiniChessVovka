# v0.2-bootstrap-150k

The end of phase 2: the full 150k teacher set (139,301 unique), mirror augmentation,
the corrected value scale, and a schedule sized to the overfit knee. Beats random
0.985 without ever losing; still short of depth-2 alpha-beta, which docs/ZERO.md
explains is a property of searchless policy play in this variant rather than a bug.

| | |
| --- | --- |
| commit | `73c622c` |
| run | `bootstrap`, epoch 14 |
| teacher | `depth8`, 132,336 training positions |
| network | 64 channels x 6 blocks, 480,910 parameters |
| encoding | 24 planes, 2196 actions |
| value scale | 2000.0 |
| mirror augmentation | True |
| val policy top-1 | 0.5081 |
| val policy top-5 | 0.8824 |
| val value MAE | 0.1702 |
| val value sign | 0.9383 |
| arena vs random | 0.985 (+194 =6 -0) |
| arena vs depth-2 alpha-beta | 0.185 (+34 =6 -160) |

## Restoring it

```bash
export MINIZERO_CHECKPOINT=$PWD/weights/v0.2-bootstrap-150k/best.pt
./play.sh          # the Engine button now plays this checkpoint
```

`nn/backend.py` otherwise picks the newest `best.pt` under
`$MINIZERO_DATA/checkpoints/*/`, so without the environment variable a later
training run takes over the GUI. That is why these are here.

Checksums in `SHA256SUMS`; the files are mode 444 to make an accidental
overwrite fail loudly.
