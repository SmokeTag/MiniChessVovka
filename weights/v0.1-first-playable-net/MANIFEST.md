# v0.1-first-playable-net

The first Minihouse Zero network that was played by a human, kept because of how it
played rather than how it scored: reasonable much of the time, with recognisably
human-looking mistakes. Trained on 48k teacher positions, before the value-scale fix,
so its eval readout is badly miscalibrated (+395cp at the opening). Read the moves.

| | |
| --- | --- |
| commit | `7b6b8c0` |
| run | `vscale`, epoch 8 |
| teacher | `depth8`, 52,260 training positions |
| network | 64 channels x 6 blocks, 480,910 parameters |
| encoding | 24 planes, 2196 actions |
| value scale | 2000.0 |
| mirror augmentation | True |
| val policy top-1 | 0.4648 |
| val policy top-5 | 0.8459 |
| val value MAE | 0.2137 |
| val value sign | 0.9195 |
| arena vs random | 0.970 |
| arena vs depth-2 alpha-beta | ~0.08 |

## Restoring it

```bash
export MINIZERO_CHECKPOINT=$PWD/weights/v0.1-first-playable-net/best.pt
./play.sh          # the Engine button now plays this checkpoint
```

`nn/backend.py` otherwise picks the newest `best.pt` under
`$MINIZERO_DATA/checkpoints/*/`, so without the environment variable a later
training run takes over the GUI. That is why these are here.

Checksums in `SHA256SUMS`; the files are mode 444 to make an accidental
overwrite fail loudly.
