#!/usr/bin/env bash
# One overnight tabula rasa run: random weights -> self-play -> train -> repeat.
# No teacher data and no evaluation gate (see nn/zero.py). Resumable: re-running
# continues from the last completed iteration rather than restarting the night.
set -uo pipefail
cd "$(dirname "$0")"

RUN="${RUN:-zero1}"
HOURS="${HOURS:-6}"
ITERS="${ITERS:-24}"
GAMES="${GAMES:-2000}"
WORKERS="${WORKERS:-14}"
SIMS="${SIMS:-400}"

LOGDIR="$HOME/.local/share/minihouse-zero/zero/$RUN"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/run.log"

echo "=== zero run '$RUN' started $(date -Is) ===" | tee -a "$LOG"
echo "    ${ITERS} iters x ${GAMES} games @ ${SIMS} sims, ${WORKERS} workers, ${HOURS}h deadline" | tee -a "$LOG"

./venv/bin/python -u -m nn.zero \
  --run "$RUN" \
  --iterations "$ITERS" \
  --games "$GAMES" \
  --workers "$WORKERS" \
  --sims "$SIMS" \
  --batch 128 \
  --window 4 \
  --max-positions 400000 \
  --epochs 6 \
  --train-batch 256 \
  --lr 2e-3 \
  --eval-every 3 \
  --eval-games 40 \
  --eval-depth 2 \
  --eval-sims 400 \
  --deadline-hours "$HOURS" 2>&1 | tee -a "$LOG"

echo "=== zero run '$RUN' finished $(date -Is) rc=$? ===" | tee -a "$LOG"
