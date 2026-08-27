#!/bin/bash
# Launch the Pygame front end for Mini Crazyhouse 6x6.
#
# Search depth is chosen in the UI (the "Depth" stepper) and remembered in
# gui_settings.json, so this script no longer claims a number it does not set.

cd "$(dirname "$0")" || exit 1

if [ ! -d "venv" ]; then
    echo "ERROR: venv not found."
    echo "  python3 -m venv venv"
    echo "  ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

if ! ./venv/bin/python -c "import minichess_engine" 2>/dev/null; then
    echo "ERROR: the Rust engine (minichess_engine) is not installed in ./venv."
    echo "  source venv/bin/activate"
    echo "  cd engine_rs && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release"
    exit 1
fi

cat <<'BANNER'
==========================================
 Mini Crazyhouse 6x6
==========================================
 Mouse   drag a piece, or click it then click a target
         click a piece in your hand strip, then a square, to drop it
 Arrows  step through the game   Home/End  jump to start/live
 U       take back one half-move (Ctrl+Z also works)
 F       flip board     H  hints     + / -  engine depth
 Ctrl+N  new game       Esc  cancel a selection / return to live

 The window is resizable; its size and your settings are remembered.
BANNER

./venv/bin/python main.py
echo
echo "Game over."
