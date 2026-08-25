#!/bin/bash
# Launch script for Mini Chess with the strong AI

echo "=========================================="
echo "Mini Chess 6x6 Crazyhouse - Strong AI"
echo "=========================================="
echo ""
echo "AI settings:"
echo "- Search depth: 6 plies (fast game)"
echo "- Search: single-threaded (parallel path disabled)"
echo "- Quiescence depth: 4"
echo "- Move cache: enabled (persisted to the DB)"
echo ""
echo "Starting the game..."
echo ""

# Change to the script's directory
cd "$(dirname "$0")"

# Check that venv exists
if [ ! -d "venv" ]; then
    echo "ERROR: venv not found!"
    echo "Create it with: python3 -m venv venv"
    echo "Then install the dependencies: ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Start the game
./venv/bin/python main.py

echo ""
echo "Game over."
