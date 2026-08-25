#!/bin/bash
# Launch script for self-play training mode

echo "=================================="
echo "Starting AI self-play training mode"
echo "=================================="
echo ""
echo "Default parameters:"
echo "  - Depth: 6"
echo "  - Exploration: 20%"
echo "  - Number of games: unlimited"
echo ""
echo "Press Ctrl+C to stop"
echo "Progress will be saved to the database"
echo ""
echo "=================================="
echo ""

# Activate the virtualenv if there is one
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start self-play training mode
python3 src/self_play.py
