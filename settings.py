# -*- coding: utf-8 -*-
"""Persisted GUI preferences (window size, search depth, toggles).

Stored next to the code as `gui_settings.json`. Every read is defensive: a
corrupt, unreadable or hand-edited file must never stop the game from starting,
so anything unexpected falls back to the default for that key alone.
"""

import json
import os

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_settings.json")

DEFAULTS = {
    "win_w": None,          # None = auto-fit to the desktop on first run
    "win_h": None,
    "ai_depth": 6,
    "white_ai": False,      # both sides start human-driven; the AI buttons opt in
    "black_ai": False,
    "show_hint": False,
    "hint_lines": 1,
    "show_eval": True,
    "board_flipped": None,  # None = orient to whichever side the human plays
    "sound": False,
}

# Depth ladder exposed by the ‹ › control. Depth 10 is the training default and
# costs ~3s median / ~30s worst per move (see CLAUDE.md) — playable, but it is
# deliberately not the interactive default.
DEPTH_CHOICES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

DEPTH_LABELS = {
    2: "very fast", 3: "very fast", 4: "fast", 5: "fast",
    6: "balanced", 7: "strong", 8: "strong", 9: "slow", 10: "very slow",
}

# How many ranked lines the hint asks for. Every value above 1 turns the hint into a
# MultiPV search, which is the only way ranks 2+ carry real scores rather than
# alpha-beta bounds -- and it is not free: measured at depth 9, 2 lines costs 2.4-3.0x
# a single-PV search and 3 lines 3.8-4.4x (see CLAUDE.md). The ladder stops at 4 because
# the hint now runs at the full selected depth, so the multiplier is paid in full, and a
# 6x6 root rarely has more than a handful of moves worth ranking anyway.
HINT_LINE_CHOICES = [1, 2, 3, 4]

HINT_LINE_LABELS = {
    1: "best move only", 2: "2.4-3x slower", 3: "~4x slower", 4: "~5x slower",
}


def load():
    """Return the saved settings merged over DEFAULTS."""
    values = dict(DEFAULTS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        if isinstance(stored, dict):
            for key in DEFAULTS:
                if key in stored:
                    values[key] = stored[key]
    except (OSError, ValueError):
        pass  # missing or corrupt: defaults are already in `values`

    # Validate the fields that would break the UI if wrong.
    if values["ai_depth"] not in DEPTH_CHOICES:
        values["ai_depth"] = DEFAULTS["ai_depth"]
    if values["hint_lines"] not in HINT_LINE_CHOICES:
        values["hint_lines"] = DEFAULTS["hint_lines"]
    for key in ("win_w", "win_h"):
        if values[key] is not None and not isinstance(values[key], int):
            values[key] = None
    return values


def save(values):
    """Write settings. Failure is non-fatal — preferences are a convenience."""
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as handle:
            json.dump({key: values.get(key, DEFAULTS[key]) for key in DEFAULTS},
                      handle, indent=2)
    except OSError as exc:
        print(f"[settings] could not save preferences: {exc}")


def _cycle(choices, current, step, fallback):
    """Move `step` places along `choices`, clamped at both ends."""
    try:
        index = choices.index(current)
    except ValueError:
        index = choices.index(fallback)
    index = max(0, min(len(choices) - 1, index + step))
    return choices[index]


def cycle_depth(current, step):
    return _cycle(DEPTH_CHOICES, current, step, DEFAULTS["ai_depth"])


def cycle_hint_lines(current, step):
    return _cycle(HINT_LINE_CHOICES, current, step, DEFAULTS["hint_lines"])
