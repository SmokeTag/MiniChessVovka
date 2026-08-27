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
    "white_ai": False,
    "black_ai": True,       # human plays White by default
    "show_hint": False,
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


def cycle_depth(current, step):
    """Move `step` places along DEPTH_CHOICES, clamped at both ends."""
    try:
        index = DEPTH_CHOICES.index(current)
    except ValueError:
        index = DEPTH_CHOICES.index(DEFAULTS["ai_depth"])
    index = max(0, min(len(DEPTH_CHOICES) - 1, index + step))
    return DEPTH_CHOICES[index]
