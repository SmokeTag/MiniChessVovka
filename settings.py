"""Persisted GUI preferences (window size, search depth, toggles).

Stored next to the code as `gui_settings.json`. Every read is defensive: a
corrupt, unreadable or hand-edited file must never stop the game from starting,
so anything unexpected falls back to the default for that key alone.
"""

import json
import os

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_settings.json")

DEFAULTS = {
    "win_w": None,
    "win_h": None,
    "ai_depth": 6,
    "white_ai": False,
    "black_ai": False,
    "show_hint": False,
    "hint_lines": 1,
    "hint_workers": 3,
    "show_eval": True,
    "board_flipped": None,
    "sound": False,
    "engine": "alphabeta",
    "net_seconds": 0.5,
}

# Which engine plays the AI's moves. "network" is the policy-value net searched with
# MCTS (nn/backend.py, docs/ZERO.md). Selecting it is what loads torch.
ENGINE_CHOICES = ["alphabeta", "network"]

ENGINE_LABELS = {
    "alphabeta": "alpha-beta search",
    "network": "network + MCTS",
}

# The network's budget, in seconds of search per move. It is a *time* rather than a
# simulation count because a simulation costs whatever the position makes it cost, so a
# fixed count is not a fixed wait — and a wait is what the user is spending. Depth does
# the same job for the alpha-beta search, and the two ladders are read the same way:
# every rung is a setting the user can see, and it is the one that runs.
NET_TIME_CHOICES = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

# Simulation counts measured from the opening on this machine (~8k/s at a batch of 16).
# They are a guide, not a promise -- that is the whole reason the setting is a clock: a
# middlegame with full hands branches wider and costs more per simulation. For scale,
# 2,400 is the count that drew level with depth-6 alpha-beta at slightly less time per
# move (docs/ZERO.md).
NET_TIME_LABELS = {
    0.1: "instant · ~800 sims",
    0.25: "fast · ~2k sims",
    0.5: "balanced · ~4k sims",
    1.0: "strong · ~8k sims",
    2.0: "very strong",
    5.0: "slow · for analysis",
}

def net_time_label(seconds):
    return NET_TIME_LABELS.get(seconds, "")

def format_net_time(seconds):
    return f"{seconds:g}s"

DEPTH_CHOICES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

DEPTH_LABELS = {
    2: "very fast", 3: "very fast", 4: "fast", 5: "fast",
    6: "balanced", 7: "strong", 8: "strong", 9: "slow", 10: "very slow",
}

HINT_LINE_CHOICES = [1, 2, 3, 4]

HINT_LINE_LABELS = {
    1: "best move only", 2: "2.4-3x slower", 3: "~4x slower", 4: "~5x slower",
}

HINT_WORKER_CHOICES = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48]

HINT_WORKER_MB = 30

def hint_worker_label(workers):
    return "one at a time" if workers == 1 else f"{workers} positions at once"

def hint_worker_choices():
    """The worker ladder, capped so it never exceeds what this machine has cores for.

    One core is left for the UI thread and the OS. A single-core machine still gets
    `[1]`, which is the old one-at-a-time behaviour.
    """
    cap = max(1, (os.cpu_count() or 2) - 1)
    allowed = [n for n in HINT_WORKER_CHOICES if n <= cap]
    return allowed or [HINT_WORKER_CHOICES[0]]

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
        pass

    if values["engine"] not in ENGINE_CHOICES:
        values["engine"] = DEFAULTS["engine"]
    if values["ai_depth"] not in DEPTH_CHOICES:
        values["ai_depth"] = DEFAULTS["ai_depth"]
    if values["net_seconds"] not in NET_TIME_CHOICES:
        values["net_seconds"] = DEFAULTS["net_seconds"]
    if values["hint_lines"] not in HINT_LINE_CHOICES:
        values["hint_lines"] = DEFAULTS["hint_lines"]
    workers = hint_worker_choices()
    if values["hint_workers"] not in workers:
        values["hint_workers"] = min(workers, key=lambda n: abs(n - _as_int(
            values["hint_workers"], DEFAULTS["hint_workers"])))
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

def _as_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback

def _cycle(choices, current, step, fallback):
    """Move `step` places along `choices`, clamped at both ends."""
    try:
        index = choices.index(current)
    except ValueError:
        index = choices.index(fallback) if fallback in choices else 0
    index = max(0, min(len(choices) - 1, index + step))
    return choices[index]

def cycle_depth(current, step):
    return _cycle(DEPTH_CHOICES, current, step, DEFAULTS["ai_depth"])

def cycle_hint_lines(current, step):
    return _cycle(HINT_LINE_CHOICES, current, step, DEFAULTS["hint_lines"])

def cycle_hint_workers(current, step):
    return _cycle(hint_worker_choices(), current, step, DEFAULTS["hint_workers"])

def cycle_net_seconds(current, step):
    return _cycle(NET_TIME_CHOICES, current, step, DEFAULTS["net_seconds"])
