#!/usr/bin/env python3
"""Control layout invariants for the Pygame front end.

`docs/GUI.md` states that hit regions are produced by the code that draws each control,
"so a control can never be clickable where it is not visible". That holds for a single
control and its own rect; it says nothing about two controls claiming the *same* rect,
which is a different failure and a silent one.

It happened: the Engine toggle and Save-position-to-book were both drawn at
`button_grid(6, 0, span=2)`. Drawing is sequential, so the later one painted over the
earlier and the Engine button was never visible; hit-testing iterates `hits['buttons']`
in insertion order and returns on the first match, and the Engine button is inserted
first — so clicking the control that read "Save position to book" toggled the engine.
One control invisible but live, the other visible but unreachable, and every unit test
that merely asked "is the button registered?" passed.

This is the machine-checkable form of the invariant. It renders a real frame and asserts
that no two hit regions overlap, under a couple of window sizes and UI states so a
collision cannot hide behind one particular geometry.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be set before pygame opens a display; there is no screen on CI or in a headless
# checkout, and this test does not need one.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

pygame = pytest.importorskip("pygame")

@pytest.fixture(scope="module")
def game():
    pygame.init()
    if not pygame.display.get_init():
        pytest.skip("no usable SDL video driver")
    import main
    return main

def _groups(frame_hits):
    """Named rects per hit group.

    Grouped, not flattened: `hits['wheel']` deliberately re-registers the stepper rects
    so the mouse wheel nudges whichever stepper it is over (docs/GUI.md), and that is a
    different input channel rather than a collision. Two entries in the *same* group are
    two things competing for the same click.
    """
    out = {}
    for group, entries in frame_hits.items():
        if not isinstance(entries, dict):
            continue
        out[group] = [(name, rect) for name, rect in entries.items()
                      if isinstance(rect, pygame.Rect)]
    return out

@pytest.mark.parametrize("size", [(1280, 1000), (1100, 1562)])
@pytest.mark.parametrize("show_hint", [True, False])
def test_no_two_controls_share_a_region(game, size, show_hint):
    g = game.Game()
    g.resize(*size)
    g.ui.show_hint = show_hint
    g.relayout()
    g.ui.can_save_book = True   # so the book button is drawn and registered
    g.ui.can_undo = True
    g.render()

    groups = _groups(g.hits)
    assert groups.get('buttons'), "the frame registered no buttons at all"

    for group, regions in groups.items():
        for i, (name_a, rect_a) in enumerate(regions):
            for name_b, rect_b in regions[i + 1:]:
                overlap = rect_a.clip(rect_b)
                assert overlap.width == 0 or overlap.height == 0, (
                    "%s/%s and %s/%s overlap at %s — the one inserted first takes the "
                    "clicks, the one drawn last takes the pixels"
                    % (group, name_a, group, name_b, overlap))

def test_the_engine_toggle_is_present_and_its_own_control(game):
    g = game.Game()
    g.resize(1280, 1000)
    g.ui.can_save_book = True
    g.render()

    buttons = g.hits['buttons']
    assert 'toggle_engine' in buttons, "the engine selector is not clickable"
    assert 'save_book' in buttons
    assert buttons['toggle_engine'] != buttons['save_book'], (
        "the engine toggle and the book button are the same rect again")
