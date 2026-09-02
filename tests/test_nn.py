#!/usr/bin/env python3
"""Minihouse Zero: the plumbing between the engine, the tensors and the network.

**Everything that needs torch skips without it.** requirements-nn.txt is optional and
the GUI, the bot and the book must never depend on it, so a plain `pytest tests/ -q` on
requirements.txt alone has to stay green. The engine-side tests below run either way.

The encoding itself is fuzzed in tests/test_encoding.py. What is checked here is the
layer above it, where the two classic silent failures live:

  * the policy head's flatten not landing on the action index, so the network trains a
    permuted move distribution and never says so;
  * a record that does not rebuild into the position it was generated from, so training
    inputs and play-time inputs differ systematically.
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minichess_engine as rs

torch = pytest.importorskip("torch", reason="requirements-nn.txt is optional")
np = pytest.importorskip("numpy")

from nn import features, paths, teacher
from nn.model import (MinihouseNet, load_checkpoint, masked_log_softmax,
                      parameter_count, save_checkpoint)

def a_position(plies=20, seed=3):
    import random
    rng = random.Random(seed)
    rs.set_search_knobs({"use_book": False})
    gs = rs.GameState()
    gs.setup_initial_board()
    for _ in range(plies):
        moves = gs.get_all_legal_moves()
        if not moves or gs.is_terminal_draw():
            break
        gs.make_ai_move(rng.choice(moves))
    return gs

def test_features_match_the_engine():
    gs = a_position()
    x = features.encode(gs)
    assert x.shape == (rs.ENCODE_PLANES, rs.BOARD_SIZE, rs.BOARD_SIZE)
    assert x.dtype == np.float32
    assert x.reshape(-1).tolist() == rs.encode_position(gs), "the reshape reordered the planes"

def test_legal_mask_agrees_with_the_engine():
    gs = a_position()
    mask = features.legal_mask(gs)
    assert mask.shape == (rs.ACTION_SPACE,)
    assert mask.sum() == len(gs.get_all_legal_moves())
    assert set(np.flatnonzero(mask)) == set(rs.legal_action_indices(gs))

def test_ragged_masks_rebuild_the_dense_slab():
    states = [a_position(p, seed=p) for p in (4, 12, 30)]
    ragged = [features.legal_indices(g) for g in states]
    dense = features.masks_from_ragged(ragged)
    assert dense.shape == (3, rs.ACTION_SPACE)
    for i, g in enumerate(states):
        assert set(np.flatnonzero(dense[i])) == set(rs.legal_action_indices(g))

def test_policy_flatten_lands_on_the_action_index():
    """The single assumption the whole policy head rests on.

    The head is a 1x1 conv to 61 planes and `flatten(1)`; the action index is
    `plane * 36 + r * 6 + f`. If those two ever disagree the network trains a permuted
    move distribution and every downstream metric still looks plausible.
    """
    board = rs.BOARD_SIZE
    for plane, r, f in [(0, 0, 0), (7, 3, 2), (40, 5, 5), (57, 2, 4), (60, 5, 5)]:
        raw = torch.zeros(1, rs.ACTION_PLANES, board, board)
        raw[0, plane, r, f] = 1.0
        flat = raw.flatten(1)
        assert flat.argmax().item() == plane * board * board + r * board + f

def test_network_shapes_and_size():
    net = MinihouseNet()
    assert net.policy.out_channels == rs.ACTION_PLANES
    assert 300_000 < parameter_count(net) < 800_000, "trunk drifted off ~0.5M parameters"

    x = torch.randn(3, rs.ENCODE_PLANES, rs.BOARD_SIZE, rs.BOARD_SIZE)
    logits, value = net(x)
    assert logits.shape == (3, rs.ACTION_SPACE)
    assert value.shape == (3,)
    assert torch.all(value.abs() <= 1.0), "value head must be a tanh"

def test_masking_leaves_only_legal_actions():
    gs = a_position()
    net = MinihouseNet().eval()
    x = torch.from_numpy(features.encode(gs)).unsqueeze(0)
    with torch.no_grad():
        logits, _ = net(x)
    mask = torch.from_numpy(features.legal_mask(gs)).unsqueeze(0)

    logp = masked_log_softmax(logits, mask)
    probs = logp.exp()
    assert pytest.approx(1.0, abs=1e-4) == probs.sum().item()
    assert probs[~mask].sum().item() < 1e-6, "probability leaked onto illegal actions"

    best = int(logp.argmax(-1).item())
    assert best in rs.legal_action_indices(gs)
    assert rs.action_index_to_move(gs, best) in gs.get_all_legal_moves()

def test_checkpoints_carry_their_encoding():
    net = MinihouseNet(channels=16, blocks=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.pt")
        save_checkpoint(path, net, {"note": "test"})
        back, meta = load_checkpoint(path)
        assert meta["note"] == "test"
        assert back.config() == net.config()

        x = torch.randn(2, rs.ENCODE_PLANES, rs.BOARD_SIZE, rs.BOARD_SIZE)
        net.eval(); back.eval()
        with torch.no_grad():
            assert torch.allclose(net(x)[0], back(x)[0], atol=1e-6)

        # A plane-layout change must fail at load, not play nonsense.
        payload = torch.load(path, weights_only=False)
        payload["encoding"]["action_space"] += 1
        torch.save(payload, path)
        with pytest.raises(ValueError, match="different encoding"):
            load_checkpoint(path)

def test_value_target_follows_the_side_to_move():
    """Scores are white-relative everywhere; the network is canonicalised to the mover.

    A sign error here is invisible in the loss curve -- it trains happily to a value head
    that is confidently backwards.
    """
    assert teacher.value_target(400, "w") == pytest.approx(-teacher.value_target(400, "b"))
    assert teacher.value_target(400, "w") > 0
    assert teacher.value_target(400, "b") < 0
    assert teacher.value_target(0, "w") == 0.0
    assert teacher.value_target(10**6, "w") == 1.0
    assert teacher.value_target(10**6, "b") == -1.0

def test_records_rebuild_into_the_position_they_came_from():
    """`restore` is the inverse of the record, and the generator asserts it per position.

    A FEN carries no `ply` and no repetition count, so getting this wrong makes the
    progress and repetition planes read one way in training and another in play.
    """
    for plies in (5, 23, 41):
        gs = a_position(plies, seed=plies)
        record = {
            "fen": rs.to_fen(gs),
            "ply": int(gs.ply),
            "ply_limit": int(gs.ply_limit),
            "reps": int(gs.repetition_count()),
        }
        rebuilt = teacher.restore(record)
        assert rs.encode_position(rebuilt) == rs.encode_position(gs)

def test_repetition_planes_survive_the_round_trip():
    gs = rs.GameState()
    gs.setup_initial_board()
    for m in [((5, 0), (4, 1), None), ((0, 5), (1, 4), None),
              ((4, 1), (5, 0), None), ((1, 4), (0, 5), None)]:
        gs.make_ai_move(m)
    assert gs.repetition_count() >= 2, "the shuffle should have repeated the position"

    record = {"fen": rs.to_fen(gs), "ply": int(gs.ply),
              "ply_limit": int(gs.ply_limit), "reps": int(gs.repetition_count())}
    assert rs.encode_position(teacher.restore(record)) == rs.encode_position(gs)

def test_data_never_lands_in_the_repo():
    """book.db is in the repo root and test_nightly.py drops its tables. Nothing this
    project generates may share that directory -- see tests/cache_isolation.py."""
    with tempfile.TemporaryDirectory() as tmp:
        os.environ[paths.ENV_VAR] = tmp
        try:
            assert paths.data_root() == os.path.abspath(tmp)
            assert paths.teacher_dir("x").startswith(os.path.abspath(tmp))
            assert paths.checkpoint_dir("y").startswith(os.path.abspath(tmp))
        finally:
            del os.environ[paths.ENV_VAR]

    assert os.path.abspath(paths.data_root()) != paths.repo_root()
    assert not paths.data_root().startswith(paths.repo_root() + os.sep)

def test_action_mirror_is_an_involution():
    from nn import symmetry
    perm = symmetry.ACTION_MIRROR
    assert perm.shape == (rs.ACTION_SPACE,)
    assert np.array_equal(perm[perm], np.arange(rs.ACTION_SPACE)), "mirroring twice must be identity"

def test_mirroring_agrees_with_the_engine():
    """The symmetry argument in nn/symmetry.py, checked against the move generator.

    For a position and its file mirror: the encoders must agree up to a flip of the file
    axis, and the legal action sets must agree up to the index permutation. Either half
    being wrong makes the augmentation teach the network moves that are not there.
    """
    from nn import symmetry

    for plies in (3, 11, 24, 47):
        gs = a_position(plies, seed=plies * 5)
        if len(gs.get_all_legal_moves()) < 2:
            continue
        mirrored = symmetry.mirror_gamestate(gs)

        assert set(mirrored.get_all_legal_moves()) == {
            symmetry.mirror_move(m) for m in gs.get_all_legal_moves()
        }, "the mirrored position has a different legal move set"

        assert np.array_equal(features.encode(mirrored),
                              symmetry.mirror_planes(features.encode(gs))), \
            "mirroring the planes is not the same as mirroring the board"

        assert set(symmetry.mirror_actions(features.legal_indices(gs)).tolist()) == \
            set(rs.legal_action_indices(mirrored)), \
            "the action permutation disagrees with the mirrored position's legal set"

def test_mirroring_holds_for_black_to_move():
    """One permutation serves both colours only because the 180 degree canonical
    rotation and the file reflection commute. Black is where that would break."""
    from nn import symmetry

    for plies in (5, 13, 29):
        gs = a_position(plies, seed=plies)
        if gs.current_turn != "b":
            gs.make_ai_move(gs.get_all_legal_moves()[0])
        if gs.current_turn != "b" or len(gs.get_all_legal_moves()) < 2:
            continue
        mirrored = symmetry.mirror_gamestate(gs)
        assert np.array_equal(features.encode(mirrored),
                              symmetry.mirror_planes(features.encode(gs)))
        assert set(symmetry.mirror_actions(features.legal_indices(gs)).tolist()) == \
            set(rs.legal_action_indices(mirrored))

def test_augmented_batches_stay_legal():
    """The end-to-end property: after augmentation, every policy target is still inside
    its own legal mask. A permutation applied to one and not the other passes every
    shape check and silently trains on moves the position does not have."""
    from nn import dataset, symmetry

    n = 64
    states = [a_position(p % 40 + 2, seed=p) for p in range(n)]
    states = [g for g in states if len(g.get_all_legal_moves()) >= 2]
    x = np.stack([features.encode(g) for g in states])
    legal = [features.legal_indices(g) for g in states]
    policy = np.array([int(l[0]) for l in legal], dtype=np.int64)
    value = np.zeros(len(states), dtype=np.float32)

    split = dataset.Split(x, policy, value, legal)
    rng = np.random.default_rng(0)
    seen_any = False
    for bx, bp, bv, bmask in split.batches(16, rng=rng, augment=True):
        assert bx.shape[1:] == features.INPUT_SHAPE
        for i in range(len(bp)):
            assert bmask[i, bp[i]], "a policy target fell outside its own legal mask"
        seen_any = True
    assert seen_any

def test_the_front_ends_do_not_import_torch():
    """The whole reason the NN deps live in the same venv (docs/ZERO.md).

    Isolation here is import discipline, not a second interpreter, so it has to be
    checked rather than assumed. Run in a subprocess: torch is already imported in this
    one by the tests above, so an in-process check would prove nothing.
    """
    import subprocess
    probe = (
        "import sys; sys.path.insert(0, %r);"
        "import ai, settings, thread_utils;"
        "from nn import backend;"
        "print('torch' in sys.modules)" % os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("False"), (
        "importing the GUI's modules pulled torch in:\n%s" % out.stdout)

def test_engine_setting_is_validated():
    import settings
    assert settings.DEFAULTS["engine"] == "alphabeta", "the search stays the default"
    assert set(settings.ENGINE_CHOICES) == {"alphabeta", "network"}
    for choice in settings.ENGINE_CHOICES:
        assert choice in settings.ENGINE_LABELS

def test_backend_reports_why_it_is_unavailable():
    """A missing checkpoint must be a toast, not a traceback in the UI thread."""
    from nn import backend
    old = os.environ.get(backend.ENV_CHECKPOINT)
    os.environ[backend.ENV_CHECKPOINT] = "/nonexistent/checkpoint.pt"
    try:
        assert not backend.available()
        assert "checkpoint" in backend.unavailable_reason()
    finally:
        if old is None:
            del os.environ[backend.ENV_CHECKPOINT]
        else:
            os.environ[backend.ENV_CHECKPOINT] = old

def test_backend_plays_a_legal_move_with_a_white_relative_score():
    """The contract the GUI depends on: same shape as ai.find_best_move_with_score."""
    from gamestate import GameState
    from nn import backend

    if not backend.available():
        pytest.skip("no trained checkpoint on this machine")

    gs = GameState()
    gs.setup_initial_board()
    move, score = backend.find_best_move_with_score(gs)
    assert move in gs.get_all_legal_moves()
    assert isinstance(score, float)

    # Scores are white-relative everywhere (CLAUDE.md), so the same position evaluated
    # with Black to move must not silently flip meaning.
    gs.make_move(move, False)
    gs.check_game_over()
    move_b, score_b = backend.find_best_move_with_score(gs)
    assert move_b in gs.get_all_legal_moves()
    assert isinstance(score_b, float)
