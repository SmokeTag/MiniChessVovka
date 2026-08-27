#!/usr/bin/env python3
"""Keep the test suite off the live `move_cache.db`.

`DB_PATH` in `engine_rs/src/cache.rs` is the hardcoded relative string
"move_cache.db", so every Rust cache call resolves it against the process CWD and
there is no way to override the path. A test that creates, drops or writes the
table therefore hits the repo-root DB -- the live self-play cache -- unless it
first moves the CWD somewhere disposable. `test_nightly.py` used to drop and
recreate the table in place, so a full pytest run silently destroyed training
data; `test_e2e.py` used to file its scratch searches into it.

chdir is the only seam available. Use it as a context manager for a single test,
or subclass `IsolatedCacheDB` for a whole TestCase.
"""

import contextlib
import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import ai


@contextlib.contextmanager
def isolated_cache_db():
    """Run the block in a throwaway CWD, so it gets its own move_cache.db."""
    prev_cwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp(prefix="minichess-cache-test-")
    try:
        os.chdir(tmp_dir)
        # Fail loudly rather than fall back to touching the real cache if the
        # chdir ever stops taking effect.
        if os.path.realpath(os.getcwd()) == os.path.realpath(REPO_ROOT):
            raise RuntimeError(
                "cache tests must not run in the repo root -- they drop the move_cache table"
            )
        yield tmp_dir
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # The Rust move cache is process-global: the block leaves it holding only
        # its own scratch rows. Reload the real cache so a later test in the same
        # pytest session does not write that scratch state back over the
        # repo-root DB.
        ai.load_move_cache_from_db()


class IsolatedCacheDB(unittest.TestCase):
    """TestCase base that wraps every test in `isolated_cache_db()`."""

    def setUp(self):
        self._cache_ctx = isolated_cache_db()
        self._cache_ctx.__enter__()

    def tearDown(self):
        self._cache_ctx.__exit__(None, None, None)
