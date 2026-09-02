"""Where Minihouse Zero keeps the things it generates.

**Never the repo root.** `tests/cache_isolation.py` exists because a stray file there
collides with `book.db`, the live opening book, and CLAUDE.md makes the rule general:
anything that persists -- teacher shards, checkpoints, replay buffers, self-play games --
lives under a configurable path outside the repository.

    MINIZERO_DATA=/mnt/big/zero ./venv/bin/python -m nn.teacher generate ...

Default is $XDG_DATA_HOME/minihouse-zero, i.e. ~/.local/share/minihouse-zero.
"""
import os

ENV_VAR = "MINIZERO_DATA"

def data_root():
    override = os.environ.get(ENV_VAR)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(xdg, "minihouse-zero")

def subdir(*parts, create=True):
    path = os.path.join(data_root(), *parts)
    if create:
        os.makedirs(path, exist_ok=True)
    return path

def teacher_dir(name="depth8", create=True):
    return subdir("teacher", name, create=create)

def checkpoint_dir(run="bootstrap", create=True):
    return subdir("checkpoints", run, create=create)

def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
