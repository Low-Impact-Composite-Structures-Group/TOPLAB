"""Helpers for suppressing console output when an analysis runs silently."""

from __future__ import annotations

import functools
import inspect
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout


@contextmanager
def silenced(enabled: bool = True):
    """Discard everything written to stdout/stderr while enabled."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        yield


def silent_when_quiet(func):
    """Silence a method when its object (or its `verbosity` argument) is 'quiet'.

    The verbosity is read from the bound `verbosity` argument if the method
    accepts one (constructors), otherwise from `self.verbosity`.
    """
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        verbosity = getattr(self, "verbosity", None)
        if "verbosity" in signature.parameters:
            bound = signature.bind_partial(self, *args, **kwargs)
            verbosity = bound.arguments.get("verbosity", signature.parameters["verbosity"].default)
        with silenced(str(verbosity).strip().lower() == "quiet"):
            return func(self, *args, **kwargs)

    return wrapper
