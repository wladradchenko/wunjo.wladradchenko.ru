# coding: utf-8
"""Redirect library logging helpers to stderr (stdout is the plugin protocol)."""

import sys

__all__ = ["rprint", "rlog"]


def rprint(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def rlog(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)
