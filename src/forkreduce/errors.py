"""Errors raised for caller mistakes.

Model, transport, and budget failures are returned as :class:`forkreduce.failures.Failure`
on a result object. The exceptions in this module mean the caller asked for something the
session cannot do, and they are raised before a model call when that is possible.
"""

from __future__ import annotations


class ForkReduceError(Exception):
    """Base class for forkreduce caller errors.

    ``code`` is an optional machine-readable label. A provider context-length
    error is a :class:`forkreduce.failures.Failure` with code ``context_overflow``,
    not an exception from this class.
    """

    code: str

    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message)
        self.code = code


class UnregisteredOutputError(ForkReduceError):
    """A query or reduce asked for an output type that was not registered at build time.

    Registering it later would change the tool set and invalidate the shared prefix cache,
    so the call fails immediately instead of silently sending a different schema.
    """


class PrefixMismatchWarning(UserWarning):
    """The supplied message history does not start with this session's frozen prefix.

    The call still runs. Provider prefix caches will miss, because the shared head of the
    prompt is no longer identical to the other branches.
    """
