"""Errors raised by the seed code. Every error a caller can act on is one of these, with a message that says what to do.

The hierarchy separates what is wrong:

* ``ConfigurationError``: a parameter or setting the caller supplied is invalid. Fix the call.
* ``DataError``: the data are missing, malformed or inconsistent (including ``UnknownAvailability``). Fix or extend
  the data; do not paper over it, because a quantity computed from bad data is still reported as a number.
* ``SourceError``: a data source could not deliver (network, provider). Usually transient; retrying may help.
* ``IntegrityError``: something that must never change was asked to change (an append-only record, a frozen forward
  binding, a completed trial given a different result). This signals a bug or a non-deterministic step.
"""

from __future__ import annotations


class AutonomousQuantError(Exception):
    """Base class for the errors this package raises on purpose."""


class ConfigurationError(AutonomousQuantError, ValueError):
    """An invalid parameter or setting."""


class DataError(AutonomousQuantError, ValueError):
    """Missing, malformed or inconsistent data."""


class UnknownAvailability(DataError):
    """An as-of read would need records whose availability time is unknown (Chapter 3)."""


class SourceError(AutonomousQuantError, RuntimeError):
    """A data source could not deliver."""


class IntegrityError(AutonomousQuantError, RuntimeError):
    """An attempt to change what must not change."""
