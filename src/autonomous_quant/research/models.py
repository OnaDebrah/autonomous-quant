"""The three model calls of Chapter 18 (match, tweak, write), with a rule-based default.

The loop, not the model, is the source of ideas: it calls a model only to *match* readings to a catalogue rule, to
*tweak* a small part of an existing rule, or to *write* a rule that is not in the catalogue. Every output is typed
and validated by code; an invalid answer is dropped, never repaired. ``RuleBased`` answers ``match`` with a fixed
rule and declines the other two, so the loop runs with no model at all. To use a language model, implement the same
three methods and keep the validation: the slot stays the same, and so do the checks around it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from ..interfaces import Reading

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleBased:
    """Trend in calm markets, mean reversion in stormy ones. A rule, stated, versioned and testable."""

    version: str = "1"

    def match(self, readings: Sequence[Reading], catalogue: Sequence[str]) -> str | None:
        state = next((r for r in readings if r.instrument == "volatility_state"), None)
        stormy = bool(state and state.evidence.get("market_stormy"))
        return "mean_reversion" if stormy else "trend"

    def tweak(self, source: str, change: str) -> str | None:
        return None  # declined: a rule-based caller does not edit code

    def write(self, specification: str) -> str | None:
        return None  # declined: full code is written only by an opted-in model, then reviewed by a person (Chapter 19)


def validated_match(model, readings: Sequence[Reading], catalogue: Sequence[str]) -> str | None:
    """Call ``match`` and keep the answer only if it names a catalogue rule. A malformed answer is dropped."""
    try:
        answer = model.match(readings, catalogue)
    except Exception:  # a failed call is no answer; the loop continues without it
        log.exception("the model's match call failed; no hypothesis from this reading")
        return None
    if answer is None:
        return None
    if not isinstance(answer, str) or answer not in catalogue:
        log.warning("dropped the model's match %r: not a catalogue rule", answer)
        return None
    return answer
