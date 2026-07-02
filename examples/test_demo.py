"""Pytest gate for the contract analyst demo.

Run: pytest examples/test_demo.py -v

The ctx fixture is auto-loaded from the contextlens pytest plugin -- no import needed.
When assert_budget fails, the enriched breakdown appears in the failure output.
"""

from __future__ import annotations

import pytest

from contextlens.assertions import assert_budget, assert_must_contain, assert_position
from contextlens.attribution import attribute
from contextlens.capture import CapturedCall, TokenSpan, UsageRecord

GOLDEN_FACT = "Contract renewal is due on 2025-03-01."
_SYSTEM_TEXT = f"You are a contract analyst. {GOLDEN_FACT}"


def _turn_5_call() -> CapturedCall:
    """Turn 5 of the contract analyst session: retrieval has grown to 4 200 tokens."""
    history_text = "User: any updates? Assistant: reviewing. " * 5
    retrieved_text = "Exhibit A clause 7.1: obligations run until termination. " * 50
    return CapturedCall(
        provider="openai",
        model="gpt-4o",
        call_index=4,
        prompt=(
            f"[system] {_SYSTEM_TEXT}\n\n"
            f"[history] {history_text}\n\n"
            f"[retrieved] {retrieved_text}"
        ),
        spans=[
            TokenSpan(component="system",    text=_SYSTEM_TEXT, token_count=20),
            TokenSpan(component="history",   text=history_text, token_count=200),
            TokenSpan(component="retrieved", text=retrieved_text, token_count=4_200),
            TokenSpan(component="formatting", text="",          token_count=6),
        ],
        usage=UsageRecord(provider="openai", input_tokens=4_426, source="response_usage"),
    )


def test_retrieval_stays_in_budget(ctx):
    """Fails: retrieved ballooned to 4 200 tokens against a 1 000-token budget."""
    call = _turn_5_call()
    ctx.add(call)
    assert_budget(attribute(call), retrieved=1_000)   # 4 200 > 1 000


def test_golden_fact_present(ctx):
    """Passes: the fact is anchored in the system prompt, not evicted."""
    call = _turn_5_call()
    ctx.add(call)
    assert_must_contain(call, GOLDEN_FACT)


def test_golden_fact_not_in_middle(ctx):
    """Passes: the fact is at position ~0.9% -- well before the 20% middle band."""
    call = _turn_5_call()
    ctx.add(call)
    assert_position(call, GOLDEN_FACT, not_in="middle")
