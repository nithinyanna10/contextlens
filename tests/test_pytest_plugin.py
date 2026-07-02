"""Tests for pytest_plugin.py — use pytester to exercise real pytest behaviour."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Piece 1 — ctx fixture isolation
# ---------------------------------------------------------------------------

def test_ctx_isolates_calls_between_tests(pytester: pytest.Pytester) -> None:
    """Two consecutive tests using ctx must not share captured calls."""
    pytester.makepyfile("""
        from contextlens.capture import CapturedCall, TokenSpan

        def _call(idx: int) -> CapturedCall:
            return CapturedCall(
                provider="openai",
                model="gpt-4o",
                call_index=idx,
                prompt="test",
                spans=[TokenSpan(component="system", text="x", token_count=1)],
                usage=None,
            )

        def test_first(ctx):
            ctx.add(_call(0))
            assert len(ctx.calls) == 1
            assert ctx.last.call_index == 0

        def test_second(ctx):
            # must not see test_first's call — fresh session
            assert len(ctx.calls) == 0
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=2)


# ---------------------------------------------------------------------------
# Piece 2 — failure enrichment
# ---------------------------------------------------------------------------

def test_assert_budget_failure_appends_breakdown(pytester: pytest.Pytester) -> None:
    """When assert_budget trips in a test using ctx, the failure output must
    include the contextlens context breakdown section with component names."""
    pytester.makepyfile("""
        from contextlens.capture import CapturedCall, TokenSpan, UsageRecord
        from contextlens.attribution import attribute
        from contextlens.assertions import assert_budget

        def test_budget_fail(ctx):
            call = CapturedCall(
                provider="openai",
                model="gpt-4o",
                call_index=0,
                prompt="[system] You are helpful.",
                spans=[
                    TokenSpan(component="system",    text="You are helpful.", token_count=100),
                    TokenSpan(component="formatting", text="",                token_count=8),
                ],
                usage=UsageRecord(
                    provider="openai", input_tokens=108, source="response_usage"
                ),
            )
            ctx.add(call)
            assert_budget(attribute(call), system=50)  # 100 > 50 → AssertionError
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(failed=1)
    output = result.stdout.str()
    assert "contextlens context breakdown" in output
    assert "system" in output


# ---------------------------------------------------------------------------
# Piece 3 — registration via entry point
# ---------------------------------------------------------------------------

def test_ctx_available_without_conftest_import(pytester: pytest.Pytester) -> None:
    """ctx fixture must be injected purely from the pytest11 entry point —
    no conftest.py import required."""
    pytester.makepyfile("""
        from contextlens.capture import CapturedCall, TokenSpan

        def test_ctx_present(ctx):
            # calls: empty list at start
            assert isinstance(ctx.calls, list)
            assert ctx.calls == []
            # last: property exists on the class
            assert isinstance(type(ctx).last, property)
    """)
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
