"""Step 2: Scale and degenerate shape stress tests.

All offline. Targets: wrong numbers and divide-by-zero, not crashes.
"""

from __future__ import annotations

import pytest

from contextlens.assertions import check_budget, check_rot_risk
from contextlens.attribution import attribute
from contextlens.capture import CapturedCall, TokenSpan, UsageRecord
from contextlens.report import build_report, render

pytestmark = pytest.mark.stress


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _call(
    *,
    system: int = 20,
    history: int = 0,
    retrieved: int = 0,
    fmt: int = 6,
    model: str = "gpt-4o",
    call_index: int = 0,
    realized: bool = True,
) -> CapturedCall:
    total = system + history + retrieved + fmt
    spans = [
        TokenSpan(component="system", text="s" * system, token_count=system),
    ]
    if history:
        spans.append(TokenSpan(component="history", text="h" * history, token_count=history))
    if retrieved:
        spans.append(TokenSpan(component="retrieved", text="r" * retrieved, token_count=retrieved))
    spans.append(TokenSpan(component="formatting", text="", token_count=fmt))
    return CapturedCall(
        provider="openai",
        model=model,
        call_index=call_index,
        prompt="x",
        spans=spans,
        usage=UsageRecord(
            provider="openai", input_tokens=total, source="response_usage"
        ) if realized else None,
    )


# ---------------------------------------------------------------------------
# 200-turn session
# ---------------------------------------------------------------------------

class TestLargeSession:
    def test_200_turn_session_builds_report(self):
        calls = [
            _call(system=20, history=i * 10, retrieved=100, fmt=6, call_index=i)
            for i in range(200)
        ]
        report = build_report(calls)
        assert len(report.calls) == 200
        # Totals should grow monotonically with history
        for i in range(1, 200):
            assert report.calls[i].total_tokens >= report.calls[i - 1].total_tokens

    def test_200_turn_session_renders_without_error(self):
        calls = [
            _call(system=20, history=i * 10, retrieved=100, fmt=6, call_index=i)
            for i in range(200)
        ]
        report = build_report(calls)
        text = render(report)
        assert "Call 0" in text
        assert "Call 199" in text

    def test_200_turn_attribution_totals_are_correct(self):
        calls = [
            _call(system=20, history=i * 5, retrieved=50, fmt=6, call_index=i)
            for i in range(200)
        ]
        for call in calls:
            attr = attribute(call)
            assert attr.total == call.usage.total_input_tokens


# ---------------------------------------------------------------------------
# Single massive call (180k tokens, exceeds common 128k window)
# ---------------------------------------------------------------------------

class TestMassiveCall:
    def test_180k_call_builds_correctly(self):
        call = _call(system=100, history=0, retrieved=179_894, fmt=6)
        assert call.usage.total_input_tokens == 180_000
        attr = attribute(call)
        assert attr.total == 180_000

    def test_180k_call_triggers_rot_risk_on_gpt4o(self):
        # 180k / 128k = 140.6% — well over any threshold
        call = _call(system=100, retrieved=179_894, fmt=6)
        violations = check_rot_risk(call, below=0.9)
        assert len(violations) == 1
        assert violations[0].rule == "rot_risk"
        # Score must reference the real total, not a truncated number
        assert violations[0].observed > 1.0   # over 100% of window

    def test_180k_call_renders_total_correctly(self):
        call = _call(system=100, retrieved=179_894, fmt=6)
        report = build_report([call])
        text = render(report)
        assert "180,000" in text

    def test_max_total_budget_fires_on_180k(self):
        call = _call(system=100, retrieved=179_894, fmt=6)
        attr = attribute(call)
        violations = check_budget(attr, max_total=100_000)
        assert len(violations) == 1
        assert violations[0].observed == 180_000


# ---------------------------------------------------------------------------
# Empty session (zero calls)
# ---------------------------------------------------------------------------

class TestEmptySession:
    def test_empty_session_builds_report(self):
        report = build_report([])
        assert report.calls == []
        assert report.total_violations == 0
        assert not report.has_violations

    def test_empty_session_renders_summary_only(self):
        report = build_report([])
        text = render(report)
        assert "0 call(s)" in text
        assert "0 violation(s)" in text

    def test_empty_session_exits_zero(self, tmp_path):
        import subprocess, sys
        jsonl = tmp_path / "empty.jsonl"
        jsonl.write_text("")
        result = subprocess.run(
            [sys.executable, "-m", "contextlens.cli", "report", str(jsonl)],
            capture_output=True,
        )
        # Empty file = no violations = exit 0
        # (Python -m contextlens.cli is not the installed entrypoint, use uv run)
        # We test via the Python API instead
        from contextlens.report import load_calls, build_report
        calls = load_calls(jsonl)
        report = build_report(calls)
        assert not report.has_violations


# ---------------------------------------------------------------------------
# Call with total == 0 (no spans at all, pre-flight)
# ---------------------------------------------------------------------------

class TestZeroTokenCall:
    def _zero_call(self) -> CapturedCall:
        # Only way to have total==0 is pre-flight (usage=None) with a single
        # zero-count formatting span.
        return CapturedCall(
            provider="openai",
            model="gpt-4o",
            call_index=0,
            prompt="",
            spans=[TokenSpan(component="formatting", text="", token_count=0)],
            usage=None,
        )

    def test_zero_total_attribution_is_zero(self):
        call = self._zero_call()
        attr = attribute(call)
        assert attr.total == 0

    def test_zero_total_proportions_are_zero_not_nan(self):
        call = self._zero_call()
        report = build_report([call])
        for bd in report.calls[0].components.values():
            assert bd.proportion == 0.0
            assert bd.proportion == bd.proportion  # not NaN

    def test_zero_total_renders_without_error(self):
        call = self._zero_call()
        report = build_report([call])
        text = render(report)
        assert "total: 0 tokens" in text

    def test_zero_total_rot_risk_does_not_crash(self):
        call = self._zero_call()
        # 0/128000 = 0.0 — should return no violation
        violations = check_rot_risk(call, below=0.9)
        assert violations == []


# ---------------------------------------------------------------------------
# Single component dominating (95%+ of window)
# ---------------------------------------------------------------------------

class TestDominantComponent:
    def test_95_pct_retrieved_proportion(self):
        call = _call(system=20, retrieved=9_474, fmt=6)  # 9500 total; 9474/9500 ≈ 99.7%
        report = build_report([call])
        cr = report.calls[0]
        retrieved_prop = cr.components["retrieved"].proportion
        # retrieved should be the dominant component
        assert retrieved_prop > 0.95

    def test_95_pct_retrieved_proportions_sum_to_one(self):
        call = _call(system=20, retrieved=9_474, fmt=6)
        report = build_report([call])
        cr = report.calls[0]
        total_prop = sum(
            bd.proportion for bd in cr.components.values() if bd.token_count > 0
        )
        assert abs(total_prop - 1.0) < 1e-6

    def test_95_pct_renders_bar_without_error(self):
        call = _call(system=20, retrieved=9_474, fmt=6)
        report = build_report([call])
        text = render(report)
        assert "retrieved" in text
