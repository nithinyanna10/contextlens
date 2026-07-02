"""Tests for report.py: serialization, ReportModel, renderer, CLI."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from contextlens.capture import CapturedCall, TokenSpan, UsageRecord
from contextlens.report import (
    Contract,
    ReportModel,
    build_report,
    dump_call,
    dump_calls,
    load_call,
    load_calls,
    render,
)


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------

def _realized_call(call_index: int = 0) -> CapturedCall:
    """28-token realized call: sys=10 hist=5 retrieved=8 fmt=5."""
    return CapturedCall(
        provider="openai",
        model="gpt-4o",
        call_index=call_index,
        prompt="[system] You are helpful.\n\n[human] Summarise.\n\n[retrieved] Doc chunk.",
        spans=[
            TokenSpan(component="system",    text="You are helpful.",  token_count=10),
            TokenSpan(component="history",   text="Summarise.",        token_count=5),
            TokenSpan(component="retrieved", text="Doc chunk.",        token_count=8),
            TokenSpan(component="formatting", text="",                 token_count=5),
        ],
        usage=UsageRecord(provider="openai", input_tokens=28, source="response_usage"),
    )


def _preflight_call(call_index: int = 1) -> CapturedCall:
    """5-token preflight call (no usage)."""
    return CapturedCall(
        provider="openai",
        model="gpt-4",
        call_index=call_index,
        prompt="[system] sys\n\n[human] hi",
        spans=[
            TokenSpan(component="system",  text="sys", token_count=3),
            TokenSpan(component="history", text="hi",  token_count=2),
        ],
        usage=None,
    )


def _big_call() -> CapturedCall:
    """120 000-token realized call on gpt-4o (> 90% of 128k context)."""
    return CapturedCall(
        provider="openai",
        model="gpt-4o",
        call_index=0,
        prompt="[system] x",
        spans=[
            TokenSpan(component="system",     text="x",  token_count=10),
            TokenSpan(component="history",    text="",   token_count=10),
            TokenSpan(component="formatting", text="",   token_count=119_980),
        ],
        usage=UsageRecord(provider="openai", input_tokens=120_000, source="response_usage"),
    )


# ---------------------------------------------------------------------------
# Step 1 — Serialization
# ---------------------------------------------------------------------------

def test_dump_call_round_trips():
    """dump_call / load_call: JSON string → CapturedCall round-trip."""
    call = _realized_call()
    assert load_call(dump_call(call)) == call


def test_dump_call_round_trips_with_usage():
    """Round-trip preserves UsageRecord fields."""
    call = _realized_call()
    loaded = load_call(dump_call(call))
    assert loaded.usage is not None
    assert loaded.usage.input_tokens == 28
    assert loaded.usage.source == "response_usage"


def test_dump_calls_round_trips_file(tmp_path: Path) -> None:
    """dump_calls / load_calls: JSONL file round-trips a list."""
    calls = [_realized_call(i) for i in range(3)]
    p = tmp_path / "calls.jsonl"
    dump_calls(calls, p)
    assert load_calls(p) == calls


# ---------------------------------------------------------------------------
# Step 2 — ReportModel
# ---------------------------------------------------------------------------

def test_build_report_proportions_and_source():
    """2-call session: correct proportions, totals, and source labels."""
    report = build_report([_realized_call(0), _preflight_call(1)])
    assert len(report.calls) == 2

    cr0 = report.calls[0]
    assert cr0.source == "exact"
    assert cr0.total_tokens == 28
    assert cr0.components["system"].token_count == 10
    assert cr0.components["system"].proportion == pytest.approx(10 / 28)
    assert cr0.components["history"].proportion == pytest.approx(5 / 28)
    assert cr0.components["retrieved"].proportion == pytest.approx(8 / 28)
    assert cr0.components["formatting"].proportion == pytest.approx(5 / 28)

    cr1 = report.calls[1]
    assert cr1.source == "estimate"
    assert cr1.total_tokens == 5
    assert cr1.components["system"].proportion == pytest.approx(3 / 5)
    assert cr1.components["history"].proportion == pytest.approx(2 / 5)


def test_build_report_formatting_non_actionable():
    """formatting component carries non_actionable=True; all others False."""
    report = build_report([_realized_call()])
    cr = report.calls[0]
    assert cr.components["formatting"].non_actionable is True
    for tag in ("system", "history", "retrieved", "tool_output", "scratchpad"):
        assert cr.components[tag].non_actionable is False


def test_build_report_rot_risk_violation():
    """120k tokens on gpt-4o (128k ctx) triggers rot_risk with default threshold."""
    # score = 120000/128000 = 0.9375, default threshold = 0.9 → violation
    report = build_report([_big_call()])
    violations = report.calls[0].violations
    assert len(violations) >= 1
    assert any(v.rule == "rot_risk" for v in violations)
    assert report.has_violations is True
    assert report.total_violations >= 1


def test_build_report_contract_budget_violation():
    """Contract per_component limit fires a budget violation."""
    contract = Contract(per_component={"system": 5})
    # realized call has system=10, limit=5 → violation
    report = build_report([_realized_call()], contract=contract)
    violations = report.calls[0].violations
    assert any(v.rule == "budget" and v.component == "system" for v in violations)


def test_build_report_no_violations_when_clean():
    """Preflight call well under rot_risk threshold has no violations."""
    report = build_report([_preflight_call()])
    assert report.calls[0].violations == []
    assert report.has_violations is False
    assert report.total_violations == 0


# ---------------------------------------------------------------------------
# Step 3 — Renderer (loose: content checks, no ANSI snapshots)
# ---------------------------------------------------------------------------

def test_render_contains_component_names_and_totals():
    """Rendered output names every non-zero component and the call total."""
    report = build_report([_realized_call()])
    out = render(report)
    # strip ANSI for text checks
    import re
    plain = re.sub(r"\033\[[0-9;]*m", "", out)
    assert "system" in plain
    assert "history" in plain
    assert "retrieved" in plain
    assert "formatting" in plain
    assert "28" in plain           # total token count


def test_render_bar_chars_present():
    """Each non-zero component contributes at least one bar character."""
    report = build_report([_realized_call()])
    out = render(report)
    assert "█" in out


def test_render_formatting_flagged_non_actionable():
    """formatting row is annotated [non-actionable] so users don't try to trim it."""
    report = build_report([_realized_call()])
    out = render(report)
    import re
    plain = re.sub(r"\033\[[0-9;]*m", "", out)
    assert "[non-actionable]" in plain


def test_render_violation_present_in_output():
    """Violation messages appear in the rendered output."""
    report = build_report([_big_call()])
    out = render(report)
    import re
    plain = re.sub(r"\033\[[0-9;]*m", "", out)
    assert "rot risk" in plain


def test_render_summary_line_reports_violations():
    """Summary line states violation count; non-zero when violations exist."""
    report = build_report([_big_call()])
    out = render(report)
    import re
    plain = re.sub(r"\033\[[0-9;]*m", "", out)
    # last line is summary: "N call(s) · M violation(s)"
    summary = [l for l in plain.splitlines() if "call(s)" in l]
    assert summary
    assert "1 violation" in summary[-1] or "violation(s)" in summary[-1]


# ---------------------------------------------------------------------------
# Step 4 — CLI
# ---------------------------------------------------------------------------

def test_cli_clean_file_exits_0(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Clean session (no violations) → exit 0; stdout contains call summary."""
    from contextlens.cli import main

    p = tmp_path / "calls.jsonl"
    dump_calls([_preflight_call()], p)

    with pytest.raises(SystemExit) as exc:
        main(["report", str(p)])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "call(s)" in out


def test_cli_violating_file_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Session with violations → exit 1; stdout contains violation text."""
    from contextlens.cli import main

    p = tmp_path / "calls.jsonl"
    dump_calls([_big_call()], p)

    with pytest.raises(SystemExit) as exc:
        main(["report", str(p)])

    assert exc.value.code == 1
    out = capsys.readouterr().out
    import re
    plain = re.sub(r"\033\[[0-9;]*m", "", out)
    assert "rot risk" in plain
