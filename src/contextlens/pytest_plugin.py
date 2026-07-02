"""pytest_plugin.py — pytest11 entry point for contextlens.

Registers:
  - ctx fixture (function-scoped ContextSession)
  - pytest_runtest_makereport hook (failure enrichment)
"""

from __future__ import annotations

import re

import pytest

from contextlens.capture import CapturedCall
from contextlens.report import build_report, render


class ContextSession:
    """Capture session for one test function.

    Public surface:
        ctx.calls            list[CapturedCall] — all calls registered this test
        ctx.add(call)        register a CapturedCall produced by any adapter
        ctx.last             calls[-1]; raises IndexError if no calls yet
    """

    def __init__(self) -> None:
        self.calls: list[CapturedCall] = []

    def add(self, call: CapturedCall) -> None:
        """Register a captured call in this test's session."""
        self.calls.append(call)

    @property
    def last(self) -> CapturedCall:
        """Most recently added call. Raises IndexError if no calls captured yet."""
        if not self.calls:
            raise IndexError("ctx.last: no calls captured yet in this test.")
        return self.calls[-1]


@pytest.fixture
def ctx() -> ContextSession:  # type: ignore[misc]
    """Function-scoped capture session. Fresh per test, isolated from siblings."""
    yield ContextSession()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # type: ignore[no-untyped-def]
    """Append per-call context breakdown to any test failure that uses ctx."""
    outcome = yield
    report = outcome.get_result()

    if report.when != "call" or not report.failed:
        return

    session_ctx = item.funcargs.get("ctx")
    if not session_ctx or not session_ctx.calls:
        return

    report_model = build_report(session_ctx.calls)
    plain = re.sub(r"\033\[[0-9;]*m", "", render(report_model))
    report.sections.append(("contextlens context breakdown", plain))
