from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from contextlens.attribution import AttributionResult
from contextlens.capture import CapturedCall


class Violation(BaseModel):
    """A single rule failure. Never raises; consumers decide what to do with it.

    rule:      dot-namespaced identifier, e.g. "budget", "budget.non_actionable"
    component: set for per-component rules (budget); None for prompt-level rules
    observed:  what was actually found
    threshold: what was required
    message:   human-readable; safe to join and surface directly
    """

    model_config = ConfigDict(frozen=True)

    rule: str
    component: str | None = None
    observed: int | float | str
    threshold: int | float | str
    message: str


# ---------------------------------------------------------------------------
# internal helper
# ---------------------------------------------------------------------------

def _raise_if(violations: list[Violation]) -> None:
    if violations:
        raise AssertionError("\n".join(v.message for v in violations))


# ---------------------------------------------------------------------------
# 1. budget
# ---------------------------------------------------------------------------

def check_budget(
    result: AttributionResult,
    max_total: int | None = None,
    **per_component: int,
) -> list[Violation]:
    """Flag components (or total) that exceed their token limit.

    Passing a limit for 'formatting' returns a non_actionable Violation and
    never counts as a real budget violation — those tokens cannot be trimmed.
    """
    violations: list[Violation] = []

    if "formatting" in per_component:
        violations.append(Violation(
            rule="budget.non_actionable",
            component="formatting",
            observed=result.components["formatting"].token_count,
            threshold=per_component["formatting"],
            message=(
                "formatting tokens (role markers, BOS, message framing) are "
                "non_actionable — the user cannot trim them. "
                "Remove the 'formatting' limit from check_budget()."
            ),
        ))

    for comp, limit in per_component.items():
        if comp == "formatting":
            continue
        actual = result.components[comp].token_count if comp in result.components else 0
        if actual > limit:
            violations.append(Violation(
                rule="budget",
                component=comp,
                observed=actual,
                threshold=limit,
                message=f"{comp}: {actual} tokens exceeds budget of {limit}",
            ))

    if max_total is not None:
        total = result.total
        if total > max_total:
            violations.append(Violation(
                rule="budget",
                component=None,
                observed=total,
                threshold=max_total,
                message=f"total: {total} tokens exceeds max_total of {max_total}",
            ))

    return violations


def assert_budget(
    result: AttributionResult,
    max_total: int | None = None,
    **per_component: int,
) -> None:
    _raise_if(check_budget(result, max_total=max_total, **per_component))


# ---------------------------------------------------------------------------
# 2. must_contain
# ---------------------------------------------------------------------------

def check_must_contain(call: CapturedCall, fact: str) -> list[Violation]:
    """Fact (substring) must appear in the assembled prompt."""
    if fact in call.prompt:
        return []
    return [Violation(
        rule="must_contain",
        component=None,
        observed="absent",
        threshold="present",
        message=f"required fact not found in prompt: {fact!r}",
    )]


def assert_must_contain(call: CapturedCall, fact: str) -> None:
    _raise_if(check_must_contain(call, fact))


# ---------------------------------------------------------------------------
# 3. must_not_contain
# ---------------------------------------------------------------------------

def check_must_not_contain(call: CapturedCall, pattern: str) -> list[Violation]:
    """Regex must not match anywhere in the assembled prompt."""
    match = re.search(pattern, call.prompt)
    if not match:
        return []
    excerpt = match.group(0)[:60]
    return [Violation(
        rule="must_not_contain",
        component=None,
        observed=excerpt,
        threshold=pattern,
        message=f"disallowed pattern {pattern!r} matched: {excerpt!r}",
    )]


def assert_must_not_contain(call: CapturedCall, pattern: str) -> None:
    _raise_if(check_must_not_contain(call, pattern))


# ---------------------------------------------------------------------------
# 4. rot_risk
# ---------------------------------------------------------------------------

# ponytail: heuristic table, not calibrated — extend as new models ship
_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4":                        8_192,
    "gpt-4-turbo":                128_000,
    "gpt-4o":                     128_000,
    "gpt-4o-mini":                128_000,
    "gpt-3.5-turbo":               16_385,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022":  200_000,
    "claude-3-opus-20240229":     200_000,
    "claude-3-sonnet-20240229":   200_000,
    "claude-3-haiku-20240307":    200_000,
}
_DEFAULT_CONTEXT = 128_000


def check_rot_risk(call: CapturedCall, below: float) -> list[Violation]:
    """Score = total_tokens / context_window. Violation if score >= below.

    Total taken from usage.total_input_tokens when realized; span sum otherwise.
    Context window is a heuristic table entry — cite this in any report.
    """
    total = (
        call.usage.total_input_tokens
        if call.usage is not None
        else sum(s.token_count for s in call.spans)
    )
    context_window = _CONTEXT_WINDOWS.get(call.model, _DEFAULT_CONTEXT)
    score = round(total / context_window, 4)
    if score < below:
        return []
    return [Violation(
        rule="rot_risk",
        component=None,
        observed=score,
        threshold=below,
        message=(
            f"rot risk {score:.2%} >= {below:.0%} "
            f"({total}/{context_window} tokens, model={call.model!r}; "
            "heuristic, not calibrated)"
        ),
    )]


def assert_rot_risk(call: CapturedCall, below: float) -> None:
    _raise_if(check_rot_risk(call, below))


# ---------------------------------------------------------------------------
# 5. position
# ---------------------------------------------------------------------------

# Middle band: central 60% of the prompt by character offset (token-offset proxy).
# ponytail: character position used as proxy for token offset — adequate for
# English prose; upgrade to per-token boundary when tokenizer grows that API.
_MIDDLE_LOWER: float = 0.20
_MIDDLE_UPPER: float = 0.80


def check_position(
    call: CapturedCall,
    fact: str,
    not_in: Literal["middle"] = "middle",
) -> list[Violation]:
    """Locate fact in the assembled prompt; flag if it lands in the danger zone.

    not_in="middle": violation if fact's start position falls in the central
    60% of the prompt (characters 20%–80%). This is the lost-in-the-middle
    zone where LLM recall degrades under context pressure.
    """
    idx = call.prompt.find(fact)
    if idx == -1:
        return []  # absent — must_contain handles that separately

    ratio = idx / max(len(call.prompt), 1)
    in_middle = _MIDDLE_LOWER < ratio < _MIDDLE_UPPER

    if not in_middle:
        return []

    return [Violation(
        rule="position",
        component=None,
        observed=f"{ratio:.0%}",
        threshold=f"outside middle ({_MIDDLE_LOWER:.0%}–{_MIDDLE_UPPER:.0%})",
        message=(
            f"fact {fact!r} found at {ratio:.0%} of prompt "
            f"(lost-in-the-middle zone: {_MIDDLE_LOWER:.0%}–{_MIDDLE_UPPER:.0%})"
        ),
    )]


def assert_position(
    call: CapturedCall,
    fact: str,
    not_in: Literal["middle"] = "middle",
) -> None:
    _raise_if(check_position(call, fact, not_in=not_in))
