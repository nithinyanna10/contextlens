"""report.py — serialization, ReportModel, renderer."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, computed_field

from contextlens.assertions import Violation, check_budget, check_rot_risk
from contextlens.attribution import SourceLabel, attribute
from contextlens.capture import CapturedCall


# ---------------------------------------------------------------------------
# Serialization (JSONL round-trip)
# ---------------------------------------------------------------------------

def dump_call(call: CapturedCall) -> str:
    """Serialize a CapturedCall to a JSON string (one JSONL line)."""
    return call.model_dump_json()


def load_call(line: str) -> CapturedCall:
    """Deserialize a CapturedCall from a JSON string."""
    return CapturedCall.model_validate_json(line)


def dump_calls(calls: list[CapturedCall], path: str | Path) -> None:
    """Write a list of CapturedCalls to a .jsonl file, one per line."""
    with open(path, "w") as f:
        for call in calls:
            f.write(dump_call(call) + "\n")


def load_calls(path: str | Path) -> list[CapturedCall]:
    """Load CapturedCalls from a .jsonl file."""
    calls = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                calls.append(load_call(line))
    return calls


# ---------------------------------------------------------------------------
# Contract — assertion thresholds loaded from JSON
# ---------------------------------------------------------------------------

class Contract(BaseModel):
    """Assertion thresholds. Serialised as JSON: Contract.model_validate_json(text)."""

    model_config = ConfigDict(frozen=True)

    rot_risk_below: float = 0.9
    max_total: int | None = None
    per_component: dict[str, int] = {}


# ---------------------------------------------------------------------------
# ReportModel — the contract the renderer and any dashboard reads
# ---------------------------------------------------------------------------

class ComponentBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    token_count: int
    proportion: float       # token_count / call total
    non_actionable: bool    # True for "formatting" — cannot be trimmed


class CallReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_index: int
    model: str
    provider: str
    source: SourceLabel     # "exact" | "estimate"
    total_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    components: dict[str, ComponentBreakdown]   # all 6 ComponentTag keys always present
    violations: list[Violation]


class ReportModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    calls: list[CallReport]

    @computed_field
    @property
    def total_violations(self) -> int:
        return sum(len(r.violations) for r in self.calls)

    @computed_field
    @property
    def has_violations(self) -> bool:
        return self.total_violations > 0


def build_report(
    calls: list[CapturedCall],
    contract: Contract | None = None,
) -> ReportModel:
    """Build a ReportModel from captured calls and an optional contract."""
    if contract is None:
        contract = Contract()

    call_reports: list[CallReport] = []
    for call in calls:
        attr = attribute(call)
        total = attr.total

        components = {
            tag: ComponentBreakdown(
                token_count=ct.token_count,
                proportion=ct.token_count / total if total > 0 else 0.0,
                non_actionable=ct.non_actionable,
            )
            for tag, ct in attr.components.items()
        }

        violations: list[Violation] = []
        violations.extend(check_rot_risk(call, below=contract.rot_risk_below))
        if contract.max_total is not None or contract.per_component:
            violations.extend(
                check_budget(attr, max_total=contract.max_total, **contract.per_component)
            )

        cache_read = call.usage.cache_read_input_tokens if call.usage else 0
        cache_creation = call.usage.cache_creation_input_tokens if call.usage else 0
        call_reports.append(CallReport(
            call_index=call.call_index,
            model=call.model,
            provider=call.provider,
            source=attr.source,
            total_tokens=total,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            components=components,
            violations=violations,
        ))

    return ReportModel(calls=call_reports)


# ---------------------------------------------------------------------------
# Renderer — ReportModel → text (ANSI, stdlib only)
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"

_COMPONENT_COLOR: dict[str, str] = {
    "system":      "\033[92m",   # bright green
    "history":     "\033[93m",   # bright yellow
    "retrieved":   "\033[96m",   # bright cyan
    "tool_output": "\033[95m",   # bright magenta
    "scratchpad":  "\033[97m",   # bright white
    "formatting":  "\033[2m",    # dim — non-actionable, cannot be trimmed
}
_BAR_WIDTH = 60
_BAR_CHAR = "█"


def _render_bar(components: dict[str, ComponentBreakdown]) -> str:
    bar = ""
    for tag, bd in components.items():
        if bd.token_count == 0:
            continue
        seg_len = max(1, round(bd.proportion * _BAR_WIDTH))
        bar += _COMPONENT_COLOR.get(tag, "") + _BAR_CHAR * seg_len + _RESET
    return bar


def render(report: ReportModel) -> str:
    """Format a ReportModel as a human-readable ANSI string."""
    lines: list[str] = []

    for cr in report.calls:
        lines.append(
            f"{_BOLD}Call {cr.call_index}  {cr.model}  [{cr.provider}]  "
            f"({cr.source}){_RESET}"
        )
        bar = _render_bar(cr.components)
        if bar:
            lines.append(f"  {bar}")
        lines.append(f"  {'component':<12}  {'tokens':>8}  {'proportion':>10}  notes")
        for tag, bd in cr.components.items():
            if bd.token_count == 0:
                continue
            color = _COMPONENT_COLOR.get(tag, "")
            flag = "  [non-actionable]" if bd.non_actionable else ""
            lines.append(
                f"  {color}{tag:<12}{_RESET}  {bd.token_count:>8,}  {bd.proportion:>9.1%}{flag}"
            )
        cache_parts: list[str] = []
        if cr.cache_read_input_tokens:
            cache_parts.append(f"{cr.cache_read_input_tokens:,} cached read")
        if cr.cache_creation_input_tokens:
            cache_parts.append(f"{cr.cache_creation_input_tokens:,} cache write")
        if cache_parts:
            new_tokens = cr.total_tokens - cr.cache_read_input_tokens - cr.cache_creation_input_tokens
            lines.append(
                f"  total: {cr.total_tokens:,} tokens"
                f"  ({' + '.join(cache_parts)} + {new_tokens:,} new)"
            )
        else:
            lines.append(f"  total: {cr.total_tokens:,} tokens")
        for v in cr.violations:
            lines.append(f"  \033[91m✗ {v.message}{_RESET}")
        lines.append("")

    n_violations = report.total_violations
    summary = f"  {len(report.calls)} call(s) · {n_violations} violation(s)"
    color = "\033[91m" if report.has_violations else "\033[92m"
    lines.append(f"{_BOLD}{color}{summary}{_RESET}")
    return "\n".join(lines)
