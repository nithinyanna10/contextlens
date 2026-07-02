# contextlens — architecture decisions

Short-form ADR record. Each entry names the decision, the reason it was made
that way, and where the constraint lives in the code.

---

## 1. usage as source of truth for totals

`CapturedCall.usage.input_tokens` is the token count that actually entered
the model. tiktoken pre-flight counts are estimates; they can be off by
5--15% for non-English text or structured data. When a realized `UsageRecord`
is present, all report totals and `check_budget` sums draw from
`usage.input_tokens`, not from summing the spans. This is why `AttributionResult`
carries a `source` label (`"exact"` vs `"estimate"`) and why the CLI renders
`(exact)` or `(estimate)` next to every call header.

Code: `capture.py` `UsageRecord.source`, `attribution.py` `AttributionResult.source`.

---

## 2. sum(spans) + formatting == usage reconciliation invariant

Per-message content is counted by tiktoken; the provider adds tokens for
role markers, conversation framing, BOS tokens, and other structural overhead
that callers never see in the raw text. Rather than hiding this gap or
pretending the estimates are exact, contextlens requires a `formatting` span
that absorbs the difference. The `CapturedCall` model validator rejects any
call where `sum(spans) != usage.input_tokens`, so every realized call is
exactly reconciled. Assertions and reports never see a silent discrepancy.
The `formatting` component is flagged `non_actionable=True` throughout;
callers cannot trim it, and `check_budget` treats a `formatting` budget limit
as a non-actionable violation rather than a real one.

Code: `capture.py` `_validate_span_total`, `attribution.py` `_NON_ACTIONABLE`,
`assertions.py` `check_budget` formatting branch.

---

## 3. check/raise split in assertions

Every rule has a `check_*` function returning `list[Violation]` and an
`assert_*` wrapper that calls `_raise_if`. The split exists because the CLI
needs to collect all violations across a session and report them together
without stopping at the first failure, while pytest tests want the standard
`AssertionError` flow. `build_report` in `report.py` uses `check_*` directly;
test code uses `assert_*`. The `Violation` type is frozen Pydantic, so it
can be serialized into the JSONL session file and reconstructed later.

Code: `assertions.py` `check_*` / `assert_*` pairs, `report.py` `build_report`.

---

## 4. function-scoped ctx fixture over module-level global

Early drafts of the langgraph integration used a module-level
`_captured_calls` list. This caused state to bleed between tests when pytest
runs workers in parallel (xdist). The fix applied to all three adapters: state
lives on the instance (`Capture.calls`, `ContextLensCallbackHandler.calls`,
`NodeCapture.calls`), not on the module. The `ctx` pytest fixture is
function-scoped for the same reason: a fresh `ContextSession` per test,
discarded after the test body, with no shared reference across tests.

Code: `pytest_plugin.py` `ContextSession` and `ctx` fixture,
`integrations/openai_sdk.py` `Capture`,
`integrations/langgraph.py` `NodeCapture`.

---

## 5. rot thresholds and char-offset position as documented heuristics

`check_rot_risk` compares total tokens against a static context-window table
keyed by model name. The table reflects published specs, not measured recall
degradation curves. The 90% default threshold is a reasonable starting point
with no empirical calibration behind it. `check_position` uses the character offset of the fact string as a proxy for
its token position in the assembled prompt. This is adequate for English prose.
The confirmed failure mode is emoji-heavy prompts: on gpt-4o (o200k_base), mixed
emoji encodes at ~1.67 tokens/char, so a fact at 4.8% by character can sit at
33% by token count — inside the lost-in-the-middle zone — and the check passes
it as safe. CJK on o200k is token-sparse (~0.5 tok/char) and moves the proxy
in the opposite direction; the false-negative risk is lower. Both functions
advertise their heuristic nature in their violation messages and in `ponytail:`
comments in the source. The upgrade path (per-token boundary lookup) is noted
but not yet built.

Code: `assertions.py` `_CONTEXT_WINDOWS`, `check_rot_risk`,
`_MIDDLE_LOWER` / `_MIDDLE_UPPER`, `check_position`.

---

## 6. CapturedCall as the frozen contract

`CapturedCall` in `capture.py` is the single serialization boundary the
whole library crosses. Every adapter produces one; every assertion, report,
and CLI consumer reads one. It is a frozen Pydantic model so that consumers
cannot mutate a captured call after the fact, which would break
reproducibility guarantees. JSONL persistence uses `model_dump_json` /
`model_validate_json`, which means the schema is the wire format. Changing any
field is a breaking change: existing JSONL files would fail to load, and every
module that reads `CapturedCall` attributes would need auditing. This is why
the field list is treated as a frozen contract in `CLAUDE.md` and requires
explicit approval before modification.

Code: `capture.py` `CapturedCall`, `report.py` `dump_call` / `load_call`.

---

## 7. Cache-aware token accounting

Anthropic prompt caching splits input tokens across three fields in the response:
`input_tokens` (new, uncached), `cache_read_input_tokens` (served from an existing
cache entry), and `cache_creation_input_tokens` (written to a new cache entry). A
naive read of `input_tokens` alone misreports the context-window occupancy of
any cached call — silently, with no error, and with a confidently wrong `(exact)`
label on the report.

`UsageRecord` stores all three fields raw, preserving what the provider reported,
and exposes a `total_input_tokens` computed field that sums them. All budget and
rot-risk math uses `total_input_tokens` because a cached token occupies context
space identically to a new one. The reconciliation invariant (`sum(spans) ==
usage.total_input_tokens`) is enforced against the total, not the raw uncached
count. Old JSONL written before this change loads correctly because the two new
fields default to 0, giving `total_input_tokens == input_tokens` — the pre-cache
behavior, not wrong, just uncached.

OpenAI's `prompt_tokens` field already represents the total; the split there
uses `prompt_tokens_details.cached_tokens` and defaults to 0 when absent, so
pre-caching OpenAI calls are unchanged.

Code: `capture.py` `UsageRecord`, `integrations/langchain.py` and
`integrations/langgraph.py` usage extraction, `assertions.py` `check_rot_risk`.

---

## 8. CLI exit codes: 0 / 1 / 2

The CLI exits with three distinct codes:
- `0`: all checks pass (or no contract supplied and rot-risk is below threshold).
- `1`: one or more violations found. The JSONL was valid; the context window
  failed the contract.
- `2`: the input file or contract file could not be read or parsed.

The distinction between 1 and 2 matters for CI: a job that fails with exit 1
means "your agent's context is out of budget"; a job that fails with exit 2 means
"the session file was not written correctly or the contract JSON is malformed."
These require different responses from the operator. Conflating them into a single
nonzero exit would make the CLI useless as a diagnostic tool in automated pipelines.
Exit 2 is accompanied by a `contextlens: error reading <file>: <message>` line on
stderr that names both the file and the parse error, so the operator does not need
to read a Python traceback to understand what went wrong.

Code: `cli.py` `main`, `tests/stress/test_malformed_jsonl.py`.
