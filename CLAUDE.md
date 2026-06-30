# CLAUDE.md — contextlens

> Read this file in full at the start of every session. Also read `AGENTS.md` and `claude/rules/`.

## What this is

`contextlens` is a **pytest plugin library** — not a service, not a server, not a web app.
It snapshots the exact tokens entering each LLM call, attributes them per source, and exposes
assertions you can gate CI on.

**Stack:** Python 3.11+ · uv · ruff · mypy · pytest · pydantic · tiktoken · anthropic

---

## Module Map

```
src/contextlens/
├── capture.py          # CapturedCall + TokenSpan — the frozen data contract
├── tokenizers.py       # Tokenizer adapter interface (tiktoken + anthropic)
├── attribution.py      # Tag spans by component (system/history/retrieved/…)
├── assertions.py       # CI-gate assertions: budget, rot, must_contain, position
├── compaction.py       # Compaction strategies (summarise, trim, drop)
├── pytest_plugin.py    # pytest11 entry point — fixtures and hooks
├── cli.py              # `contextlens` console script
├── report.py           # Human-readable report formatter
└── integrations/
    ├── langchain.py
    ├── langgraph.py
    └── openai_sdk.py
tests/
docs/architecture.md
```

---

## Frozen Contract

`CapturedCall` in `capture.py` is the **single source of truth** every other module depends on.
Do not change its fields without explicit user approval and a full Phase 1 re-green.

---

## Build Discipline

Phase 0 → Phase 1 (sequential, TDD) → Phase 2 (parallel fan-out only after Phase 1 is green).

**Phase 1 order:** tokenizers.py → capture.py → attribution.py → assertions.py

- Write the failing test first, show it, then implement until green.
- Do not proceed to the next module until the current one's tests pass.
- Do NOT parallelize Phase 0 or Phase 1 work.

---

## What Claude Must NOT Do

- Do not add any dependency not in the approved list: pydantic, tiktoken, anthropic, pytest, ruff, mypy.
  Ask before adding anything else.
- Do not change `CapturedCall` fields without approval.
- Do not skip the failing-test-first step in Phase 1.
- Do not start Phase 2 modules before Phase 1 is fully green.
- Do not add service infrastructure (HTTP, DB, Docker, queues).

---

## Code Rules

- Read `claude/rules/code-style.md` before writing any module.
- Read `claude/rules/testing.md` before writing any test.
- All public functions: typed, tested, no speculative abstraction.
- Modules stay small — if a file is approaching 150 lines, ask before expanding.

## Custom Rules

<!-- Paste project-specific rules here -->
