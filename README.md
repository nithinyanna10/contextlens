# contextlens

**pytest for your context window.**

contextlens snapshots the exact tokens entering every LLM call, attributes them
per source (system, history, retrieved, tool\_output, scratchpad), and lets you
gate CI on budget, rot risk, and whether the load-bearing fact survived.
No dashboards. No waiting for prod.

---

## The test that makes it click

```python
# examples/test_demo.py  (abridged)
from contextlens.assertions import assert_budget, assert_must_contain
from contextlens.attribution import attribute

GOLDEN_FACT = "Contract renewal is due on 2025-03-01."

def test_retrieval_stays_in_budget(ctx):
    call = _turn_5_call()
    ctx.add(call)
    assert_budget(attribute(call), retrieved=1_000)   # 4 200 > 1 000 -- fails

def test_golden_fact_present(ctx):
    call = _turn_5_call()
    ctx.add(call)
    assert_must_contain(call, GOLDEN_FACT)             # fact survived -- passes
```

When `assert_budget` trips, pytest appends the full per-call breakdown:

```
FAILED examples/test_demo.py::test_retrieval_stays_in_budget

    def test_retrieval_stays_in_budget(ctx):
        """Fails: retrieved ballooned to 4 200 tokens against a 1 000-token budget."""
        call = _turn_5_call()
        ctx.add(call)
>       assert_budget(attribute(call), retrieved=1_000)   # 4 200 > 1 000
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

E       AssertionError: retrieved: 4200 tokens exceeds budget of 1000

examples/test_demo.py:48: AssertionError
------------------------ contextlens context breakdown -------------------------
Call 4  gpt-4o  [openai]  (exact)
  ██████████████████████████████████████████████████████████████
  component       tokens  proportion  notes
  system              20       0.5%
  history            200       4.5%
  retrieved        4,200      94.9%
  formatting           6       0.1%  [non-actionable]
  total: 4,426 tokens

  1 call(s) · 0 violation(s)
```

The `0 violation(s)` in the section footer reflects the default rot-risk check only; the budget limit that failed is the assertion above it.

---

## Report

The bar is a stacked `█` block, each segment color-coded by component
(system=green, history=yellow, retrieved=cyan, formatting=dim). ANSI-stripped
below; run the command to see color and segment boundaries.

```
$ contextlens report examples/demo.jsonl --contract examples/contract.json

Call 0  gpt-4o  [openai]  (exact)
  ███████████████████████████████████████████████████████████
  component       tokens  proportion  notes
  system              20      13.7%
  history             20      13.7%
  retrieved          100      68.5%
  formatting           6       4.1%  [non-actionable]
  total: 146 tokens

Call 2  gpt-4o  [openai]  (exact)
  ████████████████████████████████████████████████████████████
  component       tokens  proportion  notes
  system              20       2.9%
  history             60       8.7%
  retrieved          600      87.5%
  formatting           6       0.9%  [non-actionable]
  total: 686 tokens

Call 4  gpt-4o  [openai]  (exact)
  ██████████████████████████████████████████████████████████████
  component       tokens  proportion  notes
  system              20       0.5%
  history            200       4.5%
  retrieved        4,200      94.9%
  formatting           6       0.1%  [non-actionable]
  total: 4,426 tokens
  ✗ retrieved: 4200 tokens exceeds budget of 1000

  3 call(s) · 1 violation(s)
```

---

## The problem

Context rot, budget bloat, and lost-in-the-middle failures get discovered
through production dashboards, after the model has already seen a degraded
window. Nothing in the standard test stack lets you write an assertion on the
assembled input window and fail CI before a regression ships.
contextlens fills that gap.

---

## Install

```
pip install contextlens
```

The pytest plugin registers automatically on install. No conftest import needed.
The `ctx` fixture is available in every test file the moment the package is
installed.

---

## Quickstart

> Project contributors: use `uv run python examples/demo.py` instead of `python`.

The `examples/` directory contains a self-contained demo of a three-turn
contract analyst agent whose retrieved context grows until a CI budget gate
trips.

**Step 1 -- generate the JSONL session file**

```bash
python examples/demo.py
# Wrote 3 calls to examples/demo.jsonl
```

**Step 2 -- inspect the report**

```bash
contextlens report examples/demo.jsonl --contract examples/contract.json
```

The contract file sets the limits:

```json
{
  "rot_risk_below": 0.9,
  "per_component": {
    "retrieved": 1000,
    "history": 500
  }
}
```

Turn 5 exceeds the `retrieved` budget by 4x. The CLI exits with code 1, so it
gates a CI step on its own without a pytest wrapper.

**Step 3 -- run the pytest gate**

```bash
pytest examples/test_demo.py -v
```

```
examples/test_demo.py::test_retrieval_stays_in_budget FAILED             [ 33%]
examples/test_demo.py::test_golden_fact_present PASSED                   [ 66%]
examples/test_demo.py::test_golden_fact_not_in_middle PASSED             [100%]

FAILED examples/test_demo.py::test_retrieval_stays_in_budget - AssertionError...
========================= 1 failed, 2 passed in 0.01s ==========================
```

Two checks pass, one fails. The failure output shows the enriched breakdown
above. Same assertions, same data, reproducible from the JSONL file without
an API call.

---

## The five assertions

All five have a `check_*` twin that returns `list[Violation]` instead of
raising, for composing multi-rule policies.

**1. assert\_budget** -- per-component or total token limits

```python
from contextlens.assertions import assert_budget
from contextlens.attribution import attribute

assert_budget(attribute(call), retrieved=2_000, history=500, max_total=4_096)
# AssertionError: retrieved: 4200 tokens exceeds budget of 2000
```

Passing a limit for `formatting` is flagged as a non-actionable violation;
those tokens (role markers, BOS, message framing) cannot be trimmed.

**2. assert\_must\_contain** -- fact survives into the assembled prompt

```python
from contextlens.assertions import assert_must_contain

assert_must_contain(call, "Contract renewal is due on 2025-03-01.")
# AssertionError: required fact not found in prompt: 'Contract renewal...'
```

Substring match, not semantic: if the exact string was reformatted or
paraphrased, this passes even though the fact may have changed meaning.

**3. assert\_must\_not\_contain** -- regex absence

```python
from contextlens.assertions import assert_must_not_contain

assert_must_not_contain(call, r"\b\d{3}-\d{2}-\d{4}\b")  # no SSNs in prompt
# AssertionError: disallowed pattern '\b\d{3}...' matched: '123-45-6789'
```

**4. assert\_rot\_risk** -- fraction of context window consumed

```python
from contextlens.assertions import assert_rot_risk

assert_rot_risk(call, below=0.9)
# AssertionError: rot risk 97.50% >= 90% (124800/128000 tokens, model='gpt-4o'; heuristic, not calibrated)
```

Score = `total_tokens / context_window`. The context window is looked up from
a static table keyed by model name; unrecognized models fall back to 128,000.
The table is not calibrated; it reflects published specs, not measured
recall curves. The violation message says so explicitly.
*ponytail: extend `_CONTEXT_WINDOWS` in assertions.py as new models ship.*

**5. assert\_position** -- fact is not in the lost-in-the-middle zone

```python
from contextlens.assertions import assert_position

assert_position(call, "Contract renewal is due on 2025-03-01.", not_in="middle")
# AssertionError: fact '...' found at 45% of prompt (lost-in-the-middle zone: 20%-80%)
```

Locates the fact by character offset and flags it if the start position falls
in the central 60% of the prompt (from 20% to 80%). Character offset is a
proxy for token offset, adequate for English prose; can give false negatives on
emoji-heavy prompts where token density diverges sharply from character density
(upgrade path: per-token boundary when tiktoken exposes that API).
*ponytail: upgrade to per-token boundary when tokenizer exposes that API.*

---

## How it works

Every LLM call is recorded as a `CapturedCall` (frozen Pydantic model in
`capture.py`): the assembled prompt string, a list of `TokenSpan` objects
each tagged with a `ComponentTag` (`system`, `history`, `retrieved`,
`tool_output`, `scratchpad`, or `formatting`), and an optional
`UsageRecord` from the provider API response. The `formatting` span absorbs
the gap between tiktoken content estimates and `usage.input_tokens`, so
`sum(all spans) == usage.input_tokens` always holds exactly, enforced by a
Pydantic model validator at construction time. When `usage` is present,
attribution is labelled `(exact)`; without it, `(estimate)` with a tiktoken
ceiling of roughly +10%. The enriched failure breakdown (the
`contextlens context breakdown` section in pytest output) only fires when the
failing test uses the `ctx` fixture; tests that call assertions directly
without `ctx.add()` get the standard pytest output.

See `docs/architecture.md` for the design decisions behind these choices.

---

## Integrations

**OpenAI SDK**

```python
from openai import OpenAI
from contextlens.integrations.openai_sdk import Capture
from contextlens.attribution import attribute
from contextlens.assertions import assert_budget

client = OpenAI()
capture = Capture(client)

capture.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a contract analyst."},
        {"role": "user",   "content": "Summarise the renewal clause."},
    ],
)
assert_budget(attribute(capture.calls[-1]), retrieved=2_000)
```

`Capture.create` is a drop-in for `client.chat.completions.create`; it
returns the original response unchanged and appends a `CapturedCall` to
`capture.calls`.

**LangChain**

```python
from langchain_openai import ChatOpenAI
from contextlens.integrations.langchain import ContextLensCallbackHandler

handler = ContextLensCallbackHandler()
chat = ChatOpenAI(model="gpt-4o")
chat.invoke(messages, config={"callbacks": [handler]})

call = handler.calls[-1]
```

If the model class is not `ChatOpenAI` or `ChatAnthropic`, contextlens
emits a loud warning and sets `provider="unknown"`. Usage stays `None`,
token counts degrade to estimates, and the `(exact)` label becomes
`(estimate)` in report output.

**LangGraph**

```python
from contextlens.integrations.langgraph import capture_node

agent_node = capture_node(call_model, model="gpt-4o", provider="openai")
# wire agent_node into the graph wherever call_model was used

# after graph.invoke(state):
call = agent_node.calls[-1]
```

`NodeCapture` stores state on the instance, not a module global, so it is
safe when tests run in parallel xdist workers.

---

## CI surfaces

contextlens has two independent CI surfaces. Use either or both.

**Surface 1 -- pytest plugin (enriched failure output)**

The plugin auto-registers when the package is installed. Every test that
uses the `ctx` fixture gets the context breakdown appended to any failure.
No YAML needed.

**Surface 2 -- CLI exit code (direct budget gate)**

```yaml
# .github/workflows/ci.yml
- name: dump session
  run: python my_agent_smoke.py   # writes session.jsonl

- name: contextlens budget check
  run: contextlens report session.jsonl --contract contract.json
  # exits 0 on clean, 1 on any violation -- job fails without extra logic
```

The `--contract` flag is optional. Without it, only the default rot-risk
check (90% threshold) runs. With it, per-component and total budget limits
are enforced in addition.

---

## Status

contextlens is MVP-complete for the token-budget and fact-survival use cases.

What works now:

- Exact token attribution for OpenAI and Anthropic calls (realized and pre-flight)
- Five CI-gateable assertions: budget, must\_contain, must\_not\_contain, rot\_risk, position
- JSONL session persistence and CLI report with contract enforcement
- pytest plugin: `ctx` fixture, failure enrichment, auto-registration

Known limits and honest notes:

- `assert_must_contain` and `assert_must_not_contain` are substring / regex
  checks, not semantic. If the fact was paraphrased into different wording,
  the check fails even if the meaning survived. If the original string is
  still present alongside a paraphrase, the check passes and misses the drift.
- `assert_position` uses character offset as a proxy for token position.
  Adequate for English prose. Can produce false negatives on emoji-heavy prompts:
  a fact at 4.8% by character can sit at 33% by token count (inside the
  lost-in-the-middle zone) and the check will pass it. CJK on gpt-4o (o200k)
  is token-sparse and less affected; emoji is the real failure mode.
- `assert_rot_risk` thresholds are heuristic. The 90% default is not
  empirically calibrated against LLM recall degradation curves.
- The context-window table in `assertions.py` covers common GPT and Claude
  models only. Extend it for other model families.

What is next:

- Compaction strategies (`compaction.py` stub): summarise / trim / drop
  passes that shrink the window to a target budget
- Semantic fact-survival check (embedding cosine) as an opt-in alternative
  to substring match
- Token-boundary position check when tiktoken exposes that API (fixes the
  emoji false-negative in `assert_position`)

---

## License

MIT License

Copyright (c) 2026 Nithin Yanna

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
