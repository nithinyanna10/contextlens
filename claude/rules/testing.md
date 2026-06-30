# Testing Rules — contextlens

> Claude reads this file before writing any test in this repo.

## Test Locations

| What you're testing | Where the test goes |
|---|---|
| Input guard, output filter, content filter | `tests/` root level |
| Retrieval and reranking | `tests/test_retrieval.py` |
| Semantic cache | `tests/test_cache.py` |
| Query routing | `tests/test_routing.py` |
| New modules not listed above | Add `test_{module_name}.py` in `tests/` |

## Mandatory Coverage

- Every function in `security/` → at least 3 tests: valid input, injection attempt, boundary value
- Every agent method → at least 1 happy path + 1 error case
- Every new service method → at least 1 happy path + 1 edge case
- Eval pipeline (`evaluation/`) → run `offline_eval.py` before marking agent work done

## Test Patterns

```python
# ── Fixtures ──────────────────────────────────────────────────────────────────
# Put shared fixtures in tests/conftest.py — not in individual test files

# ── Naming ────────────────────────────────────────────────────────────────────
# test_{what}_{condition}_returns_{expected}
# e.g. test_input_guard_injection_attempt_raises_value_error

# ── Arrange / Act / Assert — always use these three sections ──────────────────
def test_semantic_cache_hit_returns_cached_response():
    # Arrange
    cache = SemanticCache(threshold=0.92)
    cache.store("What is RAG?", "RAG stands for...")

    # Act
    result = cache.lookup("What is retrieval augmented generation?")

    # Assert
    assert result is not None
    assert "RAG" in result
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single file
pytest tests/test_retrieval.py -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

Run the full suite before marking any task complete. If a test fails that you didn't touch, flag it — don't skip it.

## Eval Tests

After any change to agent logic or prompt templates:
```bash
python evaluation/offline_eval.py
```
Commit the updated results in `evaluation/eval_results/` alongside your code.
