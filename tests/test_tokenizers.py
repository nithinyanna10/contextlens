import os

import pytest

from contextlens.tokenizers import get_tokenizer

# Exact count pre-verified against tiktoken cl100k_base (gpt-4 encoding):
# 'Hello' | ',' | ' world' | '!'  →  4 tokens
_STRING = "Hello, world!"
_OPENAI_TOKEN_COUNT = 4


def test_tiktoken_exact_count_for_fixed_string():
    tok = get_tokenizer("openai", "gpt-4")
    assert tok.count(_STRING) == _OPENAI_TOKEN_COUNT


def test_tiktoken_used_as_estimate_for_anthropic_models():
    # estimate mode: tiktoken proxy — local, free, ceiling ~10% error
    tok = get_tokenizer("anthropic", "claude-3-5-sonnet-20241022", accuracy="estimate")
    count = tok.count(_STRING)
    assert isinstance(count, int)
    assert count > 0


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="requires ANTHROPIC_API_KEY — exact mode calls messages.count_tokens()",
)
def test_anthropic_exact_count_via_api():
    tok = get_tokenizer("anthropic", "claude-3-5-sonnet-20241022", accuracy="exact")
    count = tok.count(_STRING)
    assert isinstance(count, int)
    assert count > 0
    # ponytail: pin exact int value here after first authenticated run
