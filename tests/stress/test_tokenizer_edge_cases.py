"""Step 4: Tokenizer edge cases and check_position char-offset proxy limits.

All offline (tiktoken is local). Target: document actual behavior, especially
where char-offset and token-offset diverge. These tests assert what the code
*does*, not what it ideally *should* do — where the proxy is wrong, the test
names the failure mode and the README honest-limits section covers it.

Key findings from probing (gpt-4o uses o200k_base, not cl100k):
  - CJK (你好世界) encodes at ~0.5 tok/char on o200k — token-SPARSE.
  - Mixed emoji (🔥💯🚀) encodes at ~1.67 tok/char on o200k — token-DENSE.
  - False-negative scenario: 60-char emoji prefix + 1200-char English suffix.
    Fact placed at char 60 (4.8% by char = SAFE per check_position).
    Token position: 100/301 = 33.2% (actually inside 20%-80% danger zone).
    check_position says safe. Reality: fact is in the middle band.
"""

from __future__ import annotations

import pytest

from contextlens.assertions import check_must_contain, check_position
from contextlens.capture import CapturedCall, TokenSpan
from contextlens.tokenizers import get_tokenizer

pytestmark = pytest.mark.stress


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _preflight_call(prompt: str, component: str = "system") -> CapturedCall:
    """Build a pre-flight call (no usage) for testing prompt-level assertions."""
    tok = get_tokenizer("openai", "gpt-4o")
    count = tok.count(prompt)
    return CapturedCall(
        provider="openai",
        model="gpt-4o",
        call_index=0,
        prompt=prompt,
        spans=[TokenSpan(component=component, text=prompt, token_count=count)],
        usage=None,
    )


# ---------------------------------------------------------------------------
# Tokenizer counts — actual o200k_base behavior
# ---------------------------------------------------------------------------

class TestTokenizerEdgeCounts:
    def test_empty_string_counts_zero_tokens(self):
        tok = get_tokenizer("openai", "gpt-4o")
        assert tok.count("") == 0

    def test_whitespace_only_counts_nonzero(self):
        tok = get_tokenizer("openai", "gpt-4o")
        count = tok.count("   \t\n  ")
        assert count > 0

    def test_cjk_tokens_fewer_than_chars_on_o200k(self):
        tok = get_tokenizer("openai", "gpt-4o")
        # gpt-4o uses o200k_base, which encodes CJK at ~0.5 tok/char.
        # CJK is token-SPARSE here, unlike cl100k (gpt-4) where it's token-dense.
        text = "你好世界" * 20   # 80 chars, ~40 tokens
        tokens = tok.count(text)
        chars = len(text)
        assert tokens < chars

    def test_mixed_emoji_tokens_exceed_chars(self):
        tok = get_tokenizer("openai", "gpt-4o")
        # Mixed emoji (🔥💯🚀) encodes at ~1.67 tok/char in o200k_base.
        text = "🔥💯🚀" * 20   # 60 chars, ~100 tokens
        tokens = tok.count(text)
        chars = len(text)
        assert tokens > chars

    def test_dense_code_tokens_below_chars(self):
        tok = get_tokenizer("openai", "gpt-4o")
        text = "def f(x):\n    return x + 1\n" * 20
        tokens = tok.count(text)
        chars = len(text)
        assert tokens < chars

    def test_cjk_and_emoji_ratios_differ(self):
        """Document that o200k encodes CJK and emoji at very different densities."""
        tok = get_tokenizer("openai", "gpt-4o")
        cjk_ratio = tok.count("你好世界" * 25) / (4 * 25)   # ~0.5
        emoji_ratio = tok.count("🔥💯🚀" * 20) / (3 * 20)   # ~1.67
        # Emoji is token-denser than CJK on o200k
        assert emoji_ratio > cjk_ratio
        assert emoji_ratio > 1.0
        assert cjk_ratio < 1.0


# ---------------------------------------------------------------------------
# check_position: char-offset proxy, English (expected behavior)
# ---------------------------------------------------------------------------

class TestPositionEnglish:
    def test_fact_at_start_is_safe(self):
        fact = "GOLDEN_FACT"
        prompt = fact + " " + "padding " * 500
        call = _preflight_call(prompt)
        assert check_position(call, fact, not_in="middle") == []

    def test_fact_at_end_is_safe(self):
        fact = "GOLDEN_FACT"
        prompt = "padding " * 500 + " " + fact
        call = _preflight_call(prompt)
        assert check_position(call, fact, not_in="middle") == []

    def test_fact_in_center_violates(self):
        fact = "GOLDEN_FACT"
        half = "padding " * 250
        prompt = half + fact + half
        call = _preflight_call(prompt)
        violations = check_position(call, fact, not_in="middle")
        assert len(violations) == 1
        assert violations[0].rule == "position"

    def test_absent_fact_returns_no_violation(self):
        call = _preflight_call("this prompt does not have the fact")
        violations = check_position(call, "GOLDEN_FACT", not_in="middle")
        assert violations == []


# ---------------------------------------------------------------------------
# check_position: char-offset proxy limits (documented wrong behavior)
# ---------------------------------------------------------------------------

class TestPositionCharOffsetProxyLimits:
    def test_emoji_prefix_causes_false_negative(self):
        """
        FALSE NEGATIVE — documented limit, not a bug to fix here.

        Prompt: 60 chars mixed emoji + 1200 chars English = 1260 chars total.
        Tokens: ~100 emoji tokens + ~201 English tokens = ~301 total (o200k).

        Fact ('hello world') placed at char 60 (4.8% by char = SAFE).
        Token position: 100/301 = 33.2% (in lost-in-the-middle zone 20%-80%).

        check_position returns no violation (says safe).
        Reality: fact is past the 20% token boundary.

        The failure mode: emoji at ~1.67 tok/char shifts token boundaries
        rightward relative to char position. check_position uses char offset
        as a proxy and misses this.
        """
        emoji_prefix = "🔥💯🚀" * 20   # 60 chars, ~100 tokens (1.67 tok/char)
        eng_suffix   = "hello world " * 100   # 1200 chars, ~201 tokens
        prompt = emoji_prefix + eng_suffix
        fact = "hello world"

        call = _preflight_call(prompt)

        char_ratio = prompt.find(fact) / len(prompt)
        assert char_ratio < 0.10, "setup: fact at char start of English section (< 10% char)"

        violations = check_position(call, fact, not_in="middle")
        # The proxy says safe (no violation) — this is the false negative.
        assert violations == [], (
            "Expected no violation (char-offset proxy says safe at 4.8% char), "
            "but fact is at ~33% by token count (inside 20%-80% danger zone). "
            "Documented limitation of check_position on emoji-heavy prompts."
        )

    def test_cjk_is_token_sparse_on_o200k(self):
        """
        On gpt-4o (o200k_base), CJK encodes at ~0.5 tok/char — token-sparse.
        A CJK-heavy start moves the char/token ratio in the opposite direction
        from emoji: char position overstates token position, so the proxy may
        report facts as in the middle when they are actually safe by token count.
        The false-negative from emoji and the false-positive from CJK are
        symmetric: both stem from the char/token ratio being non-uniform.
        """
        tok = get_tokenizer("openai", "gpt-4o")
        text = "你好世界" * 50   # 200 chars
        tokens = tok.count(text)
        assert tokens < len(text)   # 0.5 tok/char confirmed

    def test_emoji_tokens_exceed_chars_on_o200k(self):
        """Mixed emoji is token-dense on o200k_base (~1.67 tok/char)."""
        tok = get_tokenizer("openai", "gpt-4o")
        text = "🔥💯🚀" * 20
        tokens = tok.count(text)
        assert tokens > len(text)


# ---------------------------------------------------------------------------
# check_must_contain edge cases
# ---------------------------------------------------------------------------

class TestMustContainEdgeCases:
    def test_empty_fact_string_always_present(self):
        """Empty string is a substring of everything."""
        call = _preflight_call("any prompt content")
        assert check_must_contain(call, "") == []

    def test_fact_with_cjk_matched_by_substring(self):
        fact = "合约续签日期为2025-03-01"
        call = _preflight_call("本文件包含如下信息：" + fact + "，请注意。")
        assert check_must_contain(call, fact) == []

    def test_fact_with_cjk_absent_returns_violation(self):
        call = _preflight_call("本文件包含如下信息，请注意。")
        violations = check_must_contain(call, "合约续签日期为2025-03-01")
        assert len(violations) == 1

    def test_whitespace_only_fact(self):
        """Whitespace fact matches whitespace in prompt — not an error."""
        call = _preflight_call("hello world")
        assert check_must_contain(call, " ") == []
