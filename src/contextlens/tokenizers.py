from __future__ import annotations

from typing import Literal, Protocol

from contextlens.capture import ProviderName

AccuracyMode = Literal["estimate", "exact"]


class TokenizerAdapter(Protocol):
    def count(self, text: str) -> int: ...


class _TiktokenAdapter:
    """Local, deterministic, free. Authoritative for OpenAI; proxy for Anthropic.

    For Anthropic models, ceiling is ~10% error vs provider total, higher for
    code and non-English content. Use accuracy="exact" for authoritative counts.
    """

    _FALLBACK_ENCODING = "cl100k_base"

    def __init__(self, model: str) -> None:
        import tiktoken

        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding(self._FALLBACK_ENCODING)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


class _AnthropicAdapter:
    """Calls messages.count_tokens() — exact, but requires network + ANTHROPIC_API_KEY.

    ponytail: network call; never the default. Use accuracy="exact" to opt in.
    """

    def __init__(self, model: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model

    def count(self, text: str) -> int:
        resp = self._client.messages.count_tokens(
            model=self._model,
            messages=[{"role": "user", "content": text}],
        )
        return resp.input_tokens


def get_tokenizer(
    provider: ProviderName,
    model: str,
    accuracy: AccuracyMode = "estimate",
) -> TokenizerAdapter:
    """Return a tokenizer adapter for the given provider and model.

    accuracy="estimate"  — tiktoken, local, free. Default. ESTIMATE for Anthropic.
    accuracy="exact"     — Anthropic only; calls messages.count_tokens() over the
                           network. Requires ANTHROPIC_API_KEY. Never the default.
    """
    if provider == "openai" or accuracy == "estimate":
        # For anthropic in estimate mode, use gpt-4 encoding as proxy.
        proxy_model = model if provider == "openai" else "gpt-4"
        return _TiktokenAdapter(proxy_model)
    return _AnthropicAdapter(model)
