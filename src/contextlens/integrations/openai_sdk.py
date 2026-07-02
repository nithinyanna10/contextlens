"""OpenAI SDK integration for contextlens.

Exports:
  - build_captured_call(model, messages, response, call_index): standalone function
  - Capture: wraps an openai.OpenAI client and auto-increments call_index
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from contextlens.capture import CapturedCall, TokenSpan, UsageRecord
from contextlens.tokenizers import get_tokenizer

if TYPE_CHECKING:
    pass  # openai not imported at module level to avoid hard dependency


# Role → ComponentTag mapping per spec
ROLE_MAP: dict[str, str] = {
    "system": "system",
    "user": "history",
    "assistant": "history",
    "tool": "tool_output",
    "function": "tool_output",
}


def build_captured_call(
    model: str,
    messages: list[dict[str, Any]],
    response: Any,
    call_index: int = 0,
) -> CapturedCall:
    """Build a CapturedCall from an OpenAI chat completion response.

    Args:
        model: The model string (e.g., "gpt-4o")
        messages: The messages list passed to chat.completions.create
        response: The ChatCompletion response object
        call_index: 0-based call index within the session

    Returns:
        A CapturedCall with spans reconciled to response.usage.prompt_tokens
    """
    tokenizer = get_tokenizer("openai", model)

    # Build spans for each message
    spans: list[TokenSpan] = []
    content_total = 0

    for msg in messages:
        role = msg.get("role", "scratchpad")
        content = msg.get("content", "")

        # Determine component tag
        component = ROLE_MAP.get(role, "scratchpad")

        # Count tokens for this content
        if isinstance(content, str):
            token_count = tokenizer.count(content)
        else:
            # Handle non-string content (e.g., tool_calls, images)
            # For now, treat as empty
            token_count = 0
            content = ""

        spans.append(
            TokenSpan(
                component=component,
                text=content,
                token_count=token_count,
            )
        )
        content_total += token_count

    # Build prompt string
    prompt = "\n\n".join(
        f"[{msg.get('role', 'unknown')}] {msg.get('content', '')}"
        for msg in messages
    )

    # Handle usage
    usage: UsageRecord | None = None
    if response.usage is not None:
        total_prompt = response.usage.prompt_tokens
        # prompt_tokens_details.cached_tokens is the portion served from cache.
        # Absent on older response objects; default to 0.
        details = getattr(response.usage, "prompt_tokens_details", None)
        cache_read = int(getattr(details, "cached_tokens", 0) or 0)
        new_tokens = total_prompt - cache_read
        formatting_overhead = total_prompt - content_total
        spans.append(
            TokenSpan(
                component="formatting",
                text="",
                token_count=formatting_overhead,
            )
        )
        usage = UsageRecord(
            provider="openai",
            input_tokens=new_tokens,
            cache_read_input_tokens=cache_read,
            source="response_usage",
        )

    return CapturedCall(
        provider="openai",
        model=model,
        call_index=call_index,
        prompt=prompt,
        spans=spans,
        usage=usage,
    )


class Capture:
    """Wraps an openai.OpenAI client to automatically capture calls."""

    def __init__(self, client: Any) -> None:
        """Initialize with an OpenAI client.

        Args:
            client: An openai.OpenAI instance
        """
        self._client = client
        self.calls: list[CapturedCall] = []

    def create(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Calls client.chat.completions.create and captures the call.

        Args:
            model: Model identifier
            messages: Messages list
            **kwargs: Additional arguments to pass to create()

        Returns:
            The original response unchanged
        """
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        call_index = len(self.calls)
        captured = build_captured_call(model, messages, response, call_index)
        self.calls.append(captured)
        return response
