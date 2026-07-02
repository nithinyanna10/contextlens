from __future__ import annotations

import warnings
from typing import Any

from contextlens.capture import CapturedCall, TokenSpan, UsageRecord
from contextlens.tokenizers import get_tokenizer

# Message type to ComponentTag mapping
MESSAGE_TYPE_MAP = {
    "system": "system",
    "human": "history",
    "ai": "history",
    "tool": "tool_output",
    "function": "tool_output",
}


class ContextLensCallbackHandler:
    """LangChain callback handler that captures LLM calls for context-window analysis.

    Usage with LangChain:
        handler = ContextLensCallbackHandler()
        response = chat_model.invoke(messages, config={"callbacks": [handler]})
        calls = handler.calls  # list of CapturedCall objects
    """

    def __init__(self) -> None:
        self.calls: list[CapturedCall] = []
        self._pending_messages: list[Any] = []
        self._pending_serialized: dict[str, Any] = {}
        self._call_index: int = 0

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any
    ) -> None:
        """Capture messages and model info at the start of a chat model call.

        Args:
            serialized: Dict containing model info, typically {"name": "ChatOpenAI", "kwargs": {...}}
            messages: List of message lists; we capture messages[0] (the prompt)
            **kwargs: Additional LangChain-provided context (unused)
        """
        self._pending_serialized = serialized
        self._pending_messages = messages[0] if messages else []

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Build and store a CapturedCall when the LLM response is received.

        Args:
            response: LangChain response object with .llm_output containing usage
            **kwargs: Additional LangChain-provided context (unused)
        """
        if not self._pending_messages:
            return

        # Extract provider from serialized name
        serialized_name = self._pending_serialized.get("name", "")
        if "Anthropic" in serialized_name:
            provider = "anthropic"
        elif "OpenAI" in serialized_name:
            provider = "openai"
        else:
            warnings.warn(
                f"contextlens: unrecognized LangChain model class {serialized_name!r}; "
                "provider set to 'unknown'. Usage will be None and token counts will be "
                "estimates — assertions degraded from exact to estimate.",
                stacklevel=2,
            )
            provider = "unknown"

        # Extract model name from kwargs
        kwargs_dict = self._pending_serialized.get("kwargs", {})
        model = kwargs_dict.get("model_name") or kwargs_dict.get("model") or "unknown"

        # Extract usage from response.llm_output
        llm_output = getattr(response, "llm_output", {}) or {}
        input_tokens: int | None = None
        cache_read: int = 0
        cache_creation: int = 0

        if provider == "openai":
            token_usage = llm_output.get("token_usage", {})
            input_tokens = token_usage.get("prompt_tokens")
            # LangChain may expose cached breakdown under token_usage or nested details.
            cache_read = int(
                token_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) or 0
            )
            if input_tokens is not None:
                input_tokens = input_tokens - cache_read
        elif provider == "anthropic":
            usage_dict = llm_output.get("usage", {})
            input_tokens = usage_dict.get("input_tokens")
            cache_read = int(usage_dict.get("cache_read_input_tokens", 0) or 0)
            cache_creation = int(usage_dict.get("cache_creation_input_tokens", 0) or 0)
        # unknown: all stay None/0 → usage=None

        # Build spans from messages
        spans: list[TokenSpan] = []
        tokenizer = get_tokenizer(provider, model, accuracy="estimate")

        content_total = 0
        for msg in self._pending_messages:
            component = MESSAGE_TYPE_MAP.get(msg.type, "scratchpad")
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            token_count = tokenizer.count(text)
            spans.append(TokenSpan(component=component, text=text, token_count=token_count))
            content_total += token_count

        # Add formatting span to reconcile with usage.total_input_tokens
        if input_tokens is not None:
            total_tokens = input_tokens + cache_read + cache_creation
            formatting_overhead = total_tokens - content_total
            spans.append(
                TokenSpan(component="formatting", text="", token_count=formatting_overhead)
            )

        # Build prompt string from messages
        prompt = "\n\n".join(
            f"[{msg.type}] {msg.content}" for msg in self._pending_messages
        )

        # Build usage record
        usage = (
            UsageRecord(
                provider=provider,
                input_tokens=input_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_creation,
                source="response_usage",
            )
            if input_tokens is not None
            else None
        )

        # Build and append CapturedCall
        call = CapturedCall(
            provider=provider,
            model=model,
            call_index=self._call_index,
            prompt=prompt,
            spans=spans,
            usage=usage,
        )
        self.calls.append(call)
        self._call_index += 1

        # Reset pending state for next call
        self._pending_messages = []
        self._pending_serialized = {}
