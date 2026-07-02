"""LangGraph integration for contextlens.

Wraps LangGraph node functions to capture LLM calls. Exports:
  - capture_node(node_fn, *, model, provider): returns a NodeCapture instance
  - NodeCapture: callable wrapper; stores captured calls on .calls (instance-local)
"""

from __future__ import annotations

from typing import Any, Callable

from contextlens.capture import CapturedCall, TokenSpan, UsageRecord
from contextlens.tokenizers import get_tokenizer

MESSAGE_TYPE_MAP: dict[str, str] = {
    "system": "system",
    "human": "history",
    "ai": "history",
    "tool": "tool_output",
    "function": "tool_output",
}


class NodeCapture:
    """Wraps a LangGraph node function; stores captured calls on the instance.

    No shared global — each NodeCapture is isolated, safe for xdist-parallel CI.

    Usage:
        agent_node = capture_node(call_model, model="gpt-4o", provider="openai")
        # wire agent_node into the graph as usual
        call = agent_node.calls[0]
    """

    def __init__(
        self,
        node_fn: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        model: str,
        provider: str = "openai",
    ) -> None:
        self._node_fn = node_fn
        self._model = model
        self._provider = provider
        self.calls: list[CapturedCall] = []

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        input_messages = state.get("messages", [])
        result = self._node_fn(state)
        output_messages = result.get("messages", [])
        response_msg = output_messages[-1] if output_messages else None

        call = _build_call(
            model=self._model,
            provider=self._provider,
            input_messages=input_messages,
            response_msg=response_msg,
            call_index=len(self.calls),
        )
        self.calls.append(call)
        return result


def capture_node(
    node_fn: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    model: str,
    provider: str = "openai",
) -> NodeCapture:
    """Wrap a LangGraph node function to capture LLM calls.

    Returns a NodeCapture: callable (drop-in for node_fn) with .calls on the
    instance. No global state; safe when tests run in parallel workers.
    """
    return NodeCapture(node_fn, model=model, provider=provider)


def _extract_usage_from_response(
    response_msg: Any,
) -> tuple[int, int, int, str] | None:
    """Extract token counts from an AI response message.

    Returns (input_tokens, cache_read, cache_creation, source) or None.
    input_tokens is the raw uncached count; cache fields default to 0 when absent.

    Tries multiple paths for compatibility with different LangChain versions:
      1. response_metadata["usage"]      — Anthropic (has cache fields)
      2. response_metadata["token_usage"] — OpenAI (cached_tokens in details)
      3. usage_metadata                  — newer LangChain unified format
    """
    if response_msg is None:
        return None

    if hasattr(response_msg, "response_metadata"):
        metadata = response_msg.response_metadata
        if isinstance(metadata, dict):
            # Path 1: Anthropic via response_metadata["usage"]
            usage_dict = metadata.get("usage", {})
            if isinstance(usage_dict, dict) and "input_tokens" in usage_dict:
                return (
                    int(usage_dict["input_tokens"]),
                    int(usage_dict.get("cache_read_input_tokens", 0) or 0),
                    int(usage_dict.get("cache_creation_input_tokens", 0) or 0),
                    "response_usage",
                )

            # Path 2: OpenAI via response_metadata["token_usage"]
            token_usage = metadata.get("token_usage", {})
            if isinstance(token_usage, dict) and "prompt_tokens" in token_usage:
                total_prompt = int(token_usage["prompt_tokens"])
                cache_read = int(
                    token_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) or 0
                )
                return (total_prompt - cache_read, cache_read, 0, "response_usage")

    # Path 3: newer LangChain usage_metadata (unified format)
    if hasattr(response_msg, "usage_metadata"):
        usage_meta = response_msg.usage_metadata
        if isinstance(usage_meta, dict) and "input_tokens" in usage_meta:
            details = usage_meta.get("input_token_details", {}) or {}
            cache_read = int(details.get("cache_read", 0) or 0)
            cache_creation = int(details.get("cache_creation", 0) or 0)
            return (
                int(usage_meta["input_tokens"]),
                cache_read,
                cache_creation,
                "response_usage",
            )

    return None


def _build_call(
    model: str,
    provider: str,
    input_messages: list[Any],
    response_msg: Any,
    call_index: int,
) -> CapturedCall:
    tokenizer = get_tokenizer(provider, model)  # type: ignore[arg-type]

    spans: list[TokenSpan] = []
    content_total = 0

    for msg in input_messages:
        msg_type = getattr(msg, "type", "scratchpad")
        msg_content = getattr(msg, "content", "")
        component = MESSAGE_TYPE_MAP.get(msg_type, "scratchpad")

        if isinstance(msg_content, str):
            token_count = tokenizer.count(msg_content)
        else:
            token_count = 0
            msg_content = ""

        spans.append(TokenSpan(component=component, text=msg_content, token_count=token_count))
        content_total += token_count

    prompt = "\n\n".join(
        f"[{getattr(msg, 'type', 'unknown')}] {getattr(msg, 'content', '')}"
        for msg in input_messages
    )

    usage: UsageRecord | None = None
    if response_msg is not None:
        usage_result = _extract_usage_from_response(response_msg)
        if usage_result is not None:
            input_tokens, cache_read, cache_creation, source = usage_result
            total_tokens = input_tokens + cache_read + cache_creation
            spans.append(
                TokenSpan(component="formatting", text="", token_count=total_tokens - content_total)
            )
            usage = UsageRecord(
                provider=provider,  # type: ignore[arg-type]
                input_tokens=input_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_creation,
                source=source,  # type: ignore[arg-type]
            )

    return CapturedCall(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        call_index=call_index,
        prompt=prompt,
        spans=spans,
        usage=usage,
    )
