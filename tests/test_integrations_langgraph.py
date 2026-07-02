"""Tests for LangGraph integration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from contextlens.integrations.langgraph import NodeCapture, capture_node


def test_capture_node_comprehensive():
    """capture_node returns a NodeCapture; calls accumulate on the instance."""
    sys_msg = SimpleNamespace(type="system", content="You are a helpful agent.")
    human_msg = SimpleNamespace(type="human", content="Summarise the document.")
    tool_msg = SimpleNamespace(type="tool", content="Document text: RAG is retrieval-augmented generation.")

    ai_response = SimpleNamespace(
        type="ai",
        content="RAG stands for retrieval-augmented generation.",
        response_metadata={"token_usage": {"prompt_tokens": 55, "completion_tokens": 20}},
    )

    def node_fn(state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [ai_response]}

    node = capture_node(node_fn, model="gpt-4o", provider="openai")
    result = node({"messages": [sys_msg, human_msg, tool_msg]})

    assert result == {"messages": [ai_response]}
    assert len(node.calls) == 1
    call = node.calls[0]

    assert call.provider == "openai"
    assert call.model == "gpt-4o"
    assert call.call_index == 0
    assert call.usage is not None
    assert call.usage.source == "response_usage"
    assert call.usage.input_tokens == 55

    components = {s.component for s in call.spans}
    assert "system" in components
    assert "history" in components
    assert "tool_output" in components
    assert "formatting" in components
    assert "ai" not in components  # AI response is output, not a span

    assert sum(s.token_count for s in call.spans) == 55

    assert "[system] You are a helpful agent." in call.prompt
    assert "[human] Summarise the document." in call.prompt
    assert "[tool] Document text: RAG is retrieval-augmented generation." in call.prompt


def test_capture_node_no_usage():
    """No usage metadata → usage=None, no formatting span."""
    msg1 = SimpleNamespace(type="system", content="You are helpful.")
    msg2 = SimpleNamespace(type="human", content="Hello.")
    ai_response = SimpleNamespace(type="ai", content="Hi there!", response_metadata={})

    def node_fn(state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [ai_response]}

    node = capture_node(node_fn, model="gpt-4o", provider="openai")
    node({"messages": [msg1, msg2]})

    assert len(node.calls) == 1
    call = node.calls[0]
    assert call.usage is None
    components = {s.component for s in call.spans}
    assert "system" in components
    assert "history" in components
    assert "formatting" not in components


def test_capture_node_increments_call_index_on_instance():
    """Calling the same NodeCapture twice gives call_index 0 then 1."""

    def node_fn(state: dict[str, Any]) -> dict[str, Any]:
        msg = SimpleNamespace(
            type="ai",
            content="Response",
            response_metadata={"token_usage": {"prompt_tokens": 10}},
        )
        return {"messages": [msg]}

    node = capture_node(node_fn, model="gpt-4o", provider="openai")

    node({"messages": [SimpleNamespace(type="human", content="Q1")]})
    node({"messages": [SimpleNamespace(type="human", content="Q2")]})

    assert len(node.calls) == 2
    assert node.calls[0].call_index == 0
    assert node.calls[1].call_index == 1


def test_two_instances_do_not_share_state():
    """Two NodeCapture instances are fully isolated — no shared global."""

    def node_fn(state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [SimpleNamespace(type="ai", content="R", response_metadata={})]}

    node_a = capture_node(node_fn, model="gpt-4o", provider="openai")
    node_b = capture_node(node_fn, model="gpt-4o", provider="openai")

    node_a({"messages": [SimpleNamespace(type="human", content="A")]})

    assert len(node_a.calls) == 1
    assert len(node_b.calls) == 0  # B untouched
