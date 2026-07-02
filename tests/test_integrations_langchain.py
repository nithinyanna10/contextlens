import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from contextlens.integrations.langchain import ContextLensCallbackHandler


def test_langchain_handler_openai_path():
    """Test OpenAI message flow: capture messages, extract usage, build spans."""
    # Create mock messages
    sys_msg = SimpleNamespace(type="system", content="You are a helpful assistant.")
    human_msg = SimpleNamespace(type="human", content="What is RAG?")

    # Mock serialized dict (ChatOpenAI)
    serialized = {"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-4o"}}

    # Mock response with OpenAI llm_output structure
    response = MagicMock()
    response.llm_output = {
        "token_usage": {
            "prompt_tokens": 35,
            "completion_tokens": 10,
            "total_tokens": 45,
        },
        "model_name": "gpt-4o",
    }

    # Create handler and simulate the callback flow
    handler = ContextLensCallbackHandler()
    handler.on_chat_model_start(serialized, [[sys_msg, human_msg]])
    handler.on_llm_end(response)

    # Verify the captured call
    assert len(handler.calls) == 1
    call = handler.calls[0]

    # Provider and model
    assert call.provider == "openai"
    assert call.model == "gpt-4o"

    # Call index
    assert call.call_index == 0

    # Usage
    assert call.usage is not None
    assert call.usage.provider == "openai"
    assert call.usage.input_tokens == 35
    assert call.usage.source == "response_usage"

    # Spans
    assert len(call.spans) >= 3  # system, human, formatting
    components = [s.component for s in call.spans]
    assert "system" in components
    assert "history" in components
    assert "formatting" in components
    assert "retrieved" not in components
    assert "tool_output" not in components

    # Reconciliation: sum of all spans == input_tokens
    span_total = sum(s.token_count for s in call.spans)
    assert span_total == 35

    # Prompt format
    assert "[system]" in call.prompt
    assert "[human]" in call.prompt


def test_langchain_handler_anthropic_path():
    """Test Anthropic message flow: usage extraction from different llm_output structure."""
    # Create mock messages
    sys_msg = SimpleNamespace(type="system", content="You are helpful.")
    human_msg = SimpleNamespace(type="human", content="Hello?")

    # Mock serialized dict (ChatAnthropic)
    serialized = {"name": "ChatAnthropic", "kwargs": {"model": "claude-3-5-sonnet-20241022"}}

    # Mock response with Anthropic llm_output structure
    response = MagicMock()
    response.llm_output = {
        "usage": {
            "input_tokens": 28,
            "output_tokens": 5,
        },
        "model": "claude-3-5-sonnet-20241022",
    }

    # Create handler and simulate callback flow
    handler = ContextLensCallbackHandler()
    handler.on_chat_model_start(serialized, [[sys_msg, human_msg]])
    handler.on_llm_end(response)

    # Verify the captured call
    assert len(handler.calls) == 1
    call = handler.calls[0]

    # Provider and model
    assert call.provider == "anthropic"
    assert call.model == "claude-3-5-sonnet-20241022"

    # Usage
    assert call.usage is not None
    assert call.usage.provider == "anthropic"
    assert call.usage.input_tokens == 28
    assert call.usage.source == "response_usage"

    # Reconciliation
    span_total = sum(s.token_count for s in call.spans)
    assert span_total == 28


def test_langchain_handler_no_usage():
    """Test preflight scenario where llm_output has no usage data."""
    sys_msg = SimpleNamespace(type="system", content="System prompt.")
    human_msg = SimpleNamespace(type="human", content="User query.")

    serialized = {"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-4o"}}

    # Response with no token_usage (preflight)
    response = MagicMock()
    response.llm_output = {"model_name": "gpt-4o"}  # no token_usage

    handler = ContextLensCallbackHandler()
    handler.on_chat_model_start(serialized, [[sys_msg, human_msg]])
    handler.on_llm_end(response)

    # Should still capture a call, but with usage=None
    assert len(handler.calls) == 1
    call = handler.calls[0]
    assert call.usage is None
    # Without usage, there should be no formatting span
    assert not any(s.component == "formatting" for s in call.spans)


def test_langchain_handler_message_type_mapping():
    """Verify all message types map to correct ComponentTag."""
    messages = [
        SimpleNamespace(type="system", content="sys"),
        SimpleNamespace(type="human", content="human"),
        SimpleNamespace(type="ai", content="ai"),
        SimpleNamespace(type="tool", content="tool"),
        SimpleNamespace(type="function", content="function"),
    ]

    serialized = {"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-4"}}
    response = MagicMock()
    response.llm_output = {"token_usage": {"prompt_tokens": 100}, "model_name": "gpt-4"}

    handler = ContextLensCallbackHandler()
    handler.on_chat_model_start(serialized, [messages])
    handler.on_llm_end(response)

    call = handler.calls[0]
    component_map = {s.text: s.component for s in call.spans if s.component != "formatting"}

    assert component_map["sys"] == "system"
    assert component_map["human"] == "history"
    assert component_map["ai"] == "history"
    assert component_map["tool"] == "tool_output"
    assert component_map["function"] == "tool_output"


def test_langchain_handler_unknown_message_type():
    """Unknown message types map to 'scratchpad'."""
    msg = SimpleNamespace(type="unknown_type", content="unknown content")

    serialized = {"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-4"}}
    response = MagicMock()
    response.llm_output = {"token_usage": {"prompt_tokens": 50}, "model_name": "gpt-4"}

    handler = ContextLensCallbackHandler()
    handler.on_chat_model_start(serialized, [[msg]])
    handler.on_llm_end(response)

    call = handler.calls[0]
    # Find the span that is not formatting
    content_spans = [s for s in call.spans if s.component != "formatting"]
    assert len(content_spans) == 1
    assert content_spans[0].component == "scratchpad"


def test_langchain_handler_multiple_calls():
    """Handler maintains call_index across multiple invocations."""
    serialized = {"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-4"}}
    msg = SimpleNamespace(type="human", content="test")

    handler = ContextLensCallbackHandler()

    for i in range(3):
        response = MagicMock()
        response.llm_output = {"token_usage": {"prompt_tokens": 10}, "model_name": "gpt-4"}
        handler.on_chat_model_start(serialized, [[msg]])
        handler.on_llm_end(response)

    assert len(handler.calls) == 3
    assert handler.calls[0].call_index == 0
    assert handler.calls[1].call_index == 1
    assert handler.calls[2].call_index == 2


def test_langchain_handler_model_name_extraction():
    """Test both model_name (OpenAI) and model (Anthropic) keys."""
    msg = SimpleNamespace(type="human", content="test")

    # OpenAI uses model_name
    serialized_openai = {"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-4o"}}
    response = MagicMock()
    response.llm_output = {"token_usage": {"prompt_tokens": 10}, "model_name": "gpt-4o"}

    handler = ContextLensCallbackHandler()
    handler.on_chat_model_start(serialized_openai, [[msg]])
    handler.on_llm_end(response)

    assert handler.calls[0].model == "gpt-4o"

    # Anthropic uses model
    serialized_anthropic = {"name": "ChatAnthropic", "kwargs": {"model": "claude-3-sonnet"}}
    response.llm_output = {"usage": {"input_tokens": 10}}

    handler2 = ContextLensCallbackHandler()
    handler2.on_chat_model_start(serialized_anthropic, [[msg]])
    handler2.on_llm_end(response)

    assert handler2.calls[0].model == "claude-3-sonnet"


def test_langchain_handler_empty_messages():
    """Handler gracefully handles empty message list."""
    serialized = {"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-4"}}
    response = MagicMock()
    response.llm_output = {"token_usage": {"prompt_tokens": 0}}

    handler = ContextLensCallbackHandler()
    handler.on_chat_model_start(serialized, [])  # empty messages
    handler.on_llm_end(response)

    # Should not add a call if there are no messages
    assert len(handler.calls) == 0


def test_langchain_handler_prompt_formatting():
    """Verify prompt string format."""
    msg1 = SimpleNamespace(type="system", content="You are helpful.")
    msg2 = SimpleNamespace(type="human", content="What is RAG?")

    serialized = {"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-4"}}
    response = MagicMock()
    response.llm_output = {"token_usage": {"prompt_tokens": 100}}

    handler = ContextLensCallbackHandler()
    handler.on_chat_model_start(serialized, [[msg1, msg2]])
    handler.on_llm_end(response)

    prompt = handler.calls[0].prompt
    assert "[system] You are helpful." in prompt
    assert "[human] What is RAG?" in prompt
    assert "\n\n" in prompt


def test_langchain_handler_warns_on_unrecognized_provider():
    """Unknown class name fires a UserWarning naming the class."""
    handler = ContextLensCallbackHandler()
    serialized = {"name": "CustomLLMWrapper", "kwargs": {"model_name": "custom-model"}}
    msg = SimpleNamespace(type="human", content="Hello.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        handler.on_chat_model_start(serialized, [[msg]])
        response = MagicMock()
        response.llm_output = {}  # no usage — pre-flight fallback
        handler.on_llm_end(response)

    assert len(caught) == 1
    w = caught[0]
    assert issubclass(w.category, UserWarning)
    assert "CustomLLMWrapper" in str(w.message)

    # Call produced; provider is "unknown", usage is None (degraded to estimate)
    assert len(handler.calls) == 1
    assert handler.calls[0].provider == "unknown"
    assert handler.calls[0].usage is None
