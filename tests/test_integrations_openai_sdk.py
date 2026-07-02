"""Tests for openai_sdk integration."""

from __future__ import annotations

from unittest.mock import MagicMock

from contextlens.integrations.openai_sdk import Capture, build_captured_call


def test_build_captured_call_with_system_and_user_messages():
    """Test build_captured_call with system and user messages."""
    # Build a mock response with usage (no caching)
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 42
    mock_response.usage.prompt_tokens_details = None
    mock_response.model = "gpt-4o"

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
    ]

    call = build_captured_call("gpt-4o", messages, mock_response, call_index=0)

    # Assertions
    assert call.provider == "openai"
    assert call.model == "gpt-4o"
    assert call.call_index == 0
    assert call.usage is not None
    assert call.usage.source == "response_usage"
    assert call.usage.input_tokens == 42

    # Check span components
    components = {s.component for s in call.spans}
    assert "system" in components
    assert "history" in components
    assert "formatting" in components
    assert "retrieved" not in components
    assert "tool_output" not in components

    # Reconciliation: sum of all spans == total_input_tokens (no cache here, so == 42)
    span_sum = sum(s.token_count for s in call.spans)
    assert span_sum == call.usage.total_input_tokens
    assert call.usage.total_input_tokens == 42


def test_capture_class_auto_increments_call_index():
    """Test that Capture class auto-increments call_index across calls."""
    # Build a mock client
    mock_client = MagicMock()

    # Create mock responses for two calls
    mock_response1 = MagicMock()
    mock_response1.usage.prompt_tokens = 20
    mock_response1.usage.prompt_tokens_details = None
    mock_response1.model = "gpt-4o"

    mock_response2 = MagicMock()
    mock_response2.usage.prompt_tokens = 30
    mock_response2.usage.prompt_tokens_details = None
    mock_response2.model = "gpt-4o"

    mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]

    # Create Capture wrapper
    capture = Capture(mock_client)

    messages1 = [{"role": "user", "content": "First call"}]
    messages2 = [{"role": "user", "content": "Second call"}]

    # Make two calls
    response1 = capture.create(model="gpt-4o", messages=messages1)
    response2 = capture.create(model="gpt-4o", messages=messages2)

    # Verify responses are unchanged
    assert response1 is mock_response1
    assert response2 is mock_response2

    # Verify call_index auto-incremented
    assert len(capture.calls) == 2
    assert capture.calls[0].call_index == 0
    assert capture.calls[1].call_index == 1

    # Verify both calls are captured
    assert capture.calls[0].usage.input_tokens == 20
    assert capture.calls[1].usage.input_tokens == 30
