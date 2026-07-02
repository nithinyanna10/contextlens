"""Step 3: Malformed JSONL and CLI robustness stress tests.

All offline. Target: fails loud with nonzero exit and a message that names
the problem. Never parses into wrong numbers.

The key distinction:
  - Missing optional field with a default (e.g. cache fields) → loads fine.
  - Missing required field or invalid JSON → fails loud, exit 2, stderr names it.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from contextlens.report import dump_call, load_call, load_calls
from contextlens.capture import CapturedCall, TokenSpan, UsageRecord

pytestmark = pytest.mark.stress


# ---------------------------------------------------------------------------
# helper — minimal valid serialized CapturedCall
# ---------------------------------------------------------------------------

def _valid_line() -> str:
    call = CapturedCall(
        provider="openai",
        model="gpt-4o",
        call_index=0,
        prompt="hello",
        spans=[
            TokenSpan(component="system", text="hello", token_count=2),
            TokenSpan(component="formatting", text="", token_count=4),
        ],
        usage=UsageRecord(provider="openai", input_tokens=6, source="response_usage"),
    )
    return dump_call(call)


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", "contextlens"] + args,
        capture_output=True,
        text=True,
        cwd="/Users/nithinyanna/Downloads/local/contextlens",
    )


# ---------------------------------------------------------------------------
# Load-level: what load_call does with bad input
# ---------------------------------------------------------------------------

class TestLoadCallRejection:
    def test_truncated_json_raises_validation_error(self):
        with pytest.raises(ValidationError, match="JSON"):
            load_call('{"provider":"openai","model":"gpt-4o"')

    def test_valid_json_wrong_shape_raises_validation_error(self):
        with pytest.raises(ValidationError, match="Field required"):
            load_call('{"provider":"openai","model":"gpt-4o","call_index":0}')

    def test_garbage_string_raises_validation_error(self):
        with pytest.raises(ValidationError):
            load_call("not json at all")

    def test_valid_json_missing_cache_fields_loads_fine(self):
        """Missing cache fields have defaults — this is legal (old JSONL compatibility)."""
        line = json.dumps({
            "provider": "openai",
            "model": "gpt-4o",
            "call_index": 0,
            "prompt": "hello",
            "spans": [
                {"component": "system", "text": "hello", "token_count": 2},
                {"component": "formatting", "text": "", "token_count": 4},
            ],
            "usage": {
                "provider": "openai",
                "input_tokens": 6,
                "source": "response_usage",
                # no cache_read_input_tokens, no cache_creation_input_tokens
            },
        })
        call = load_call(line)
        assert call.usage.cache_read_input_tokens == 0
        assert call.usage.cache_creation_input_tokens == 0
        assert call.usage.total_input_tokens == 6  # not wrong, just uncached

    def test_missing_optional_usage_field_loads_fine(self):
        """usage=None is valid (pre-flight call)."""
        line = json.dumps({
            "provider": "openai",
            "model": "gpt-4o",
            "call_index": 0,
            "prompt": "hello",
            "spans": [{"component": "system", "text": "hello", "token_count": 2}],
        })
        call = load_call(line)
        assert call.usage is None


class TestLoadCallsFile:
    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert load_calls(f) == []

    def test_blank_lines_are_skipped(self, tmp_path):
        f = tmp_path / "blanks.jsonl"
        f.write_text("\n\n" + _valid_line() + "\n\n")
        calls = load_calls(f)
        assert len(calls) == 1

    def test_mixed_valid_garbage_raises_on_garbage_line(self, tmp_path):
        f = tmp_path / "mixed.jsonl"
        f.write_text(_valid_line() + "\nnot json\n" + _valid_line())
        with pytest.raises(ValidationError):
            load_calls(f)

    def test_garbage_line_does_not_silently_skip(self, tmp_path):
        """Must raise, not return a partial result with 2 calls instead of 3."""
        f = tmp_path / "partial.jsonl"
        f.write_text(_valid_line() + "\n{\"x\":1}\n" + _valid_line())
        with pytest.raises(ValidationError):
            load_calls(f)


# ---------------------------------------------------------------------------
# CLI level: exit codes and stderr messages
# ---------------------------------------------------------------------------

class TestCliErrors:
    def test_empty_file_exits_zero(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        result = _run_cli(["report", str(f)])
        assert result.returncode == 0

    def test_garbage_file_exits_two_and_names_file(self, tmp_path):
        f = tmp_path / "garbage.jsonl"
        f.write_text("not json at all")
        result = _run_cli(["report", str(f)])
        assert result.returncode == 2
        assert "contextlens: error reading" in result.stderr
        assert str(f) in result.stderr

    def test_truncated_json_exits_two(self, tmp_path):
        f = tmp_path / "truncated.jsonl"
        f.write_text('{"provider":"openai","model":"gpt-4o"')
        result = _run_cli(["report", str(f)])
        assert result.returncode == 2
        assert "contextlens: error reading" in result.stderr

    def test_wrong_shape_exits_two(self, tmp_path):
        f = tmp_path / "wrong_shape.jsonl"
        f.write_text('{"provider":"openai","model":"gpt-4o","call_index":0}')
        result = _run_cli(["report", str(f)])
        assert result.returncode == 2
        assert "contextlens: error reading" in result.stderr

    def test_valid_file_no_violations_exits_zero(self, tmp_path):
        f = tmp_path / "valid.jsonl"
        f.write_text(_valid_line())
        result = _run_cli(["report", str(f)])
        assert result.returncode == 0

    def test_old_jsonl_without_cache_fields_exits_zero(self, tmp_path):
        """Old JSONL (pre-cache-accounting) must load and run without error."""
        f = tmp_path / "old.jsonl"
        f.write_text(json.dumps({
            "provider": "openai",
            "model": "gpt-4o",
            "call_index": 0,
            "prompt": "hello",
            "spans": [
                {"component": "system", "text": "hello", "token_count": 2},
                {"component": "formatting", "text": "", "token_count": 4},
            ],
            "usage": {
                "provider": "openai",
                "input_tokens": 6,
                "source": "response_usage",
            },
        }))
        result = _run_cli(["report", str(f)])
        assert result.returncode == 0
        assert "contextlens: error" not in result.stderr

    def test_nonexistent_file_exits_nonzero(self, tmp_path):
        result = _run_cli(["report", str(tmp_path / "nonexistent.jsonl")])
        assert result.returncode != 0

    def test_violations_exit_one_not_two(self, tmp_path):
        """Exit 1 = violations. Exit 2 = input error. Must be distinct.

        Default contract runs rot_risk_below=0.9. gpt-4o window = 128k.
        Use 120k tokens = 93.75% > 90% to trigger a violation.
        """
        call = CapturedCall(
            provider="openai",
            model="gpt-4o",
            call_index=0,
            prompt="x",
            spans=[
                TokenSpan(component="system", text="x", token_count=119_994),
                TokenSpan(component="formatting", text="", token_count=6),
            ],
            usage=UsageRecord(provider="openai", input_tokens=120_000, source="response_usage"),
        )
        f = tmp_path / "violations.jsonl"
        f.write_text(dump_call(call))
        result = _run_cli(["report", str(f)])
        assert result.returncode == 1  # violations, not input error
