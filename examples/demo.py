"""Synthetic multi-turn contract analyst demo.

Run:
    python examples/demo.py            # writes examples/demo.jsonl
    contextlens report examples/demo.jsonl --contract examples/contract.json
    pytest examples/test_demo.py -v
"""

from __future__ import annotations

from pathlib import Path

from contextlens.capture import CapturedCall, TokenSpan, UsageRecord
from contextlens.report import dump_calls

GOLDEN_FACT = "Contract renewal is due on 2025-03-01."
_SYSTEM_TEXT = f"You are a contract analyst. {GOLDEN_FACT}"


def make_call(call_index: int, history_tokens: int, retrieved_tokens: int) -> CapturedCall:
    system_tokens = 20
    fmt_tokens = 6
    total = system_tokens + history_tokens + retrieved_tokens + fmt_tokens
    return CapturedCall(
        provider="openai",
        model="gpt-4o",
        call_index=call_index,
        prompt=(
            f"[system] {_SYSTEM_TEXT}\n\n"
            f"[history] {'User: any updates? Assistant: reviewing. ' * max(1, history_tokens // 8)}\n\n"
            f"[retrieved] {'Exhibit A clause 7.1: obligations run until termination. ' * max(1, retrieved_tokens // 9)}"
        ),
        spans=[
            TokenSpan(component="system",    text=_SYSTEM_TEXT, token_count=system_tokens),
            TokenSpan(component="history",   text="",           token_count=history_tokens),
            TokenSpan(component="retrieved", text="",           token_count=retrieved_tokens),
            TokenSpan(component="formatting", text="",          token_count=fmt_tokens),
        ],
        usage=UsageRecord(provider="openai", input_tokens=total, source="response_usage"),
    )


if __name__ == "__main__":
    calls = [
        make_call(0, history_tokens=20,  retrieved_tokens=100),   # turn 1:  146 tokens, fine
        make_call(2, history_tokens=60,  retrieved_tokens=600),   # turn 3:  686 tokens, fine
        make_call(4, history_tokens=200, retrieved_tokens=4_200), # turn 5: 4426 tokens, over budget
    ]
    out = Path("examples/demo.jsonl")
    dump_calls(calls, out)
    print(f"Wrote {len(calls)} calls to {out}")
    print("Next: contextlens report examples/demo.jsonl --contract examples/contract.json")
