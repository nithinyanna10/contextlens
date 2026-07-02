"""cli.py — `contextlens` console script entry point."""

from __future__ import annotations

import argparse
import sys

from contextlens.report import Contract, build_report, load_calls, render


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `contextlens` console script."""
    parser = argparse.ArgumentParser(
        prog="contextlens",
        description="Context-window linter for LLM agents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rp = sub.add_parser("report", help="Render a context-window report from a JSONL file.")
    rp.add_argument("file", help="Path to .jsonl file of captured calls.")
    rp.add_argument("--contract", metavar="FILE", help="Path to JSON contract file.")

    args = parser.parse_args(argv)

    try:
        calls = load_calls(args.file)
    except Exception as exc:
        print(f"contextlens: error reading {args.file}: {exc}", file=sys.stderr)
        sys.exit(2)

    contract: Contract | None = None
    if args.contract:
        try:
            with open(args.contract) as f:
                contract = Contract.model_validate_json(f.read())
        except Exception as exc:
            print(f"contextlens: error reading contract {args.contract}: {exc}", file=sys.stderr)
            sys.exit(2)

    report = build_report(calls, contract=contract)
    print(render(report))

    sys.exit(1 if report.has_violations else 0)
