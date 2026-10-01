"""Command-line entry point: `python -m job_assistant <command>`."""

from __future__ import annotations

import argparse
import sys

from job_assistant.config import ConfigError, load_settings


def cmd_check(_: argparse.Namespace) -> int:
    from anthropic import APIConnectionError, APIStatusError, AuthenticationError

    from job_assistant.llm import check_connection

    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    print(f"Model:      {settings.model}")
    print(f"Data dir:   {settings.data_dir}")
    print(f"Output dir: {settings.output_dir}")
    try:
        reply = check_connection(settings)
    except AuthenticationError:
        print("Authentication failed: check ANTHROPIC_API_KEY in .env.", file=sys.stderr)
        return 1
    except APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}", file=sys.stderr)
        return 1
    except APIConnectionError:
        print("Could not reach the Anthropic API (network problem).", file=sys.stderr)
        return 1

    print(f"Claude replied: {reply!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="job_assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Verify .env config and the connection to Claude").set_defaults(
        func=cmd_check
    )
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
