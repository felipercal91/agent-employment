"""Command-line entry point: `python -m job_assistant <command>`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from job_assistant.config import ConfigError, Settings, load_settings


def _settings_or_exit() -> Settings:
    try:
        return load_settings()
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        raise SystemExit(2)


def cmd_check(_: argparse.Namespace) -> int:
    from anthropic import APIConnectionError, APIStatusError, AuthenticationError

    from job_assistant.llm import check_connection

    settings = _settings_or_exit()
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


def cmd_index(_: argparse.Namespace) -> int:
    from job_assistant.profile.loaders import load_documents
    from job_assistant.profile.store import ProfileStore, persistent_client

    settings = _settings_or_exit()
    documents = load_documents(settings.data_dir)
    if not documents:
        print(f"No documents found in subfolders of {settings.data_dir} (see data/README.md).")
        return 1

    for doc in documents:
        print(f"  [{doc.category}] {doc.source} ({len(doc.text)} chars)")
    store = ProfileStore(persistent_client(settings.vectorstore_dir))
    chunks = store.rebuild(documents)
    per_category = Counter(doc.category for doc in documents)
    summary = ", ".join(f"{cat}: {n}" for cat, n in sorted(per_category.items()))
    print(f"Indexed {len(documents)} documents ({summary}) into {chunks} chunks.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from job_assistant.agents.profile_agent import ProfileAgent
    from job_assistant.llm import get_chat_model
    from job_assistant.profile.store import ProfileStore, persistent_client

    settings = _settings_or_exit()
    store = ProfileStore(persistent_client(settings.vectorstore_dir))
    if store.count() == 0:
        print("The index is empty. Run `python -m job_assistant index` first.")
        return 1

    agent = ProfileAgent(store, get_chat_model(settings))
    result = agent.ask(args.question, category=args.category)
    print(result.answer)
    if result.grounded:
        print("\nSources consulted:")
        for source in result.sources:
            print(f"  - {source}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from job_assistant.agents.requirements_agent import RequirementsAgent
    from job_assistant.llm import get_chat_model
    from job_assistant.pipeline import evaluate_posting
    from job_assistant.profile.candidate import ProfileNotFoundError, load_candidate_profile
    from job_assistant.profile.store import ProfileStore, persistent_client

    settings = _settings_or_exit()
    try:
        profile = load_candidate_profile(settings.data_dir)
    except ProfileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    posting_text = Path(args.posting).read_text(encoding="utf-8")

    store = ProfileStore(persistent_client(settings.vectorstore_dir))
    result = evaluate_posting(
        posting_text, profile, RequirementsAgent(get_chat_model(settings)), store
    )

    fit = result.fit
    print(f"FIT SCORE: {fit.score}%")
    print(f"  Matched: {', '.join(fit.matched) or '-'}")
    print(f"  Missing: {', '.join(fit.missing) or '-'}")
    if fit.years_required is not None:
        status = "ok" if fit.years_ok else "BELOW"
        print(f"  Years:   posting asks {fit.years_required}, profile has {profile.years_experience} ({status})")

    req = result.requirements
    print(f"\nREQUIREMENTS CHECK: {req.role_title}")
    marks = {"met": "OK  ", "not_met": "FAIL", "unclear": "??  "}
    for r in req.requirements:
        print(f"  [{marks[r.status]}] ({r.kind}) {r.requirement}\n         {r.evidence}")

    if result.decision == "stop":
        print("\nDECISION: STOP - hard mismatch on must-have requirement(s):")
        for r in req.hard_mismatches:
            print(f"  - {r.requirement}")
        print("No material will be prepared for this posting.")
        return 3
    print("\nDECISION: CONTINUE - no hard mismatches found.")
    if req.unclear_must_haves:
        print("Review these must-haves manually (not enough information in your profile):")
        for r in req.unclear_must_haves:
            print(f"  - {r.requirement}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(prog="job_assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Verify .env config and the connection to Claude").set_defaults(
        func=cmd_check
    )
    sub.add_parser("index", help="(Re)build the local index from the data folder").set_defaults(
        func=cmd_index
    )
    ask = sub.add_parser("ask", help="Ask the Profile agent a question about your background")
    ask.add_argument("question")
    ask.add_argument("--category", help="Restrict to one data subfolder, e.g. cv or feedback")
    ask.set_defaults(func=cmd_ask)

    evaluate = sub.add_parser("evaluate", help="Fit score + requirements check for a posting file")
    evaluate.add_argument("posting", help="Path to a text file with the job posting")
    evaluate.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
