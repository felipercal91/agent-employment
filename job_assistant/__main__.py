"""Command-line entry point: `python -m job_assistant <command>`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from job_assistant.config import ConfigError, Settings, load_settings
from job_assistant.tracker.excel import Tracker, TrackerError, TrackerRow


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
    from job_assistant.pipeline import add_to_tracker, evaluate_posting
    from job_assistant.profile.candidate import ProfileNotFoundError, load_candidate_profile
    from job_assistant.profile.store import ProfileStore, persistent_client

    settings = _settings_or_exit()
    try:
        profile = load_candidate_profile(settings.data_dir)
    except ProfileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    posting_text = Path(args.posting).read_text(encoding="utf-8")

    tracker = Tracker(settings.tracker_path, settings.tracker_sheet) if settings.tracker_path.is_file() else None
    if args.add and tracker is None:
        print(f"Tracker not found at {settings.tracker_path}.", file=sys.stderr)
        return 2
    # Cheap duplicate check before spending an LLM call.
    if tracker and (args.company or args.link):
        early = tracker.find_duplicates(args.company or "", link=args.link or "")
        if early.is_duplicate:
            _print_rows("ALREADY IN TRACKER - skipping evaluation:", early.duplicates)
            return 4
        if early.same_company:
            _print_rows("Note: same company already in tracker (other roles):", early.same_company)

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
    print(f"\nREQUIREMENTS CHECK: {req.role_title} @ {args.company or req.company or '?'}")
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

    if args.add and tracker is not None:
        try:
            row = add_to_tracker(
                tracker, result, settings.tracker_initial_status, args.company, args.link or ""
            )
        except TrackerError as e:
            print(f"\nTracker: not added - {e}", file=sys.stderr)
            return 4
        print(f"\nTracker: added to sheet '{row.sheet}', row {row.row} (status: {row.status}).")
    return 0


def _print_rows(title: str, rows: list[TrackerRow]) -> None:
    print(title)
    for r in rows:
        print(f"  - [{r.sheet} r{r.row}] {r.company} | {r.role} | {r.status} | {r.link}")


def cmd_tracker(args: argparse.Namespace) -> int:
    settings = _settings_or_exit()
    if args.action == "init":
        from job_assistant.tracker.template import create_tracker_template

        try:
            path = create_tracker_template(settings.tracker_path)
        except FileExistsError as e:
            print(e, file=sys.stderr)
            return 2
        print(f"Created {path} (sheets: Aplicaciones, Reclutadores).")
        return 0
    try:
        tracker = Tracker(settings.tracker_path, settings.tracker_sheet)
    except TrackerError as e:
        print(e, file=sys.stderr)
        return 2

    if args.action == "list":
        rows = tracker.rows()
        for layout in tracker.layouts():
            cols = ", ".join(f"{f}={c}" for f, c in layout.columns.items())
            n = sum(1 for r in rows if r.sheet == layout.name)
            print(f"Sheet '{layout.name}': header row {layout.header_row}, {n} rows ({cols})")
        _print_rows("Rows:", rows)
        return 0

    check = tracker.find_duplicates(args.company or "", args.role or "", args.link or "")
    if check.is_duplicate:
        _print_rows("DUPLICATE:", check.duplicates)
        return 4
    if check.same_company:
        _print_rows("Same company, different role:", check.same_company)
    print("Not in tracker.")
    return 0


def cmd_tailor(args: argparse.Namespace) -> int:
    from pathlib import Path

    from job_assistant.agents.requirements_agent import RequirementsAgent
    from job_assistant.agents.tailoring_agent import TailoringAgent
    from job_assistant.llm import get_chat_model
    from job_assistant.pipeline import tailor_posting
    from job_assistant.profile.candidate import ProfileNotFoundError, load_candidate_profile
    from job_assistant.profile.loaders import load_documents
    from job_assistant.profile.store import ProfileStore, persistent_client

    settings = _settings_or_exit()
    try:
        profile = load_candidate_profile(settings.data_dir)
    except ProfileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    posting_text = Path(args.posting).read_text(encoding="utf-8")
    cvs = [d for d in load_documents(settings.data_dir) if d.category == "cv"]
    llm = get_chat_model(settings, max_tokens=16000)

    outcome = tailor_posting(
        posting_text,
        profile,
        RequirementsAgent(llm),
        TailoringAgent(llm),
        cvs,
        settings.output_dir,
        store=ProfileStore(persistent_client(settings.vectorstore_dir)),
        company=args.company,
        language=args.lang,
    )
    req = outcome.evaluation.requirements
    if outcome.stopped:
        print(f"STOP: {req.role_title} has hard mismatches; no CV prepared:")
        for r in req.hard_mismatches:
            print(f"  - {r.requirement}: {r.evidence}")
        return 3

    assert outcome.market is not None and outcome.draft_dir is not None
    print(f"Tailored CV for {req.role_title} ({outcome.market.language}: {outcome.market.reason})")
    print(f"Base CV: {outcome.base_cv or '(none)'} | Fit: {outcome.evaluation.fit.score}%")
    if outcome.flags:
        print("Flags to review:")
        for flag in outcome.flags:
            print(f"  ! {flag}")
    print(f"\nDraft PENDING APPROVAL in: {outcome.draft_dir}")
    print("  cv.md      -> tailored sections")
    print("  review.md  -> evidence per bullet, gaps not claimed, flags")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; French/Spanish accents need UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
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
    evaluate.add_argument("--company", help="Company name (used for duplicate check / tracker row)")
    evaluate.add_argument("--link", help="Posting URL (used for duplicate check / tracker row)")
    evaluate.add_argument("--add", action="store_true", help="Add to the tracker if it passes")
    evaluate.set_defaults(func=cmd_evaluate)

    tailor = sub.add_parser("tailor", help="Evaluate a posting and draft a tailored CV (pending approval)")
    tailor.add_argument("posting", help="Path to a text file with the job posting")
    tailor.add_argument("--company", help="Company name (otherwise taken from the posting)")
    tailor.add_argument("--lang", choices=["en", "fr"], help="Force the CV language")
    tailor.set_defaults(func=cmd_tailor)

    tracker = sub.add_parser("tracker", help="Inspect the Excel tracker")
    tracker.add_argument("action", choices=["init", "list", "check"])
    tracker.add_argument("--company")
    tracker.add_argument("--role")
    tracker.add_argument("--link")
    tracker.set_defaults(func=cmd_tracker)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
