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


def cmd_answer(args: argparse.Namespace) -> int:
    from pathlib import Path

    from job_assistant.agents.qa_agent import (
        QAAgent,
        flag_counts,
        parse_questions,
        render_answers_markdown,
    )
    from job_assistant.llm import get_chat_model
    from job_assistant.profile.candidate import ProfileNotFoundError, load_candidate_profile
    from job_assistant.profile.loaders import load_documents
    from job_assistant.tailoring.drafts import Draft, write_draft
    from job_assistant.tailoring.market import detect_market, is_french_text

    settings = _settings_or_exit()
    try:
        profile = load_candidate_profile(settings.data_dir)
    except ProfileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    questions = parse_questions(Path(args.questions).read_text(encoding="utf-8"))
    if not questions:
        print("No questions found in the file.", file=sys.stderr)
        return 2
    posting_text = Path(args.posting).read_text(encoding="utf-8") if args.posting else ""

    if args.lang:
        language = args.lang
    elif is_french_text("\n".join(questions)):
        language = "fr"
    else:
        language = detect_market(posting_text, company=args.company or "").language

    history = [d for d in load_documents(settings.data_dir) if d.category == "qa_history"]
    answers, issues = QAAgent(get_chat_model(settings, max_tokens=16000)).answer(
        questions, profile, history, language, posting_text
    )
    company, role = args.company or "unknown", args.role or "application"
    folder = write_draft(
        settings.output_dir,
        Draft(
            company=company, role=role, kind="form_answers",
            files={"form_answers.md": render_answers_markdown(answers, company, role, language)},
            meta={"language": language, "flags": flag_counts(answers)},
        ),
    )
    counts = flag_counts(answers)
    print(f"Drafted {len(answers.answers)} answers ({language}); past answers used: {len(history)}")
    print(f"  premise mismatches: {counts['premise']} | need your input: {counts['input']} "
          f"| your decision: {counts['decision']}")
    for a in answers.answers:
        if a.premise_issue:
            print(f"  ! {a.question[:70]}\n      -> {a.premise_issue}")
    print(f"\nDraft PENDING APPROVAL in: {folder / 'form_answers.md'}")
    return 0


def cmd_contacts(args: argparse.Namespace) -> int:
    from datetime import date

    from job_assistant.tracker.contacts import ContactBook

    settings = _settings_or_exit()
    book = ContactBook(settings.tracker_path)
    try:
        if args.action == "add":
            if not args.name:
                print("--name is required.", file=sys.stderr)
                return 2
            book.add(args.name, agency=args.agency or "", end_client=args.client or "",
                     role=args.role or "", channel=args.channel or "",
                     status=args.status or "Activo", next_action=args.next or "",
                     notes=args.note or "")
            print(f"Added contact {args.name}.")
            return 0
        if args.action == "touch":
            c = book.touch(args.name, status=args.status, next_action=args.next, note=args.note)
            print(f"Updated {c.name}: last contact {c.last_contact}, status {c.status}, next: {c.next_action or '-'}")
            return 0
    except TrackerError as e:
        print(e, file=sys.stderr)
        return 2

    today = date.today()
    stale = {c.row for c, _ in book.stale(settings.followup_stale_days, today)}
    contacts = book.all()
    if not contacts:
        print("No contacts yet. Add one with: contacts add --name ...")
    for c in contacts:
        since = c.days_since_contact(today)
        mark = "STALE " if c.row in stale else "      "
        print(f"{mark}{c.name} | {c.agency or '-'} -> {c.end_client or '-'} | {c.role or '-'} | "
              f"{c.status or '-'} | {since if since is not None else '?'}d | next: {c.next_action or '-'}")
    return 0


def cmd_followup(args: argparse.Namespace) -> int:
    from datetime import date

    from job_assistant.agents.followup_agent import FollowUpAgent
    from job_assistant.llm import get_chat_model
    from job_assistant.pipeline import (
        FollowUpBlocked,
        draft_followup,
        due_applications,
        followup_language,
        target_from_application,
        target_from_contact,
    )
    from job_assistant.profile.candidate import ProfileNotFoundError, load_candidate_profile
    from job_assistant.tracker.contacts import (
        FOLLOWUP_DISCARDED,
        FOLLOWUP_SENT,
        ContactBook,
        FollowUpLog,
    )

    settings = _settings_or_exit()
    book, log = ContactBook(settings.tracker_path), FollowUpLog(settings.tracker_path)
    days = settings.followup_stale_days
    today = date.today()

    if args.action == "due":
        stale = book.stale(days, today)
        apps = due_applications(Tracker(settings.tracker_path), days, today) if settings.tracker_path.is_file() else []
        print(f"Recruiters with no contact for more than {days} days:")
        for c, since in stale:
            blocked = " (follow-up pending/recent)" if log.recent(c.name, c.company, settings.followup_cooldown_days) else ""
            print(f"  - {c.name} ({c.company or '-'}): {since if since is not None else '?'}d, next: {c.next_action or '-'}{blocked}")
        if not stale:
            print("  none")
        print(f"Applications waiting for more than {days} days:")
        for row, waited in apps:
            blocked = " (follow-up pending/recent)" if log.recent("", row.company, settings.followup_cooldown_days) else ""
            print(f"  - {row.company} | {row.role} | {row.status} | {waited}d{blocked}")
        if not apps:
            print("  none")
        return 0

    try:
        if args.action in {"sent", "discard"}:
            if not args.to:
                print("--to is required.", file=sys.stderr)
                return 2
            status = FOLLOWUP_SENT if args.action == "sent" else FOLLOWUP_DISCARDED
            entry = log.set_status(args.to, status)
            print(f"Marked follow-up to {entry.recipient or entry.company} as {status}.")
            if status == FOLLOWUP_SENT and entry.recipient:
                book.touch(entry.recipient, note="Follow-up sent")
                print(f"Contact {entry.recipient}: last contact set to today.")
            return 0

        # action == "draft"
        if bool(args.contact) == bool(args.company):
            print("Use exactly one of --contact or --company.", file=sys.stderr)
            return 2
        if args.contact:
            target, location = target_from_contact(book.find(args.contact), today), ""
        else:
            matches = [r for r in Tracker(settings.tracker_path).rows()
                       if r.company.lower() == args.company.lower()]
            if not matches:
                print(f"No application for '{args.company}' in the tracker.", file=sys.stderr)
                return 2
            target, location = target_from_application(matches[-1], today), matches[-1].location
        profile = load_candidate_profile(settings.data_dir)
    except (TrackerError, ProfileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 2

    language = args.lang or followup_language(target, location)
    try:
        folder = draft_followup(
            target, profile, FollowUpAgent(get_chat_model(settings)), log,
            settings.output_dir, settings.followup_cooldown_days, language,
        )
    except FollowUpBlocked as blocked:
        print("NOT DRAFTED: a follow-up for this recipient/company is pending or recent:")
        for e in blocked.entries:
            print(f"  - {e.date} {e.status}: {e.recipient or e.company} ({e.file})")
        print("Send it and run `followup sent --to ...`, or `followup discard --to ...`.")
        return 4
    print((folder / "message.md").read_text(encoding="utf-8"))
    print(f"Draft PENDING APPROVAL in: {folder} (logged in sheet 'Seguimientos' as Borrador)")
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

    answer = sub.add_parser("answer", help="Draft answers to application-form questions")
    answer.add_argument("questions", help="Text file: questions separated by blank lines")
    answer.add_argument("--posting", help="Optional posting file for context")
    answer.add_argument("--company")
    answer.add_argument("--role")
    answer.add_argument("--lang", choices=["en", "fr"])
    answer.set_defaults(func=cmd_answer)

    contacts = sub.add_parser("contacts", help="Recruiter CRM (sheet 'Reclutadores')")
    contacts.add_argument("action", choices=["list", "add", "touch"])
    contacts.add_argument("name", nargs="?", help="Contact name (for touch)")
    contacts.add_argument("--name", dest="name_opt", help="Contact name (for add)")
    contacts.add_argument("--agency")
    contacts.add_argument("--client", help="End client")
    contacts.add_argument("--role")
    contacts.add_argument("--channel", help="Email or LinkedIn URL")
    contacts.add_argument("--status")
    contacts.add_argument("--next", help="Next action")
    contacts.add_argument("--note")
    contacts.set_defaults(func=cmd_contacts)

    followup = sub.add_parser("followup", help="Follow-up drafts (never sent automatically)")
    followup.add_argument("action", choices=["due", "draft", "sent", "discard"])
    followup.add_argument("--contact", help="Recruiter name (draft)")
    followup.add_argument("--company", help="Company of a tracked application (draft)")
    followup.add_argument("--to", help="Recipient or company (sent/discard)")
    followup.add_argument("--lang", choices=["en", "fr"])
    followup.set_defaults(func=cmd_followup)

    tracker = sub.add_parser("tracker", help="Inspect the Excel tracker")
    tracker.add_argument("action", choices=["init", "list", "check"])
    tracker.add_argument("--company")
    tracker.add_argument("--role")
    tracker.add_argument("--link")
    tracker.set_defaults(func=cmd_tracker)

    args = parser.parse_args(argv)
    if args.command == "contacts":
        args.name = args.name_opt or args.name
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
