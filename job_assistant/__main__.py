"""Command-line entry point: `python -m job_assistant <command>`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter

from job_assistant.config import PROFILE_ENV, ConfigError, Settings, load_settings
from job_assistant.cv.pipeline import DEFAULT_MAX_ROUNDS
from job_assistant.tracker.excel import Tracker, TrackerError, TrackerRow


def _settings_or_exit() -> Settings:
    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        raise SystemExit(2)
    if settings.profile:  # make it obvious whose data this run is touching
        print(f"[profile: {settings.profile}]", file=sys.stderr)
    return settings


def cmd_check(args: argparse.Namespace) -> int:
    from anthropic import APIConnectionError, APIStatusError, AuthenticationError

    from job_assistant.llm import check_connection

    settings = _settings_or_exit()
    provider = args.provider or settings.writer_provider
    if provider == "deepseek":
        print(f"Model:      {settings.deepseek_model} (DeepSeek)")
        try:
            reply = check_connection(settings, provider="deepseek")
        except ConfigError as e:
            print(e, file=sys.stderr)
            return 2
        except Exception as e:
            print(f"DeepSeek error: {_short(e)}", file=sys.stderr)
            return 1
        print(f"DeepSeek replied: {reply!r}")
        return 0
    print(f"Model:      {settings.model}")
    print(f"Data dir:   {settings.data_dir}")
    print(f"Output dir: {settings.output_dir}")
    try:
        reply = check_connection(settings)
    except ConfigError as e:  # e.g. the Claude API is switched off in .env
        print(e, file=sys.stderr)
        return 2
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
    from job_assistant.profile.loaders import PROFILE_CATEGORIES, load_documents
    from job_assistant.profile.store import ProfileStore, persistent_client

    settings = _settings_or_exit()
    documents = load_documents(settings.data_dir, PROFILE_CATEGORIES)
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

    agent = ProfileAgent(store, get_chat_model(settings, provider=settings.review_provider))
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
        posting_text, profile,
        RequirementsAgent(get_chat_model(settings, provider=settings.review_provider)), store
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
    cvs = load_documents(settings.data_dir, {"cv"})
    writer_llm = get_chat_model(settings, max_tokens=16000, provider=settings.writer_provider)
    review_llm = get_chat_model(settings, max_tokens=16000, provider=settings.review_provider)

    outcome = tailor_posting(
        posting_text,
        profile,
        RequirementsAgent(review_llm),
        TailoringAgent(writer_llm),
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

    history = load_documents(settings.data_dir, {"qa_history"})
    answers, issues = QAAgent(
        get_chat_model(settings, max_tokens=16000, provider=settings.writer_provider)
    ).answer(
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
            target, profile,
            FollowUpAgent(get_chat_model(settings, provider=settings.writer_provider)), log,
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


def cmd_discover(args: argparse.Namespace) -> int:
    from datetime import date

    from job_assistant.discovery.discover import discover, load_config, render_report, save_posting
    from job_assistant.discovery.sources import archive_inbox_file, fetch_boards, read_inbox
    from job_assistant.profile.candidate import ProfileNotFoundError, load_candidate_profile

    settings = _settings_or_exit()
    try:
        profile = load_candidate_profile(settings.data_dir)
    except ProfileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    config = load_config(settings.data_dir)
    inbox = read_inbox(settings.data_dir / "inbox")
    postings, errors = list(inbox), []
    if config.boards and not args.no_boards:
        print(f"Reading {len(config.boards)} company job boards (public APIs, read-only)...")
        fetched, errors = fetch_boards(config.boards)
        postings += fetched
    print(f"Postings read: {len(postings)} ({len(inbox)} from inbox)")

    tracker = Tracker(settings.tracker_path, settings.tracker_sheet) if settings.tracker_path.is_file() else None
    result = discover(postings, profile, config, tracker, errors)

    today = date.today()
    run_dir = settings.output_dir / "discovery" / today.isoformat()
    saved = {
        (c.posting.url or c.posting.title): save_posting(c.posting, run_dir / "postings")
        for c in result.candidates
    }
    report = render_report(result, today, saved)

    if args.check and result.candidates:
        report += _check_candidates(args, settings, profile, tracker, result.candidates[: args.top], saved)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    for posting in inbox:
        if posting.inbox_file is not None:
            archive_inbox_file(posting.inbox_file)

    print(f"New relevant: {len(result.candidates)} | already tracked: {len(result.already_tracked)} "
          f"| filtered: {result.filtered_out} | below min fit: {len(result.below_min_fit)}")
    for c in result.candidates[:10]:
        print(f"  {c.fit.score:3d}%  {c.posting.title} @ {c.posting.company} ({c.posting.location})")
    for e in result.errors:
        print(f"  ! {e}")
    print(f"\nReport: {run_dir / 'report.md'}")
    return 0


def _check_candidates(args: argparse.Namespace, settings: Settings, profile: object,
                      tracker: Tracker | None, candidates: list, saved: dict) -> str:
    from job_assistant.agents.requirements_agent import RequirementsAgent
    from job_assistant.llm import get_chat_model
    from job_assistant.pipeline import add_to_tracker, evaluate_posting
    from job_assistant.profile.store import ProfileStore, persistent_client

    agent = RequirementsAgent(get_chat_model(settings, provider=settings.review_provider))
    store = ProfileStore(persistent_client(settings.vectorstore_dir))
    lines = ["", "## Requirements check (top candidates)", ""]
    print(f"\nRequirements check on top {len(candidates)}:")
    for c in candidates:
        p = c.posting
        text = f"{p.title}\nCompany: {p.company}\nLocation: {p.location}\n\n{p.text}"
        evaluation = evaluate_posting(text, profile, agent, store)  # type: ignore[arg-type]
        decision = evaluation.decision.upper()
        line = f"- **{decision}** · {p.title} @ {p.company}"
        if evaluation.decision == "stop":
            line += " · " + "; ".join(r.requirement for r in evaluation.requirements.hard_mismatches)
        elif args.add and tracker is not None:
            try:
                row = add_to_tracker(tracker, evaluation, settings.tracker_initial_status,
                                     p.company or None, p.url)
                line += f" · added to tracker ('{row.sheet}' row {row.row})"
            except TrackerError as e:
                line += f" · not added: {e}"
        print(f"  {line.lstrip('- ').replace('**', '')}")
        lines.append(line)
    return "\n".join(lines) + "\n"


def _is_account_error(exc: BaseException) -> bool:
    """Errors that no retry of the next row can fix: missing credits, invalid key, no access."""
    from anthropic import AuthenticationError, PermissionDeniedError

    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return True
    text = str(exc).lower()
    return any(k in text for k in ("credit balance", "billing", "invalid x-api-key", "insufficient balance",
                                   "authentication fails", "incorrect api key"))


def _short(exc: BaseException) -> str:
    text = str(exc)
    for marker in ("'message': '", '"message": "'):
        if marker in text:
            return text.split(marker, 1)[1].split("'", 1)[0].split('"', 1)[0]
    return text[:160]


def cmd_postular(args: argparse.Namespace) -> int:
    from datetime import date

    from job_assistant.agents.hr_reviewer import PROVIDER_SCORE_OFFSET, HRReviewerAgent
    from job_assistant.agents.requirements_agent import RequirementsAgent
    from job_assistant.agents.resume_writer import ResumeWriterAgent
    from job_assistant.cv import postular
    from job_assistant.cv.master import MasterNotFoundError, load_master
    from job_assistant.usage import UsageGroup, UsageTracker
    from job_assistant.llm import get_chat_model
    from job_assistant.profile.candidate import ProfileNotFoundError, load_candidate_profile
    from job_assistant.profile.store import ProfileStore, persistent_client

    settings = _settings_or_exit()
    path = settings.tracker_path
    if not path.is_file():
        print(f"Tracker not found at {path}. Run `tracker init` first.", file=sys.stderr)
        return 2
    try:
        if postular.ensure_sheet(path):
            print(f"Created sheet '{postular.SHEET}' in {path.name}. Add your postings (at least the "
                  "Link column) and run `postular` again.")
            return 0
        master = load_master(settings.data_dir)
        profile = load_candidate_profile(settings.data_dir)
    except (TrackerError, MasterNotFoundError, ProfileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 2

    rows = postular.pending_rows(path, args.row)
    if not rows:
        print("Nothing to do: no rows with Estado empty / Nuevo / Regenerar / Forzar.")
        return 0

    writer_provider = args.writer_provider or settings.writer_provider
    review_provider = args.review_provider or settings.review_provider
    writer_model = (settings.deepseek_writer_model or settings.deepseek_model
                    if writer_provider == "deepseek" else settings.model)
    review_model = settings.deepseek_model if review_provider == "deepseek" else settings.model
    writer_tracker, review_tracker = UsageTracker(writer_model), UsageTracker(review_model)
    usage_tracker = UsageGroup([writer_tracker, review_tracker])
    try:
        llm = get_chat_model(settings, max_tokens=16000, callbacks=[writer_tracker],
                             provider=writer_provider, thinking=settings.writer_thinking,
                             model=settings.deepseek_writer_model)
        light = get_chat_model(settings, max_tokens=16000, callbacks=[review_tracker],
                               effort=settings.review_effort, provider=review_provider,
                               thinking=settings.review_thinking)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    print(f"Providers: writer = {writer_provider} ({writer_model}), "
          f"requirements + HR review = {review_provider} ({review_model})")
    agents = dict(
        requirements=RequirementsAgent(light), writer=ResumeWriterAgent(llm),
        reviewer=HRReviewerAgent(light, PROVIDER_SCORE_OFFSET[review_provider]),
    )
    store = ProfileStore(persistent_client(settings.vectorstore_dir))
    run_dir = postular.next_run_folder(settings.output_dir, date.today())
    print(f"Processing {len(rows)} posting(s). Each one: requirements check, writer <-> HR review "
          f"(max {args.max_rounds} rounds), PDF.")
    failures = 0
    for row in rows:
        label = row.get("company") or row.get("link") or f"row {row['_row']}"
        print(f"\n[row {row['_row']}] {label}")
        before = usage_tracker.snapshot()
        try:
            out = postular.process_row(
                row, path=path, master=master, profile=profile, output_dir=settings.output_dir,
                store=store, max_rounds=args.max_rounds, run_dir=run_dir, **agents,
            )
        except TrackerError as e:  # Excel open, etc.: stop, nothing else can be written
            print(f"  ! {e}", file=sys.stderr)
            return 2
        except Exception as e:
            if _is_account_error(e):  # no credits / bad key: every next row would fail the same way
                print(f"  ! Stopped: the Anthropic API refused the request ({_short(e)}).\n"
                      "    Check your credits/API key, then run `postular` again: pending rows are kept.",
                      file=sys.stderr)
                return 2
            logging.getLogger(__name__).exception("Row %s failed", row["_row"])
            postular.table(path).update(row["_row"], {"status": postular.STATUS_ERROR,
                                                      "hr_notes": f"Error: {type(e).__name__}. See console."})
            print(f"  ! failed: {type(e).__name__}: {e}", file=sys.stderr)
            failures += 1
            continue
        score = f" | HR {out.score}/10" if out.score is not None else ""
        print(f"  {out.status}: {out.company} – {out.role}{score}")
        print(f"  Usage: {(usage_tracker.snapshot() - before).describe()}")
        if out.cv_path:
            print(f"  CV: {out.cv_path}")
        else:
            print(f"  {out.detail}")
    if run_dir.exists():
        print(f"\nCVs of this run: {run_dir}")
    print(f"\nTotal usage: {usage_tracker.snapshot().describe()}")
    summary = getattr(args, "summary", None)
    if summary is not None:  # the `daily` command reads these for its report
        summary.update(usage=usage_tracker.snapshot().describe(),
                       run_dir=run_dir if run_dir.exists() else None, failures=failures)
    print(f"Done. Results written to sheet '{postular.SHEET}' in {path}.")
    return 1 if failures else 0


def cmd_daily(args: argparse.Namespace) -> int:
    """Search new postings -> add them to Postular -> write the CVs -> Spanish report."""
    from datetime import date

    from job_assistant.cv import postular
    from job_assistant.discovery import daily
    from job_assistant.discovery.discover import load_config
    from job_assistant.profile.candidate import ProfileNotFoundError, load_candidate_profile
    from job_assistant.tracker.contacts import ContactBook
    from job_assistant.tracker.workbook import WorkbookError, ensure_writable

    settings = _settings_or_exit()
    path, today = settings.tracker_path, date.today()
    if not path.is_file():
        print(f"Tracker not found at {path}. Run `tracker init` first.", file=sys.stderr)
        return 2
    try:
        ensure_writable(path)  # fail before spending anything if Excel has the file open
        postular.ensure_sheet(path)
        profile = load_candidate_profile(settings.data_dir)
    except (WorkbookError, TrackerError, ProfileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 2

    config = load_config(settings.data_dir)
    if args.no_boards:
        config.boards = []
    print(f"Searching ({len(config.boards)} company boards + data/inbox)...")
    known = daily.KnownPostings(Tracker(path, settings.tracker_sheet), path)
    search = daily.search_new(profile, config, known, settings.data_dir / "inbox", today)
    try:
        rows = daily.add_to_postular(path, search.to_add)
    except (WorkbookError, TrackerError) as e:
        print(e, file=sys.stderr)
        return 2
    daily.archive_handled_inbox(search)
    print(f"Read {search.postings_read} | new added to Postular: {len(rows)} | already tracked: "
          f"{len(search.discovery.already_tracked)} | stale: {search.stale} | filtered: {search.discovery.filtered_out}")

    summary: dict = {}
    code = 0
    if not args.no_cv:
        post_args = argparse.Namespace(row=None, max_rounds=args.max_rounds, writer_provider=None,
                                       review_provider=None, summary=summary)
        code = cmd_postular(post_args)

    notes = [] if code == 0 else ["La generación de CVs terminó con errores: revisa la consola y la hoja Postular."]
    due = [f"{c.name} ({c.company or c.agency or '-'}): {since if since is not None else '?'} días"
           for c, since in ContactBook(path).stale(settings.followup_stale_days, today)]
    report = daily.DailyReport(
        today=today, search=search, added=len(rows), results=daily.todays_results(path, today),
        usage=summary.get("usage", ""), run_dir=summary.get("run_dir"), recruiters_due=due, notes=notes,
    )
    out = daily.write_report(settings.output_dir, today, daily.render_daily_report(report))
    print(f"\nReport: {out}")
    return code


def cmd_setup(args: argparse.Namespace) -> int:
    """Create the folders and starter files for a person; safe to run again."""
    from job_assistant.config import PROJECT_ROOT, profile_dir
    from job_assistant.setup_profile import scaffold

    settings = _settings_or_exit()
    who = f"profile '{settings.profile}'" if settings.profile else "the default profile"
    examples = PROJECT_ROOT / "examples" / (args.template or "")
    if not examples.is_dir():
        print(f"Unknown template '{args.template}'.", file=sys.stderr)
        return 2
    result = scaffold(
        settings.data_dir, settings.tracker_path, examples,
        profile_dir(settings.profile) / ".env" if settings.profile else None,
    )
    flag = f"--profile {settings.profile} " if settings.profile else ""
    print(f"Setup for {who}: {settings.data_dir}")
    print(f"  created: {', '.join(result.created) or '-'}")
    print(f"  kept (already there, not touched): {', '.join(result.kept) or '-'}")
    print(
        "\nNext steps, in this order:\n"
        f"  1. Copy the CVs into {settings.data_dir / 'cv'}\n"
        f"  2. Fill in {settings.data_dir / 'profile.toml'} (name, skills, certifications, situation)\n"
        f"  3. Fill in {settings.data_dir / 'cv_master.toml'} (employers, dates, education: the facts a CV\n"
        "     can never change). Until then the system refuses to generate CVs: it still holds the example.\n"
        f"  4. python -m job_assistant {flag}index\n"
        f"  5. Add postings to the 'Postular' sheet of {settings.tracker_path}\n"
        f"  6. python -m job_assistant {flag}postular"
    )
    return 0


def cmd_base(args: argparse.Namespace) -> int:
    """Build the person's base CV from cv_master.toml: no model, no cost."""
    from job_assistant.cv.base import build_base_cv
    from job_assistant.cv.master import MasterNotFoundError, load_master

    settings = _settings_or_exit()
    try:
        master = load_master(settings.data_dir)
    except MasterNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    lang = args.lang or "en"
    if lang == "fr" and not master.has_language("fr"):
        print("There is no French version in cv_master.toml; building the English one.", file=sys.stderr)
        lang = "en"
    pdf, pages = build_base_cv(master, lang, settings.output_dir)
    print(f"Base CV ({lang}, {pages} page(s)): {pdf}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Strict second opinion (Claude) on generated CVs, before the candidate sends them."""
    from job_assistant.agents.hr_reviewer import PROVIDER_SCORE_OFFSET, HRReviewerAgent
    from job_assistant.cv import postular
    from job_assistant.cv.master import MasterNotFoundError, load_master
    from job_assistant.llm import get_chat_model
    from job_assistant.usage import UsageTracker

    settings = _settings_or_exit()
    path = settings.tracker_path
    if not path.is_file():
        print(f"Tracker not found at {path}.", file=sys.stderr)
        return 2
    try:
        master = load_master(settings.data_dir)
    except MasterNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    rows = [r for r in postular.table(path).rows() if r.get("cv_file")]
    if args.row is not None:
        rows = [r for r in rows if r["_row"] == args.row]
    else:  # default: the CVs the cheap reviewer called ready; those are the ones to double check
        rows = [r for r in rows if str(r.get("status") or "") == postular.STATUS_READY]
    if not rows:
        print("Nothing to audit: no rows with a generated CV (use --row N for a specific one).")
        return 0

    provider = args.provider or settings.review_provider
    model = settings.deepseek_model if provider == "deepseek" else settings.model
    tracker = UsageTracker(model)
    try:
        llm = get_chat_model(settings, max_tokens=16000, callbacks=[tracker],
                             effort=settings.review_effort, provider=provider,
                             thinking=settings.review_thinking)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    # Scores are calibrated to Claude's scale (DeepSeek reads about 0.8 more generous).
    reviewer = HRReviewerAgent(llm, PROVIDER_SCORE_OFFSET[provider])
    print(f"Auditing {len(rows)} CV(s) with {provider} ({model}).")
    for row in rows:
        before = tracker.snapshot()
        try:
            out = postular.audit_row(row, path=path, master=master, reviewer=reviewer,
                                     output_dir=settings.output_dir, reviewer_name=provider.capitalize())
        except TrackerError as e:
            print(f"  ! {e}", file=sys.stderr)
            return 2
        except Exception as e:
            if _is_account_error(e):
                print(f"  ! Stopped: the API refused the request ({_short(e)}).", file=sys.stderr)
                return 2
            print(f"  ! row {row['_row']} failed: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        score = f" | {out.score}/10" if out.score is not None else ""
        print(f"[row {out.row}] {out.company} – {out.role}: {out.status}{score}")
        print(f"  {(tracker.snapshot() - before).describe()}")
        print(f"  {out.report or out.detail}")
    print(f"\nTotal usage: {tracker.snapshot().describe()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; French/Spanish accents need UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(prog="job_assistant")
    parser.add_argument("--profile", help="Work on another person's data (profiles/<name>/); "
                        "without it, the default person. Same as setting JOB_ASSISTANT_PROFILE.")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="Create the folders and starter files for a person")
    setup.add_argument("--template", choices=["compliance"],
                       help="Starter files for a field other than software development")
    setup.set_defaults(func=cmd_setup)

    check = sub.add_parser("check", help="Verify .env config and the connection to the model provider")
    check.add_argument("--provider", choices=["anthropic", "deepseek"], default=None,
                       help="Provider to test (default: the one used for the CV writer)")
    check.set_defaults(func=cmd_check)
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

    discover = sub.add_parser("discover", help="Find new postings (inbox + company job boards)")
    discover.add_argument("--no-boards", action="store_true", help="Only process data/inbox/")
    discover.add_argument("--check", action="store_true", help="Run the requirements check on the top N")
    discover.add_argument("--top", type=int, default=5)
    discover.add_argument("--add", action="store_true", help="With --check: add passing postings to the tracker")
    discover.set_defaults(func=cmd_discover)

    post = sub.add_parser("postular", help="Write a CV for each posting listed in sheet 'Postular'")
    post.add_argument("--row", type=int, help="Process only this Excel row (any status)")
    post.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                      help=f"Writer <-> HR review rounds (default {DEFAULT_MAX_ROUNDS})")
    post.add_argument("--writer-provider", choices=["anthropic", "deepseek"],
                      help="Override WRITER_PROVIDER for this run")
    post.add_argument("--review-provider", choices=["anthropic", "deepseek"],
                      help="Override REVIEW_PROVIDER for this run")
    post.set_defaults(func=cmd_postular)

    daily = sub.add_parser("daily", help="Search new postings, add them to Postular, write the CVs, report")
    daily.add_argument("--no-boards", action="store_true", help="Only use data/inbox/")
    daily.add_argument("--no-cv", action="store_true", help="Search and fill the sheet, skip the CVs")
    daily.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    daily.set_defaults(func=cmd_daily)

    base = sub.add_parser("base", help="Build the base CV from cv_master.toml (no AI, no cost)")
    base.add_argument("--lang", choices=["en", "fr"], help="Language (default: English)")
    base.set_defaults(func=cmd_base)

    audit = sub.add_parser("audit", help="Strict Claude review of generated CVs before you send them")
    audit.add_argument("--row", type=int, help="Excel row of the CV to audit (default: every 'CV listo')")
    audit.add_argument("--provider", choices=["anthropic", "deepseek"],
                       help="Reviewer (default: REVIEW_PROVIDER from .env)")
    audit.set_defaults(func=cmd_audit)

    tracker = sub.add_parser("tracker", help="Inspect the Excel tracker")
    tracker.add_argument("action", choices=["init", "list", "check"])
    tracker.add_argument("--company")
    tracker.add_argument("--role")
    tracker.add_argument("--link")
    tracker.set_defaults(func=cmd_tracker)

    args = parser.parse_args(argv)
    if args.profile:
        os.environ[PROFILE_ENV] = args.profile
    if args.command == "contacts":
        args.name = args.name_opt or args.name
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
