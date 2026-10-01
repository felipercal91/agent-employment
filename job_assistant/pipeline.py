"""Evaluation flow for one posting: fit score -> requirements check -> continue or stop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from job_assistant.agents.requirements_agent import RequirementsAgent, RequirementsAssessment
from job_assistant.agents.tailoring_agent import TailoredCV, TailoringAgent
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.profile.loaders import SourceDocument
from job_assistant.profile.store import ProfileStore
from job_assistant.scoring.fit import FitResult, score_fit
from job_assistant.tailoring.cv_select import select_base_cv
from job_assistant.tailoring.drafts import Draft, write_draft
from job_assistant.tailoring.guardrails import (
    fix_quebec_title,
    protected_title_mentions,
    ungrounded_keywords,
)
from job_assistant.tailoring.market import Language, Market, detect_market
from job_assistant.tracker.excel import NewEntry, Tracker, TrackerError, TrackerRow


@dataclass(frozen=True)
class PostingEvaluation:
    fit: FitResult
    requirements: RequirementsAssessment

    @property
    def decision(self) -> str:
        return "stop" if self.requirements.should_stop else "continue"


def evaluate_posting(
    posting_text: str,
    profile: CandidateProfile,
    requirements_agent: RequirementsAgent,
    store: ProfileStore | None = None,
) -> PostingEvaluation:
    fit = score_fit(posting_text, profile)
    # Give the checker real evidence from the documents when the index exists.
    excerpts = store.search(posting_text[:2000], k=6) if store is not None else []
    requirements = requirements_agent.check(posting_text, profile, excerpts)
    return PostingEvaluation(fit=fit, requirements=requirements)


_SECTION_TITLES = {
    "en": ("Professional Summary", "Relevant Experience Highlights", "Technical Skills"),
    "fr": ("Profil professionnel", "Expérience pertinente", "Compétences techniques"),
}


@dataclass(frozen=True)
class TailoringOutcome:
    evaluation: PostingEvaluation
    market: Market | None = None
    base_cv: str | None = None  # source path of the base CV used
    cv: TailoredCV | None = None
    flags: list[str] = field(default_factory=list)
    draft_dir: Path | None = None

    @property
    def stopped(self) -> bool:
        return self.cv is None


def apply_guardrails(
    cv: TailoredCV, market: Market, source_texts: list[str]
) -> tuple[TailoredCV, list[str]]:
    flags: list[str] = []
    if market.language == "fr":
        fixed = fix_quebec_title(cv.headline)
        if fixed != cv.headline:
            flags.append(f"Headline changed for Quebec: '{cv.headline}' -> '{fixed}'")
            cv = cv.model_copy(update={"headline": fixed})
        body = "\n".join([cv.summary, *(b.text for b in cv.experience_bullets)])
        for sentence in protected_title_mentions(body):
            flags.append(f"Uses protected title 'ingénieur' (fix before sending): {sentence}")
    draft_text = "\n".join(
        [cv.headline, cv.summary, *(b.text for b in cv.experience_bullets), *cv.skills]
    )
    for keyword in ungrounded_keywords(draft_text, source_texts):
        flags.append(f"'{keyword}' is not in your CV/documents/profile: remove it or confirm it")
    return cv, flags


def render_cv_markdown(cv: TailoredCV, language: Language) -> str:
    summary_t, exp_t, skills_t = _SECTION_TITLES[language]
    bullets = "\n".join(f"- {b.text}" for b in cv.experience_bullets)
    return (
        f"# {cv.headline}\n\n## {summary_t}\n\n{cv.summary}\n\n"
        f"## {exp_t}\n\n{bullets}\n\n## {skills_t}\n\n{', '.join(cv.skills)}\n"
    )


def render_review_markdown(outcome: TailoringOutcome, company: str) -> str:
    cv, ev = outcome.cv, outcome.evaluation
    assert cv is not None and outcome.market is not None
    lines = [
        f"# Review: {ev.requirements.role_title} @ {company}",
        "",
        "**Status: PENDING YOUR APPROVAL**. Nothing has been sent. Edit `cv.md` freely.",
        "",
        f"- Language: **{outcome.market.language}** ({outcome.market.reason})",
        f"- Base CV: `{outcome.base_cv or 'none found - built from profile/excerpts only'}`",
        f"- Fit score: {ev.fit.score}% (missing: {', '.join(ev.fit.missing) or '-'})",
        "",
        "## Flags (check these first)",
        "",
        *([f"- ⚠ {f}" for f in outcome.flags] or ["- none"]),
        "",
        "## Gaps NOT claimed in the CV",
        "",
        *([f"- {g}" for g in cv.gaps] or ["- none"]),
        "",
        "## Bullet evidence",
        "",
        *[f"- {b.text}\n  - _source:_ {b.evidence}" for b in cv.experience_bullets],
        "",
        "## Notes from the tailoring agent",
        "",
        cv.reviewer_notes,
        "",
    ]
    return "\n".join(lines)


def tailor_posting(
    posting_text: str,
    profile: CandidateProfile,
    requirements_agent: RequirementsAgent,
    tailoring_agent: TailoringAgent,
    cvs: list[SourceDocument],
    output_dir: Path,
    store: ProfileStore | None = None,
    company: str | None = None,
    language: Language | None = None,
) -> TailoringOutcome:
    """Evaluate the posting; if it passes, write a tailored CV draft pending approval."""
    evaluation = evaluate_posting(posting_text, profile, requirements_agent, store)
    if evaluation.decision == "stop":
        return TailoringOutcome(evaluation=evaluation)

    company_name = (company or evaluation.requirements.company or "unknown").strip()
    market = detect_market(posting_text, evaluation.requirements.location, company_name)
    if language and language != market.language:
        market = Market(language, market.is_quebec, f"forced to '{language}' by --lang")

    wants_kafka = "Kafka" in evaluation.fit.found
    base = select_base_cv(cvs, market.language, wants_kafka)
    excerpts = (
        [h for h in store.search(posting_text[:2000], k=10) if h.category != "feedback"]
        if store is not None
        else []
    )
    cv = tailoring_agent.tailor(
        posting_text, market, base.text if base else "(no base CV available)", profile,
        evaluation.fit, evaluation.requirements, excerpts,
    )
    sources = [base.text if base else "", *(h.text for h in excerpts), "\n".join(profile.skills)]
    cv, flags = apply_guardrails(cv, market, sources)
    if base is None:
        flags.insert(0, f"No {market.language} base CV in data/cv/: draft built from profile/excerpts only")

    outcome = TailoringOutcome(
        evaluation=evaluation, market=market, base_cv=base.source if base else None,
        cv=cv, flags=flags,
    )
    draft = Draft(
        company=company_name,
        role=evaluation.requirements.role_title,
        kind="cv",
        files={
            "cv.md": render_cv_markdown(cv, market.language),
            "review.md": render_review_markdown(outcome, company_name),
        },
        meta={
            "language": market.language,
            "base_cv": outcome.base_cv,
            "fit_score": evaluation.fit.score,
            "flags": flags,
        },
    )
    return TailoringOutcome(**{**outcome.__dict__, "draft_dir": write_draft(output_dir, draft)})


def tracker_notes(evaluation: PostingEvaluation) -> str:
    parts = []
    if evaluation.fit.missing:
        parts.append(f"Missing keywords: {', '.join(evaluation.fit.missing)}")
    unclear = evaluation.requirements.unclear_must_haves
    if unclear:
        parts.append(f"Verify must-haves: {'; '.join(r.requirement for r in unclear)}")
    return " | ".join(parts)


def add_to_tracker(
    tracker: Tracker,
    evaluation: PostingEvaluation,
    status: str,
    company: str | None = None,
    link: str = "",
) -> TrackerRow:
    """Record a posting that passed the requirements check. Refuses stopped postings."""
    if evaluation.decision != "continue":
        raise TrackerError("Posting did not pass the requirements check; not adding it.")
    name = (company or evaluation.requirements.company).strip()
    if not name:
        raise TrackerError("Company name unknown: pass --company.")
    return tracker.add(
        NewEntry(
            company=name,
            role=evaluation.requirements.role_title,
            link=link,
            fit=evaluation.fit.score,
            status=status,
            notes=tracker_notes(evaluation),
            location=evaluation.requirements.location,
        )
    )
