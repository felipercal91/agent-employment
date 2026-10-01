"""Evaluation flow for one posting: fit score -> requirements check -> continue or stop."""

from __future__ import annotations

from dataclasses import dataclass

from job_assistant.agents.requirements_agent import RequirementsAgent, RequirementsAssessment
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.profile.store import ProfileStore
from job_assistant.scoring.fit import FitResult, score_fit
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
        )
    )
