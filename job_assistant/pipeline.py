"""Evaluation flow for one posting: fit score -> requirements check -> continue or stop."""

from __future__ import annotations

from dataclasses import dataclass

from job_assistant.agents.requirements_agent import RequirementsAgent, RequirementsAssessment
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.profile.store import ProfileStore
from job_assistant.scoring.fit import FitResult, score_fit


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
