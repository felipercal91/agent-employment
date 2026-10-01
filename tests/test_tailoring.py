import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from job_assistant.agents.requirements_agent import Requirement, RequirementsAssessment
from job_assistant.agents.tailoring_agent import TailoredBullet, TailoredCV
from job_assistant.pipeline import apply_guardrails, render_cv_markdown, tailor_posting
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.profile.loaders import SourceDocument
from job_assistant.tailoring.cv_select import cv_language, has_kafka_variant, select_base_cv
from job_assistant.tailoring.drafts import Draft, slugify, write_draft
from job_assistant.tailoring.guardrails import (
    fix_quebec_title,
    protected_title_mentions,
    ungrounded_keywords,
)
from job_assistant.tailoring.market import Market, detect_market

# ---- market detection --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "location", "company", "language"),
    [
        ("Java developer", "Toronto, ON", "", "en"),
        ("Java developer", "Montréal, QC", "", "fr"),
        ("Java developer", "Montreal", "", "fr"),
        ("Java developer", "Laval, Quebec", "", "fr"),
        ("Java developer", "Remote (Canada)", "Desjardins", "fr"),
        ("Nous cherchons un développeur Java avec de l'expérience pour notre équipe dans les "
         "services financiers. Vous avez une maîtrise des microservices.", "", "", "fr"),
        ("Hybrid role, office in Montreal, QC", "", "", "fr"),
        ("Remote across Canada. Our team is in Toronto.", "Remote", "Shopify", "en"),
        ("Clients in Quebec and Ontario", "Toronto, ON", "", "en"),  # stated location wins
    ],
)
def test_detect_market(text: str, location: str, company: str, language: str) -> None:
    assert detect_market(text, location, company).language == language


# ---- base CV selection -------------------------------------------------------


def doc(source: str, text: str = "Java developer with experience") -> SourceDocument:
    return SourceDocument(text=text, source=source, category="cv")


def test_cv_language_and_kafka_variant() -> None:
    assert cv_language(doc("cv/CV_Felipe_FR.docx")) == "fr"
    assert cv_language(doc("cv/resume-english.pdf")) == "en"
    assert cv_language(doc("cv/cv.pdf", "Nous avons une équipe avec expérience pour les "
                           "projets dans les services et compétences")) == "fr"
    assert has_kafka_variant(doc("cv/cv_en_kafka.pdf")) is True
    assert has_kafka_variant(doc("cv/cv_en_sin_kafka.pdf")) is False
    assert has_kafka_variant(doc("cv/cv_en_no-kafka.pdf")) is False
    assert has_kafka_variant(doc("cv/cv_en.pdf")) is None


def test_select_base_cv() -> None:
    cvs = [doc("cv/cv_en.pdf"), doc("cv/cv_en_kafka.pdf"), doc("cv/cv_en_sin_kafka.pdf"),
           doc("cv/cv_fr.docx")]
    assert select_base_cv(cvs, "en", wants_kafka=True).source == "cv/cv_en_kafka.pdf"
    assert select_base_cv(cvs, "en", wants_kafka=False).source == "cv/cv_en_sin_kafka.pdf"
    assert select_base_cv(cvs, "fr", wants_kafka=True).source == "cv/cv_fr.docx"
    assert select_base_cv([doc("cv/cv_en.pdf")], "fr", wants_kafka=False) is None


# ---- guardrails --------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "fixed"),
    [
        ("Ingénieur logiciel Java senior", "Développeur logiciel Java senior"),
        ("Ingénieure backend", "Développeuse backend"),
        ("Senior Backend Engineer", "Senior Backend Développeur"),
        ("Développeur Java senior", "Développeur Java senior"),
    ],
)
def test_fix_quebec_title(title: str, fixed: str) -> None:
    assert fix_quebec_title(title) == fixed


def test_protected_title_mentions_ignores_ingenierie() -> None:
    text = "Expérience en ingénierie logicielle. J'ai travaillé comme ingénieur chez X."
    assert protected_title_mentions(text) == ["J'ai travaillé comme ingénieur chez X."]


def test_ungrounded_keywords() -> None:
    sources = ["Java, Spring Boot and Kafka at a bank", "Docker Compose"]
    draft = "Built Kafka microservices in Java on Kubernetes and AWS, packaged with Docker"
    assert ungrounded_keywords(draft, sources) == ["Microservices", "Kubernetes", "AWS"]


def tailored(headline: str = "Senior Java Developer", bullet: str = "Built Java services") -> TailoredCV:
    return TailoredCV(
        headline=headline, summary="Summary.",
        experience_bullets=[TailoredBullet(text=bullet, evidence="cv")],
        skills=["Java"], gaps=["Kubernetes"], reviewer_notes="ok",
    )


def test_apply_guardrails_quebec() -> None:
    market = Market("fr", True, "test")
    cv, flags = apply_guardrails(
        tailored("Ingénieur Java senior", "Rôle d'ingénieur principal sur des services Java"),
        market, ["Java"],
    )
    assert cv.headline == "Développeur Java senior"
    assert any("Headline changed" in f for f in flags)
    assert any("protected title" in f for f in flags)


def test_apply_guardrails_flags_fabricated_tech() -> None:
    _, flags = apply_guardrails(tailored(bullet="Ran Java on Kubernetes"), Market("en", False, ""), ["Java"])
    assert flags == ["'Kubernetes' is not in your CV/documents/profile: remove it or confirm it"]


def test_render_cv_markdown_french_sections() -> None:
    md = render_cv_markdown(tailored("Développeur Java"), "fr")
    assert md.startswith("# Développeur Java")
    assert "## Compétences techniques" in md and "- Built Java services" in md


# ---- drafts ------------------------------------------------------------------


def test_slugify() -> None:
    assert slugify("Banque Nationale du Canada / Développeur Java") == "banque-nationale-du-canada-developpeur"


def test_write_draft_pending(tmp_path: Path) -> None:
    folder = write_draft(
        tmp_path, Draft(company="Acme", role="Java Dev", kind="cv", files={"cv.md": "# x"}),
        now=datetime(2026, 9, 30, 10, 0),
    )
    assert folder == tmp_path / "pending" / "2026-09-30_acme_java-dev"
    meta = json.loads((folder / "cv.meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "pending_approval" and meta["company"] == "Acme"


# ---- pipeline ----------------------------------------------------------------


class FakeRequirements:
    def __init__(self, stop: bool, location: str = "Toronto, ON") -> None:
        self.stop, self.location = stop, location

    def check(self, *args: Any, **kwargs: Any) -> RequirementsAssessment:
        return RequirementsAssessment(
            company="Acme", role_title="Senior Java Developer", location=self.location,
            requirements=[Requirement(requirement="CPO", kind="must_have",
                                      status="not_met" if self.stop else "met", evidence="e")],
        )


class FakeTailoring:
    def __init__(self, cv: TailoredCV) -> None:
        self.cv = cv
        self.calls: list[dict[str, Any]] = []

    def tailor(self, posting_text: str, market: Market, base_cv: str, *args: Any) -> TailoredCV:
        self.calls.append({"market": market, "base_cv": base_cv})
        return self.cv


PROFILE = CandidateProfile(name="T", title="Dev", years_experience=13, skills=["Java", "Kafka"])
CVS = [doc("cv/cv_en_kafka.md", "Java and Kafka in banking"), doc("cv/cv_fr.md", "Développeur Java")]


def test_pipeline_stops_without_calling_tailoring(tmp_path: Path) -> None:
    tailoring = FakeTailoring(tailored())
    outcome = tailor_posting("Kafka", PROFILE, FakeRequirements(stop=True), tailoring, CVS, tmp_path)  # type: ignore[arg-type]
    assert outcome.stopped and tailoring.calls == []
    assert not (tmp_path / "pending").exists()


def test_pipeline_quebec_uses_french_cv_and_writes_draft(tmp_path: Path) -> None:
    tailoring = FakeTailoring(tailored("Ingénieur Java"))
    outcome = tailor_posting(
        "Java Kafka", PROFILE, FakeRequirements(stop=False, location="Montréal, QC"),  # type: ignore[arg-type]
        tailoring, CVS, tmp_path,  # type: ignore[arg-type]
    )
    assert tailoring.calls[0]["market"].language == "fr"
    assert tailoring.calls[0]["base_cv"] == "Développeur Java"
    assert outcome.cv is not None and outcome.cv.headline == "Développeur Java"
    assert outcome.draft_dir is not None
    assert (outcome.draft_dir / "cv.md").read_text(encoding="utf-8").startswith("# Développeur Java")
    review = (outcome.draft_dir / "review.md").read_text(encoding="utf-8")
    assert "PENDING YOUR APPROVAL" in review and "Kubernetes" in review  # gap listed


def test_pipeline_lang_override(tmp_path: Path) -> None:
    tailoring = FakeTailoring(tailored())
    tailor_posting("Java", PROFILE, FakeRequirements(False, "Montréal, QC"), tailoring, CVS,  # type: ignore[arg-type]
                   tmp_path, language="en")
    assert tailoring.calls[0]["market"].language == "en"
    assert tailoring.calls[0]["base_cv"] == "Java and Kafka in banking"
