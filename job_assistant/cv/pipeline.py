"""Writer -> checks -> HR reviewer loop for one job description, ending in a PDF."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from job_assistant.agents.hr_reviewer import (
    HRReview,
    HRReviewerAgent,
    feedback_lines,
    passes,
)
from job_assistant.agents.resume_writer import ResumeWriterAgent
from job_assistant.cv.checks import Finding, near_duplicate_check, run_checks, tailored_body
from job_assistant.cv.content import ResumeContent, normalize
from job_assistant.cv.master import CVMaster
from job_assistant.cv.render import build_html, render_pdf
from job_assistant.cv.roles import RoleBase, ungrounded_terms
from job_assistant.cv.structure import apply_role_structure
from job_assistant.tailoring.market import Language

DEFAULT_MAX_ROUNDS = 2  # each extra round is a full writer + reviewer call; round 3 rarely improved
# Measured on 18 postings: a second round clearly helps drafts that score below 6.5 (+0.8 on
# average) and does not help the rest (about 0). Drafts at or above this stay as they are.
RETRY_BELOW_AVERAGE = 6.5

Renderer = Callable[[str, Path], int]


@dataclass
class RoundResult:
    number: int
    average: float
    minimum: int
    passed: bool
    serious_issues: int


@dataclass
class ResumeResult:
    content: ResumeContent
    review: HRReview
    findings: list[Finding]
    passed: bool
    pdf_path: Path
    pages: int
    language: Language
    rounds: list[RoundResult] = field(default_factory=list)


def generate_resume(
    jd_text: str,
    master: CVMaster,
    lang: Language,
    writer: ResumeWriterAgent,
    reviewer: HRReviewerAgent,
    out_pdf: Path,
    candidate_notes: str = "",
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    renderer: Renderer = render_pdf,
    quebec: bool = False,
    role_base: RoleBase | None = None,
    sibling_texts: list[tuple[str, str]] | None = None,
) -> ResumeResult:
    """Draft, check and review up to `max_rounds` times; the BEST round is the one rendered.

    Best = passed first, then fewest blocker/major problems, then highest average score.
    A revision can make things worse, so the last draft is not trusted blindly.
    """
    source = master.source_text(lang)
    content: ResumeContent | None = None
    feedback: list[str] | None = None
    rounds: list[RoundResult] = []
    best: tuple[tuple[bool, int, float], ResumeResult] | None = None

    for number in range(1, max_rounds + 1):
        extra = {"role_base": role_base} if role_base is not None else {}
        content = writer.write(jd_text, master, lang, candidate_notes, content, feedback, quebec, **extra)
        content, fixes = normalize(content, master, lang)
        if role_base is not None:
            content, _restored = apply_role_structure(content, master, lang, role_base.role)
            if role_base.fixed_headline:
                content = content.model_copy(update={"title": role_base.role.headline})
        pages = renderer(build_html(master, content, lang), out_pdf)
        findings = run_checks(content, lang, source, jd_text, pages, quebec, master)
        findings += [Finding("minor", fix, "Keep to the facts' structure.") for fix in fixes]
        if role_base is not None:
            claimed = ungrounded_terms(role_base.role, content.all_text() + "\n" + " ".join(content.tools), source)
            findings += [Finding("blocker", f"Claims '{t}', which is not in the candidate's facts.",
                                 f"Remove '{t}' and reframe the transferable experience instead.") for t in claimed]
        master_body = ""
        if role_base is not None and role_base.master_json:
            try:
                master_body = tailored_body(ResumeContent.model_validate_json(role_base.master_json))
            except Exception:
                master_body = role_base.master_json
        dup_issues = near_duplicate_check(content, master_text=master_body, sibling_texts=sibling_texts)
        findings += [
            Finding("blocker", msg,
                    "Tailor genuinely: mirror this posting's top 8-10 keywords across summary, competencies, and bullets using true facts from cv_master.toml.")
            for msg in dup_issues
        ]
        review = reviewer.review(jd_text, content, master, lang, findings)
        ok = passes(review, findings)
        serious = sum(1 for i in review.issues if i.severity != "minor") + sum(
            1 for f in findings if f.severity != "minor"
        )
        rounds.append(RoundResult(number, review.scores.average(), review.scores.minimum(), ok, serious))
        result = ResumeResult(content=content, review=review, findings=findings, passed=ok,
                              pdf_path=out_pdf, pages=pages, language=lang)
        key = (ok, -serious, review.scores.average())
        if best is None or key > best[0]:
            best = (key, result)
        has_blocker = any(i.severity == "blocker" for i in review.issues) or any(
            f.severity == "blocker" for f in findings
        )
        if ok or (review.scores.average() >= RETRY_BELOW_AVERAGE and not has_blocker):
            break
        feedback = feedback_lines(review, findings)

    assert best is not None
    chosen = best[1]
    if chosen.content is not content:  # an earlier round won: put its PDF back on disk
        chosen.pages = renderer(build_html(master, chosen.content, lang), out_pdf)
    chosen.rounds = rounds
    return chosen


def render_review_report(result: ResumeResult, company: str, role: str, link: str,
                         requirement_notes: list[str]) -> str:
    r = result.review
    s = r.scores
    shift = s.adjusted(0.0)
    raw_note = f" (el modelo dio {s.raw_average()}; calibrado {shift:+.1f})" if shift else ""
    lines = [
        f"# Revisión HR: {role} @ {company}",
        "",
        f"**Resultado: {'APROBADO' if result.passed else 'REVISAR MANUALMENTE'}** · "
        f"puntaje promedio {s.average()}/10{raw_note} · {len(result.rounds)} ronda(s) · {result.pages} página(s) · "
        f"idioma {result.language}",
        "",
        f"Vacante: {link or '(sin link)'}  ",
        f"CV: `{result.pdf_path.name}`. Nada se ha enviado: revísalo antes de postular.",
        "",
        "## Notas del fiscalizador",
        "",
        r.notes_for_candidate,
        "",
        "## Puntajes",
        "",
        "| Criterio | Puntaje |",
        "| --- | --- |",
        f"| Cobertura de requisitos | {s.requirements_coverage} |",
        f"| Veracidad | {s.truthfulness} |",
        f"| Tono humano | {s.human_tone} |",
        f"| Claridad | {s.clarity} |",
        f"| Palabras clave ATS | {s.ats_keywords} |",
        f"| *Tu calce con la vacante (informativo, incluye brechas)* | *{s.candidate_fit}* |",
        "",
        "## Rondas",
        "",
        *[f"- Ronda {x.number}: promedio {x.average}, mínimo {x.minimum}, "
          f"problemas serios {x.serious_issues}, {'aprobada' if x.passed else 'devuelta al redactor'}"
          for x in result.rounds],
        "",
        "El PDF corresponde a la mejor ronda (las observaciones de abajo son de esa ronda).",
        "",
        "## Observaciones pendientes (última ronda)",
        "",
        *([f"- [{i.severity}] {i.where}: {i.problem} → {i.fix}" for i in r.issues]
          + [f"- [{f.severity}] {f.problem} → {f.fix}" for f in result.findings] or ["- ninguna"]),
        "",
        "## Requisitos que no puedes cubrir con la verdad (no están en el CV)",
        "",
        *([f"- {g}" for g in r.gaps_cannot_fix] or ["- ninguno"]),
        "",
        "## Requisitos a verificar (revisión de requisitos)",
        "",
        *([f"- {n}" for n in requirement_notes] or ["- ninguno"]),
        "",
        "## Preguntas probables en la entrevista",
        "",
        *[f"- {q}" for q in r.likely_interview_questions],
        "",
    ]
    return "\n".join(lines)
