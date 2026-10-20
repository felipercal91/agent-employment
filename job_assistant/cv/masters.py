"""Phase 1: one master resume per target role, plus an index and a gaps report.

A master is written by the same writer <-> HR reviewer loop as any resume, against a synthetic
posting built from the role (its ATS keywords and emphasis) instead of a real job description.
Phase 2 (postular) then starts from the matching master and adapts it to the real posting.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from job_assistant.agents.hr_reviewer import HRReviewerAgent
from job_assistant.agents.resume_writer import ResumeWriterAgent
from job_assistant.cv.master import CVMaster
from job_assistant.cv.pipeline import DEFAULT_MAX_ROUNDS, Renderer, generate_resume
from job_assistant.cv.render import render_pdf
from job_assistant.cv.roles import INDEX_FILENAME, Role, RoleBase, coverage, synthetic_posting

REPORT_FILENAME = "ROLES_REPORT.md"


@dataclass
class MasterResult:
    role: Role
    pdf: Path
    content_file: Path
    score: float
    pages: int
    passed: bool
    covered: list[str]
    gaps: list[str]
    interview_gaps: list[str] = field(default_factory=list)   # what the reviewer says cannot be covered
    questions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


def build_master(
    role: Role, master: CVMaster, writer: ResumeWriterAgent, reviewer: HRReviewerAgent, out_dir: Path,
    max_rounds: int = DEFAULT_MAX_ROUNDS, renderer: Renderer = render_pdf,
) -> MasterResult:
    pdf = out_dir / role.pdf_name
    result = generate_resume(
        synthetic_posting(role), master, "en", writer, reviewer, pdf, "", max_rounds, renderer,
        role_base=RoleBase(role),
    )
    content_file = out_dir / role.content_name
    content_file.write_text(result.content.model_dump_json(indent=2), encoding="utf-8")
    covered, gaps = coverage(role, master.source_text("en"))
    observations = [f"[{i.severity}] {i.problem}" for i in result.review.issues] + [
        f"[{f.severity}] {f.problem}" for f in result.findings
    ]
    return MasterResult(
        role=role, pdf=pdf, content_file=content_file, score=result.review.scores.average(),
        pages=result.pages, passed=result.passed, covered=covered, gaps=gaps,
        interview_gaps=list(result.review.gaps_cannot_fix), questions=list(result.review.likely_interview_questions),
        observations=observations,
    )


def build_masters(
    roles: list[Role], master: CVMaster, writer: ResumeWriterAgent, reviewer: HRReviewerAgent, out_dir: Path,
    max_rounds: int = DEFAULT_MAX_ROUNDS, renderer: Renderer = render_pdf,
    progress: Callable[[str], None] = lambda _: None, all_roles: list[Role] | None = None,
) -> list[MasterResult]:
    """Write the masters of `roles`. The index and the report keep the entries of every other role,
    so rebuilding one role never erases the rest (`all_roles` gives the full order)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for role in roles:
        progress(role.name)
        results.append(build_master(role, master, writer, reviewer, out_dir, max_rounds, renderer))
    order = [r.id for r in (all_roles or roles)]
    write_index(out_dir, results, order)
    write_report(out_dir, results, order)
    return results


def _read_index(out_dir: Path) -> list[dict]:
    path = out_dir / INDEX_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def write_index(out_dir: Path, results: list[MasterResult], order: list[str] | None = None) -> Path:
    """roles_index.json: role -> file, plus the title keywords used to match postings to a role.

    Entries of roles that were not rebuilt are kept; `order` (role ids) sets the final order.
    """
    entries = {e["id"]: e for e in _read_index(out_dir)}
    for r in results:
        entries[r.role.id] = {
            "id": r.role.id, "name": r.role.name, "file": r.pdf.name, "content_file": r.content_file.name,
            "title_keywords": list(r.role.title_keywords), "ats_keywords": list(r.role.ats_keywords),
        }
    ranked = sorted(entries, key=lambda i: order.index(i) if order and i in order else len(order or []))
    path = out_dir / INDEX_FILENAME
    path.write_text(json.dumps([entries[i] for i in ranked], indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_master_content(out_dir: Path, role: Role) -> str:
    """The saved master content of `role` as JSON text; empty when that master was not built yet."""
    path = out_dir / role.content_name
    return path.read_text(encoding="utf-8") if path.is_file() else ""


REPORT_HEADER = (
    "# Master resumes by role\n\nCovered = the keyword is supported by cv_master.toml. Gap = it is not: the resume "
    "does not claim it, and you should be ready to address it in an interview.\n"
)


def render_section(r: MasterResult) -> str:
    lines = [
        f"## {r.role.name}", "",
        f"File: `{r.pdf.name}` · HR score {r.score}/10 · {r.pages} page(s) · "
        f"{'approved' if r.passed else 'needs review'}", "",
        "| ATS keyword | Status |", "| --- | --- |",
        *[f"| {k} | covered |" for k in r.covered],
        *[f"| {k} | **gap** |" for k in r.gaps],
        "",
    ]
    if r.interview_gaps:
        lines += ["Gaps to be ready for in interviews (reviewer):", *[f"- {g}" for g in r.interview_gaps], ""]
    if r.observations:
        lines += ["Open observations:", *[f"- {o}" for o in r.observations], ""]
    return "\n".join(lines)


def render_report(results: list[MasterResult]) -> str:
    return REPORT_HEADER + "\n" + "\n".join(render_section(r) for r in results)


def write_report(out_dir: Path, results: list[MasterResult], order: list[str]) -> Path:
    """ROLES_REPORT.md: the sections of rebuilt roles replace their old ones; the others stay."""
    path = out_dir / REPORT_FILENAME
    sections: dict[str, str] = {}
    if path.is_file():
        for block in path.read_text(encoding="utf-8").split("\n## ")[1:]:
            sections[block.split("\n")[0]] = "## " + block.rstrip("\n")
    for r in results:
        sections[r.role.name] = render_section(r).rstrip("\n")
    names = {r.role.id: r.role.name for r in results}
    known = {e["id"]: e["name"] for e in _read_index(out_dir)}
    ranked = [n for n in (known.get(i) or names.get(i) for i in order) if n]
    body = [sections[n] for n in ranked if n in sections] + [s for n, s in sections.items() if n not in ranked]
    path.write_text(REPORT_HEADER + "\n" + "\n\n".join(body) + "\n", encoding="utf-8")
    return path
