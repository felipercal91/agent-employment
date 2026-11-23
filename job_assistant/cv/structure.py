"""Structure a role dictates on a generated resume, enforced in code after the writer.

`exclude_jobs`: jobs the role's resume never shows (the writer is told, but normalize() puts every
job back, so the removal happens here). `must_include`: "job_id:term" rules; when the job has no
bullet mentioning the term, the master's own bullet for it is appended verbatim.
"""

from __future__ import annotations

from job_assistant.cv.base import split_lead
from job_assistant.cv.content import ResumeContent
from job_assistant.cv.master import CVMaster
from job_assistant.cv.roles import Role
from job_assistant.tailoring.market import Language


def apply_role_structure(
    content: ResumeContent, master: CVMaster, lang: Language, role: Role,
) -> tuple[ResumeContent, list[str]]:
    """Drop the excluded jobs and restore the required bullets. Returns the content and what was added."""
    facts = {j.id: j.bullets for j in master.jobs(lang)}
    added: list[str] = []
    entries = []
    for entry in content.experience:
        if entry.job_id in role.exclude_jobs:
            continue
        bullets = list(entry.bullets)
        for rule in role.must_include:
            job_id, _, term = rule.partition(":")
            if job_id != entry.job_id or not term:
                continue
            if any(term.casefold() in b.text().casefold() for b in bullets):
                continue
            source = next((b for b in facts.get(job_id, []) if term.casefold() in b.casefold()), None)
            if source is not None:
                bullets.append(split_lead(source))
                added.append(f"{job_id}: required bullet about '{term}' restored")
        entries.append(entry.model_copy(update={"bullets": bullets}))
    return content.model_copy(update={"experience": entries}), added
