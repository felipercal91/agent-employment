"""Target roles: one master resume per role, and matching a posting's title to the closest role."""

from __future__ import annotations

import re
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROLES_FILENAME = "roles.toml"
INDEX_FILENAME = "roles_index.json"
MASTER_FOLDER = "master_resumes"
TITLE_HIT = 10  # base points for a title keyword found in the posting's title
ATS_CAP = 10    # at most this many points from the posting's text


@dataclass(frozen=True)
class Role:
    id: str
    name: str
    file_name: str          # used in Master_<file_name>_Resume_EN.pdf
    headline: str
    title_keywords: tuple[str, ...]
    ats_keywords: tuple[str, ...]
    focus: str
    verify_terms: tuple[str, ...] = ()  # claims that must appear in the facts to be used

    @property
    def pdf_name(self) -> str:
        return f"Master_{self.file_name}_Resume_EN.pdf"

    @property
    def content_name(self) -> str:
        return f"Master_{self.file_name}_Resume_EN.json"


@dataclass(frozen=True)
class RoleMatch:
    role: Role
    score: int
    reason: str  # in Spanish: it is written into the tracker for the user
    runner_up: Role | None = None


@dataclass(frozen=True)
class RoleBase:
    """The role a resume targets, plus that role's master resume (empty when writing the master itself)."""

    role: Role
    master_json: str = ""


class RolesNotFoundError(FileNotFoundError):
    pass


def load_roles(data_dir: Path) -> list[Role]:
    path = data_dir / ROLES_FILENAME
    if not path.is_file():
        raise RolesNotFoundError(f"{path} not found: it lists the target roles for the master resumes.")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    roles = [
        Role(
            id=r["id"], name=r["name"], file_name=r["file_name"], headline=r["headline"],
            title_keywords=tuple(r["title_keywords"]), ats_keywords=tuple(r["ats_keywords"]), focus=r["focus"],
            verify_terms=tuple(r.get("verify_terms", [])),
        )
        for r in raw.get("roles", [])
    ]
    if not roles:
        raise RolesNotFoundError(f"{path} has no [[roles]].")
    return roles


def _fold(text: str) -> str:
    raw = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", raw).casefold().strip()


def _has_term(folded_text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(_fold(term))}(?![a-z0-9])", folded_text) is not None


def _ats_hits(role: Role, folded_text: str) -> int:
    return sum(1 for k in role.ats_keywords if _has_term(folded_text, k))


def match_role(title: str, jd_text: str, roles: list[Role]) -> RoleMatch:
    """The role whose title keywords best match `title`; the posting's text breaks ties.

    A longer, more specific title keyword wins ("Guidewire Business Analyst" over "Business Analyst").
    """
    folded_title, folded_jd = _fold(title), _fold(jd_text)
    scored: list[tuple[int, int, int, Role, str | None]] = []
    for order, role in enumerate(roles):
        hits = [k for k in role.title_keywords if _has_term(folded_title, k)]
        best = max(hits, key=lambda k: len(k.split()), default=None)
        title_points = TITLE_HIT + len(best.split()) if best else 0
        ats = _ats_hits(role, folded_jd)
        scored.append((title_points + min(ats, ATS_CAP), title_points, -order, role, best))
    scored.sort(key=lambda s: (s[0], s[1], s[2]), reverse=True)
    top, second = scored[0], scored[1] if len(scored) > 1 else None
    role, matched_term = top[3], top[4]
    ats = _ats_hits(role, folded_jd)
    if matched_term:
        reason = f"El título coincide con «{matched_term}»; {ats} de {len(role.ats_keywords)} palabras clave del rol aparecen en la vacante."
    elif top[0] > 0:
        reason = f"Ningún título coincide; se eligió por contenido: {ats} de {len(role.ats_keywords)} palabras clave del rol aparecen en la vacante."
    else:
        reason = "Sin coincidencia de título ni de palabras clave; se usó el primer rol como base."
    runner = second[3] if second and second[0] > 0 and second[0] >= top[0] - 3 else None
    if runner:
        reason += f" También calzaba «{runner.name}»; se prefirió este por ser más cercano al título."
    return RoleMatch(role, top[0], reason, runner)


_SUFFIXES = ("ations", "ation", "ments", "ment", "ing", "ers", "ed", "es", "ful", "er", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _same_word(a: str, b: str) -> bool:
    """Same word in another grammatical form: mentoring/mentored, KPI/KPIs, REST/RESTful, integration/integrated."""
    sa, sb = _stem(a), _stem(b)
    close = min(len(sa), len(sb)) >= 6 and abs(len(sa) - len(sb)) <= 2  # integr/integrat, never service/servicenow
    return sa == sb or (close and (sa.startswith(sb) or sb.startswith(sa)))


def _tokens(text: str) -> list[str]:
    words = (w.strip(".") for w in re.split(r"[^a-z0-9+#.]+", text))  # keep "node.js", drop a sentence's period
    return [w for w in words if len(w) >= 2]


def coverage(role: Role, source_text: str) -> tuple[list[str], list[str]]:
    """Split the role's ATS keywords into (covered by the candidate's facts, gaps).

    A keyword is covered when one line of the facts contains every word of it, in any grammatical form.
    """
    lines = [_tokens(_fold(line)) for line in source_text.splitlines() if line.strip()]
    covered, gaps = [], []
    for keyword in role.ats_keywords:
        wanted = _tokens(_fold(keyword))
        found = any(all(any(_same_word(w, t) for t in line) for w in wanted) for line in lines)
        (covered if found else gaps).append(keyword)
    return covered, gaps


def ungrounded_terms(role: Role, resume_text: str, source_text: str) -> list[str]:
    """Role terms that appear in the resume but nowhere in the candidate's facts (claims he cannot make)."""
    resume, source = _fold(resume_text), _fold(source_text)
    return [t for t in role.verify_terms if _has_term(resume, t) and not _has_term(source, t)]


def synthetic_posting(role: Role) -> str:
    """A posting-like text for the role, used instead of a real job description to write its master."""
    return (
        f"Target role: {role.name}\nLocation: Canada\n\n"
        f"Typical requirements and keywords for this role: {', '.join(role.ats_keywords)}.\n\n"
        f"What this resume must show: {role.focus}\n\n"
        f"Suggested headline: {role.headline}\n"
    )
