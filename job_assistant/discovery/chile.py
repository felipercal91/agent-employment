"""Chile search: GetOnBoard's public API, 100% remote postings only, for the candidate's role families.

Read-only: one request per search term, a descriptive User-Agent, a pause between requests. Nothing is
applied to. LinkedIn, Computrabajo, Laborum and Trabajando.com have no public API and forbid scraping: those
postings reach the tracker another way (Grok, or the user pasting a row with Country = Chile).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from job_assistant.cv import postular
from job_assistant.cv.roles import Role, _title_has, _fold
from job_assistant.discovery.sources import USER_AGENT, Posting, html_to_text
from job_assistant.tailoring.chile import (
    COUNTRY_CHILE, SPANISH_TITLE_KEYWORDS, check_fully_remote, extract_details, spanish_roles,
)

logger = logging.getLogger(__name__)

API = "https://www.getonbrd.com/api/v0/search/jobs"
SOURCE = "getonboard"
PER_PAGE = 50
MAX_PAGES = 2
MAX_DESCRIPTION_CHARS = 30000  # an Excel cell holds 32,767 characters
# One search per role family (GetOnBoard searches title and text).
QUERIES = (
    "java", "backend", "analista funcional", "business analyst", "analista de negocios", "analista de sistemas",
    "qa", "analista qa", "data analyst", "analista de datos", "bi", "power bi", "soporte aplicaciones",
    "tech lead", "líder técnico", "consultor funcional", "implementation consultant",
)
# remote_modality values that are NOT fully remote. "remote_local" (remote inside a country) and
# "fully_remote" pass this gate; the text check and the countries decide the rest.
_NOT_REMOTE = {"hybrid_remote", "no_remote", "hybrid", "on_site", "onsite"}
_OPEN_ZONES = ("latam", "latin", "worldwide", "anywhere", "remote", "sin restric", "no_restrictions", "global")


@dataclass(frozen=True)
class ChilePosting:
    posting: Posting
    modality: str
    countries: str
    salary_usd: str


def _text(attrs: dict[str, Any]) -> str:
    """The posting's sections as plain text, in reading order."""
    parts = []
    for headline, body in (
        ("", attrs.get("projects")), (attrs.get("functions_headline") or "Funciones", attrs.get("functions")),
        (attrs.get("description_headline") or "Requisitos", attrs.get("description")),
        (attrs.get("desirable_headline") or "Deseable", attrs.get("desirable")),
        (attrs.get("benefits_headline") or "Beneficios", attrs.get("benefits")),
    ):
        text = html_to_text(str(body or ""))
        if text:
            parts.append(f"{headline}\n{text}".strip())
    return "\n\n".join(parts)


def parse_job(item: dict[str, Any]) -> ChilePosting | None:
    """One GetOnBoard job as a posting, or None when it is not a candidate for Chile."""
    attrs = item.get("attributes") or {}
    if not attrs.get("remote", False):
        return None
    modality = str(attrs.get("remote_modality") or "").lower()
    if modality in _NOT_REMOTE:
        return None
    countries = [str(c) for c in attrs.get("countries") or []]
    zone = str(attrs.get("remote_zone") or "").lower()
    open_to_chile = (
        any(c.casefold() == "chile" for c in countries)
        or any(k in zone for k in _OPEN_ZONES)
        or any(any(k in c.casefold() for k in _OPEN_ZONES) for c in countries)
    )
    if not open_to_chile:
        return None
    company = ((attrs.get("company") or {}).get("data") or {}).get("attributes", {}).get("name", "")
    published = attrs.get("published_at")
    posted = datetime.fromtimestamp(int(published), tz=timezone.utc).date().isoformat() if published else ""
    salary = ""
    if attrs.get("min_salary") or attrs.get("max_salary"):
        salary = f"USD {attrs.get('min_salary') or '?'} - {attrs.get('max_salary') or '?'}"
    text = _text(attrs)
    if salary:
        text += f"\n\nSueldo publicado en GetOnBoard (USD, no CLP): {salary}"
    posting = Posting(
        source=SOURCE, company=company, title=str(attrs.get("title") or ""),
        location=", ".join(countries) or "Remoto", url=(item.get("links") or {}).get("public_url", ""),
        text=text, posted_at=posted,
    )
    return ChilePosting(posting, modality, ", ".join(countries), salary)


def matches_a_role(title: str, roles: list[Role]) -> bool:
    """True when the posting's title names one of the candidate's role families (English or Spanish)."""
    folded = _fold(title)
    return any(_title_has(folded, k) for role in spanish_roles(roles) for k in role.title_keywords)


def fetch(queries: tuple[str, ...], client: httpx.Client, pages: int = MAX_PAGES, delay: float = 1.0) -> tuple[list[dict[str, Any]], list[str]]:
    """Every result of every search term (deduplicated by id). Returns jobs and errors; never raises."""
    seen: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    first = True
    for query in queries:
        for page in range(1, pages + 1):
            if not first and delay:
                time.sleep(delay)
            first = False
            try:
                r = client.get(API, params={"query": query, "per_page": PER_PAGE, "page": page,
                                            "expand": json.dumps(["company"])})
                r.raise_for_status()
                body = r.json()
            except (httpx.HTTPError, ValueError) as e:
                errors.append(f"GetOnBoard '{query}' p{page}: {e}")
                break
            for item in body.get("data", []):
                seen.setdefault(str(item.get("id")), item)
            if page >= int((body.get("meta") or {}).get("total_pages") or 1):
                break
    return list(seen.values()), errors


@dataclass
class ChileSearch:
    found: int  # jobs returned by the searches
    candidates: list[ChilePosting]  # remote, open to Chile, in a role family, not yet in the tracker
    discarded: dict[str, int]  # reason -> count
    errors: list[str]


def search(path: Path, roles: list[Role], client: httpx.Client | None = None, queries: tuple[str, ...] = QUERIES,
           delay: float = 1.0) -> ChileSearch:
    own = client is None
    client = client or httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT})
    try:
        items, errors = fetch(queries, client, delay=delay)
    finally:
        if own:
            client.close()
    known = {str(r.get("link") or "").strip() for r in postular.table(path).rows()}
    discarded = {"no remoto o no abierto a Chile": 0, "texto menciona híbrido/presencial": 0,
                 "no es de tus familias de rol": 0, "ya está en el tracker": 0}
    candidates: list[ChilePosting] = []
    for item in items:
        parsed = parse_job(item)
        if parsed is None:
            discarded["no remoto o no abierto a Chile"] += 1
            continue
        p = parsed.posting
        if not check_fully_remote(p.text + "\n" + p.location).fully_remote:
            discarded["texto menciona híbrido/presencial"] += 1
        elif not matches_a_role(p.title, roles):
            discarded["no es de tus familias de rol"] += 1
        elif p.url in known:
            discarded["ya está en el tracker"] += 1
        else:
            candidates.append(parsed)
    return ChileSearch(len(items), candidates, discarded, errors)


def add_to_tracker(path: Path, candidates: list[ChilePosting], sheet: str = postular.SHEET) -> list[int]:
    """Append the candidates to Postular with Country = Chile, in ONE write (one backup)."""
    records = []
    for c in candidates:
        p = c.posting
        found = extract_details(p.text)
        records.append({
            "link": p.url or None, "company": p.company or None, "role": p.title or None,
            "location": p.location or None, "description": p.text[:MAX_DESCRIPTION_CHARS] or None,
            "status": "Nuevo", "source": p.source, "country": COUNTRY_CHILE,
            "requires_residence": found.requires_residence, "contract": found.contract, "salary_clp": found.salary_clp,
        })
    return postular.table(path, sheet).append_many(records) if records else []
