"""Posting sources: the manual inbox and public ATS job-board APIs."""

from __future__ import annotations

import html
import logging
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from job_assistant.profile.loaders import read_text

logger = logging.getLogger(__name__)

USER_AGENT = "personal-job-search-assistant/0.1 (read-only; one request per company)"
REQUEST_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class Posting:
    source: str  # "inbox" | "greenhouse" | "lever" | "ashby"
    company: str
    title: str
    location: str
    url: str
    text: str
    posted_at: str = ""  # ISO date when known
    inbox_file: Path | None = None


def html_to_text(raw: str) -> str:
    text = html.unescape(raw or "")
    text = re.sub(r"<\s*(br|/p|/li|/h\d|/div)\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*li[^>]*>", "- ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


# ---- inbox ---------------------------------------------------------------------

_HEADER = re.compile(r"^(company|empresa|title|puesto|role|location|ubicaci[oó]n|url|link)\s*:\s*(.+)$", re.IGNORECASE)
_HEADER_FIELD = {
    "company": "company", "empresa": "company", "title": "title", "puesto": "title",
    "role": "title", "location": "location", "ubicacion": "location", "ubicación": "location",
    "url": "url", "link": "url",
}


def parse_inbox_file(path: Path) -> Posting:
    """A saved posting. Optional `Key: value` header lines (Company/Title/Location/URL) on top."""
    text = read_text(path).strip()
    fields: dict[str, str] = {}
    body_lines = []
    in_header = True
    for line in text.splitlines():
        match = _HEADER.match(line.strip()) if in_header else None
        if match:
            key = _HEADER_FIELD[match.group(1).lower()]
            fields.setdefault(key, match.group(2).strip())
            continue
        if line.strip():
            in_header = False
        body_lines.append(line)
    body = "\n".join(body_lines).strip()
    title = fields.get("title") or next((l.strip() for l in body.splitlines() if l.strip()), path.stem)
    if not fields.get("url"):
        url = re.search(r"https?://\S+", text)
        fields["url"] = url.group(0) if url else ""
    return Posting(
        source="inbox", company=fields.get("company", ""), title=title[:150],
        location=fields.get("location", ""), url=fields["url"], text=body, inbox_file=path,
    )


def read_inbox(inbox_dir: Path) -> list[Posting]:
    if not inbox_dir.is_dir():
        return []
    postings = []
    for path in sorted(inbox_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
            try:
                postings.append(parse_inbox_file(path))
            except Exception as e:
                logger.warning("Skipping inbox file %s: %s", path.name, e)
    return postings


def archive_inbox_file(path: Path) -> Path:
    target_dir = path.parent / "processed"
    target_dir.mkdir(exist_ok=True)
    target = target_dir / path.name
    if target.exists():
        target = target_dir / f"{path.stem}-{datetime.now():%Y%m%d%H%M%S}{path.suffix}"
    shutil.move(str(path), target)
    return target


# ---- public ATS boards ------------------------------------------------------------


def _iso_from_ms(ms: Any) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError):
        return ""


def parse_greenhouse(company: str, data: dict[str, Any]) -> list[Posting]:
    return [
        Posting(
            source="greenhouse", company=company, title=j.get("title", ""),
            location=(j.get("location") or {}).get("name", ""), url=j.get("absolute_url", ""),
            text=html_to_text(j.get("content", "")), posted_at=(j.get("updated_at") or "")[:10],
        )
        for j in data.get("jobs", [])
    ]


def parse_lever(company: str, data: list[dict[str, Any]]) -> list[Posting]:
    postings = []
    for j in data:
        cats = j.get("categories") or {}
        sections = [
            f"{s.get('text', '')}\n{html_to_text(s.get('content', ''))}" for s in j.get("lists", [])
        ]
        text = "\n\n".join(
            p for p in [j.get("descriptionPlain", ""), *sections, j.get("additionalPlain", "")] if p
        )
        location = cats.get("location", "") or ", ".join(cats.get("allLocations", []) or [])
        if j.get("workplaceType") == "remote" and "remote" not in location.lower():
            location = f"{location} (Remote)".strip()
        postings.append(Posting(
            source="lever", company=company, title=j.get("text", ""), location=location,
            url=j.get("hostedUrl", ""), text=text, posted_at=_iso_from_ms(j.get("createdAt")),
        ))
    return postings


def parse_ashby(company: str, data: dict[str, Any]) -> list[Posting]:
    postings = []
    for j in data.get("jobs", []):
        locations = [j.get("location", "")] + [
            s.get("location", "") for s in j.get("secondaryLocations", []) or []
        ]
        location = ", ".join(l for l in locations if l)
        if j.get("isRemote") and "remote" not in location.lower():
            location = f"{location} (Remote)".strip()
        postings.append(Posting(
            source="ashby", company=company, title=j.get("title", ""), location=location,
            url=j.get("jobUrl", ""), text=j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml", "")),
            posted_at=(j.get("publishedAt") or "")[:10],
        ))
    return postings


BOARD_URLS = {
    "greenhouse": "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true",
    "lever": "https://api.lever.co/v0/postings/{token}?mode=json",
    "ashby": "https://api.ashbyhq.com/posting-api/job-board/{token}",
}
_PARSERS = {"greenhouse": parse_greenhouse, "lever": parse_lever, "ashby": parse_ashby}


@dataclass(frozen=True)
class Board:
    company: str
    ats: str  # greenhouse | lever | ashby
    token: str  # the company's board identifier in that ATS


def fetch_board(board: Board, client: httpx.Client) -> list[Posting]:
    if board.ats not in BOARD_URLS:
        raise ValueError(f"Unsupported ATS '{board.ats}' (use greenhouse, lever or ashby)")
    response = client.get(BOARD_URLS[board.ats].format(token=board.token))
    response.raise_for_status()
    return _PARSERS[board.ats](board.company, response.json())


def fetch_boards(
    boards: list[Board], client: httpx.Client | None = None, delay: float = REQUEST_DELAY_SECONDS
) -> tuple[list[Posting], list[str]]:
    """Fetch every board once. Returns postings and human-readable errors (never raises)."""
    own = client is None
    client = client or httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT})
    postings: list[Posting] = []
    errors: list[str] = []
    try:
        for i, board in enumerate(boards):
            if i and delay:
                time.sleep(delay)
            try:
                postings.extend(fetch_board(board, client))
            except httpx.HTTPStatusError as e:
                errors.append(f"{board.company} ({board.ats}/{board.token}): HTTP {e.response.status_code}")
            except (httpx.HTTPError, ValueError) as e:
                errors.append(f"{board.company} ({board.ats}/{board.token}): {e}")
    finally:
        if own:
            client.close()
    return postings, errors
