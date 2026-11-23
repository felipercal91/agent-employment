"""Filter, de-duplicate and rank discovered postings. Deterministic; no LLM involved."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from job_assistant.discovery.sources import Board, Posting
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.scoring.fit import FitResult, extract_keywords, score_fit
from job_assistant.tailoring.drafts import slugify
from job_assistant.tracker.excel import Tracker, normalize_company, normalize_link, normalize_role

CONFIG_FILENAME = "discovery.toml"

DEFAULT_TITLE_INCLUDE = [r"\bjava\b", r"back[- ]?end", r"d[ée]veloppeur", r"spring", r"software (developer|engineer)"]
DEFAULT_TITLE_EXCLUDE = [r"\bjunior\b", r"\bjr\.?\b", r"\bintern", r"\bstage\b", r"\bstagiaire\b", r"\bco-?op\b",
                         r"new grad", r"\bmobile\b", r"front[- ]?end",
                         r"\bmanager\b", r"\bdirector\b", r"\bsales\b", r"\bandroid\b", r"\bios\b"]
# Toronto (+GTA), London ON, Montréal (+Laval), Québec City, and remote within Canada.
DEFAULT_LOCATIONS = [
    r"toronto", r"\bgta\b", r"mississauga",
    r"london,?\s*(on|ontario)\b",                      # not London, UK
    r"montr[ée]al", r"\blaval\b",
    r"qu[ée]bec\s*city", r"ville de qu[ée]bec", r"qu[ée]bec,\s*qc",
    r"(^|[;,/(]|\bou\b|\bor\b)\s*qu[ée]bec\s*($|[;,/)(])",  # "Québec" as a city name
    r"remote.*canada", r"canada.*remote",
]
# A board posting must mention at least one of these (canonical keywords from scoring.fit).
DEFAULT_REQUIRE_ANY = ["Java", "Spring Boot", "Spring", "Kotlin"]


@dataclass
class DiscoveryConfig:
    boards: list[Board] = field(default_factory=list)
    title_include: list[str] = field(default_factory=lambda: list(DEFAULT_TITLE_INCLUDE))
    title_exclude: list[str] = field(default_factory=lambda: list(DEFAULT_TITLE_EXCLUDE))
    locations: list[str] = field(default_factory=lambda: list(DEFAULT_LOCATIONS))
    min_fit: int = 50
    require_any_keyword: list[str] = field(default_factory=lambda: list(DEFAULT_REQUIRE_ANY))
    max_age_days: int = 14  # board postings published longer ago are skipped (when the date is known)
    daily_max_new: int = 15  # at most this many new postings are added to the Postular sheet per day
    salary_floor_cad: int = 85_000  # minimum annual CAD salary (range overlap qualifies); below goes to Revisar


def load_config(data_dir: Path) -> DiscoveryConfig:
    path = data_dir / CONFIG_FILENAME
    if not path.is_file():
        return DiscoveryConfig()
    with path.open("rb") as f:
        raw = tomllib.load(f)
    defaults = DiscoveryConfig()
    filters = raw.get("filters", {})
    return DiscoveryConfig(
        boards=[Board(b["company"], b["ats"].lower(), b["token"]) for b in raw.get("boards", [])],
        title_include=filters.get("title_include", defaults.title_include),
        title_exclude=filters.get("title_exclude", defaults.title_exclude),
        locations=filters.get("locations", defaults.locations),
        min_fit=int(filters.get("min_fit", defaults.min_fit)),
        require_any_keyword=filters.get("require_any_keyword", defaults.require_any_keyword),
        max_age_days=int(filters.get("max_age_days", defaults.max_age_days)),
        daily_max_new=int(filters.get("daily_max_new", defaults.daily_max_new)),
        salary_floor_cad=int(filters.get("salary_floor_cad", defaults.salary_floor_cad)),
    )


@dataclass(frozen=True)
class Candidate:
    posting: Posting
    fit: FitResult


@dataclass
class DiscoveryResult:
    candidates: list[Candidate] = field(default_factory=list)  # ranked, new, relevant
    already_tracked: list[Posting] = field(default_factory=list)
    filtered_out: int = 0
    below_min_fit: list[Candidate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def is_relevant(posting: Posting, config: DiscoveryConfig) -> bool:
    """Inbox postings were chosen by the user, so only board postings are filtered."""
    if posting.source == "inbox":
        return True
    if not _any(config.title_include, posting.title) or _any(config.title_exclude, posting.title):
        return False
    if not _any(config.locations, posting.location):
        return False
    if config.require_any_keyword:
        found = set(extract_keywords(f"{posting.title}\n{posting.text}"))
        return bool(found & set(config.require_any_keyword))
    return True


def discover(
    postings: list[Posting],
    profile: CandidateProfile,
    config: DiscoveryConfig,
    tracker: Tracker | None,
    errors: list[str] | None = None,
) -> DiscoveryResult:
    result = DiscoveryResult(errors=list(errors or []))
    tracked = tracker.rows() if tracker else []
    tracked_links = {normalize_link(r.link) for r in tracked if r.link}
    tracked_pairs = {(normalize_company(r.company), normalize_role(r.role)) for r in tracked}
    seen: set[str] = set()

    for posting in postings:
        if not is_relevant(posting, config):
            result.filtered_out += 1
            continue
        link = normalize_link(posting.url)
        pair = (normalize_company(posting.company), normalize_role(posting.title))
        if (link and link in tracked_links) or (pair[0] and pair in tracked_pairs):
            result.already_tracked.append(posting)
            continue
        key = link or f"{pair[0]}|{pair[1]}"
        if key in seen:
            continue
        seen.add(key)
        candidate = Candidate(posting, score_fit(f"{posting.title}\n{posting.text}", profile))
        if posting.source != "inbox" and candidate.fit.score < config.min_fit:
            result.below_min_fit.append(candidate)
        else:
            result.candidates.append(candidate)

    result.candidates.sort(key=lambda c: (-c.fit.score, c.posting.company))
    return result


def save_posting(posting: Posting, folder: Path) -> Path:
    """Save a posting as a text file usable by `evaluate` / `tailor`."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slugify(posting.company or 'unknown')}_{slugify(posting.title)}.txt"
    header = [f"Company: {posting.company}", f"Title: {posting.title}",
              f"Location: {posting.location}", f"URL: {posting.url}", ""]
    path.write_text("\n".join(header) + posting.text + "\n", encoding="utf-8")
    return path


def render_report(result: DiscoveryResult, today: date, saved: dict[str, Path]) -> str:
    lines = [f"# Discovery report {today.isoformat()}", ""]
    lines.append(
        f"New relevant postings: **{len(result.candidates)}** · already in tracker: "
        f"{len(result.already_tracked)} · filtered out: {result.filtered_out} · "
        f"below min fit: {len(result.below_min_fit)}"
    )
    lines.append("")
    if result.errors:
        lines += ["## Source errors", "", *[f"- {e}" for e in result.errors], ""]
    lines += ["## New postings (best fit first)", ""]
    if not result.candidates:
        lines.append("- none")
    for c in result.candidates:
        p = c.posting
        lines += [
            f"### {c.fit.score}% · {p.title} @ {p.company or '?'}",
            f"- {p.location or 'location n/a'} · {p.source}{' · ' + p.posted_at if p.posted_at else ''}",
            f"- {p.url or '(no url)'}",
            f"- Missing keywords: {', '.join(c.fit.missing) or '-'}",
            f"- Saved as: `{saved.get(p.url or p.title, '')}`",
            "",
        ]
    if result.below_min_fit:
        lines += ["## Below minimum fit (for reference)", ""]
        lines += [f"- {c.fit.score}% · {c.posting.title} @ {c.posting.company} · {c.posting.url}"
                  for c in result.below_min_fit]
    return "\n".join(lines) + "\n"
