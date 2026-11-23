"""Unified search: gathers new job postings across Government of Canada (DND / GC Jobs / Job Bank),
company ATS boards (Greenhouse, Lever, Ashby), and Chile remote (GetOnBoard), and logs them to tracker.xlsx.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from job_assistant.cv import postular
from job_assistant.cv.roles import Role, match_role
from job_assistant.discovery import chile
from job_assistant.discovery.discover import DiscoveryConfig, load_config
from job_assistant.discovery.jobbank import search_jobbank
from job_assistant.discovery.sources import Posting, archive_inbox_file, fetch_boards, read_inbox
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.tailoring.chile import COUNTRY_CHILE
from job_assistant.tracker.excel import Tracker
from job_assistant.tracker.workbook import as_date

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 30000

ACTIVE_INTERVIEWS_EXCLUDE = (
    "opentext", "cgi", "insurance systems", "isi", "insurance systems inc",
    "wissen", "morgan stanley", "citi", "citigroup"
)

EXCLUDED_CORE_STACK_REGEX = r"\b(c#|\.net|dotnet|golang|go developer|python developer|python engineer|node\.?js|php|ruby|salesforce|apex|netsuite|servicenow|sap developer|workday developer|data scientist|machine learning engineer|ml engineer|ai engineer|rag engineer)\b"
CLEARANCE_EXCLUDE_REGEX = r"\b(secret clearance|top secret|reliability status|reliability clearance|cbc bilingual|u\.?s\.? citizenship)\b"
SALARY_FLOOR_CAD = 85_000

_HOURLY_RE = re.compile(
    r"(?:CA?\$|\$|CAD\s*)\s*(\d{2,3}(?:\.\d{1,2})?)\s*(?:[-–—to]+\s*(?:CA?\$|\$|CAD\s*)?\s*(\d{2,3}(?:\.\d{1,2})?))?\s*(?:/\s*(?:hr|hour)|per\s+hour|an\s+hour|hourly)",
    re.IGNORECASE,
)
_ANNUAL_RANGE_RE = re.compile(
    r"(?:CA?\$|\$|CAD\s*)\s*(\d{2,3}(?:,\d{3})+|\d{2,3}\s*[kK])\s*(?:[-–—]|to)\s*(?:CA?\$|\$|CAD\s*)?\s*(\d{2,3}(?:,\d{3})+|\d{2,3}\s*[kK])(?:\s*(?:CAD|a\s+year|per\s+year|/\s*(?:yr|year)|annually|base))?",
    re.IGNORECASE,
)
_ANNUAL_SINGLE_RE = re.compile(
    r"(?:CA?\$|\$|CAD\s*)\s*(\d{2,3}(?:,\d{3})+|\d{2,3}\s*[kK])\s*(?:CAD\s*)?(?:a\s+year|per\s+year|/\s*(?:yr|year)|annually|base\s+salary)",
    re.IGNORECASE,
)
_CONTRACT_RE = re.compile(r"\b(contract(?:or)?|hourly|fixed[- ]term)\b", re.IGNORECASE)


def _to_annual_int(raw: str) -> int:
    s = raw.strip().lower().replace(",", "").replace(" ", "")
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    return int(float(s))


@dataclass(frozen=True)
class SalaryInfo:
    display: str
    annual_low: int | None
    annual_high: int | None
    is_hourly_or_contract: bool
    below_floor: bool
    needs_review: bool


def parse_salary(text: str, floor_cad: int = SALARY_FLOOR_CAD) -> SalaryInfo:
    """Extract posted salary range or hourly rate and classify against the CAD floor."""
    blob = text or ""
    is_contract = bool(_CONTRACT_RE.search(blob))

    m_hr = _HOURLY_RE.search(blob)
    if m_hr:
        display = re.sub(r"\s+", " ", m_hr.group(0)).strip()
        return SalaryInfo(
            display=display,
            annual_low=None,
            annual_high=None,
            is_hourly_or_contract=True,
            below_floor=False,
            needs_review=True,
        )

    for m in _ANNUAL_RANGE_RE.finditer(blob):
        low = _to_annual_int(m.group(1))
        high = _to_annual_int(m.group(2))
        if low > high:
            low, high = high, low
        if 35_000 <= low <= 400_000 and 35_000 <= high <= 500_000:
            below = high < floor_cad
            display = f"${low:,}–${high:,} CAD"
            return SalaryInfo(
                display=display,
                annual_low=low,
                annual_high=high,
                is_hourly_or_contract=is_contract,
                below_floor=below,
                needs_review=below or is_contract,
            )

    for m in _ANNUAL_SINGLE_RE.finditer(blob):
        val = _to_annual_int(m.group(1))
        if 35_000 <= val <= 400_000:
            below = val < floor_cad
            display = f"${val:,} CAD"
            return SalaryInfo(
                display=display,
                annual_low=val,
                annual_high=val,
                is_hourly_or_contract=is_contract,
                below_floor=below,
                needs_review=below or is_contract,
            )

    return SalaryInfo(
        display="Not posted",
        annual_low=None,
        annual_high=None,
        is_hourly_or_contract=is_contract,
        below_floor=False,
        needs_review=is_contract,
    )



@dataclass
class SearchResult:
    postings_found: int = 0
    canada_added: int = 0
    chile_added: int = 0
    duplicates_skipped: int = 0
    filtered_out: int = 0
    errors: list[str] = field(default_factory=list)


def is_excluded_title(title: str, excludes: list[str]) -> bool:
    import re
    for pattern in excludes:
        if re.search(pattern, title, re.IGNORECASE):
            return True
    return False


def is_allowed_location(location: str, location_patterns: list[str]) -> bool:
    if not location:
        return True
    import re
    loc = location.lower()
    if any(k in loc for k in ("remote", "canada", "ontario", "toronto", "on", "qc", "quebec", "montreal")):
        return True
    return any(re.search(pat, loc, re.IGNORECASE) for pat in location_patterns)


def run_unified_search(
    path: Path,
    profile: CandidateProfile,
    roles: list[Role],
    config: DiscoveryConfig,
    inbox_dir: Path,
    today: date | None = None,
    top_canada: int = 20,
    top_chile: int = 10,
    include_jobbank: bool = True,
    include_boards: bool = True,
    include_chile: bool = True,
) -> SearchResult:
    today = today or date.today()
    result = SearchResult()

    postular.ensure_sheet(path)
    postular.ensure_columns(path, postular.CHILE_SHEET)

    # 1. Gather all existing known URLs / titles from tracker to skip duplicates
    tracker = Tracker(path, postular.SHEET) if path.is_file() else None
    existing_urls: set[str] = set()
    existing_keys: set[tuple[str, str]] = set()

    if tracker:
        for r in tracker.rows():
            if r.link:
                existing_urls.add(r.link.strip().rstrip("/"))
            if r.company and r.role:
                existing_keys.add((r.company.strip().lower(), r.role.strip().lower()))

    # Check Postular and Postular Chile sheets
    for sheet_name in (postular.SHEET, postular.CHILE_SHEET):
        table = postular.table(path, sheet_name)
        for r in table.rows():
            if r.get("link"):
                existing_urls.add(str(r["link"]).strip().rstrip("/"))
            if r.get("company") and r.get("role"):
                existing_keys.add((str(r["company"]).strip().lower(), str(r["role"]).strip().lower()))

    # 2. Fetch Canada sources (Job Bank + ATS boards + Inbox)
    canada_postings: list[Posting] = []

    # Inbox
    inbox = read_inbox(inbox_dir)
    canada_postings.extend(inbox)

    # Company Boards (Greenhouse, Lever, Ashby)
    if include_boards and config.boards:
        boards_postings, boards_errors = fetch_boards(config.boards)
        canada_postings.extend(boards_postings)
        result.errors.extend(boards_errors)

    # Job Bank Canada (National Defence, Federal Government, Toronto tech)
    if include_jobbank:
        jb_postings, jb_errors = search_jobbank(limit_per_query=6)
        canada_postings.extend(jb_postings)
        result.errors.extend(jb_errors)

    result.postings_found += len(canada_postings)

    # Filter Canada candidates
    canada_records: list[dict[str, Any]] = []
    for p in canada_postings:
        url_clean = (p.url or "").strip().rstrip("/")
        comp_clean = (p.company or "").strip().lower()
        role_clean = (p.title or "").strip().lower()

        if any(ex in comp_clean for ex in ACTIVE_INTERVIEWS_EXCLUDE):
            result.filtered_out += 1
            continue

        if re.search(EXCLUDED_CORE_STACK_REGEX, p.title, re.IGNORECASE) and not re.search(r"\bjava\b", p.title, re.IGNORECASE):
            result.filtered_out += 1
            continue

        if re.search(CLEARANCE_EXCLUDE_REGEX, (p.title + " " + (p.text or "")), re.IGNORECASE):
            result.filtered_out += 1
            continue

        # Pure Data Analyst, Data Engineer, BI Analyst or Business Analyst roles without Java are excluded
        if re.search(r"\b(data analyst|data engineer|bi analyst|business intelligence analyst|business analyst)\b", p.title, re.IGNORECASE):
            if not re.search(r"\bjava\b", (p.title + " " + (p.text or "")), re.IGNORECASE):
                result.filtered_out += 1
                continue

        # Systems Analyst / Technical Analyst only if Java/SQL-heavy
        if re.search(r"\b(systems? analyst|technical analyst)\b", p.title, re.IGNORECASE):
            blob = p.title + " " + (p.text or "")
            if not (re.search(r"\bjava\b", blob, re.IGNORECASE) and re.search(r"\b(sql|oracle|pl/sql)\b", blob, re.IGNORECASE)):
                result.filtered_out += 1
                continue

        # Exclude postings older than max_age_days (default 14)
        posted_dt = as_date(p.posted_at) if p.posted_at else None
        if posted_dt and (today - posted_dt).days > config.max_age_days:
            result.filtered_out += 1
            continue

        if url_clean and url_clean in existing_urls:
            result.duplicates_skipped += 1
            continue
        if comp_clean and role_clean and (comp_clean, role_clean) in existing_keys:
            result.duplicates_skipped += 1
            continue

        if is_excluded_title(p.title, config.title_exclude):
            result.filtered_out += 1
            continue

        if not is_allowed_location(p.location, config.locations):
            result.filtered_out += 1
            continue

        # Must strictly match one of candidate's role families
        if not chile.matches_a_role(p.title, roles):
            result.filtered_out += 1
            continue

        # Match to one of candidate's roles
        matched = match_role(p.title, p.text, roles)
        if not matched:
            result.filtered_out += 1
            continue

        salary_info = parse_salary(
            f"{p.title}\n{p.text or ''}",
            floor_cad=getattr(config, "salary_floor_cad", SALARY_FLOOR_CAD),
        )
        status = postular.STATUS_REVIEW if salary_info.needs_review else "Nuevo"
        flags: list[str] = []
        if salary_info.below_floor:
            flags.append(f"below {getattr(config, 'salary_floor_cad', SALARY_FLOOR_CAD) // 1000}k")
        if salary_info.is_hourly_or_contract:
            flags.append("hourly/contract")
        notes = f"Flags: {', '.join(flags)}" if flags else None

        if url_clean:
            existing_urls.add(url_clean)
        if comp_clean and role_clean:
            existing_keys.add((comp_clean, role_clean))

        canada_records.append({
            "link": p.url or None,
            "company": p.company or None,
            "role": p.title or None,
            "location": p.location or "Toronto, ON",
            "country": "Canadá",
            "notes": notes,
            "description": p.text[:MAX_DESCRIPTION_CHARS] or None,
            "status": status,
            "source": p.source,
            "posted": posted_dt or today,
            "master_role": matched.role.name,
            "salary": salary_info.display,
        })
        if len(canada_records) >= top_canada:
            break

    if canada_records:
        postular.table(path, postular.SHEET).append_many(canada_records)
        result.canada_added = len(canada_records)

    # Archive handled inbox files
    for p in inbox:
        if p.inbox_file and p.inbox_file.exists():
            archive_inbox_file(p.inbox_file)

    # 3. Chile Remote Search (GetOnBoard)
    if include_chile:
        chile_res = chile.search(path, roles)
        result.postings_found += chile_res.found
        result.errors.extend(chile_res.errors)

        chile_candidates = chile_res.candidates[:top_chile]
        if chile_candidates:
            added_chile = chile.add_to_tracker(path, chile_candidates, sheet=postular.CHILE_SHEET)
            result.chile_added = len(added_chile)

    return result
