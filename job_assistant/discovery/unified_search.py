"""Unified search: gathers new job postings across Government of Canada (DND / GC Jobs / Job Bank),
company ATS boards (Greenhouse, Lever, Ashby), and Chile remote (GetOnBoard), ranks them by fit
and target-company priority, enforces caps and exclusions, and logs them to tracker.xlsx.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from job_assistant.cv import postular
from job_assistant.cv.roles import Role, match_role
from job_assistant.discovery import chile
from job_assistant.discovery.discover import DiscoveryConfig
from job_assistant.discovery.jobbank import search_jobbank
from job_assistant.discovery.sources import Posting, archive_inbox_file, fetch_boards, read_inbox
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.scoring.fit import score_fit
from job_assistant.tailoring.chile import COUNTRY_CHILE
from job_assistant.tracker.excel import Tracker, normalize_company
from job_assistant.tracker.workbook import as_date

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 30000
SALARY_FLOOR_CAD = 85_000
DEFAULT_TOP_CANADA = 8  # target 5-8 strong jobs per day
COMPANY_CAP_30D = 2     # max 2 applications per company every 30 days
COMPANY_CAP_WINDOW_DAYS = 30
REJECTION_WINDOW_DAYS = 90
MAX_ANALYST_PER_DAY = 1

# Whole-company-name patterns (never raw substrings like "isi" or "citi" that catch unrelated firms).
ACTIVE_INTERVIEW_PATTERNS = (
    re.compile(r"\bcgi(?:\s+(?:inc|group|canada))?\b", re.IGNORECASE),
    re.compile(r"\binsurance\s+systems(?:\s+inc\.?)?\b", re.IGNORECASE),
    re.compile(r"\bisi\b", re.IGNORECASE),
    re.compile(r"\bwissen(?:\s+(?:technology|infotech|inc\.?))?\b", re.IGNORECASE),
    re.compile(r"\bmorgan\s+stanley\b", re.IGNORECASE),
    re.compile(r"\bciti(?:bank|group)?\b", re.IGNORECASE),
    re.compile(r"\bopen\s*text\b", re.IGNORECASE),
)

EXCLUDED_CORE_STACK_REGEX = re.compile(
    r"\b(c#|\.net|dotnet|golang|go developer|python developer|python engineer|node\.?js|php|ruby|"
    r"salesforce|apex|netsuite|servicenow|sap developer|workday developer|data scientist|"
    r"machine learning engineer|ml engineer|ai engineer|rag engineer)\b",
    re.IGNORECASE,
)
CLEARANCE_EXCLUDE_REGEX = re.compile(
    r"\b(secret clearance|top secret|reliability status|reliability clearance|cbc bilingual|u\.?s\.? citizenship)\b",
    re.IGNORECASE,
)

# Priority Canadian insurers, insurance-tech platforms, and financial services (+15 boost).
PRIORITY_INSURANCE_FINANCE_RE = re.compile(
    r"\b(intact|definity|manulife|sun\s*life|canada\s*life|desjardins|co-?operators|aviva|rsa|"
    r"economical|wawanesa|equitable\s*life|guidewire|duck\s*creek|allianz|rbc|royal\s+bank|"
    r"td\s*bank|td\s+securities|bmo|bank\s+of\s+montreal|scotiabank|cibc|national\s+bank|"
    r"eq\s*bank|equitable\s+bank|questrade|wealthsimple|moneris|interac|fidelity)\b",
    re.IGNORECASE,
)

# Consulting firms with Java practices (+8 boost).
PRIORITY_CONSULTING_RE = re.compile(
    r"\b(accenture|deloitte|\bey\b|ernst\s*&\s*young|kpmg|pwc|capgemini|infosys|\btcs\b|"
    r"tata\s+consultancy|cognizant|capco|levio|\bltm\b)\b",
    re.IGNORECASE,
)

# US big tech where Staff/Principal roles should be skipped.
US_BIG_TECH_RE = re.compile(
    r"\b(google|amazon|aws|meta|facebook|apple|microsoft|netflix|stripe|coinbase|okta|affirm|"
    r"instacart|doordash|lyft|uber|block|square|robinhood|databricks|snowflake|mongodb|"
    r"elastic|braze|toast|datadog|nvidia|salesforce)\b",
    re.IGNORECASE,
)
STAFF_PRINCIPAL_RE = re.compile(r"\b(staff|principal|distinguished|fellow)\b", re.IGNORECASE)
ANALYST_ROLE_RE = re.compile(
    r"\b(analyst|analyste|qa|quality\s+assurance|sdet|business\s+systems|data\s+engineer)\b",
    re.IGNORECASE,
)
REJECTION_STATUS_RE = re.compile(r"\b(rechaz\w*|reject\w*|no\s+seleccionad\w*|declined)\b", re.IGNORECASE)

# Whole-token Canadian location matchers (never raw substring "on" which matches Boston/Houston/London).
_CANADA_LOC_RE = re.compile(
    r"\b(?:toronto|ontario|gta|mississauga|oakville|markham|brampton|vaughan|north\s+york|"
    r"etobicoke|scarborough|waterloo|kitchener|ottawa|montr[ée]al|laval|qu[ée]bec|canada|"
    r"remote\s*[-–—,/]?\s*canada|canada\s*[-–—,/]?\s*remote)\b"
    r"|(?:,\s*|\(\s*)(?:on|qc)(?:\b|\))",
    re.IGNORECASE,
)
_NON_CANADA_LOC_RE = re.compile(
    r"\b(?:boston|houston|london,?\s*(?:uk|england|united\s+kingdom)|united\s+kingdom|\buk\b|"
    r"united\s+states|\busa\b|new\s+york|san\s+francisco|seattle|austin|chicago|california|"
    r"texas|massachusetts|europe|berlin|amsterdam|dublin|india|singapore|australia)\b",
    re.IGNORECASE,
)

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
    suggested_ask: str
    annual_low: int | None
    annual_high: int | None
    is_hourly_or_contract: bool
    below_floor: bool
    needs_review: bool

    @property
    def column_value(self) -> str:
        return f"{self.display} (ask: {self.suggested_ask})"


def parse_salary(text: str, floor_cad: int = SALARY_FLOOR_CAD) -> SalaryInfo:
    """Extract posted salary range or hourly rate, compute midpoint ask, and classify against floor.

    Rules:
    - Range posted -> suggest midpoint (never above the posted high; floored at `floor_cad` when overlapping).
    - Single salary posted -> suggest that value.
    - Not posted -> suggest $95,000–$100,000 CAD.
    - Below floor_cad (85k) or hourly/contract -> needs_review = True (logged as Estado = "Revisar").
    """
    blob = text or ""
    is_contract = bool(_CONTRACT_RE.search(blob))

    m_hr = _HOURLY_RE.search(blob)
    if m_hr:
        display = re.sub(r"\s+", " ", m_hr.group(0)).strip()
        return SalaryInfo(
            display=display,
            suggested_ask=display,
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
            midpoint = (low + high) // 2
            # Never suggest above the posted high; when overlapping floor, respect floor_cad.
            ask_val = midpoint if below else min(high, max(floor_cad, midpoint))
            display = f"${low:,}–${high:,} CAD"
            return SalaryInfo(
                display=display,
                suggested_ask=f"${ask_val:,} CAD",
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
                suggested_ask=display,
                annual_low=val,
                annual_high=val,
                is_hourly_or_contract=is_contract,
                below_floor=below,
                needs_review=below or is_contract,
            )

    return SalaryInfo(
        display="Not posted",
        suggested_ask="$95,000–$100,000 CAD",
        annual_low=None,
        annual_high=None,
        is_hourly_or_contract=is_contract,
        below_floor=False,
        needs_review=is_contract,
    )


def is_excluded_company(company: str) -> bool:
    """True when `company` matches an active-interview exclusion as a whole company name."""
    name = (company or "").strip()
    if not name:
        return False
    return any(pat.search(name) for pat in ACTIVE_INTERVIEW_PATTERNS)


def is_excluded_title(title: str, excludes: list[str]) -> bool:
    for pattern in excludes:
        if re.search(pattern, title, re.IGNORECASE):
            return True
    return False


def is_allowed_location(location: str, location_patterns: list[str] | None = None) -> bool:
    """Match whole Canadian location tokens only. Rejects Boston, Houston, London UK, etc."""
    if not location or not location.strip():
        return True
    loc = location.strip()
    has_canada = bool(_CANADA_LOC_RE.search(loc))
    if not has_canada and location_patterns:
        has_canada = any(re.search(pat, loc, re.IGNORECASE) for pat in location_patterns)
    if has_canada:
        return True
    if _NON_CANADA_LOC_RE.search(loc):
        return False
    return False


def is_us_big_tech_staff(company: str, title: str) -> bool:
    """True for Staff/Principal roles at US big tech companies."""
    return bool(US_BIG_TECH_RE.search(company or "") and STAFF_PRINCIPAL_RE.search(title or ""))


def priority_boost(company: str, text: str = "") -> int:
    """Boost Canadian insurers & financial services (+15) and consulting firms (+8)."""
    blob = f"{company} {text[:800]}"
    if PRIORITY_INSURANCE_FINANCE_RE.search(blob):
        return 15
    if PRIORITY_CONSULTING_RE.search(company or ""):
        return 8
    return 0


@dataclass
class SearchResult:
    postings_found: int = 0
    canada_added: int = 0
    chile_added: int = 0
    duplicates_skipped: int = 0
    filtered_out: int = 0
    filtered_reasons: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def record_filtered(self, reason: str) -> None:
        self.filtered_out += 1
        self.filtered_reasons[reason] = self.filtered_reasons.get(reason, 0) + 1


def _collect_tracker_history(
    path: Path,
    today: date,
) -> tuple[set[str], set[tuple[str, str]], Counter[str], set[str]]:
    """Return (existing_urls, existing_keys, recent_company_counts_30d, rejected_companies_90d)."""
    existing_urls: set[str] = set()
    existing_keys: set[tuple[str, str]] = set()
    recent_30d: Counter[str] = Counter()
    rejected_90d: set[str] = set()

    def _inspect_row(url: str, comp: str, role: str, status: str, dt_val: Any) -> None:
        u = (url or "").strip().rstrip("/")
        c = (comp or "").strip().lower()
        r = (role or "").strip().lower()
        if u:
            existing_urls.add(u)
        if c and r:
            existing_keys.add((c, r))
        norm_c = normalize_company(comp) if comp else ""
        dt = as_date(dt_val) if dt_val else None
        if norm_c and dt is not None:
            age = (today - dt).days
            if 0 <= age <= COMPANY_CAP_WINDOW_DAYS:
                recent_30d[norm_c] += 1
            if 0 <= age <= REJECTION_WINDOW_DAYS and REJECTION_STATUS_RE.search(status or ""):
                rejected_90d.add(norm_c)

    if path.is_file():
        tracker = Tracker(path, postular.SHEET)
        for row in tracker.rows():
            _inspect_row(row.link, row.company, row.role, row.status, row.date)

        for sheet_name in (postular.SHEET, postular.CHILE_SHEET):
            try:
                table = postular.table(path, sheet_name)
                for row in table.rows():
                    dt_val = row.get("processed") or row.get("posted")
                    _inspect_row(
                        str(row.get("link") or ""),
                        str(row.get("company") or ""),
                        str(row.get("role") or ""),
                        str(row.get("status") or ""),
                        dt_val,
                    )
            except Exception:
                continue

    return existing_urls, existing_keys, recent_30d, rejected_90d


def run_unified_search(
    path: Path,
    profile: CandidateProfile,
    roles: list[Role],
    config: DiscoveryConfig,
    inbox_dir: Path,
    today: date | None = None,
    top_canada: int = DEFAULT_TOP_CANADA,
    top_chile: int = 5,
    include_jobbank: bool = True,
    include_boards: bool = True,
    include_chile: bool = True,
) -> SearchResult:
    today = today or date.today()
    result = SearchResult()

    postular.ensure_sheet(path)
    postular.ensure_columns(path, postular.CHILE_SHEET)

    # 1. Gather existing URLs/keys, 30-day company counts, and 90-day rejections from tracker
    existing_urls, existing_keys, company_counts_30d, rejected_90d = _collect_tracker_history(path, today)

    # 2. Fetch Canada sources (Job Bank + ATS boards + Inbox)
    canada_postings: list[Posting] = []
    inbox = read_inbox(inbox_dir)
    canada_postings.extend(inbox)

    if include_boards and config.boards:
        boards_postings, boards_errors = fetch_boards(config.boards)
        canada_postings.extend(boards_postings)
        result.errors.extend(boards_errors)

    if include_jobbank:
        jb_postings, jb_errors = search_jobbank(limit_per_query=6)
        canada_postings.extend(jb_postings)
        result.errors.extend(jb_errors)

    result.postings_found += len(canada_postings)

    # 3. Filter and score ALL candidates BEFORE applying the daily cap
    scored_candidates: list[tuple[int, int, bool, str, dict[str, Any]]] = []
    seen_batch_urls: set[str] = set()
    seen_batch_keys: set[tuple[str, str]] = set()

    for p in canada_postings:
        url_clean = (p.url or "").strip().rstrip("/")
        comp_clean = (p.company or "").strip().lower()
        role_clean = (p.title or "").strip().lower()
        norm_comp = normalize_company(p.company or "")
        blob = f"{p.title}\n{p.text or ''}"

        if is_excluded_company(p.company or ""):
            result.record_filtered("active_interview_company")
            continue

        if norm_comp and norm_comp in rejected_90d:
            result.record_filtered("rejected_last_90d")
            continue

        if is_us_big_tech_staff(p.company or "", p.title or ""):
            result.record_filtered("us_big_tech_staff")
            continue

        if EXCLUDED_CORE_STACK_REGEX.search(p.title) and not re.search(r"\bjava\b", p.title, re.IGNORECASE):
            result.record_filtered("excluded_core_stack")
            continue

        if CLEARANCE_EXCLUDE_REGEX.search(blob):
            result.record_filtered("clearance_or_citizenship")
            continue

        # Pure Data Analyst, Data Engineer, BI Analyst or Business Analyst roles without Java are excluded
        if re.search(r"\b(data analyst|data engineer|bi analyst|business intelligence analyst|business analyst)\b", p.title, re.IGNORECASE):
            if not re.search(r"\bjava\b", blob, re.IGNORECASE):
                result.record_filtered("pure_data_ba_no_java")
                continue

        # Systems Analyst / Technical Analyst / QA Analyst only if Java + SQL-heavy
        if ANALYST_ROLE_RE.search(p.title):
            if not (re.search(r"\bjava\b", blob, re.IGNORECASE) and re.search(r"\b(sql|oracle|pl/sql)\b", blob, re.IGNORECASE)):
                result.record_filtered("analyst_not_java_sql")
                continue

        # Exclude postings older than max_age_days (default 14)
        posted_dt = as_date(p.posted_at) if p.posted_at else None
        if posted_dt and (today - posted_dt).days > config.max_age_days:
            result.record_filtered("older_than_14d")
            continue

        if (url_clean and (url_clean in existing_urls or url_clean in seen_batch_urls)) or (
            comp_clean and role_clean and ((comp_clean, role_clean) in existing_keys or (comp_clean, role_clean) in seen_batch_keys)
        ):
            result.duplicates_skipped += 1
            continue

        if is_excluded_title(p.title, config.title_exclude):
            result.record_filtered("excluded_title")
            continue

        if not is_allowed_location(p.location, config.locations):
            result.record_filtered("outside_location")
            continue

        if not chile.matches_a_role(p.title, roles):
            result.record_filtered("outside_role_families")
            continue

        matched = match_role(p.title, p.text, roles)
        if not matched:
            result.record_filtered("no_role_match")
            continue

        # Compute deterministic fit score and drop below min_fit BEFORE ranking/capping
        fit = score_fit(blob, profile)
        if fit.score < config.min_fit:
            result.record_filtered("below_min_fit")
            continue

        salary_info = parse_salary(
            blob,
            floor_cad=getattr(config, "salary_floor_cad", SALARY_FLOOR_CAD),
        )
        status = postular.STATUS_REVIEW if salary_info.needs_review else "Nuevo"
        flags: list[str] = []
        if salary_info.below_floor:
            flags.append(f"below {getattr(config, 'salary_floor_cad', SALARY_FLOOR_CAD) // 1000}k")
        if salary_info.is_hourly_or_contract:
            flags.append("hourly/contract")
        notes = f"Fit {fit.score}% | Suggested ask: {salary_info.suggested_ask}"
        if flags:
            notes += f" | Flags: {', '.join(flags)}"

        if url_clean:
            seen_batch_urls.add(url_clean)
        if comp_clean and role_clean:
            seen_batch_keys.add((comp_clean, role_clean))

        is_analyst = bool(ANALYST_ROLE_RE.search(p.title))
        boost = priority_boost(p.company or "", p.text or "")
        record = {
            "link": p.url or None,
            "company": p.company or None,
            "role": p.title or None,
            "location": p.location or "Toronto, ON",
            "country": "Canadá",
            "notes": notes,
            "description": p.text[:MAX_DESCRIPTION_CHARS] or None,
            "status": status,
            "fit": fit.score,
            "source": p.source,
            "posted": posted_dt or today,
            "master_role": matched.role.name,
            "salary": salary_info.column_value,
        }
        scored_candidates.append((fit.score + boost, fit.score, is_analyst, norm_comp, record))

    # 4. Sort by priority-adjusted score then raw fit descending, then enforce company cap, analyst cap, and daily cap
    scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    canada_records: list[dict[str, Any]] = []
    analyst_added = 0
    for _rank_score, _raw_fit, is_analyst, norm_comp, record in scored_candidates:
        if norm_comp and company_counts_30d[norm_comp] >= COMPANY_CAP_30D:
            result.record_filtered("company_cap_30d")
            continue
        if is_analyst and analyst_added >= MAX_ANALYST_PER_DAY:
            result.record_filtered("analyst_daily_cap")
            continue
        canada_records.append(record)
        if norm_comp:
            company_counts_30d[norm_comp] += 1
        if is_analyst:
            analyst_added += 1
        if len(canada_records) >= top_canada:
            break

    if canada_records:
        postular.table(path, postular.SHEET).append_many(canada_records)
        result.canada_added = len(canada_records)

    # Archive handled inbox files
    for p in inbox:
        if p.inbox_file and p.inbox_file.exists():
            archive_inbox_file(p.inbox_file)

    # 5. Chile Remote Search (GetOnBoard)
    if include_chile:
        chile_res = chile.search(path, roles)
        result.postings_found += chile_res.found
        result.errors.extend(chile_res.errors)

        chile_candidates = chile_res.candidates[:top_chile]
        if chile_candidates:
            added_chile = chile.add_to_tracker(path, chile_candidates, sheet=postular.CHILE_SHEET)
            result.chile_added = len(added_chile)

    return result
