"""Tests for the 13 Job Assistant pipeline improvements (search + resume generation)."""

from __future__ import annotations

import pytest

from job_assistant.cv.checks import near_duplicate_check, tailored_body
from job_assistant.cv.content import Bullet, JobBullets, ResumeContent
from job_assistant.discovery.unified_search import (
    is_allowed_location,
    is_excluded_company,
    is_us_big_tech_staff,
    parse_salary,
    priority_boost,
)
from job_assistant.llm import invoke_with_backoff


def _sample_cv(summary: str, bullet_rest: str) -> ResumeContent:
    return ResumeContent(
        title="Senior Java Backend Developer",
        summary=summary,
        competencies=["Java 8/17", "Spring Boot microservices", "REST APIs", "Oracle PL/SQL"],
        experience=[
            JobBullets(
                job_id="ust",
                bullets=[
                    Bullet(lead="Built production Java microservices", rest=bullet_rest),
                ],
            )
        ],
        tools=["Java 8/17", "Spring Boot", "Oracle PL/SQL"],
        methodologies=["Agile", "Scrum"],
    )


# --- #6: Location filter whole-token bug fix ---
@pytest.mark.parametrize("loc,expected", [
    ("Toronto, ON", True),
    ("Mississauga, ON", True),
    ("Ontario, Canada", True),
    ("GTA", True),
    ("Remote - Canada", True),
    ("Canada (Remote)", True),
    ("Montreal, QC", True),
    ("Boston, MA", False),
    ("Houston, TX", False),
    ("London, UK", False),
    ("London, United Kingdom", False),
    ("New York, NY", False),
    ("Berlin, Germany", False),
])
def test_location_filter_whole_tokens(loc: str, expected: bool) -> None:
    assert is_allowed_location(loc) is expected


# --- #7: Active interviews whole-company-name matching ---
@pytest.mark.parametrize("company,expected", [
    ("CGI", True),
    ("CGI Inc.", True),
    ("Insurance Systems Inc", True),
    ("Insurance Systems", True),
    ("ISI", True),
    ("Wissen", True),
    ("Wissen Technology", True),
    ("Morgan Stanley", True),
    ("Citi", True),
    ("Citibank", True),
    ("Citigroup", True),
    ("OpenText", True),
    ("Open Text", True),
    # Must NOT match substrings inside unrelated company names:
    ("Citrix", False),
    ("Citizens Bank", False),
    ("Vision Financial", False),
    ("Decision Analytics", False),
    ("Posit PBC", False),
    ("Isilon Systems", False),
    ("Definity Financial", False),
    ("Intact Financial", False),
])
def test_excluded_company_whole_names(company: str, expected: bool) -> None:
    assert is_excluded_company(company) is expected


# --- #9: Targeting priority & US big tech Staff/Principal skip ---
def test_targeting_priority_and_us_big_tech_skip() -> None:
    assert priority_boost("Intact Financial") == 15
    assert priority_boost("Definity") == 15
    assert priority_boost("Manulife") == 15
    assert priority_boost("Sun Life") == 15
    assert priority_boost("Capco") == 8
    assert priority_boost("Deloitte") == 8
    assert priority_boost("Unknown Startup") == 0

    assert is_us_big_tech_staff("Stripe", "Staff Software Engineer") is True
    assert is_us_big_tech_staff("Google", "Principal Software Engineer") is True
    assert is_us_big_tech_staff("Stripe", "Senior Software Engineer") is False
    assert is_us_big_tech_staff("Definity", "Principal Java Developer") is False


# --- #11: Salary midpoint suggestion & 85k floor ---
def test_salary_midpoint_and_floor() -> None:
    s1 = parse_salary("Salary: $100,000 - $120,000 CAD")
    assert s1.suggested_ask == "$110,000 CAD"
    assert s1.below_floor is False
    assert s1.needs_review is False

    # Range overlaps 85k+: midpoint 87,500 qualifies and never exceeds 95,000
    s2 = parse_salary("$80,000 - $95,000 a year")
    assert s2.suggested_ask == "$87,500 CAD"
    assert s2.below_floor is False
    assert s2.needs_review is False

    # Range overlaps 85k+ with midpoint below 85k: clamped to 85k (never above 88k)
    s3 = parse_salary("$80,000 - $88,000 CAD")
    assert s3.suggested_ask == "$85,000 CAD"
    assert s3.below_floor is False

    # Below 85k: flagged for Revisar
    s4 = parse_salary("$70,000 - $82,000 CAD")
    assert s4.below_floor is True
    assert s4.needs_review is True
    assert s4.suggested_ask == "$76,000 CAD"

    # Not posted: suggest 95k-100k CAD
    s5 = parse_salary("Senior Java Developer in Toronto")
    assert s5.display == "Not posted"
    assert s5.suggested_ask == "$95,000–$100,000 CAD"
    assert s5.needs_review is False


# --- #2: Near-duplicate tailoring detection ---
def test_near_duplicate_detection() -> None:
    cv1 = _sample_cv(
        "Software engineer with 13+ years in insurance technology, building Java 8/17 and 5 years with Spring Boot.",
        "using Java 17, Spring Boot microservices and Oracle PL/SQL tuning with EXPLAIN PLAN.",
    )
    cv1_copy = _sample_cv(
        "Software engineer with 13+ years in insurance technology, building Java 8/17 and 5 years with Spring Boot.",
        "using Java 17, Spring Boot microservices and Oracle PL/SQL tuning with EXPLAIN PLAN.",
    )
    cv_distinct = _sample_cv(
        "Production support engineer with 13+ years in Java and Oracle PL/SQL diagnostics for claims systems.",
        "resolving L2/L3 production incidents, optimizing SQL queries, and supporting REST API integrations.",
    )
    # Identical to role master -> flagged
    assert near_duplicate_check(cv1, master_text=tailored_body(cv1_copy))
    # Identical to same-day sibling -> flagged
    assert near_duplicate_check(cv1, sibling_texts=[("CompanyA_Role", tailored_body(cv1_copy))])
    # Genuinely tailored -> passes
    assert near_duplicate_check(cv_distinct, sibling_texts=[("CompanyA_Role", tailored_body(cv1))]) == []


# --- #12: Retries with exponential backoff ---
def test_invoke_with_backoff_retries_transient_then_succeeds() -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("DeepSeek connection reset")
        return "OK"

    res = invoke_with_backoff(flaky, max_attempts=4, base_delay=1.5, sleep_fn=sleeps.append)
    assert res == "OK"
    assert calls["n"] == 3
    assert sleeps == [1.5, 3.0]


def test_invoke_with_backoff_does_not_retry_non_transient() -> None:
    sleeps: list[float] = []

    def bad() -> str:
        raise ValueError("Invalid JSON schema")

    with pytest.raises(ValueError):
        invoke_with_backoff(bad, max_attempts=4, base_delay=1.0, sleep_fn=sleeps.append)
    assert sleeps == []
