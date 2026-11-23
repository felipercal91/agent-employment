"""Job Bank Canada client: searches federal government, defence and private postings (read-only).

Parses search results and individual postings using Job Bank's public HTML and Schema.org metadata.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from job_assistant.discovery.sources import Posting, html_to_text

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
BASE_URL = "https://www.jobbank.gc.ca"
SEARCH_URL = f"{BASE_URL}/jobsearch/jobsearch"

# Standard search terms mapped to candidate's target roles and government targets
DEFAULT_JOBBANK_QUERIES = [
    {"query": "National Defence", "location": ""},
    {"query": "Defence IT", "location": ""},
    {"query": "Java Developer", "location": "Toronto, ON"},
    {"query": "Java Developer", "location": "Canada"},
    {"query": "Business Systems Analyst", "location": "Toronto, ON"},
    {"query": "Business Analyst", "location": "Toronto, ON"},
    {"query": "Quality Assurance Analyst", "location": "Toronto, ON"},
    {"query": "Data Analyst", "location": "Toronto, ON"},
    {"query": "Application Support", "location": "Toronto, ON"},
]


def fetch_job_detail(client: httpx.Client, job_id: str) -> dict[str, str]:
    """Fetch the full description and structured metadata for a Job Bank job posting."""
    url = f"{BASE_URL}/jobsearch/jobposting/{job_id}"
    try:
        r = client.get(url, timeout=15)
        if r.status_code != 200:
            return {}
        text = r.text

        title_m = re.search(r'<span[^>]*property="title"[^>]*>([^<]+)</span>', text)
        biz_m = re.search(r'<span[^>]*property="hiringOrganization"[^>]*>(.*?)</span>', text, re.DOTALL)
        loc_m = re.search(r'<span[^>]*property="joblocation"[^>]*>(.*?)</span>', text, re.DOTALL)
        date_m = re.search(r'<span[^>]*property="datePosted"[^>]*>([^<]+)</span>', text)
        desc_m = re.search(r'<div[^>]*property="description"[^>]*>(.*?)</div>', text, re.DOTALL)

        title = title_m.group(1).strip() if title_m else ""
        company = html_to_text(biz_m.group(1)) if biz_m else ""
        location = html_to_text(loc_m.group(1)) if loc_m else ""
        posted = date_m.group(1).strip() if date_m else ""
        raw_desc = desc_m.group(1) if desc_m else ""
        description = html_to_text(raw_desc) if raw_desc else ""

        return {
            "title": title,
            "company": company,
            "location": location,
            "posted": posted,
            "description": description,
            "url": url,
        }
    except Exception as e:
        logger.debug("Could not fetch Job Bank detail %s: %s", job_id, e)
        return {}


def search_jobbank(
    queries: list[dict[str, str]] | None = None,
    limit_per_query: int = 5,
    client: httpx.Client | None = None,
) -> tuple[list[Posting], list[str]]:
    """Search Job Bank across queries and return structured Postings with full descriptions."""
    own_client = client is None
    client = client or httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    postings: list[Posting] = []
    seen_ids: set[str] = set()
    errors: list[str] = []

    target_queries = queries or DEFAULT_JOBBANK_QUERIES

    try:
        for q_item in target_queries:
            query = q_item["query"]
            location = q_item.get("location", "")
            params = {"searchstring": query}
            if location:
                params["locationstring"] = location

            try:
                r = client.get(SEARCH_URL, params=params, timeout=15)
                if r.status_code != 200:
                    errors.append(f"JobBank search '{query}': HTTP {r.status_code}")
                    continue

                articles = re.findall(r'<article[^>]*id="article-(\d+)"[^>]*>(.*?)</article>', r.text, re.DOTALL)
                for aid, body in articles[:limit_per_query]:
                    if aid in seen_ids:
                        continue
                    seen_ids.add(aid)

                    # Extract card fallback info
                    title_m = re.search(r'<span class="noctitle">\s*([^<]+)\s*</span>', body)
                    biz_m = re.search(r'<li class="business">\s*([^<]+)\s*</li>', body)
                    loc_m = re.search(r'<li class="location">\s*([^<]+)\s*</li>', body)
                    date_m = re.search(r'<li class="date">\s*([^<]+)\s*</li>', body)

                    fallback_title = title_m.group(1).strip() if title_m else "IT Position"
                    fallback_company = (
                        biz_m.group(1).strip()
                        if biz_m
                        else "Department of National Defence" if "Defence" in query else "Government of Canada"
                    )
                    fallback_loc = loc_m.group(1).strip() if loc_m else location or "Canada"
                    fallback_posted = date_m.group(1).strip() if date_m else ""

                    # Fetch full JD
                    detail = fetch_job_detail(client, aid)
                    title = detail.get("title") or fallback_title
                    company = detail.get("company") or fallback_company
                    loc = detail.get("location") or fallback_loc
                    posted = detail.get("posted") or fallback_posted
                    desc = detail.get("description") or f"{title} at {company} ({loc})"
                    url = f"{BASE_URL}/jobsearch/jobposting/{aid}"

                    postings.append(
                        Posting(
                            source="jobbank",
                            company=company,
                            title=title,
                            location=loc,
                            url=url,
                            text=desc,
                            posted_at=posted,
                        )
                    )
            except Exception as e:
                errors.append(f"JobBank '{query}': {e}")
    finally:
        if own_client:
            client.close()

    return postings, errors
