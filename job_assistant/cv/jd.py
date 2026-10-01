"""Get a job description from a link the user listed: one read-only request per link.

ATS-hosted postings (Greenhouse, Lever, Ashby) are read through their public JSON APIs.
Other career pages: the schema.org JobPosting embedded in the page, else the page text.
LinkedIn/Indeed usually refuse anonymous requests: the user pastes the text instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

import httpx

from job_assistant.discovery.sources import USER_AGENT, html_to_text

MIN_DESCRIPTION_CHARS = 400
BLOCKED_HOSTS = ("linkedin.com", "indeed.com", "glassdoor.")


class JobDescriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class JobDescription:
    text: str
    title: str = ""
    company: str = ""
    location: str = ""
    source: str = ""  # how it was obtained


def _greenhouse(url: str, client: httpx.Client) -> JobDescription | None:
    parts = urlsplit(url)
    m = re.search(r"greenhouse\.io/([^/]+)/jobs/(\d+)", parts.netloc + parts.path)
    if not m:
        return None
    r = client.get(f"https://boards-api.greenhouse.io/v1/boards/{m.group(1)}/jobs/{m.group(2)}")
    r.raise_for_status()
    j = r.json()
    return JobDescription(
        text=html_to_text(j.get("content", "")), title=j.get("title", ""),
        location=(j.get("location") or {}).get("name", ""), source="greenhouse api",
    )


def _lever(url: str, client: httpx.Client) -> JobDescription | None:
    m = re.search(r"jobs\.lever\.co/([^/]+)/([0-9a-f-]{36})", url)
    if not m:
        return None
    r = client.get(f"https://api.lever.co/v0/postings/{m.group(1)}/{m.group(2)}?mode=json")
    r.raise_for_status()
    j = r.json()
    sections = [f"{s.get('text', '')}\n{html_to_text(s.get('content', ''))}" for s in j.get("lists", [])]
    text = "\n\n".join(p for p in [j.get("descriptionPlain", ""), *sections, j.get("additionalPlain", "")] if p)
    return JobDescription(
        text=text, title=j.get("text", ""),
        location=(j.get("categories") or {}).get("location", ""), source="lever api",
    )


def _ashby(url: str, client: httpx.Client) -> JobDescription | None:
    m = re.search(r"jobs\.ashbyhq\.com/([^/]+)/([0-9a-f-]{36})", url)
    if not m:
        return None
    r = client.get(f"https://api.ashbyhq.com/posting-api/job-board/{m.group(1)}")
    r.raise_for_status()
    for j in r.json().get("jobs", []):
        if m.group(2) in (j.get("jobUrl") or "") or j.get("id") == m.group(2):
            return JobDescription(
                text=j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml", "")),
                title=j.get("title", ""), location=j.get("location", ""), source="ashby api",
            )
    raise JobDescriptionError("Posting not found on the Ashby board (closed?).")


def _json_ld(page: str) -> JobDescription | None:
    for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', page, re.S | re.I):
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data]) if isinstance(data, dict) else []
        for item in items:
            if isinstance(item, dict) and item.get("@type") == "JobPosting":
                org = item.get("hiringOrganization") or {}
                loc = item.get("jobLocation") or {}
                if isinstance(loc, list):
                    loc = loc[0] if loc else {}
                addr = (loc.get("address") or {}) if isinstance(loc, dict) else {}
                return JobDescription(
                    text=html_to_text(item.get("description", "")), title=item.get("title", ""),
                    company=org.get("name", "") if isinstance(org, dict) else "",
                    location=", ".join(
                        x for x in [addr.get("addressLocality", ""), addr.get("addressRegion", "")] if x
                    ),
                    source="page (schema.org JobPosting)",
                )
    return None


def _page_text(page: str) -> str:
    page = re.sub(r"<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)
    return html_to_text(page)


def fetch_job_description(url: str, client: httpx.Client | None = None) -> JobDescription:
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise JobDescriptionError("Not a link (must start with http:// or https://).")
    host = urlsplit(url).netloc.lower()
    if any(b in host for b in BLOCKED_HOSTS):
        raise JobDescriptionError(
            f"{host} does not allow automated reading: paste the description in the sheet."
        )
    own = client is None
    client = client or httpx.Client(timeout=25, follow_redirects=True, headers={"User-Agent": USER_AGENT})
    try:
        for reader in (_greenhouse, _lever, _ashby):
            jd = reader(url, client)
            if jd:
                break
        else:
            # Greenhouse-hosted posting embedded on a company site (?gh_jid=...)
            gh = parse_qs(urlsplit(url).query).get("gh_jid")
            r = client.get(url)
            r.raise_for_status()
            jd = _json_ld(r.text) or JobDescription(text=_page_text(r.text), source="page text")
            if gh and len(jd.text) < MIN_DESCRIPTION_CHARS:
                raise JobDescriptionError("Could not read the embedded posting: paste the description.")
    except httpx.HTTPStatusError as e:
        raise JobDescriptionError(f"The page answered HTTP {e.response.status_code}.") from e
    except httpx.HTTPError as e:
        raise JobDescriptionError(f"Could not reach the page ({type(e).__name__}).") from e
    finally:
        if own:
            client.close()
    if len(jd.text) < MIN_DESCRIPTION_CHARS:
        raise JobDescriptionError("The page did not contain a readable job description.")
    return jd
