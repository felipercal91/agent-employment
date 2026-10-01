"""Decide the language of employer-facing documents: French for Quebec postings, else English."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Language = Literal["en", "fr"]

_QUEBEC_PLACES = re.compile(
    r"\b(qu[eé]bec|montr[eé]al|laval|gatineau|sherbrooke|longueuil|trois[- ]rivi[eè]res|"
    r"saguenay|l[eé]vis|terrebonne|brossard|boucherville|saint[- ]laurent)\b",
    re.IGNORECASE,
)
_QC_CODE = re.compile(r"(?:,\s*|\b)QC\b")  # case-sensitive: "Montreal, QC"

# Employers headquartered in Quebec whose postings are typically for Quebec offices.
QUEBEC_COMPANIES = {
    "desjardins", "mouvement desjardins", "banque nationale", "national bank",
    "national bank of canada", "hydro-quebec", "hydro-québec", "loto-quebec", "loto-québec",
    "bombardier", "coveo", "nuvei", "lightspeed", "videotron", "vidéotron", "quebecor",
    "québecor", "saq", "caisse de depot", "cdpq", "ssq", "beneva", "la capitale",
}

_FRENCH_WORDS = re.compile(
    r"\b(nous|vous|votre|vos|avec|pour|dans|les|des|une|est|sont|expérience|compétences|"
    r"poste|équipe|développeur|exigences|atouts|connaissance|maîtrise)\b",
    re.IGNORECASE,
)
_ENGLISH_WORDS = re.compile(
    r"\b(we|you|your|with|for|the|and|are|is|experience|skills|role|team|developer|"
    r"requirements|knowledge)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Market:
    language: Language
    is_quebec: bool
    reason: str


def is_french_text(text: str) -> bool:
    fr = len(_FRENCH_WORDS.findall(text))
    en = len(_ENGLISH_WORDS.findall(text))
    return fr >= 5 and fr > en


def detect_market(posting_text: str, location: str = "", company: str = "") -> Market:
    """Quebec if the location, the company or the posting language says so."""
    if location and (_QUEBEC_PLACES.search(location) or _QC_CODE.search(location)):
        return Market("fr", True, f"location '{location}' is in Quebec")
    if company and company.strip().lower() in QUEBEC_COMPANIES:
        return Market("fr", True, f"'{company}' is a Quebec employer")
    if is_french_text(posting_text):
        return Market("fr", True, "posting is written in French")
    match = _QUEBEC_PLACES.search(posting_text) or _QC_CODE.search(posting_text)
    if match and not location:
        return Market("fr", True, f"posting mentions '{match.group(0).strip(', ')}'")
    return Market("en", False, "not a Quebec posting (default English)")
