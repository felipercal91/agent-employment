"""Chile market: which postings qualify (100% remote) and what to record about each one.

Everything here is deterministic text analysis (no AI, no cost). Canada never goes through it: a row is
treated as Chile only when its Country column says Chile.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

COUNTRY_CANADA = "Canadá"
COUNTRY_CHILE = "Chile"
NOT_STATED = "No indicado"
YES, NO = "Sí", "No"


def _fold(text: str) -> str:
    raw = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", raw).casefold()


def is_chile(country: object) -> bool:
    """True when the row's Country column says Chile. Empty means Canada (every row from before this column)."""
    return _fold(str(country or "")).strip() == "chile"


# ---- 100% remote filter -------------------------------------------------------------------------

_FULL_REMOTE = re.compile(
    r"100\s?%\s*(?:remot|a distancia|teletrabaj|online)|full[- ]?remote|fully[- ]remote|totalmente remot"
    r"|remoto al 100|trabajo remoto completo|remote[- ]first|modalidad remota|100% remote"
)
_REMOTE = re.compile(r"\bremot[oa]s?\b|\bremote\b|teletrabajo|trabajo a distancia|work from home")
_NOT_REMOTE = re.compile(
    r"\bhibrid[oa]s?\b|\bhybrid\b|\bpresencial(?:es|idad)?\b|\bon[- ]?site\b|\bin[- ]office\b|modalidad mixta"
    r"|\bsemipresencial\b|dias? en (?:la )?oficina|trabajo en oficina"
)
# "no es presencial", "sin presencialidad": the posting is saying it is NOT on-site.
_NEGATED = re.compile(
    r"(?:\bno\b|\bsin\b|\bni\b)\s+(?:es\s+|requiere\s+|exige\s+|hay\s+|necesit\w+\s+)?"
    r"(?:\w+\s+)?(?:hibrid\w*|presencial\w*|on[- ]?site|oficina)"
)


@dataclass(frozen=True)
class RemoteVerdict:
    fully_remote: bool
    reason: str  # in Spanish: it is written into the tracker for the user


def check_fully_remote(text: str) -> RemoteVerdict:
    """100% remote only. A posting that mentions hybrid or on-site work (without saying it is not) is out."""
    folded = _fold(text)
    if _FULL_REMOTE.search(folded):
        return RemoteVerdict(True, "La vacante dice 100% remoto.")
    cleaned = _NEGATED.sub(" ", folded)
    on_site = _NOT_REMOTE.search(cleaned)
    if on_site:
        return RemoteVerdict(False, f"Descartada: la vacante menciona «{on_site.group(0)}» (se piden solo trabajos 100% remotos).")
    if _REMOTE.search(folded):
        return RemoteVerdict(True, "La vacante es remota y no menciona presencialidad.")
    return RemoteVerdict(False, "Descartada: la vacante no dice que sea remota.")


# ---- details recorded for each Chilean posting ---------------------------------------------------

_RESIDENCE_YES = re.compile(
    r"(?:residir|residente|radicad[oa]|vivir|residencia|domicilio|domiciliad[oa]|basad[oa]|ubicad[oa])\s+(?:\w+\s+){0,3}?en\s+chile"
    r"|(?:debes?|deberas?|requiere|requisito|necesario|excluyente)[^.]{0,40}(?:vivir|residir)[^.]{0,20}chile"
    r"|solo (?:candidatos|postulantes|personas)[^.]{0,30}(?:en|de) chile|chile\s+(?:solo|unicamente)"
)
_RESIDENCE_NO = re.compile(
    r"cualquier (?:parte|lugar|pais)|desde cualquier|anywhere|latam|latinoamerica|worldwide|"
    r"sin restriccion(?:es)? geografica"
)
_CONTRACTS = (
    ("Indefinido", re.compile(r"\bindefinid[oa]\b|contrato permanente|\bpermanent\b")),
    ("Plazo fijo", re.compile(r"plazo fijo|contrato temporal|\bfixed[- ]term\b|por proyecto|por obra")),
    ("Honorarios", re.compile(r"honorarios|\bfreelance\b|boleta|\bcontractor\b|independiente")),
)
_MONEY = re.compile(r"(?<![a-z])(?:clp\s*)?\$\s*(\d{1,3}(?:[.,]\d{3}){2,}|\d{6,9})(?:\s*(?:-|–|a|al?)\s*\$?\s*(\d{1,3}(?:[.,]\d{3}){2,}|\d{6,9}))?")
_MILLONES = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:a|-|–)?\s*(\d+(?:[.,]\d+)?)?\s*millon(?:es)?")


@dataclass(frozen=True)
class ChileDetails:
    requires_residence: str
    contract: str
    salary_clp: str


def _int(raw: str) -> int:
    return int(re.sub(r"[.,]", "", raw))


def _clp(n: int) -> str:
    return "$" + f"{n:,}".replace(",", ".")


def _salary(folded: str) -> str:
    """The posted salary in CLP, or NOT_STATED. Amounts tied to US dollars are not CLP and are ignored."""
    for m in _MONEY.finditer(folded):
        before = folded[max(0, m.start() - 6): m.start()]
        after = folded[m.end(): m.end() + 12]
        if "us" in before[-3:] or "usd" in after or "dolar" in after:
            continue
        low, high = _int(m.group(1)), _int(m.group(2)) if m.group(2) else None
        if low < 200_000:  # not a monthly CLP salary (a fee, a bonus, a typo)
            continue
        around = folded[max(0, m.start() - 25): m.end() + 60]
        kind = " líquidos" if "liquid" in around else " brutos" if "bruto" in around else ""
        return f"{_clp(low)}" + (f" - {_clp(high)}" if high and high >= low else "") + f" CLP{kind}"
    m = _MILLONES.search(folded)
    if m and "clp" in folded or (m and "pesos" in folded):
        low = int(float(m.group(1).replace(",", ".")) * 1_000_000)
        high = int(float(m.group(2).replace(",", ".")) * 1_000_000) if m.group(2) else None
        if low >= 200_000:
            return _clp(low) + (f" - {_clp(high)}" if high else "") + " CLP"
    return NOT_STATED


def extract_details(text: str) -> ChileDetails:
    folded = _fold(text)
    if _RESIDENCE_YES.search(folded):
        residence = YES
    elif _RESIDENCE_NO.search(folded):
        residence = NO
    else:
        residence = NOT_STATED
    found = [name for name, pattern in _CONTRACTS if pattern.search(folded)]
    return ChileDetails(residence, " / ".join(found) or NOT_STATED, _salary(folded))


# ---- role matching for Spanish titles ------------------------------------------------------------

# Spanish job titles per role id. They are added to the roles only for Chilean rows, so Canada's matching
# (roles.toml, English and French titles) is exactly as before.
SPANISH_TITLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "bsa-insurance": ("Analista Funcional", "Analista de Negocios", "Analista de Sistemas", "Analista de Requerimientos",
                      "Analista Funcional de Seguros", "Analista Senior de Negocios"),
    "tech-ba-fintech": ("Analista Técnico", "Analista de Negocios Técnico", "Analista Técnico de Negocios"),
    "app-support-java": ("Soporte de Aplicaciones", "Ingeniero de Soporte", "Soporte Aplicativo", "Analista de Soporte",
                         "Soporte Productivo", "Soporte N2", "Soporte N3", "Ingeniero de Soporte de Aplicaciones"),
    "implementation-consultant": ("Consultor de Implementación", "Consultor Funcional", "Consultor de Soluciones",
                                  "Consultor Técnico", "Consultor de Negocios"),
    "data-bi-analyst": ("Analista de Datos", "Analista BI", "Analista de Inteligencia de Negocios", "Desarrollador BI",
                        "Ingeniero de Datos", "Analista de Reportería", "Inteligencia de Negocios"),
    "qa-analyst": ("Analista QA", "Analista de Calidad", "Aseguramiento de Calidad", "Ingeniero QA", "Tester",
                   "Analista de Pruebas", "QA"),
    "java-backend": ("Desarrollador Java", "Desarrollador Backend", "Ingeniero de Software", "Desarrollador de Software",
                     "Desarrollador Full Stack", "Programador Java", "Ingeniero Java", "Desarrollador Senior"),
    "tech-lead-java": ("Líder Técnico", "Jefe Técnico", "Líder de Desarrollo", "Líder de Equipo Técnico",
                       "Líder Técnico Java", "Tech Lead"),
}


def spanish_roles(roles: list) -> list:
    """The same roles with Spanish title keywords added, for matching a Chilean posting's title."""
    from dataclasses import replace

    return [replace(r, title_keywords=tuple(r.title_keywords) + SPANISH_TITLE_KEYWORDS.get(r.id, ())) for r in roles]
