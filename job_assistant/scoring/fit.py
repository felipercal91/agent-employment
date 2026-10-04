"""Explicit, deterministic fit score: posting keywords vs. the candidate's real skills.

No LLM is involved, so the same posting always gets the same score and the reasoning is
inspectable: which technical keywords the posting mentions, which ones the profile covers,
which ones are missing, and whether the requested years of experience are met.

Formula:
    keyword_score = matched / found            (100 if the posting names no known keyword)
    score         = keyword_score - 15 if the posting asks for more years than the profile has
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from job_assistant.profile.candidate import CandidateProfile

YEARS_PENALTY = 15

# Canonical keyword -> aliases (matched case-insensitively as whole tokens).
# Keep canonical names readable: they are shown to the user in "missing" lists.
VOCABULARY: dict[str, list[str]] = {
    "Java": ["java", "java 8", "java 11", "java 17", "java 21", "jdk"],
    "Spring Boot": ["spring boot", "springboot"],
    "Spring": ["spring framework", "spring mvc", "spring cloud", "spring data", "spring security"],
    "Microservices": ["microservices", "microservice", "micro-services", "microservicios"],
    "REST APIs": ["rest", "restful", "rest api", "rest apis", "api rest"],
    "GraphQL": ["graphql"],
    "gRPC": ["grpc"],
    "Kafka": ["kafka", "apache kafka"],
    "RabbitMQ": ["rabbitmq"],
    "JMS": ["jms", "ibm mq", "activemq"],
    "Hibernate/JPA": ["hibernate", "jpa"],
    "SQL": ["sql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "Oracle": ["oracle", "pl/sql"],
    "MySQL": ["mysql"],
    "SQL Server": ["sql server", "mssql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "Cassandra": ["cassandra"],
    "Elasticsearch": ["elasticsearch", "elastic search", "opensearch"],
    "Docker": ["docker", "containers", "containerization"],
    "Kubernetes": ["kubernetes", "k8s", "openshift", "aks", "eks", "gke"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Terraform": ["terraform"],
    "CI/CD": ["ci/cd", "ci-cd", "continuous integration", "continuous delivery"],
    "Jenkins": ["jenkins"],
    "GitHub Actions": ["github actions"],
    "GitLab CI": ["gitlab ci", "gitlab-ci"],
    "Git": ["git"],
    "Maven": ["maven"],
    "Gradle": ["gradle"],
    "JUnit": ["junit"],
    "Mockito": ["mockito"],
    "TDD": ["tdd", "test-driven", "test driven"],
    "Kotlin": ["kotlin"],
    "Scala": ["scala"],
    "Python": ["python"],
    "Go": ["golang"],
    "C#/.NET": ["c#", ".net", "dotnet"],
    "JavaScript": ["javascript"],
    "TypeScript": ["typescript"],
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular"],
    "Node.js": ["node.js", "nodejs"],
    "Linux": ["linux", "unix"],
    "OAuth2/OIDC": ["oauth", "oauth2", "openid connect", "oidc", "jwt"],
    "Event-driven architecture": ["event-driven", "event driven", "event sourcing", "cqrs"],
    "Distributed systems": ["distributed systems"],
    "Observability": ["observability", "prometheus", "grafana", "datadog", "splunk", "new relic"],
    "Agile/Scrum": ["agile", "scrum", "kanban"],
    "LLM/GenAI": ["llm", "llms", "genai", "generative ai", "langchain", "rag"],
    "Machine learning": ["machine learning"],
}

# Compliance / risk / regulatory vocabulary. Always active, for every profile: it lets the
# "technologies not in your facts" check catch an invented regulation, tool or certification in
# a compliance CV exactly as it catches an invented framework in a developer CV.
COMPLIANCE_VOCABULARY: dict[str, list[str]] = {
    "AML": ["aml", "anti-money laundering", "anti money laundering", "money laundering",
            "lutte contre le blanchiment", "blanchiment d'argent", "lcbc"],
    "KYC": ["kyc", "know your customer", "connaissance du client"],
    "CDD/EDD": ["cdd", "edd", "customer due diligence", "enhanced due diligence", "due diligence"],
    "Sanctions screening": ["sanctions", "sanctions screening", "ofac", "watchlist", "watch list"],
    "Transaction monitoring": ["transaction monitoring", "surveillance des transactions"],
    "Suspicious transaction reporting": ["str", "sar", "suspicious transaction", "suspicious activity"],
    "FINTRAC": ["fintrac", "cetrf"],
    "PCMLTFA": ["pcmltfa", "lrpcfat"],
    "OSFI": ["osfi", "bsif"],
    "Regulatory reporting": ["regulatory reporting", "regulatory filings", "déclarations réglementaires"],
    "Regulatory change management": ["regulatory change", "regulatory horizon"],
    "Privacy / data protection": ["pipeda", "gdpr", "privacy", "data protection", "loi 25", "law 25",
                                  "protection des renseignements"],
    "SOX / internal controls": ["sox", "sarbanes-oxley", "internal controls", "internal control",
                                "contrôles internes"],
    "Basel": ["basel", "bâle"],
    "FATCA/CRS": ["fatca", "crs"],
    "Anti-bribery": ["fcpa", "cfpoa", "anti-bribery", "anti-corruption", "abac"],
    "Fraud prevention": ["fraud prevention", "fraud detection", "financial crime", "financial crimes",
                         "criminalité financière"],
    "Internal audit": ["internal audit", "audit interne", "audit"],
    "Risk assessment": ["risk assessment", "risk assessments", "risk-based approach", "évaluation des risques"],
    "Third-party risk": ["third-party risk", "third party risk", "vendor risk", "risque tiers"],
    "Operational risk": ["operational risk", "risque opérationnel"],
    "Policy & procedures": ["policies and procedures", "policy development", "politiques et procédures"],
    "Investigations": ["investigations", "enquêtes"],
    "Complaints handling": ["complaints handling", "complaint management", "gestion des plaintes"],
    "ISO 27001 / SOC 2": ["iso 27001", "soc 2", "soc2"],
    "Actimize": ["actimize"],
    "FircoSoft": ["fircosoft", "firco"],
    "World-Check": ["world-check", "worldcheck", "world check"],
    "LexisNexis": ["lexisnexis", "lexis nexis", "bridger"],
    "GRC tools": ["archer", "servicenow grc", "metricstream", "onetrust", "grc"],
    # Certifications: a CV must never claim one that is not in the candidate's facts.
    "PMP": ["pmp"],
    "CAMS": ["cams"],
    "CFE": ["cfe"],
    "CRCM": ["crcm"],
    "CIPP": ["cipp", "cipp/c"],
    "FRM": ["frm"],
    "CISA": ["cisa"],
    "CISSP": ["cissp"],
    "CPA": ["cpa"],
    "CFA": ["cfa"],
}
VOCABULARY.update(COMPLIANCE_VOCABULARY)

_TOKEN =r"(?<![\w.#+-]){}(?![\w#+-]|\.\w)"
_PATTERNS: dict[str, re.Pattern[str]] = {
    canonical: re.compile(
        "|".join(_TOKEN.format(re.escape(alias)) for alias in sorted(aliases, key=len, reverse=True)),
        re.IGNORECASE,
    )
    for canonical, aliases in VOCABULARY.items()
}

_YEARS = re.compile(
    r"(?<!\d)(\d{1,2})(?!\d)\s*\+?\s*(?:-|to|à)?\s*(?:\d{1,2}\s*)?\+?\s*(?:years?|yrs?|ans|años)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FitResult:
    score: int  # 0-100
    found: list[str]  # canonical keywords mentioned by the posting
    matched: list[str]
    missing: list[str]
    years_required: int | None
    years_ok: bool


def extract_keywords(text: str) -> list[str]:
    return [canonical for canonical, pattern in _PATTERNS.items() if pattern.search(text)]


def extract_years_required(text: str) -> int | None:
    """Largest plausible 'N years' figure in the posting (capped to avoid company-age noise)."""
    values = [int(m.group(1)) for m in _YEARS.finditer(text)]
    values = [v for v in values if 1 <= v <= 20]
    return max(values) if values else None


def profile_keywords(profile: CandidateProfile) -> set[str]:
    """Canonical keywords covered by the profile's skills (skills are matched via the vocabulary)."""
    return set(extract_keywords("\n".join(profile.skills)))


def score_fit(posting_text: str, profile: CandidateProfile) -> FitResult:
    found = extract_keywords(posting_text)
    have = profile_keywords(profile)
    matched = [k for k in found if k in have]
    missing = [k for k in found if k not in have]

    keyword_score = 100.0 if not found else 100.0 * len(matched) / len(found)
    years_required = extract_years_required(posting_text)
    years_ok = years_required is None or profile.years_experience >= years_required
    score = keyword_score - (0 if years_ok else YEARS_PENALTY)

    return FitResult(
        score=max(0, round(score)),
        found=found,
        matched=matched,
        missing=missing,
        years_required=years_required,
        years_ok=years_ok,
    )
