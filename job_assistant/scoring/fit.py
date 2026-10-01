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

_TOKEN = r"(?<![\w.#+-]){}(?![\w#+-]|\.\w)"
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
