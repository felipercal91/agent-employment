"""Pick the base CV variant (language, with/without Kafka) from the files in data/cv/."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from job_assistant.profile.loaders import SourceDocument
from job_assistant.tailoring.market import Language, is_french_text

_FR = re.compile(r"(^|[^a-z])(fr|french|francais|français|quebec|québec)([^a-z]|$)")
_EN = re.compile(r"(^|[^a-z])(en|english|ingles|inglés)([^a-z]|$)")
_NO_KAFKA = re.compile(r"(no|sin|without|sans|wo)[^a-z]*kafka")


def cv_language(doc: SourceDocument) -> Language:
    name = PurePosixPath(doc.source).stem.lower()
    if _FR.search(name):
        return "fr"
    if _EN.search(name):
        return "en"
    return "fr" if is_french_text(doc.text) else "en"


def has_kafka_variant(doc: SourceDocument) -> bool | None:
    """True = Kafka variant, False = explicitly without Kafka, None = not a Kafka variant."""
    name = PurePosixPath(doc.source).stem.lower()
    if _NO_KAFKA.search(name):
        return False
    if "kafka" in name:
        return True
    return None


def select_base_cv(
    cvs: list[SourceDocument], language: Language, wants_kafka: bool
) -> SourceDocument | None:
    """Best CV for the language; among those, prefer the matching Kafka variant."""
    candidates = [d for d in cvs if cv_language(d) == language]
    if not candidates:
        return None

    def rank(doc: SourceDocument) -> int:
        variant = has_kafka_variant(doc)
        if variant is None:
            return 1  # generic CV
        return 0 if variant == wants_kafka else 2

    return sorted(candidates, key=lambda d: (rank(d), d.source))[0]
