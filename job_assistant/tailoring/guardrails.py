"""Deterministic checks on generated CV text, applied after the LLM and before you review it."""

from __future__ import annotations

import re

from job_assistant.scoring.fit import extract_keywords

# "Ingénieur" is a protected title in Quebec (Ordre des ingénieurs du Québec).
_INGENIEUR = re.compile(r"\bing[ée]nieur(e)?(s)?\b", re.IGNORECASE)
_ENGINEER = re.compile(r"\bengineer\b", re.IGNORECASE)


def fix_quebec_title(title: str) -> str:
    """Replace Ingénieur/Engineer with Développeur in a French CV headline."""
    title = _INGENIEUR.sub(lambda m: "Développeuse" if m.group(1) else "Développeur", title)
    return _ENGINEER.sub("Développeur", title)


def protected_title_mentions(text: str) -> list[str]:
    """Sentences using 'ingénieur' as a title (to review by hand; 'ingénierie' is fine)."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?\n])", text)
        if _INGENIEUR.search(sentence)
    ]


def ungrounded_keywords(generated_text: str, source_texts: list[str]) -> list[str]:
    """Technologies the draft mentions that appear in none of the source material.

    Sources are the base CV, the retrieved excerpts and the profile skills; anything else
    in the draft is a likely fabrication and must be reviewed (or removed) by hand.
    """
    grounded = set(extract_keywords("\n".join(source_texts)))
    return [k for k in extract_keywords(generated_text) if k not in grounded]
