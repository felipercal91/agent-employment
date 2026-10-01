"""Read the user's documents (CVs, projects, interview feedback) from the data folder.

The category of a document is the name of its first-level subfolder under `data/`
(e.g. `data/cv/cv_en.pdf` -> "cv"). Files directly under `data/` are skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
# Subfolders that describe the candidate. Others (inbox, postings, qa_history, contacts...)
# hold job postings or operational data and must not be indexed as profile evidence.
PROFILE_CATEGORIES = frozenset({"cv", "projects", "feedback"})


@dataclass(frozen=True)
class SourceDocument:
    text: str
    source: str  # path relative to the data folder, with forward slashes
    category: str


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def _read_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    return path.read_text(encoding="utf-8", errors="replace")


def load_documents(
    data_dir: Path, categories: frozenset[str] | set[str] | None = None
) -> list[SourceDocument]:
    """Load supported files in subfolders of `data_dir` (only `categories`, if given)."""
    documents: list[SourceDocument] = []
    if not data_dir.is_dir():
        return documents

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(data_dir)
        if len(relative.parts) < 2 or path.name.startswith("~$"):
            continue
        if categories is not None and relative.parts[0] not in categories:
            continue
        try:
            text = read_text(path).strip()
        except Exception as e:
            logger.warning("Skipping %s: could not read it (%s)", relative, e)
            continue
        if not text:
            logger.warning("Skipping %s: no extractable text (scanned PDF?)", relative)
            continue
        documents.append(
            SourceDocument(text=text, source=relative.as_posix(), category=relative.parts[0])
        )
    return documents
