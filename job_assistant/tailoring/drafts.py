"""Write tailored material to output/pending/<draft>/ in the `pending_approval` state.

Nothing here sends anything anywhere: a draft only leaves this folder when you copy it.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PENDING_DIR = "pending"
STATUS_PENDING = "pending_approval"


def slugify(text: str, max_len: int = 40) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    if len(slug) > max_len:  # cut at a word boundary
        slug = slug[: max_len + 1].rsplit("-", 1)[0]
    return slug or "untitled"


@dataclass
class Draft:
    company: str
    role: str
    kind: str  # "cv", later also "cover_letter", "message", "form_answers"
    files: dict[str, str]  # file name -> content
    meta: dict[str, object] = field(default_factory=dict)


def write_draft(output_dir: Path, draft: Draft, now: datetime | None = None) -> Path:
    now = now or datetime.now()
    folder = (
        output_dir
        / PENDING_DIR
        / f"{now:%Y-%m-%d}_{slugify(draft.company)}_{slugify(draft.role)}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in draft.files.items():
        (folder / name).write_text(content, encoding="utf-8")
    meta = {
        "status": STATUS_PENDING,
        "kind": draft.kind,
        "company": draft.company,
        "role": draft.role,
        "created_at": now.isoformat(timespec="seconds"),
        **draft.meta,
    }
    (folder / f"{draft.kind}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return folder
