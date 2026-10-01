import hashlib
import math
import re
from pathlib import Path
from typing import Any

import chromadb
import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings as ChromaSettings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from job_assistant.agents.profile_agent import NOT_FOUND, ProfileAgent, build_messages
from job_assistant.profile.loaders import load_documents
from job_assistant.profile.store import Hit, ProfileStore, split_documents


class HashingEmbedding(EmbeddingFunction[Documents]):
    """Deterministic bag-of-words embedding so tests need no model download."""

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    @staticmethod
    def name() -> str:
        return "test-hashing"

    def get_config(self) -> dict[str, Any]:
        return {"dims": self.dims}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashingEmbedding":
        return HashingEmbedding(config["dims"])

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vec = [0.0] * self.dims
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "cv").mkdir(parents=True)
    (root / "projects").mkdir()
    (root / "feedback").mkdir()
    (root / "README.md").write_text("not a profile document", encoding="utf-8")
    (root / "cv" / "cv_en.md").write_text(
        "Senior Java developer. Spring Boot microservices for banking.", encoding="utf-8"
    )
    (root / "projects" / "minibank.txt").write_text(
        "MiniBank: accounts and transfers service built with Spring Boot and PostgreSQL.",
        encoding="utf-8",
    )
    (root / "feedback" / "altimetrik.md").write_text(
        "Altimetrik interview: strong banking domain knowledge, did not finish live coding exercise.",
        encoding="utf-8",
    )
    (root / "cv" / "empty.txt").write_text("   ", encoding="utf-8")
    (root / "cv" / "photo.png").write_bytes(b"\x89PNG")
    return root


@pytest.fixture
def store() -> ProfileStore:
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    for c in client.list_collections():
        client.delete_collection(c.name)
    return ProfileStore(client, embedding_function=HashingEmbedding())


def test_load_documents_categories_and_skips(data_dir: Path) -> None:
    docs = load_documents(data_dir)
    assert sorted(d.source for d in docs) == [
        "cv/cv_en.md",
        "feedback/altimetrik.md",
        "projects/minibank.txt",
    ]
    assert {d.source: d.category for d in docs}["feedback/altimetrik.md"] == "feedback"


def test_load_docx(tmp_path: Path) -> None:
    from docx import Document

    (tmp_path / "cv").mkdir()
    doc = Document()
    doc.add_paragraph("Développeur Java senior")
    doc.save(str(tmp_path / "cv" / "cv_fr.docx"))
    [loaded] = load_documents(tmp_path)
    assert "Développeur Java senior" in loaded.text


def test_load_missing_dir(tmp_path: Path) -> None:
    assert load_documents(tmp_path / "nope") == []


def test_split_long_document_keeps_metadata(data_dir: Path) -> None:
    from job_assistant.profile.loaders import SourceDocument

    long_doc = SourceDocument(text="word " * 1000, source="cv/long.md", category="cv")
    chunks = split_documents([long_doc])
    assert len(chunks) > 1
    assert all(c.source == "cv/long.md" and c.category == "cv" for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_rebuild_and_search(store: ProfileStore, data_dir: Path) -> None:
    assert store.search("anything") == []
    assert store.rebuild(load_documents(data_dir)) == 3
    hits = store.search("live coding exercise interview", k=1)
    assert hits[0].source == "feedback/altimetrik.md"
    only_cv = store.search("Spring Boot", category="cv")
    assert {h.category for h in only_cv} == {"cv"}


def test_rebuild_replaces_previous_index(store: ProfileStore, data_dir: Path) -> None:
    docs = load_documents(data_dir)
    store.rebuild(docs)
    store.rebuild(docs[:1])
    assert store.count() == 1


def test_build_messages_numbers_excerpts() -> None:
    hits = [Hit("text A", "cv/a.md", "cv", 0.1), Hit("text B", "feedback/b.md", "feedback", 0.2)]
    system, human = build_messages("What is my stack?", hits)
    assert "ONLY" in system.content
    assert "[1] (source: cv/a.md" in human.content
    assert "[2] (source: feedback/b.md" in human.content
    assert human.content.endswith("Question: What is my stack?")


def test_agent_does_not_call_llm_without_context(store: ProfileStore) -> None:
    llm = FakeListChatModel(responses=["I invented this"])
    result = ProfileAgent(store, llm).ask("Do I know Kubernetes?")
    assert result.answer == NOT_FOUND
    assert result.grounded is False


def test_agent_returns_answer_and_sources(store: ProfileStore, data_dir: Path) -> None:
    store.rebuild(load_documents(data_dir))
    llm = FakeListChatModel(responses=["You built MiniBank with Spring Boot [1]."])
    result = ProfileAgent(store, llm, k=2).ask("What is MiniBank?")
    assert result.grounded is True
    assert "MiniBank" in result.answer
    assert "projects/minibank.txt" in result.sources


def test_agent_marks_not_found_answer_as_ungrounded(store: ProfileStore, data_dir: Path) -> None:
    store.rebuild(load_documents(data_dir))
    llm = FakeListChatModel(responses=[NOT_FOUND])
    assert ProfileAgent(store, llm).ask("Do I have a PMP?").grounded is False
