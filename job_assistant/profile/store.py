"""Local Chroma vector store over the user's documents.

Embeddings are computed on this machine (Chroma's default ONNX MiniLM model), so document
text is never sent to a paid embedding service. Chroma telemetry is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from job_assistant.profile.loaders import SourceDocument

COLLECTION_NAME = "profile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    category: str


@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    category: str
    distance: float


def split_documents(documents: list[SourceDocument]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[Chunk] = []
    for doc in documents:
        for i, piece in enumerate(splitter.split_text(doc.text)):
            chunks.append(
                Chunk(id=f"{doc.source}#{i}", text=piece, source=doc.source, category=doc.category)
            )
    return chunks


def persistent_client(path: Path) -> ClientAPI:
    return chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )


class ProfileStore:
    def __init__(self, client: ClientAPI, embedding_function: Any | None = None) -> None:
        self._client = client
        self._embedding_function = (
            embedding_function or chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        )

    def _collection(self) -> Any:
        return self._client.get_or_create_collection(
            COLLECTION_NAME,
            embedding_function=self._embedding_function,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def rebuild(self, documents: list[SourceDocument], batch_size: int = 64) -> int:
        """Drop the current index and re-index `documents`. Returns the number of chunks."""
        if COLLECTION_NAME in [c.name for c in self._client.list_collections()]:
            self._client.delete_collection(COLLECTION_NAME)
        collection = self._collection()
        chunks = split_documents(documents)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{"source": c.source, "category": c.category} for c in batch],
            )
        return len(chunks)

    def count(self) -> int:
        return self._collection().count()

    def search(self, query: str, k: int = 6, category: str | None = None) -> list[Hit]:
        collection = self._collection()
        total = collection.count()
        if total == 0:
            return []
        result = collection.query(
            query_texts=[query],
            n_results=min(k, total),
            where={"category": category} if category else None,
        )
        return [
            Hit(text=text, source=meta["source"], category=meta["category"], distance=dist)
            for text, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
