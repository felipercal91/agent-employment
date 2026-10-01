"""Profile / RAG agent: answers context questions strictly from the user's real documents."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from job_assistant.llm import message_text
from job_assistant.profile.store import Hit, ProfileStore

NOT_FOUND = "Not found in your documents."

SYSTEM_PROMPT = f"""You answer questions about one specific job candidate, using ONLY the \
numbered excerpts from the candidate's own documents (CVs, project notes, interview feedback) \
provided in the user message.

Rules:
- Every factual claim must be supported by an excerpt; cite it inline as [n].
- Never infer, embellish or generalize experience, skills, years, employers, titles or \
certifications that the excerpts do not state explicitly.
- If the excerpts only partially answer the question, answer that part and say plainly what \
is not covered.
- If the excerpts do not answer the question at all, reply exactly: "{NOT_FOUND}"
- Answer in the same language as the question. Be concise."""


@dataclass(frozen=True)
class ProfileAnswer:
    answer: str
    sources: list[str]
    grounded: bool  # False when nothing relevant was retrieved or the model found no support


def build_messages(question: str, hits: list[Hit]) -> list[BaseMessage]:
    excerpts = "\n\n".join(
        f"[{i}] (source: {hit.source}, category: {hit.category})\n{hit.text}"
        for i, hit in enumerate(hits, start=1)
    )
    return [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"<excerpts>\n{excerpts}\n</excerpts>\n\nQuestion: {question}"),
    ]


class ProfileAgent:
    def __init__(self, store: ProfileStore, llm: BaseChatModel, k: int = 6) -> None:
        self._store = store
        self._llm = llm
        self._k = k

    def ask(self, question: str, category: str | None = None) -> ProfileAnswer:
        hits = self._store.search(question, k=self._k, category=category)
        if not hits:
            # Nothing indexed (or nothing in that category): don't let the model improvise.
            return ProfileAnswer(answer=NOT_FOUND, sources=[], grounded=False)

        answer = message_text(self._llm.invoke(build_messages(question, hits))).strip()
        sources = list(dict.fromkeys(hit.source for hit in hits))
        return ProfileAnswer(answer=answer, sources=sources, grounded=answer != NOT_FOUND)
