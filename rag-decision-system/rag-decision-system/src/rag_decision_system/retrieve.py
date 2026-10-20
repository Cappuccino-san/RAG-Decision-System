"""Retrieval: semantic search over the policy corpus.

Kept as its own thin module on purpose: retrieval quality is measured
separately from decision quality in the eval (a wrong decision can come
from bad retrieval or from bad rules, and you need to know which).
"""

from __future__ import annotations

from .schemas import RetrievedChunk
from .vectorstore import InMemoryVectorStore


class Retriever:
    """Query the vector store for the policy chunks most relevant to a request."""

    def __init__(self, store: InMemoryVectorStore, top_k: int = 5) -> None:
        self._store = store
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the most relevant chunks for a query, best first."""
        return self._store.search(query, top_k=top_k or self._top_k)
