"""Vector store: holds chunks + their embeddings and answers similarity search.

The default implementation is in-memory with JSON persistence — zero
dependencies, and the right size for a policy corpus of this scale. It sits
behind the same small interface a real vector DB would: `add`, `search`,
`save`, `load`. Swapping in Chroma (the 'vectordb' extra) or pgvector means
writing one more class with those four methods; retrieval and everything
downstream do not change. The tradeoff is documented in DESIGN.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from .embeddings import Embedder, cosine_similarity
from .schemas import PolicyChunk, RetrievedChunk


class InMemoryVectorStore:
    """Cosine-similarity search over chunk embeddings held in memory."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._chunks: list[PolicyChunk] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[PolicyChunk]) -> None:
        """Embed and store chunks."""
        for chunk in chunks:
            self._chunks.append(chunk)
            self._vectors.append(self._embedder.embed(chunk.text))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks, best first."""
        query_vector = self._embedder.embed(query)
        scored = [
            RetrievedChunk(chunk=chunk, score=cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def save(self, path: str | Path) -> None:
        """Persist the index as JSON so the API can load it without re-embedding."""
        payload = {
            "chunks": [c.model_dump() for c in self._chunks],
            "vectors": self._vectors,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> InMemoryVectorStore:
        """Load a previously saved index."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(embedder)
        store._chunks = [PolicyChunk.model_validate(c) for c in payload["chunks"]]
        store._vectors = payload["vectors"]
        return store
