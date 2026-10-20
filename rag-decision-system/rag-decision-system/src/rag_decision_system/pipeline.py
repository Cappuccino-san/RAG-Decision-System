"""Pipeline: the stages wired together.

    policy documents
      -> chunk -> embed -> vector store          (build_index, once)

    request
      -> retrieve (semantic search, top-k chunks)
      -> confidence (computed from the retrieval evidence)
      -> decide (rules by default; LLM optional, guardrailed)
      -> DecisionResult (decision, rationale, citations, risk flags, confidence)
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .decide.llm import get_decider
from .embeddings import get_embedder
from .ingest.chunker import load_and_chunk
from .retrieve import Retriever
from .schemas import DecisionResult, RetrievedChunk
from .scoring import compute_confidence
from .vectorstore import InMemoryVectorStore


class DecisionSystem:
    """End-to-end RAG decision pipeline over a policy corpus."""

    def __init__(self, settings: Settings | None = None, store: InMemoryVectorStore | None = None):
        self._settings = settings or Settings.from_env()
        self._embedder = get_embedder(self._settings)
        self._store = store or InMemoryVectorStore(self._embedder)
        self._retriever = Retriever(self._store, top_k=self._settings.top_k)
        self._decider = get_decider(self._settings)

    def build_index(self, policy_dir: str | None = None) -> int:
        """Chunk + embed every policy document. Returns the chunk count."""
        chunks = load_and_chunk(policy_dir or self._settings.policy_dir)
        self._store = InMemoryVectorStore(self._embedder)
        self._store.add(chunks)
        self._retriever = Retriever(self._store, top_k=self._settings.top_k)
        return len(chunks)

    def save_index(self, path: str | None = None) -> None:
        """Persist the built index to disk."""
        self._store.save(path or self._settings.index_path)

    def load_index(self, path: str | None = None) -> None:
        """Load a previously built index from disk."""
        self._store = InMemoryVectorStore.load(path or self._settings.index_path, self._embedder)
        self._retriever = Retriever(self._store, top_k=self._settings.top_k)

    def index_exists(self, path: str | None = None) -> bool:
        return Path(path or self._settings.index_path).exists()

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Semantic search only — no decision."""
        return self._retriever.retrieve(query, top_k=top_k)

    def decide(self, request_text: str) -> DecisionResult:
        """Retrieve evidence, score confidence, and return a structured decision."""
        retrieved = self._retriever.retrieve(request_text)
        confidence = compute_confidence(retrieved)
        return self._decider.decide(request_text, retrieved, confidence)
