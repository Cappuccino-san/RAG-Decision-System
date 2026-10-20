"""Embeddings: turn text into vectors for semantic search.

Three interchangeable modes behind one small interface (`embed`):
- "hashing" (default): a deterministic, dependency-free stand-in. It hashes
  word tokens into a fixed-size vector, so it captures lexical overlap, not
  true semantics. It exists so the pipeline, tests, CI, and the eval all run
  offline for free — and so the embedding layer can be benchmarked honestly.
- "local": sentence-transformers (real semantic embeddings, runs on CPU).
- "openai": OpenAI embeddings API.

Swapping modes changes exactly one thing — vector quality — and the eval
harness exists to measure how much that matters. See DESIGN.md.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

DIMENSIONS = 384
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    """Anything that can turn text into a fixed-size vector."""

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for one text."""
        ...


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (offline fallback)."""

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for token in _tokens(text):
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % DIMENSIONS
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize(vector)


class LocalEmbedder:
    """sentence-transformers embedder (install the 'local-embeddings' extra)."""

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "Local embeddings need: pip install -e '.[local-embeddings]'"
            ) from exc
        self._model = SentenceTransformer(model)

    def embed(self, text: str) -> list[float]:
        return _normalize(self._model.encode(text).tolist())


class OpenAIEmbedder:
    """OpenAI embeddings API (install the 'llm' extra for langchain-openai)."""

    def __init__(self, model: str) -> None:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("OpenAI embeddings need: pip install -e '.[llm]'") from exc
        self._embeddings = OpenAIEmbeddings(model=model)

    def embed(self, text: str) -> list[float]:
        return _normalize(self._embeddings.embed_query(text))


def get_embedder(settings) -> Embedder:
    """Return the embedder for the configured mode."""
    if settings.embedder_mode == "local":
        return LocalEmbedder(settings.local_embedding_model)
    if settings.embedder_mode == "openai":
        return OpenAIEmbedder(settings.embedding_model)
    return HashingEmbedder()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two normalized vectors (a dot product), clamped to 0..1."""
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))
