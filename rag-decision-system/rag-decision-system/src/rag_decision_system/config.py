"""Configuration, loaded from environment variables.

Defaults run the whole system offline with no API key: hashing embeddings
(a deterministic stand-in, see embeddings.py) and the rules decider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Frozen so a run's config cannot change mid-pipeline."""

    embedder_mode: str = "hashing"  # "hashing" | "local" | "openai"
    decider_mode: str = "rules"  # "rules" | "llm"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-4o-mini"
    top_k: int = 5
    confidence_threshold: float = 0.30
    index_path: str = "data/index.json"
    policy_dir: str = "data/policies"

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the environment (see .env.example)."""
        return cls(
            embedder_mode=os.getenv("EMBEDDER_MODE", "hashing"),
            decider_mode=os.getenv("DECIDER_MODE", "rules"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            local_embedding_model=os.getenv(
                "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            top_k=int(os.getenv("TOP_K", "5")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.30")),
            index_path=os.getenv("INDEX_PATH", "data/index.json"),
            policy_dir=os.getenv("POLICY_DIR", "data/policies"),
        )
