"""Tests for retrieval and the confidence score."""

import pytest

from rag_decision_system.embeddings import HashingEmbedder, cosine_similarity
from rag_decision_system.schemas import PolicyChunk, RetrievedChunk
from rag_decision_system.scoring import compute_confidence
from rag_decision_system.vectorstore import InMemoryVectorStore


def make_chunk(doc: str, text: str, idx: int = 0) -> PolicyChunk:
    return PolicyChunk(chunk_id=f"{doc}-{idx}", document=doc, section="S", text=text)


def test_identical_text_scores_highest():
    store = InMemoryVectorStore(HashingEmbedder())
    store.add([
        make_chunk("vendor.md", "vendors need a signed data processing agreement", 0),
        make_chunk("remote.md", "remote work requires a company vpn connection", 1),
    ])
    results = store.search("signed data processing agreement for vendors", top_k=1)
    assert results[0].chunk.document == "vendor.md"


def test_cosine_of_identical_vectors_is_one():
    embedder = HashingEmbedder()
    vector = embedder.embed("hello world")
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_confidence_is_zero_with_no_evidence():
    assert compute_confidence([]) == 0.0


def test_strong_evidence_beats_weak_evidence():
    strong = [RetrievedChunk(chunk=make_chunk("a.md", "x"), score=0.9),
              RetrievedChunk(chunk=make_chunk("b.md", "y"), score=0.85)]
    weak = [RetrievedChunk(chunk=make_chunk("a.md", "x"), score=0.1)]
    assert compute_confidence(strong) > compute_confidence(weak)


def test_confidence_stays_in_range():
    results = [RetrievedChunk(chunk=make_chunk("a.md", "x"), score=1.0)]
    score = compute_confidence(results)
    assert 0.0 <= score <= 1.0
