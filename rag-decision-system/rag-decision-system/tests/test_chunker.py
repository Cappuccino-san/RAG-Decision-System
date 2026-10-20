"""Tests for chunking: deterministic, section-aware, size-bounded."""

from rag_decision_system.ingest.chunker import chunk_document

DOC = """# Test Policy

## First Section
Alpha paragraph one is here.

Beta paragraph two follows it.

## Second Section
Gamma paragraph is in the second section.
"""


def test_chunks_keep_their_section():
    chunks = chunk_document("test.md", DOC)
    sections = {c.section for c in chunks}
    assert "First Section" in sections
    assert "Second Section" in sections


def test_chunk_ids_are_stable_and_unique():
    first = [c.chunk_id for c in chunk_document("test.md", DOC)]
    second = [c.chunk_id for c in chunk_document("test.md", DOC)]
    assert first == second  # deterministic
    assert len(first) == len(set(first))  # unique


def test_long_sections_are_split():
    long_doc = "# P\n\n## S\n" + "\n\n".join(f"Paragraph number {i} with some text." for i in range(40))
    chunks = chunk_document("long.md", long_doc, max_chars=200, overlap_chars=20)
    assert len(chunks) > 1
