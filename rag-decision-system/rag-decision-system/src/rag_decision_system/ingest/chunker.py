"""Document loading and chunking.

Chunking strategy (documented because it is a real design tradeoff):
- Split first by Markdown heading, so every chunk keeps the section it came
  from — that section becomes part of the citation.
- Pack the paragraphs under a heading into chunks of at most `max_chars`,
  with `overlap_chars` carried over so a rule that straddles a boundary is
  still retrievable whole from at least one chunk.
- Fully deterministic: same document + same settings = same chunks, which
  is what makes the eval reproducible.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..schemas import PolicyChunk


def load_documents(policy_dir: str | Path) -> list[tuple[str, str]]:
    """Return (document_name, text) for every .md file in the policy directory."""
    directory = Path(policy_dir)
    return [
        (path.name, path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.md"))
    ]


def _sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into (section_heading, body) pairs."""
    parts: list[tuple[str, str]] = []
    heading, body_lines = "Introduction", []
    for line in text.splitlines():
        match = re.match(r"^#{1,3}\s+(.*)", line)
        if match:
            if body_lines:
                parts.append((heading, "\n".join(body_lines).strip()))
            heading, body_lines = match.group(1).strip(), []
        else:
            body_lines.append(line)
    if body_lines:
        parts.append((heading, "\n".join(body_lines).strip()))
    return [(h, b) for h, b in parts if b]


def chunk_document(
    document: str, text: str, max_chars: int = 600, overlap_chars: int = 100
) -> list[PolicyChunk]:
    """Split one document into overlapping, section-aware chunks."""
    chunks: list[PolicyChunk] = []
    for section, body in _sections(text):
        paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
        buffer = ""
        for paragraph in paragraphs:
            if buffer and len(buffer) + len(paragraph) + 1 > max_chars:
                chunks.append(_make_chunk(document, section, buffer, len(chunks)))
                buffer = buffer[-overlap_chars:] + " " + paragraph if overlap_chars else paragraph
            else:
                buffer = f"{buffer} {paragraph}".strip()
        if buffer:
            chunks.append(_make_chunk(document, section, buffer, len(chunks)))
    return chunks


def _make_chunk(document: str, section: str, text: str, index: int) -> PolicyChunk:
    stem = document.removesuffix(".md")
    return PolicyChunk(chunk_id=f"{stem}-{index}", document=document, section=section, text=text)


def load_and_chunk(policy_dir: str | Path) -> list[PolicyChunk]:
    """Load every policy document and return all chunks."""
    chunks: list[PolicyChunk] = []
    for name, text in load_documents(policy_dir):
        chunks.extend(chunk_document(name, text))
    return chunks
