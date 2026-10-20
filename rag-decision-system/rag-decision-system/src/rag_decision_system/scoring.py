"""Confidence scoring: how much should anyone trust a decision here?

The score is computed from the retrieval evidence, never asked from a model
("how confident are you?" is not a measurement). Three signals:

- top_score (60%): how well the single best chunk matches the request.
- mean_top_score (25%): whether the evidence is consistently on-topic,
  not one lucky chunk.
- coverage (15%): whether the evidence spans more than one document —
  a decision resting on a single source is weaker than one corroborated
  by two policies.

The formula is deliberately simple, documented, and unit-tested, so its
behavior can be argued about and tuned with the eval — that is the point.
"""

from __future__ import annotations

from .schemas import RetrievedChunk


def compute_confidence(retrieved: list[RetrievedChunk], top_n: int = 3) -> float:
    """Combine retrieval signals into a 0..1 confidence score."""
    if not retrieved:
        return 0.0

    top_score = retrieved[0].score
    head = retrieved[:top_n]
    mean_top = sum(r.score for r in head) / len(head)
    distinct_docs = len({r.chunk.document for r in head})
    coverage = distinct_docs / len(head)

    confidence = 0.60 * top_score + 0.25 * mean_top + 0.15 * coverage
    return round(max(0.0, min(1.0, confidence)), 4)
