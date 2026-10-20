"""Data schemas — the contract at every boundary of the system.

Downstream systems do not consume paragraphs; they consume decisions. That
is why the output of this project is the DecisionResult model below, and why
every field an automated system might act on is typed and validated here.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(str, Enum):
    """The four outcomes a request can receive."""

    APPROVE = "approve"
    FLAG = "flag"  # allowed only with conditions / follow-up
    REJECT = "reject"  # a policy rule prohibits this
    NEEDS_HUMAN_REVIEW = "needs_human_review"  # evidence too weak to decide


class PolicyChunk(BaseModel):
    """One chunk of a policy document, ready to embed."""

    chunk_id: str
    document: str  # source file name, e.g. "vendor_risk.md"
    section: str  # heading the chunk came from
    text: str


class RetrievedChunk(BaseModel):
    """A chunk returned by semantic search, with its similarity score."""

    chunk: PolicyChunk
    score: float  # cosine similarity, 0..1


class Citation(BaseModel):
    """A pointer back to the exact policy evidence behind a decision."""

    document: str
    section: str
    quote: str
    score: float


class DecisionRequest(BaseModel):
    """A proposed action to be checked against the policies."""

    request: str
    context: str | None = None


class DecisionResult(BaseModel):
    """The structured output of the whole system."""

    request: str
    decision: Decision
    rationale: str
    citations: list[Citation] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
