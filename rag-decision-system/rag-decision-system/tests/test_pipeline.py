"""End-to-end tests over the real sample policy corpus (offline embedder)."""

import pytest

from rag_decision_system.config import Settings
from rag_decision_system.pipeline import DecisionSystem
from rag_decision_system.schemas import Decision


@pytest.fixture(scope="module")
def system() -> DecisionSystem:
    instance = DecisionSystem(settings=Settings(embedder_mode="hashing", decider_mode="rules"))
    instance.build_index()
    return instance


def test_prohibited_request_is_rejected_with_citations(system):
    result = system.decide("Store the team passwords in a shared spreadsheet")
    assert result.decision == Decision.REJECT
    assert result.citations  # every decision carries its evidence
    assert result.risk_flags


def test_compliant_request_is_approved(system):
    result = system.decide(
        "Share customer data with the vendor after a security review is complete "
        "and a signed DPA is in place, using encryption"
    )
    assert result.decision == Decision.APPROVE


def test_off_topic_request_needs_human_review(system):
    result = system.decide("What snacks should we order for the office kitchen next month?")
    assert result.decision == Decision.NEEDS_HUMAN_REVIEW


def test_search_returns_relevant_document(system):
    results = system.search("vendor data processing agreement security review", top_k=3)
    assert results
    assert any(r.chunk.document == "vendor_risk.md" for r in results)
