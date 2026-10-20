"""Tests for the decision rules — the judgment layer of the system."""

from rag_decision_system.decide.rules import RulesDecider, detect_flags
from rag_decision_system.schemas import Decision, PolicyChunk, RetrievedChunk

EVIDENCE = [
    RetrievedChunk(
        chunk=PolicyChunk(chunk_id="p-0", document="vendor_risk.md", section="S", text="policy text"),
        score=0.8,
    )
]


def decide(text: str, confidence: float = 0.9) -> Decision:
    return RulesDecider(confidence_threshold=0.30).decide(text, EVIDENCE, confidence).decision


def test_passwords_in_spreadsheet_is_rejected():
    assert decide("Store team passwords in a shared spreadsheet") == Decision.REJECT


def test_vendor_without_dpa_is_rejected():
    text = "Send customer data to the vendor without a DPA and no security review"
    assert decide(text) == Decision.REJECT


def test_contractor_access_is_flagged():
    assert decide("Give a contractor access to the production database") == Decision.FLAG


def test_stated_conditions_suppress_the_flag():
    text = "Share data with the vendor after a security review and a signed DPA is in place"
    assert decide(text) == Decision.APPROVE


def test_weak_evidence_goes_to_human_review():
    assert decide("Reorganize the office seating plan", confidence=0.05) == Decision.NEEDS_HUMAN_REVIEW


def test_reject_beats_weak_evidence():
    """A clear prohibition is rejected even when retrieval is weak."""
    assert decide("Store passwords in plain text", confidence=0.0) == Decision.REJECT


def test_clean_request_is_approved():
    assert decide("Enable MFA for admin access after manager approval") == Decision.APPROVE


def test_flags_are_named():
    names = [f.name for f in detect_flags("Store passwords in a spreadsheet")]
    assert "plaintext_password_storage" in names
