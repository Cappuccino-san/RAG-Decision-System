"""Deterministic risk flags and the rules decider.

Same philosophy as a business-rules engine: the *decision* is made by
ordered, named, testable rules — not by asking a model to vibes-check a
policy. Retrieval supplies the evidence (citations); these rules supply
the judgment, and every flag they raise is recorded by name.

Two severities:
- "reject" flags match a direct prohibition in the policies
  (e.g. passwords in a spreadsheet, vendor sharing without a DPA).
- "flag" flags match something that is only allowed with a condition the
  request does not state (e.g. granting a contractor access with no mention
  of manager approval) — allowed, but only with follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import Citation, Decision, DecisionResult, RetrievedChunk


@dataclass(frozen=True)
class FlagRule:
    """One named risk rule. `all_of` groups must each have a keyword present."""

    name: str
    severity: str  # "reject" | "flag"
    all_of: tuple[tuple[str, ...], ...]
    explanation: str


# Each inner tuple is an OR group; every group must match (AND of ORs).
FLAG_RULES: list[FlagRule] = [
    FlagRule(
        name="plaintext_password_storage",
        severity="reject",
        all_of=(("password",), ("plain text", "spreadsheet", "email", "sticky note")),
        explanation="Passwords may only be stored in the approved password manager.",
    ),
    FlagRule(
        name="shared_account_or_password",
        severity="reject",
        all_of=(("share", "shared", "sharing"), ("account", "password", "login")),
        explanation="Shared accounts and shared passwords are prohibited; every user needs a unique account.",
    ),
    FlagRule(
        name="customer_data_without_encryption",
        severity="reject",
        all_of=(
            ("customer data", "personal data", "customer report", "customer records",
             "customer export", "customer list"),
            ("without encryption", "unencrypted"),
        ),
        explanation="Customer and personal data must be encrypted in transit and at rest.",
    ),
    FlagRule(
        name="vendor_sharing_without_dpa_or_review",
        severity="reject",
        all_of=(
            ("vendor", "third party", "third-party"),
            ("without a dpa", "without dpa", "no dpa", "without a security review",
             "without security review", "no security review", "skip the review"),
        ),
        explanation="Vendors may only receive data after a security review and a signed DPA.",
    ),
    FlagRule(
        name="customer_data_on_personal_storage",
        severity="reject",
        all_of=(("customer data",), ("personal drive", "personal cloud", "personal laptop", "usb")),
        explanation="Customer data may only be stored in approved company systems.",
    ),
    FlagRule(
        name="company_data_for_personal_use",
        severity="reject",
        all_of=(("company data", "customer data", "customer list"), ("personal project", "side project")),
        explanation="Company and customer data may only be used for company business.",
    ),
    FlagRule(
        name="unapproved_software_install",
        severity="reject",
        all_of=(("install",), ("unapproved", "not approved", "from a forum")),
        explanation="Only approved software may be installed on company devices.",
    ),
    FlagRule(
        name="external_data_sharing_needs_verification",
        severity="flag",
        all_of=(("vendor", "third party", "third-party", "external", "auditor"),
                ("share", "send", "give", "provide")),
        explanation="External sharing is allowed only after vendor review and a signed DPA — verify both first.",
    ),
    FlagRule(
        name="third_party_access_without_stated_approval",
        severity="flag",
        all_of=(("contractor", "vendor", "consultant"), ("access",)),
        explanation="Third-party access requires manager approval and least-privilege scoping — neither is stated.",
    ),
]


def detect_flags(request_text: str) -> list[FlagRule]:
    """Return every flag rule matched by a request, in rule order.

    A "flag" rule is suppressed when the request already states the
    condition it exists to check for — flagging a request that says
    "after security review and a signed DPA" would just train users to
    ignore the flags.
    """
    text = request_text.lower()
    matched = []
    for rule in FLAG_RULES:
        if not all(any(keyword in text for keyword in group) for group in rule.all_of):
            continue
        if rule.name == "external_data_sharing_needs_verification" and (
            "signed dpa" in text and "security review" in text
        ):
            continue
        if rule.name == "third_party_access_without_stated_approval" and (
            "manager approval" in text or "approved by" in text
        ):
            continue
        matched.append(rule)
    return matched


def make_citations(retrieved: list[RetrievedChunk], limit: int = 3) -> list[Citation]:
    """Build citations from the top retrieved chunks."""
    return [
        Citation(
            document=r.chunk.document,
            section=r.chunk.section,
            quote=r.chunk.text[:200],
            score=round(r.score, 4),
        )
        for r in retrieved[:limit]
    ]


class RulesDecider:
    """Decide from flags + confidence, in a fixed, documented order.

    1. Any reject-level flag -> REJECT (a prohibition beats weak retrieval:
       the request is refused and a human sees the flag either way).
    2. Confidence below threshold -> NEEDS_HUMAN_REVIEW (evidence too weak
       to automate a decision).
    3. Any flag-level flag -> FLAG (proceed only with the stated condition).
    4. Otherwise -> APPROVE.
    """

    def __init__(self, confidence_threshold: float = 0.30) -> None:
        self._threshold = confidence_threshold

    def decide(
        self, request_text: str, retrieved: list[RetrievedChunk], confidence: float
    ) -> DecisionResult:
        flags = detect_flags(request_text)
        flag_names = [f.name for f in flags]
        citations = make_citations(retrieved)

        reject_flags = [f for f in flags if f.severity == "reject"]
        if reject_flags:
            return DecisionResult(
                request=request_text,
                decision=Decision.REJECT,
                rationale="; ".join(f.explanation for f in reject_flags),
                citations=citations,
                risk_flags=flag_names,
                confidence=confidence,
            )

        if confidence < self._threshold:
            return DecisionResult(
                request=request_text,
                decision=Decision.NEEDS_HUMAN_REVIEW,
                rationale=(
                    f"Retrieval confidence {confidence:.2f} is below the "
                    f"{self._threshold:.2f} threshold; the policy evidence is "
                    "too weak to decide automatically."
                ),
                citations=citations,
                risk_flags=flag_names,
                confidence=confidence,
            )

        soft_flags = [f for f in flags if f.severity == "flag"]
        if soft_flags:
            return DecisionResult(
                request=request_text,
                decision=Decision.FLAG,
                rationale="; ".join(f.explanation for f in soft_flags),
                citations=citations,
                risk_flags=flag_names,
                confidence=confidence,
            )

        return DecisionResult(
            request=request_text,
            decision=Decision.APPROVE,
            rationale="No prohibitions matched and the retrieved policy evidence supports the request.",
            citations=citations,
            risk_flags=[],
            confidence=confidence,
        )
