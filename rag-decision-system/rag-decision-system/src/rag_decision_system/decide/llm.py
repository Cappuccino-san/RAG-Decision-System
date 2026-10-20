"""LangChain LLM decider — an alternative decider with hard guardrails.

The LLM receives the request plus the retrieved policy chunks and returns a
structured DecisionResult via `with_structured_output`. Two guardrails are
applied in code, whatever the model returns:

1. A reject-level rule flag overrides an LLM approval — a model may not
   talk its way past a written prohibition.
2. Low retrieval confidence downgrades any decision to
   needs_human_review — the model only ever sees the retrieved evidence,
   so weak evidence means a weak basis for deciding.

LangChain is imported lazily; the rules decider remains the default and
the one used in tests and CI.
"""

from __future__ import annotations

from ..schemas import Decision, DecisionResult, RetrievedChunk
from .rules import RulesDecider, detect_flags

SYSTEM_PROMPT = """You decide whether a proposed action complies with company
policy, using ONLY the policy excerpts provided. Return a decision of
approve, flag, reject, or needs_human_review, a short rationale, and the
risk flags you relied on. If the excerpts do not cover the request, choose
needs_human_review."""


class LLMDecider:
    """Decide with a LangChain chat model, validated by the rules guardrails."""

    def __init__(self, model: str = "gpt-4o-mini", confidence_threshold: float = 0.30) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("LLM mode needs the 'llm' extra: pip install -e '.[llm]'") from exc
        llm = ChatOpenAI(model=model, temperature=0)
        self._chain = llm.with_structured_output(DecisionResult)
        self._threshold = confidence_threshold
        self._rules = RulesDecider(confidence_threshold=confidence_threshold)

    def decide(
        self, request_text: str, retrieved: list[RetrievedChunk], confidence: float
    ) -> DecisionResult:
        # Guardrail 1: reject-level flags are decided by the rules, full stop.
        flags = detect_flags(request_text)
        if any(f.severity == "reject" for f in flags):
            return self._rules.decide(request_text, retrieved, confidence)

        evidence = "\n\n".join(
            f"[{r.chunk.document} — {r.chunk.section}] {r.chunk.text}" for r in retrieved
        )
        prompt = f"{SYSTEM_PROMPT}\n\nRequest: {request_text}\n\nPolicy excerpts:\n{evidence}"
        result = DecisionResult.model_validate(self._chain.invoke(prompt))
        result.request = request_text
        result.confidence = confidence  # our computed score, not the model's

        # Guardrail 2: weak evidence downgrades to a human.
        if confidence < self._threshold and result.decision != Decision.REJECT:
            result.decision = Decision.NEEDS_HUMAN_REVIEW
            result.rationale += " (Downgraded: retrieval confidence below threshold.)"
        return result


def get_decider(settings):
    """Return the decider for the configured mode."""
    if settings.decider_mode == "llm":
        return LLMDecider(model=settings.llm_model, confidence_threshold=settings.confidence_threshold)
    return RulesDecider(confidence_threshold=settings.confidence_threshold)
