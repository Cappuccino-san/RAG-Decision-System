"""Evaluation harness.

Scores the system on the labeled set in questions.json along two separate
axes, because they fail for different reasons:

- Retrieval hit rate: was the expected policy document in the top-k chunks?
  (If not, the embedding/chunking layer needs work.)
- Decision accuracy: did the final decision match the expected one?
  (If retrieval hit but the decision missed, the rules need work.)

Usage:
    python eval/evaluate.py
Run it in each embedding mode to compare them — that comparison is the
honest way to justify a "semantic search" claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_decision_system.config import Settings  # noqa: E402
from rag_decision_system.pipeline import DecisionSystem  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    settings = Settings.from_env()
    system = DecisionSystem(settings=settings)
    chunk_count = system.build_index(str(ROOT / "data" / "policies"))
    cases = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))

    decision_hits = 0
    retrieval_hits = 0
    retrieval_cases = 0

    print(f"Indexed {chunk_count} chunks | embedder={settings.embedder_mode} decider={settings.decider_mode}\n")
    for case in cases:
        result = system.decide(case["request"])
        retrieved = system.search(case["request"])
        decision_ok = result.decision.value == case["expected_decision"]

        retrieval_ok = None
        if case["expected_document"]:
            retrieval_cases += 1
            retrieval_ok = any(r.chunk.document == case["expected_document"] for r in retrieved)
            retrieval_hits += int(retrieval_ok)

        decision_hits += int(decision_ok)
        print(
            f"[{'PASS' if decision_ok else 'FAIL'}] expected={case['expected_decision']:<19} "
            f"got={result.decision.value:<19} conf={result.confidence:.2f} "
            f"retrieval={'-' if retrieval_ok is None else ('hit' if retrieval_ok else 'MISS')}"
        )
        print(f"       {case['request']}")

    total = len(cases)
    print(f"\nDecision accuracy : {decision_hits}/{total} = {decision_hits / total:.0%}")
    if retrieval_cases:
        print(f"Retrieval hit rate: {retrieval_hits}/{retrieval_cases} = {retrieval_hits / retrieval_cases:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
