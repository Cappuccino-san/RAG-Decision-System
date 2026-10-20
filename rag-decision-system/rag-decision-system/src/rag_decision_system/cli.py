"""Command-line demo: build the index and decide the sample requests.

Usage:
    python -m rag_decision_system.cli data/sample_requests.json
Each sample request is a JSON object: {"request": "..."}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import Settings
from .pipeline import DecisionSystem


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    path = argv[0] if argv else "data/sample_requests.json"

    system = DecisionSystem(settings=Settings.from_env())
    chunk_count = system.build_index()
    print(f"Indexed {chunk_count} policy chunks.\n")

    requests = json.loads(Path(path).read_text(encoding="utf-8"))
    for entry in requests:
        result = system.decide(entry["request"])
        flags = ", ".join(result.risk_flags) if result.risk_flags else "-"
        top_cite = (
            f"{result.citations[0].document} / {result.citations[0].section}"
            if result.citations
            else "-"
        )
        print(f"REQUEST : {entry['request']}")
        print(
            f"DECISION: {result.decision.value:<20} confidence={result.confidence:.2f}  "
            f"flags={flags}"
        )
        print(f"EVIDENCE: {top_cite}")
        print(f"WHY     : {result.rationale}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
