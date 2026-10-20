#!/usr/bin/env bash
# Build the index and run the demo against the sample requests (no API key needed).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m rag_decision_system.cli data/sample_requests.json
