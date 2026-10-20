RAG-Powered Decision System
Semantic search over policy documents that returns a structured decision —
`approve`, `flag`, `reject`, or `needs_human_review` — with citations to
the exact policy evidence and a confidence score computed from the
retrieval, not from asking a model how sure it feels.
> A RAG chatbot gives you a paragraph and hopes you're convinced. Systems
> that act on an answer need a schema, evidence, and a number they can set
> a threshold on. That's what this returns.
What it does
Ingest — policy Markdown is split into section-aware, overlapping chunks.
Embed + index — chunks are embedded and stored in a vector store.
Retrieve — a proposed action is embedded and matched by cosine
similarity (semantic search) against the policies.
Score — confidence is computed from the evidence: best-match strength,
consistency across the top results, and source coverage.
Decide — named, deterministic rules check the request against the
prohibitions and conditions in the policies. An optional LangChain LLM
decider is available, guardrailed so it cannot override a prohibition
or a low-confidence escalation.
Return — one structured JSON decision with rationale, risk flags,
and citations.
Quick start (fully offline, no API key)
```bash
pip install -e ".[dev]"
python -m rag_decision_system.cli data/sample_requests.json
python eval/evaluate.py
pytest -q
```
Run the API:
```bash
uvicorn rag_decision_system.api:app --reload
# POST /decide  {"request": "Store team passwords in a shared spreadsheet"}
# POST /search  {"request": "vendor data processing agreement"}
```
Run with Docker:
```bash
docker build -t rag-decision-system .
docker run -p 8000:8000 rag-decision-system
```
Example output
```json
{
  "request": "Send the customer data export to our new vendor without a DPA...",
  "decision": "reject",
  "rationale": "Vendors may only receive data after a security review and a signed DPA.",
  "risk_flags": ["vendor_sharing_without_dpa_or_review", "external_data_sharing_needs_verification"],
  "confidence": 0.61,
  "citations": [{"document": "vendor_risk.md", "section": "Before sharing data with a vendor", "score": 0.71, "quote": "..."}]
}
```
Embedding modes
Mode	What it is	When to use it
`hashing` (default)	Deterministic offline stand-in — lexical overlap, not true semantics	Tests, CI, demos with no setup
`local`	sentence-transformers (`all-MiniLM-L6-v2`)	Real semantic search on your own machine
`openai`	OpenAI embeddings API	Hosted quality without local models
Set `EMBEDDER_MODE` in `.env` and compare them with the eval — the lift from
real embeddings is measured, not assumed.
Evaluation
```bash
python eval/evaluate.py
# Decision accuracy : 9/9
# Retrieval hit rate: 8/8   (in hashing mode; your numbers are printed live)
```
The eval scores retrieval and decisions separately: a miss tells you whether
the embedding/chunking layer or the decision rules need the work.
See DESIGN.md for the design decisions and tradeoffs.
Architecture
Module	Job
`schemas.py`	Pydantic models — `DecisionResult` is the product
`ingest/chunker.py`	Section-aware, overlapping, deterministic chunking
`embeddings.py`	Hashing / sentence-transformers / OpenAI behind one interface
`vectorstore.py`	In-memory vector store with JSON persistence (Chroma/pgvector-swappable)
`retrieve.py`	Semantic search, measured separately from deciding
`scoring.py`	Confidence from retrieval evidence — documented formula
`decide/rules.py`	Named risk-flag rules + the ordered decider
`decide/llm.py`	Optional LangChain decider, guardrailed by the rules
`pipeline.py`	Wires the stages together
`api.py`	FastAPI: `/decide`, `/search`, `/reindex`, `/health`
Tech
Python · RAG · embeddings / semantic search · LangChain (structured outputs) ·
Pydantic · FastAPI · pytest · Docker · GitHub Actions
Limitations
The bundled policies are synthetic examples, and the keyword risk rules are
deliberately simple — they demonstrate the pattern (named, testable rules +
citations), not production policy coverage. In production: a managed vector
DB, a larger labeled eval, and per-policy rule ownership. See DESIGN.md.
