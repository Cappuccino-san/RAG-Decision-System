"""FastAPI surface for the decision system.

Run locally with:
    uvicorn rag_decision_system.api:app --reload
The index is built from data/policies on startup if no saved index exists.
Then open http://127.0.0.1:8000/docs for the interactive API page.
"""

from __future__ import annotations

from fastapi import FastAPI

from .config import Settings
from .pipeline import DecisionSystem
from .schemas import DecisionRequest, DecisionResult, RetrievedChunk

settings = Settings.from_env()
system = DecisionSystem(settings=settings)
if system.index_exists():
    system.load_index()
else:
    system.build_index()
    system.save_index()

app = FastAPI(title="RAG Decision System", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok", "embedder": settings.embedder_mode, "decider": settings.decider_mode}


@app.post("/search", response_model=list[RetrievedChunk])
def search(request: DecisionRequest) -> list[RetrievedChunk]:
    """Semantic search over the policies — retrieval only, no decision."""
    return system.search(request.request)


@app.post("/decide", response_model=DecisionResult)
def decide(request: DecisionRequest) -> DecisionResult:
    """Return a structured, cited, confidence-scored decision for a request."""
    text = request.request if not request.context else f"{request.request} {request.context}"
    return system.decide(text)


@app.post("/reindex")
def reindex() -> dict[str, int]:
    """Rebuild the index from the policy directory."""
    count = system.build_index()
    system.save_index()
    return {"chunks_indexed": count}
