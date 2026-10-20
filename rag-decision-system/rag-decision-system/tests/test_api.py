"""Smoke tests for the FastAPI surface."""

from fastapi.testclient import TestClient

from rag_decision_system.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_decide_endpoint_returns_structured_decision():
    response = client.post(
        "/decide", json={"request": "Store team passwords in a shared spreadsheet"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "reject"
    assert body["citations"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_search_endpoint_returns_chunks():
    response = client.post("/search", json={"request": "password manager requirements"})
    assert response.status_code == 200
    assert len(response.json()) >= 1
