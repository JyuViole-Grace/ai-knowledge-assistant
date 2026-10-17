"""API tests. Skipped automatically when fastapi/httpx are not installed."""

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import app.api as api  # noqa: E402
from app.schemas import AnswerResult, Evidence  # noqa: E402


class FakeAssistant:
    def __init__(self, config):
        self.config = config
        self.documents = [SimpleNamespace(title="book.txt")]
        self.chunks = [object(), object()]
        self.doc_by_id = {"d1": SimpleNamespace(title="book.txt")}

    def answer_question(self, question):
        evidence = Evidence(
            evidence_id="E1",
            claim="claim",
            quote="a verified quote",
            doc_id="d1",
            chunk_id="d1:0",
            source_start=1,
            source_end=17,
        )
        return AnswerResult(answer=f"Answer to {question} [E1]", evidence=[evidence])


@pytest.fixture
def client(monkeypatch, tmp_path):
    (tmp_path / "book.txt").write_text("hello", encoding="utf-8")
    monkeypatch.setenv("RAG_DOCS_ROOT", str(tmp_path))
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.setattr(api, "RAGAssistant", FakeAssistant)
    monkeypatch.setattr(api, "_assistant", None)
    return TestClient(api.app)


def test_health_before_and_after_indexing(client):
    assert client.get("/health").json() == {"status": "ok", "indexed": False}

    assert client.post("/index", json={"docs_path": "book.txt"}).status_code == 200

    body = client.get("/health").json()
    assert body["indexed"] is True and body["chunks"] == 2


def test_ask_requires_an_index(client):
    response = client.post("/ask", json={"question": "Why?"})
    assert response.status_code == 409


def test_index_then_ask_returns_answer_and_evidence(client):
    client.post("/index", json={"docs_path": "."})

    response = client.post("/ask", json={"question": "Why?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Answer to Why? [E1]"
    assert body["evidence"][0]["source"] == "book.txt"
    assert body["evidence"][0]["quote"] == "a verified quote"


def test_index_rejects_paths_outside_the_root(client):
    assert client.post("/index", json={"docs_path": ".."}).status_code == 403
    assert client.post("/index", json={"docs_path": "/etc"}).status_code == 403
    assert client.post("/index", json={"docs_path": "missing"}).status_code == 404


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "s3cret")

    assert client.post("/index", json={"docs_path": "."}).status_code == 401
    assert client.get("/health").status_code == 200  # health stays open

    headers = {"X-API-Key": "s3cret"}
    assert client.post("/index", json={"docs_path": "."}, headers=headers).status_code == 200
    assert client.post("/ask", json={"question": "Why?"}, headers=headers).status_code == 200
    assert client.post("/ask", json={"question": "Why?"}, headers={"X-API-Key": "bad"}).status_code == 401


def test_llm_outage_maps_to_502(client, monkeypatch):
    client.post("/index", json={"docs_path": "."})

    def boom(self, question):
        raise RuntimeError("Ollama connection failed")

    monkeypatch.setattr(FakeAssistant, "answer_question", boom)

    assert client.post("/ask", json={"question": "Why?"}).status_code == 502
