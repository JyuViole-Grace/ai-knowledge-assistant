"""FastAPI interface for the AI Knowledge Assistant.

Environment variables:
    RAG_DOCS_ROOT  Directory the API may index (default: ``documents``).
                   Clients can only index paths inside it.
    RAG_API_KEY    If set, every request except /health must send it in the
                   ``X-API-Key`` header.
    OLLAMA_URL     Ollama server (default: http://localhost:11434).
"""

from __future__ import annotations

import os
import secrets
from threading import Lock, RLock
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .assistant import RAGAssistant
from .config import Config, setup_logging
from .security import resolve_docs_path

setup_logging()

app = FastAPI(
    title="AI Knowledge Assistant API",
    version="1.1.0",
    description="Local multilingual RAG API with verified source citations.",
)

_assistant: Optional[RAGAssistant] = None
_state_lock = RLock()   # guards swapping/reading the active index (short)
_index_lock = Lock()    # serializes expensive index builds


def docs_root() -> str:
    return os.getenv("RAG_DOCS_ROOT", "documents")


def require_api_key(
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    expected = os.getenv("RAG_API_KEY")

    if expected and not (
        x_api_key and secrets.compare_digest(x_api_key, expected)
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class IndexRequest(BaseModel):
    """Parameters used to build an in-memory document index."""

    docs_path: str = Field(
        default=".",
        max_length=500,
        description=(
            "File or directory to index, relative to the server's "
            "documents root. Paths outside the root are rejected."
        ),
    )
    model: str = Field(
        default="qwen2.5:7b",
        max_length=100,
        description="Local Ollama model used for evidence and answers.",
    )
    use_reranker: bool = True
    use_cache: bool = True


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to answer using the indexed documents.",
    )


class EvidenceResponse(BaseModel):
    id: str
    quote: str
    source: str
    chunk_id: str
    source_start: int
    source_end: int


class AskResponse(BaseModel):
    answer: str
    evidence: list[EvidenceResponse]


def _index_summary(assistant: RAGAssistant) -> dict[str, object]:
    return {
        "status": "ready",
        "documents": len(assistant.documents),
        "chunks": len(assistant.chunks),
        "sources": [document.title for document in assistant.documents],
    }


@app.get("/health")
def health() -> dict[str, object]:
    """Return service and in-memory index status."""
    with _state_lock:
        assistant = _assistant

    if assistant is None:
        return {"status": "ok", "indexed": False}

    return {"indexed": True, **_index_summary(assistant)}


@app.post("/index", dependencies=[Depends(require_api_key)])
def index_documents(request: IndexRequest) -> dict[str, object]:
    """Load documents and build a new in-memory retrieval index."""
    global _assistant

    try:
        path = resolve_docs_path(request.docs_path, docs_root())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        config = Config(
            docs=str(path),
            model=request.model,
            use_reranker=request.use_reranker,
            use_cache=request.use_cache,
        )

        # Build first; only replace a healthy existing index on success.
        with _index_lock:
            new_assistant = RAGAssistant(config)

        with _state_lock:
            _assistant = new_assistant

        return _index_summary(new_assistant)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Indexing failed: {exc}",
        ) from exc


@app.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(require_api_key)],
)
def ask_question(request: AskRequest) -> AskResponse:
    """Answer a question using the latest in-memory document index.

    The lock is held only to read the current index, not while the LLM
    works, so concurrent requests are processed in parallel.
    """
    with _state_lock:
        assistant = _assistant

    if assistant is None:
        raise HTTPException(
            status_code=409,
            detail="No documents are indexed. Call POST /index first.",
        )

    try:
        result = assistant.answer_question(request.question)
    except RuntimeError as exc:
        # e.g. Ollama is not reachable
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    evidence = [
        EvidenceResponse(
            id=item.evidence_id,
            quote=item.quote,
            source=assistant.doc_by_id[item.doc_id].title,
            chunk_id=item.chunk_id,
            source_start=item.source_start,
            source_end=item.source_end,
        )
        for item in result.evidence
    ]

    return AskResponse(answer=result.answer, evidence=evidence)
