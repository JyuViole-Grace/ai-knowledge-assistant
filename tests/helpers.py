"""Shared test doubles: a fake LLM and a model-free assistant stub."""

from app.answer import AnswerMixin
from app.config import Config
from app.evidence import EvidenceMixin
from app.schemas import Chunk, Document, SearchHit


class FakeLLM:
    """Deterministic stand-in for OllamaClient driven by a callable."""

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def generate(self, prompt, system="", json_mode=False):
        self.calls.append(prompt)
        return self.responder(prompt)


def make_document(text, doc_id="d1", title="book.txt"):
    return Document(
        doc_id=doc_id,
        path=f"/docs/{title}",
        title=title,
        text=text,
        fingerprint="fp",
    )


class StubAssistant(EvidenceMixin, AnswerMixin):
    """Evidence + answer logic without embeddings or Ollama."""

    def __init__(self, text, llm, **config):
        self.config = Config(**config)
        self.document = make_document(text)
        self.doc_by_id = {self.document.doc_id: self.document}
        self.llm = llm


def make_hit(document, start, end, ordinal=0, score=0.5):
    chunk = Chunk(
        chunk_id=f"{document.doc_id}:{ordinal}",
        doc_id=document.doc_id,
        text=document.text[start:end],
        start=start,
        end=end,
        ordinal=ordinal,
    )
    return SearchHit(chunk=chunk, rerank_score=score)
