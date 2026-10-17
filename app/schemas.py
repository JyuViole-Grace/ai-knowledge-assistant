"""Shared data models for the RAG pipeline."""

from dataclasses import dataclass


@dataclass
class Document:
    """A loaded source document."""

    doc_id: str
    path: str
    title: str
    text: str
    fingerprint: str


@dataclass
class Chunk:
    """A retrievable section of a source document."""

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    ordinal: int


@dataclass
class SearchHit:
    """A chunk and its retrieval scores."""

    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class Evidence:
    """A verified quote used to support an answer."""

    evidence_id: str
    claim: str
    quote: str
    doc_id: str
    chunk_id: str
    source_start: int
    source_end: int
    directness: int = 3
    retrieval_score: float = 0.0


@dataclass
class AnswerResult:
    """The outcome of answering one question (no shared mutable state)."""

    answer: str
    evidence: list[Evidence]
