"""The RAG assistant: wires ingestion, retrieval, evidence and answering."""

from typing import Optional

import numpy as np

from .answer import AnswerMixin
from .config import Config, log
from .evidence import EvidenceMixin
from .ingestion import load_documents, make_chunks
from .llm import OllamaClient
from .retrieval import RetrievalMixin
from .schemas import AnswerResult, Evidence


class RAGAssistant(RetrievalMixin, EvidenceMixin, AnswerMixin):
    def __init__(self, config: Config):
        self.config = config
        self.documents = load_documents(config.docs, config.debug)
        self.doc_by_id = {
            document.doc_id: document for document in self.documents
        }

        self.chunks = make_chunks(
            self.documents,
            config.chunk_size,
            config.chunk_overlap,
        )

        self.last_evidence: list[Evidence] = []
        self.embedding_model = None
        self.reranker = None
        self.embeddings: Optional[np.ndarray] = None
        self.bm25 = None

        self.llm = OllamaClient(
            config.ollama_url,
            config.model,
            config.timeout,
            config.num_ctx,
        )

        # The verification step may use a different (e.g. stronger) model.
        self.verify_llm = (
            OllamaClient(
                config.ollama_url,
                config.verify_model,
                config.timeout,
                config.num_ctx,
            )
            if config.verify_model and config.verify_model != config.model
            else self.llm
        )

        log(f"[CHUNK] Created: {len(self.chunks)}", config.debug)

        self.initialize_retrieval()

    def answer_question(self, question: str) -> AnswerResult:
        """Run the full pipeline and return the result.

        This method keeps no shared mutable state and prints nothing, so it
        is safe to call from several threads (e.g. the HTTP API).
        """
        hits = self.retrieve(question)
        candidates = self.extract_evidence(question, hits)
        verified = self.verify_evidence(question, candidates)
        evidence = self.select_evidence(verified)
        answer = self.generate_answer(question, evidence)

        return AnswerResult(answer=answer, evidence=evidence)

    def ask(self, question: str) -> str:
        """CLI-friendly wrapper: prints the answer and remembers evidence."""
        print("\n" + "=" * 78)
        print(f"[QUESTION] {question}")

        result = self.answer_question(question)
        self.last_evidence = result.evidence

        print("\n[ANSWER]")
        print(result.answer)

        if result.evidence:
            print("\n[VERIFIED EVIDENCE]")

            for item in result.evidence:
                document = self.doc_by_id[item.doc_id]

                print(f"[{item.evidence_id}] {item.quote}")
                print(
                    f"    Source: {document.title} | "
                    f"chunk={item.chunk_id} | "
                    f"chars={item.source_start}-{item.source_end}"
                )

        print("=" * 78)

        return result.answer
