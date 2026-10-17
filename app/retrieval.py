"""Hybrid retrieval: dense embeddings + BM25 fused with RRF, then rerank."""

import json
import logging
import re
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:  # pragma: no cover - depends on the environment
    CrossEncoder = None
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - depends on the environment
    BM25Okapi = None

from .config import log, stable_id, tokenize
from .ingestion import INGESTION_VERSION
from .schemas import SearchHit

RRF_K = 60


class RetrievalMixin:
    def bm25_tokens(self, text: str) -> list[str]:
        """Tokens for BM25, optionally cut to ``bm25_prefix`` characters."""
        tokens = tokenize(text)
        size = self.config.bm25_prefix

        return [token[:size] for token in tokens] if size > 0 else tokens

    def cache_path(self) -> Path:
        manifest = {
            "embedding_model": self.config.embedding_model,
            "ingestion_version": INGESTION_VERSION,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "documents": sorted(
                [
                    {
                        "path": document.path,
                        "fingerprint": document.fingerprint,
                    }
                    for document in self.documents
                ],
                key=lambda item: item["path"],
            ),
        }

        key = stable_id(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True),
            40,
        )

        folder = Path(self.config.cache_dir)
        folder.mkdir(parents=True, exist_ok=True)

        return folder / f"embeddings_{key}.npz"

    def initialize_retrieval(self) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "Install sentence-transformers before running this project."
            )

        self.embedding_model = SentenceTransformer(
            self.config.embedding_model
        )

        cache_file = self.cache_path()
        loaded = False

        if self.config.use_cache and cache_file.exists():
            try:
                cached = np.load(cache_file, allow_pickle=False)
                embeddings = cached["embeddings"]

                if len(embeddings) == len(self.chunks):
                    self.embeddings = embeddings
                    loaded = True
                    log(f"[CACHE] Loaded: {cache_file}", self.config.debug)

            except Exception as exc:
                log(f"[CACHE][WARN] {exc}", self.config.debug, logging.WARNING)

        if not loaded:
            self.embeddings = self.embedding_model.encode(
                [chunk.text for chunk in self.chunks],
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=True,
            )

            if self.config.use_cache:
                np.savez_compressed(cache_file, embeddings=self.embeddings)
                log(f"[CACHE] Saved: {cache_file}", self.config.debug)

        if BM25Okapi is not None:
            self.bm25 = BM25Okapi(
                [self.bm25_tokens(chunk.text) for chunk in self.chunks]
            )
            log("[RETRIEVAL] BM25 enabled", self.config.debug)

        if self.config.use_reranker and CrossEncoder is not None:
            try:
                self.reranker = CrossEncoder(self.config.reranker_model)
                log("[RERANK] Enabled", self.config.debug)
            except Exception as exc:
                log(f"[RERANK][WARN] {exc}", self.config.debug, logging.WARNING)

    def retrieve(self, question: str) -> list[SearchHit]:
        if not self.chunks:
            return []

        count = min(
            len(self.chunks),
            max(self.config.retrieve_k * 3, self.config.retrieve_k),
        )

        # Dense ranking (skipped for BM25-only ablations).
        dense_scores = np.zeros(len(self.chunks))
        dense_rank: dict[int, int] = {}

        if self.config.use_dense:
            if self.embedding_model is None or self.embeddings is None:
                raise RuntimeError("Retrieval index is not initialized.")

            query_vector = self.embedding_model.encode(
                [question],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]

            dense_scores = np.dot(self.embeddings, query_vector)
            dense_order = np.argsort(dense_scores)[::-1][:count]
            dense_rank = {
                int(index): rank + 1
                for rank, index in enumerate(dense_order)
            }

        # BM25 ranking. Chunks with a zero score share no query term, so
        # they are left out instead of receiving an arbitrary tie rank.
        bm25_scores = np.zeros(len(self.chunks))
        bm25_rank: dict[int, int] = {}

        if self.config.use_bm25 and self.bm25 is not None:
            bm25_scores = np.asarray(
                self.bm25.get_scores(self.bm25_tokens(question))
            )
            bm25_order = [
                int(index)
                for index in np.argsort(bm25_scores)[::-1][:count]
                if bm25_scores[index] > 0
            ]
            bm25_rank = {
                index: rank + 1
                for rank, index in enumerate(bm25_order)
            }

        hits: list[SearchHit] = []

        for index in set(dense_rank) | set(bm25_rank):
            rrf_score = (
                1 / (RRF_K + dense_rank[index]) if index in dense_rank else 0
            ) + (
                1 / (RRF_K + bm25_rank[index]) if index in bm25_rank else 0
            )

            hits.append(
                SearchHit(
                    chunk=self.chunks[index],
                    dense_score=float(dense_scores[index]),
                    bm25_score=float(bm25_scores[index]),
                    rrf_score=rrf_score,
                    rerank_score=rrf_score,
                )
            )

        hits.sort(key=lambda hit: hit.rrf_score, reverse=True)
        hits = hits[:self.config.retrieve_k]

        if self.config.use_reranker and self.reranker is not None and hits:
            try:
                scores = self.reranker.predict(
                    [[question, hit.chunk.text] for hit in hits],
                    show_progress_bar=False,
                )

                for hit, score in zip(hits, scores):
                    hit.rerank_score = float(score)

                hits.sort(key=lambda hit: hit.rerank_score, reverse=True)

            except Exception as exc:
                log(f"[RERANK][WARN] {exc}", self.config.debug, logging.WARNING)

        hits = hits[:self.config.rerank_k]

        for hit in hits:
            preview = re.sub(r"\s+", " ", hit.chunk.text[:120])

            log(
                f"[SEARCH] {hit.chunk.chunk_id} "
                f"score={hit.rerank_score:.4f} :: {preview}",
                self.config.debug,
            )

        return hits
