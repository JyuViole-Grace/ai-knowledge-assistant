import numpy as np

from app.config import Config
from app.retrieval import RetrievalMixin
from app.schemas import Chunk


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.array([[1.0, 0.0, 0.0, 0.0]])


class FakeBM25:
    def __init__(self, scores):
        self.scores = np.array(scores, dtype=float)
        self.last_tokens = None

    def get_scores(self, tokens):
        self.last_tokens = tokens
        return self.scores


class FakeReranker:
    def __init__(self, scores=None, error=None):
        self.scores = scores
        self.error = error
        self.calls = 0

    def predict(self, pairs, **kwargs):
        self.calls += 1

        if self.error:
            raise self.error

        return np.array(self.scores[: len(pairs)])


class Retriever(RetrievalMixin):
    """Four chunks; dense order is 0 > 1 > 2 > 3, only chunk 2 matches BM25."""

    def __init__(self, **config):
        self.config = Config(**config)
        self.chunks = [
            Chunk(f"d:{i}", "d", f"chunk {i}", i * 10, i * 10 + 10, i)
            for i in range(4)
        ]
        self.embedding_model = FakeEncoder()
        self.embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.6, 0.0, 0.0],
                [0.6, 0.0, 0.8, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.bm25 = FakeBM25([0, 0, 5, 0])
        self.reranker = None


def order(hits):
    return [hit.chunk.ordinal for hit in hits]


def test_dense_only_ranks_by_similarity():
    retriever = Retriever(use_bm25=False, use_reranker=False)
    assert order(retriever.retrieve("q")) == [0, 1, 2, 3]


def test_bm25_only_drops_chunks_with_zero_score():
    retriever = Retriever(use_dense=False, use_reranker=False)

    hits = retriever.retrieve("q")

    assert order(hits) == [2]
    assert retriever.embedding_model.calls == 0  # dense not computed


def test_hybrid_rrf_promotes_chunk_found_by_both_methods():
    retriever = Retriever(use_reranker=False)

    hits = retriever.retrieve("q")

    assert order(hits) == [2, 0, 1, 3]
    assert hits[0].rrf_score > hits[1].rrf_score


def test_reranker_reorders_hits_when_enabled():
    retriever = Retriever(use_reranker=True)
    # scores are applied to the RRF order [2, 0, 1, 3]
    retriever.reranker = FakeReranker(scores=[0.1, 0.2, 0.3, 0.9])

    hits = retriever.retrieve("q")

    assert order(hits)[0] == 3
    assert retriever.reranker.calls == 1


def test_reranker_is_ignored_when_disabled_in_config():
    retriever = Retriever(use_reranker=False)
    retriever.reranker = FakeReranker(scores=[0.1, 0.2, 0.3, 0.9])

    assert order(retriever.retrieve("q")) == [2, 0, 1, 3]
    assert retriever.reranker.calls == 0


def test_reranker_failure_falls_back_to_rrf_order():
    retriever = Retriever(use_reranker=True)
    retriever.reranker = FakeReranker(error=RuntimeError("boom"))

    assert order(retriever.retrieve("q")) == [2, 0, 1, 3]


def test_result_is_limited_by_rerank_k():
    retriever = Retriever(use_reranker=False, rerank_k=2)
    assert len(retriever.retrieve("q")) == 2


def test_no_chunks_returns_empty_list():
    retriever = Retriever()
    retriever.chunks = []
    assert retriever.retrieve("q") == []


def test_bm25_tokens_are_plain_by_default():
    retriever = Retriever()
    assert retriever.bm25_tokens("Adulazione e adulatori") == ["adulazione", "e", "adulatori"]


def test_bm25_prefix_merges_inflected_forms():
    retriever = Retriever(bm25_prefix=5)

    assert retriever.bm25_tokens("adulazione adulazioni adulatori") == ["adula", "adula", "adula"]
    assert retriever.bm25_tokens("Государь государя государю") == ["госуд", "госуд", "госуд"]
    assert retriever.bm25_tokens("il re") == ["il", "re"]  # short words are untouched


def test_retrieve_queries_bm25_with_the_same_tokenization_as_the_index():
    retriever = Retriever(bm25_prefix=4, use_reranker=False)

    retriever.retrieve("Adulazione dei principi")

    assert retriever.bm25.last_tokens == ["adul", "dei", "prin"]
