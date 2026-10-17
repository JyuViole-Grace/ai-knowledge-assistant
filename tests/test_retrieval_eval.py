import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Config
from app.retrieval_eval import (
    GoldQuestion,
    evidence_hits_gold,
    first_hit_rank,
    load_gold,
    run_retrieval_eval,
    score_ranks,
)
from tests.helpers import make_document, make_hit

def test_score_ranks_recall_and_mrr():
    scores = score_ranks([1, 3, None, 12], ks=(1, 3, 5, 10))

    assert scores["R@1"] == 0.25
    assert scores["R@3"] == 0.5
    assert scores["R@5"] == 0.5
    assert scores["R@10"] == 0.5
    # rank 12 is beyond the cutoff (10) and does not count
    assert scores["MRR"] == pytest.approx((1 + 1 / 3) / 4)


def test_score_ranks_rejects_empty_input():
    with pytest.raises(ValueError):
        score_ranks([])


def test_first_hit_rank_requires_enough_overlap_and_right_document():
    document = make_document("x" * 1000)
    other = make_document("x" * 1000, doc_id="d2", title="other.txt")
    spans = [(100, 200)]

    far = make_hit(document, 400, 600)
    barely = make_hit(document, 190, 400)      # covers 10% of the span
    enough = make_hit(document, 140, 400)      # covers 60% of the span
    wrong_doc = make_hit(other, 0, 1000)

    assert first_hit_rank([far, wrong_doc], "d1", spans) is None
    assert first_hit_rank([far, barely], "d1", spans) is None
    assert first_hit_rank([far, enough], "d1", spans) == 2
    assert first_hit_rank([wrong_doc, enough], "d1", spans) == 2


def write_gold(directory, items):
    path = Path(directory) / "gold.json"
    path.write_text(json.dumps({"questions": items}), encoding="utf-8")
    return str(path)


def test_load_gold_reports_every_unresolvable_phrase():
    document = make_document("The quick brown fox jumps over the lazy dog today.")
    assistant = SimpleNamespace(documents=[document])

    with tempfile.TemporaryDirectory() as directory:
        gold_file = write_gold(
            directory,
            [
                {"question": "q1", "doc": "book.txt", "gold": ["brown fox jumps over"]},
                {"question": "q2", "doc": "book.txt", "gold": ["this is not in the text"]},
                {"question": "q3", "doc": "missing.txt", "gold": ["whatever phrase"]},
            ],
        )

        with pytest.raises(ValueError) as error:
            load_gold(assistant, gold_file)

    message = str(error.value)
    assert "#2" in message and "#3" in message
    assert "#1" not in message


class FakeRetriever:
    """Returns the gold chunk at a rank that depends on the active mode."""

    def __init__(self, document, start, end):
        self.config = Config(rerank_k=10)
        self.documents = [document]
        self.bm25 = object()
        self.reranker = object()
        self._document = document
        self._start, self._end = start, end

    def retrieve(self, question):
        config = self.config

        if config.use_reranker:
            rank = 1
        elif config.use_dense and config.use_bm25:
            rank = 2
        elif config.use_bm25:
            rank = 4
        else:
            rank = 6

        document = self._document
        filler = make_hit(document, 0, 5)
        hits = [filler] * (rank - 1)
        hits.append(make_hit(document, self._start, self._end))
        return hits


def test_run_retrieval_eval_scores_modes_and_restores_config():
    text = "Filler words. " * 5 + "The gold sentence about mercenary armies lives here." + " Tail." * 5
    document = make_document(text)
    start = text.index("The gold sentence")
    end = start + len("The gold sentence about mercenary armies lives here.")
    assistant = FakeRetriever(document, start, end)
    assistant.config.debug = True

    with tempfile.TemporaryDirectory() as directory:
        gold_file = write_gold(
            directory,
            [
                {
                    "question": "q",
                    "doc": "book.txt",
                    "gold": ["The gold sentence about mercenary armies lives here."],
                }
            ],
        )
        rows = run_retrieval_eval(assistant, gold_file)

    by_mode = {row["mode"]: row for row in rows}

    assert by_mode["dense only"]["R@5"] == 0.0
    assert by_mode["dense only"]["R@10"] == 1.0
    assert by_mode["BM25 only"]["R@5"] == 1.0
    assert by_mode["hybrid (RRF)"]["R@3"] == 1.0
    assert by_mode["hybrid + rerank"]["R@1"] == 1.0
    assert by_mode["hybrid + rerank"]["MRR"] == 1.0

    # per-question first-hit ranks are exposed for diagnostics
    assert by_mode["dense only"]["ranks"] == [6]
    assert by_mode["BM25 only"]["ranks"] == [4]
    assert by_mode["hybrid (RRF)"]["ranks"] == [2]
    assert by_mode["hybrid + rerank"]["ranks"] == [1]

    # config must be left exactly as it was
    assert assistant.config.use_dense and assistant.config.use_bm25
    assert assistant.config.use_reranker
    assert assistant.config.debug is True


def test_run_retrieval_eval_skips_modes_without_components():
    text = "Intro. The gold sentence about mercenary armies lives here. End."
    document = make_document(text)
    start = text.index("The gold")
    end = start + len("The gold sentence about mercenary armies lives here.")
    assistant = FakeRetriever(document, start, end)
    assistant.reranker = None
    assistant.bm25 = None

    with tempfile.TemporaryDirectory() as directory:
        gold_file = write_gold(
            directory,
            [{"question": "q", "doc": "book.txt",
              "gold": ["The gold sentence about mercenary armies lives here."]}],
        )
        rows = run_retrieval_eval(assistant, gold_file)

    assert [row["mode"] for row in rows] == ["dense only"]



def make_evidence(start, end, doc_id="d1", evidence_id="E1"):
    from app.schemas import Evidence

    return Evidence(
        evidence_id=evidence_id,
        claim="claim",
        quote="q",
        doc_id=doc_id,
        chunk_id="d1:0",
        source_start=start,
        source_end=end,
    )


def test_evidence_hits_gold_requires_same_document_and_overlap():
    gold = GoldQuestion("q", "d1", [(100, 200)])

    inside = make_evidence(120, 180)             # fully inside the gold span
    containing = make_evidence(0, 1000)          # long quote containing the span
    barely = make_evidence(195, 400)             # touches only the last 5 chars
    far = make_evidence(500, 600)
    other_doc = make_evidence(100, 200, doc_id="d2")

    flags = evidence_hits_gold(
        [inside, containing, barely, far, other_doc],
        gold,
    )

    assert flags == [True, True, False, False, False]
    assert evidence_hits_gold([], gold) == []
