import json
import tempfile
from pathlib import Path

import pytest

from app.evaluation import run_tests
from app.schemas import Evidence
from tests.helpers import make_document

TEXT = (
    "Intro sentence about gardens. "
    "The gold sentence about mercenary armies lives right here. "
    "Another unrelated sentence about rivers and mountains."
)
GOLD = "The gold sentence about mercenary armies lives right here."
START = TEXT.index(GOLD)
END = START + len(GOLD)


class FakeAssistant:
    """Answers every question with one evidence item at a chosen span."""

    def __init__(self, start, end, answer="It is dangerous [E1]."):
        self.document = make_document(TEXT)
        self.documents = [self.document]
        self.doc_by_id = {self.document.doc_id: self.document}
        self.span = None if start is None else (start, end)
        self.answer = answer
        self.last_evidence = []

    def ask(self, question):
        if self.span is None:
            self.last_evidence = []
            return "No evidence."

        self.last_evidence = [
            Evidence(
                evidence_id="E1",
                claim="claim",
                quote=TEXT[self.span[0]:self.span[1]],
                doc_id="d1",
                chunk_id="d1:0",
                source_start=self.span[0],
                source_end=self.span[1],
            )
        ]
        return self.answer


def write_files(directory, tests):
    root = Path(directory)
    (root / "tests.json").write_text(json.dumps({"questions": tests}), encoding="utf-8")
    (root / "gold.json").write_text(
        json.dumps(
            {
                "questions": [
                    {"question": "Why?", "doc": "book.txt", "gold": [GOLD]}
                ]
            }
        ),
        encoding="utf-8",
    )
    return str(root / "tests.json"), str(root / "gold.json")


def test_strict_mode_passes_when_evidence_is_grounded_in_gold():
    with tempfile.TemporaryDirectory() as directory:
        tests, gold = write_files(directory, [{"question": "Why?"}])
        run_tests(FakeAssistant(START, END), tests, gold)  # must not raise


def test_strict_mode_fails_well_formed_answer_citing_wrong_passage():
    with tempfile.TemporaryDirectory() as directory:
        tests, gold = write_files(directory, [{"question": "Why?"}])
        wrong_place = FakeAssistant(0, 25)  # valid quote, but not the gold one

        with pytest.raises(RuntimeError):
            run_tests(wrong_place, tests, gold)


def test_lenient_mode_still_accepts_the_same_wrong_passage():
    # Without a gold file only citation/source are checked (old behaviour).
    with tempfile.TemporaryDirectory() as directory:
        tests, _ = write_files(directory, [{"question": "Why?"}])
        run_tests(FakeAssistant(0, 25), tests)  # must not raise


def test_missing_citation_fails_even_when_grounded():
    with tempfile.TemporaryDirectory() as directory:
        tests, gold = write_files(directory, [{"question": "Why?"}])
        uncited = FakeAssistant(START, END, answer="It is dangerous.")

        with pytest.raises(RuntimeError):
            run_tests(uncited, tests, gold)


def test_abstention_cases_ignore_gold_and_require_no_evidence():
    with tempfile.TemporaryDirectory() as directory:
        tests, gold = write_files(
            directory,
            [{"question": "Unanswerable?", "expect_abstain": True}],
        )

        run_tests(FakeAssistant(None, None), tests, gold)  # abstains -> pass

        with pytest.raises(RuntimeError):
            run_tests(FakeAssistant(START, END), tests, gold)  # answered -> fail
