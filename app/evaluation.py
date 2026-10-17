"""End-to-end smoke tests (answers, citations, sources, abstention)."""

import json
import re
import statistics
import time
from pathlib import Path
from typing import Optional

from .config import source_matches
from .retrieval_eval import evidence_hits_gold, load_gold


def run_tests(
    assistant,
    test_file: str,
    gold_file: Optional[str] = None,
) -> None:
    """Run end-to-end tests against the assistant.

    Without ``gold_file`` a case passes when the answer has a valid citation
    and comes from the expected source. With ``gold_file`` (the same files
    used by the retrieval evaluation) it must also be *grounded*: at least
    one verified evidence quote has to overlap a gold passage. That catches
    well-formed answers that cite the wrong passage.
    """
    path = Path(test_file)

    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    tests = data if isinstance(data, list) else data.get("questions", [])

    if not tests:
        raise RuntimeError("No test cases found.")

    gold_by_question = {}

    if gold_file:
        gold_by_question = {
            item.question: item for item in load_gold(assistant, gold_file)
        }

    passed = 0
    failed = 0
    durations: list[float] = []
    grounded = 0
    graded = 0
    precisions: list[float] = []

    for index, item in enumerate(tests, 1):
        if isinstance(item, str):
            item = {"question": item}

        question = str(item.get("question", "")).strip()

        if not question:
            continue

        # Negative case: the documents do NOT contain the answer, so the
        # assistant must abstain (no verified evidence at all).
        expect_abstain = bool(item.get("expect_abstain", False))

        expected_sources = [
            str(source) for source in item.get("expected_sources", [])
        ]

        required_phrases = [
            str(phrase).lower() for phrase in item.get("must_include", [])
        ]

        print(f"\n[TEST {index}] {question}")

        started = time.perf_counter()
        answer = assistant.ask(question)
        durations.append(time.perf_counter() - started)

        if expect_abstain:
            if not assistant.last_evidence:
                passed += 1
                print(f"[TEST {index}] PASS (abstained as expected)")
            else:
                failed += 1
                print(f"[TEST {index}] FAIL (expected abstention)")
            continue

        actual_sources = {
            assistant.doc_by_id[evidence.doc_id].title
            for evidence in assistant.last_evidence
        }

        source_ok = not expected_sources or any(
            source_matches(expected, actual)
            for expected in expected_sources
            for actual in actual_sources
        )

        text_ok = all(phrase in answer.lower() for phrase in required_phrases)

        citation_ok = bool(
            re.search(r"\[E\d+(?:\s*,\s*E\d+)*\]", answer)
        )

        gold = gold_by_question.get(question)
        gold_ok = True
        flags: list[bool] = []

        if gold is not None:
            flags = evidence_hits_gold(assistant.last_evidence, gold)
            gold_ok = any(flags)
            graded += 1
            grounded += int(gold_ok)

            if flags:
                precisions.append(sum(flags) / len(flags))
            else:
                precisions.append(0.0)

            hit_ids = [
                evidence.evidence_id
                for evidence, flag in zip(assistant.last_evidence, flags)
                if flag
            ]
            print(
                "[GOLD] "
                + (f"HIT ({', '.join(hit_ids)})" if gold_ok else "MISS")
            )

        if source_ok and text_ok and citation_ok and gold_ok:
            passed += 1
            print(f"[TEST {index}] PASS")
        else:
            failed += 1
            print(f"[TEST {index}] FAIL")

            if not source_ok:
                print(f"Expected: {expected_sources}")
                print(f"Actual: {sorted(actual_sources)}")

            if not text_ok:
                print(f"Missing phrases: {required_phrases}")

            if not citation_ok:
                print("No citations in answer")

            if not gold_ok:
                print("Evidence does not overlap any gold passage")

    print(f"\n[TEST SUMMARY] PASS={passed}, FAIL={failed}")

    if graded:
        print(
            f"[GOLD] grounded answers: {grounded}/{graded} "
            f"({grounded / graded:.0%}), "
            f"mean evidence precision: {statistics.mean(precisions):.0%}"
        )

    if durations:
        print(
            "[LATENCY] "
            f"mean={statistics.mean(durations):.1f}s "
            f"median={statistics.median(durations):.1f}s "
            f"max={max(durations):.1f}s per question"
        )

    if failed:
        raise RuntimeError(f"Tests failed: {failed}")
