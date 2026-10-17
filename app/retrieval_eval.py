"""Retrieval-only evaluation against gold passages.

No LLM is needed: for every question the gold file lists exact phrases of
the source document that contain the answer. A retrieved chunk counts as a
hit when it covers at least half of a gold phrase. The same index is scored
in several modes (dense, BM25, hybrid, hybrid + rerank) so the contribution
of each component is measurable.

Gold file format::

    {"questions": [
        {"question": "...", "doc": "file name",
         "gold": ["exact phrase copied from the document", "..."]}
    ]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .config import source_matches
from .quotes import recover_source_quote

DEFAULT_KS = (1, 3, 5, 10)
MIN_GOLD_CHARS = 8
MIN_COVERAGE = 0.5

MODES = [
    ("dense only", {"use_dense": True, "use_bm25": False, "use_reranker": False}),
    ("BM25 only", {"use_dense": False, "use_bm25": True, "use_reranker": False}),
    ("hybrid (RRF)", {"use_dense": True, "use_bm25": True, "use_reranker": False}),
    ("hybrid + rerank", {"use_dense": True, "use_bm25": True, "use_reranker": True}),
]


@dataclass
class GoldQuestion:
    question: str
    doc_id: str
    spans: list[tuple[int, int]]


def load_gold(assistant, gold_file: str) -> list[GoldQuestion]:
    """Read the gold file and locate every gold phrase in its document.

    Raises ``ValueError`` listing every phrase that cannot be found, so a
    broken gold set can never silently produce misleading metrics.
    """
    path = Path(gold_file)

    if not path.exists():
        raise FileNotFoundError(f"Gold file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("questions", [])

    if not items:
        raise RuntimeError("No gold questions found.")

    gold: list[GoldQuestion] = []
    problems: list[str] = []

    for number, item in enumerate(items, 1):
        question = str(item.get("question", "")).strip()
        doc_name = str(item.get("doc", ""))
        phrases = [str(phrase) for phrase in item.get("gold", [])]

        document = next(
            (
                doc
                for doc in assistant.documents
                if source_matches(doc_name, doc.title)
            ),
            None,
        )

        if document is None:
            problems.append(f"#{number}: document {doc_name!r} is not indexed")
            continue

        spans: list[tuple[int, int]] = []

        for phrase in phrases:
            located = recover_source_quote(document.text, phrase, MIN_GOLD_CHARS)

            if located is None:
                problems.append(f"#{number}: gold phrase not found: {phrase!r}")
            else:
                spans.append((located[1], located[2]))

        if not spans:
            problems.append(f"#{number}: no usable gold phrase")
            continue

        gold.append(GoldQuestion(question, document.doc_id, spans))

    if problems:
        raise ValueError("Invalid gold file:\n" + "\n".join(problems))

    return gold


def first_hit_rank(hits, doc_id: str, spans: Sequence[tuple[int, int]]) -> Optional[int]:
    """1-based rank of the first chunk covering a gold span, else None."""
    for rank, hit in enumerate(hits, 1):
        chunk = hit.chunk

        if chunk.doc_id != doc_id:
            continue

        for start, end in spans:
            overlap = min(chunk.end, end) - max(chunk.start, start)

            if overlap >= MIN_COVERAGE * (end - start):
                return rank

    return None


def evidence_hits_gold(evidence, gold: GoldQuestion) -> list[bool]:
    """For each verified evidence item: does it overlap a gold passage?

    An evidence quote counts when it shares at least half of the shorter of
    the two spans with a gold span of the same document. This is a strict,
    conservative check: the gold list is not exhaustive, so a correct answer
    drawn from some other valid passage is reported as a miss.
    """
    flags: list[bool] = []

    for item in evidence:
        hit = False

        if item.doc_id == gold.doc_id:
            length = item.source_end - item.source_start

            for start, end in gold.spans:
                overlap = min(item.source_end, end) - max(item.source_start, start)

                if overlap > 0 and overlap >= MIN_COVERAGE * min(length, end - start):
                    hit = True
                    break

        flags.append(hit)

    return flags


def score_ranks(
    ranks: Sequence[Optional[int]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """Recall@k for each k plus MRR (restricted to the largest k)."""
    total = len(ranks)

    if total == 0:
        raise ValueError("No ranks to score.")

    cutoff = max(ks)
    scores = {
        f"R@{k}": sum(1 for r in ranks if r is not None and r <= k) / total
        for k in ks
    }
    scores["MRR"] = (
        sum(1 / r for r in ranks if r is not None and r <= cutoff) / total
    )

    return scores


def run_retrieval_eval(
    assistant,
    gold_file: str,
    ks: Sequence[int] = DEFAULT_KS,
) -> list[dict]:
    """Score every retrieval mode on the gold set and print a table."""
    gold = load_gold(assistant, gold_file)
    ks = tuple(k for k in ks if k <= assistant.config.rerank_k)

    config = assistant.config
    saved = (
        config.use_dense,
        config.use_bm25,
        config.use_reranker,
        config.debug,
    )

    rows: list[dict] = []
    skipped: list[str] = []

    try:
        config.debug = False

        for name, flags in MODES:
            if flags["use_bm25"] and assistant.bm25 is None:
                skipped.append(f"{name} (rank-bm25 not installed)")
                continue

            if flags["use_reranker"] and assistant.reranker is None:
                skipped.append(f"{name} (reranker not loaded)")
                continue

            config.use_dense = flags["use_dense"]
            config.use_bm25 = flags["use_bm25"]
            config.use_reranker = flags["use_reranker"]

            ranks = [
                first_hit_rank(
                    assistant.retrieve(item.question),
                    item.doc_id,
                    item.spans,
                )
                for item in gold
            ]

            rows.append({"mode": name, **score_ranks(ranks, ks), "ranks": ranks})

    finally:
        (
            config.use_dense,
            config.use_bm25,
            config.use_reranker,
            config.debug,
        ) = saved

    columns = [f"R@{k}" for k in ks] + ["MRR"]

    print(f"\n[RETRIEVAL EVAL] {len(gold)} questions from {gold_file}")
    print("| Mode | " + " | ".join(columns) + " |")
    print("| --- | " + " | ".join("---:" for _ in columns) + " |")

    for row in rows:
        cells = [
            f"{row[column]:.3f}" if column == "MRR" else f"{row[column]:.0%}"
            for column in columns
        ]
        print(f"| {row['mode']} | " + " | ".join(cells) + " |")

    if rows:
        print("\nFirst-hit rank per question ('-' = not in the top results):")
        print("| # | Question | " + " | ".join(row["mode"] for row in rows) + " |")
        print("| ---: | --- | " + " | ".join("---:" for _ in rows) + " |")

        for number, item in enumerate(gold):
            question = item.question
            question = question if len(question) <= 55 else question[:52] + "..."
            cells = [
                str(row["ranks"][number]) if row["ranks"][number] else "-"
                for row in rows
            ]
            print(f"| {number + 1} | {question} | " + " | ".join(cells) + " |")

    for note in skipped:
        print(f"[skipped] {note}")

    return rows
