"""LLM evidence extraction and verification with exact-quote recovery."""

import logging
from concurrent.futures import ThreadPoolExecutor
from string import Template
from typing import Optional

from .config import log, parse_json_object, tokenize
from .llm import ModelNotFoundError
from .quotes import recover_source_quote
from .schemas import Evidence, SearchHit

EXTRACT_SYSTEM = "You extract grounded evidence. Return only valid JSON."

EXTRACT_PROMPT = Template('''Question:
$question

Source passage:
"""
$passage
"""

Return JSON only:
{
  "relevant": true,
  "items": [
    {
      "claim": "short answer claim",
      "quote": "exact continuous quote copied from the source passage",
      "directness": 5
    }
  ]
}

Rules:
- Return relevant=false if the passage does not answer the question.
- Copy quote exactly from the passage.
- Do not paraphrase the quote.
- Do not use ellipses.
- Do not use external knowledge.
- directness must be an integer from 1 to 5.
''')

VERIFY_SYSTEM = "Return only valid JSON."

# One candidate per call: a short prompt cannot overflow the model's context
# and its verdict cannot be lost behind other candidates in a long list.
VERIFY_PROMPT = Template('''Question:
$question

Claim:
$claim

Quote from the source:
"""
$quote
"""

Return JSON only:
{"supported": true, "relevant": true}

supported=true only when the quote itself supports the claim.
relevant=true only when the quote helps answer the question.
''')


class EvidenceMixin:
    def _extract_from_hit(
        self,
        question: str,
        hit: SearchHit,
    ) -> list[Evidence]:
        """Ask the LLM for quotes in one chunk and keep only exact ones."""
        chunk = hit.chunk
        document = self.doc_by_id[chunk.doc_id]
        prompt = EXTRACT_PROMPT.substitute(
            question=question,
            passage=chunk.text,
        )

        try:
            result = parse_json_object(
                self.llm.generate(
                    prompt,
                    system=EXTRACT_SYSTEM,
                    json_mode=True,
                )
            )
        except ModelNotFoundError:
            raise  # a setup error: stop instead of answering "no evidence"
        except Exception as exc:
            log(f"[EXTRACT][WARN] {exc}", self.config.debug, logging.WARNING)
            return []

        items = result.get("items", [])

        if result.get("relevant") is not True or not isinstance(items, list):
            return []

        found: list[Evidence] = []

        for item in items[:3]:
            if not isinstance(item, dict):
                continue

            claim = str(item.get("claim", "")).strip()
            quote = str(item.get("quote", "")).strip()

            if not claim or not quote:
                continue

            recovered = recover_source_quote(
                document.text,
                quote,
                self.config.min_quote_chars,
            )

            if recovered is None:
                log(
                    f"[QUOTE] REJECT {chunk.chunk_id} :: {quote[:100]!r}",
                    self.config.debug,
                )
                continue

            exact_quote, start, end = recovered

            if end <= chunk.start or start >= chunk.end:
                continue

            try:
                directness = int(item.get("directness", 3))
            except (ValueError, TypeError):
                directness = 3

            found.append(
                Evidence(
                    evidence_id="",
                    claim=claim,
                    quote=exact_quote,
                    doc_id=document.doc_id,
                    chunk_id=chunk.chunk_id,
                    source_start=start,
                    source_end=end,
                    directness=max(1, min(5, directness)),
                    retrieval_score=hit.rerank_score,
                )
            )

        return found

    def extract_evidence(
        self,
        question: str,
        hits: list[SearchHit],
    ) -> list[Evidence]:
        """Extract candidate evidence from every hit.

        One LLM call per chunk is the slowest part of the pipeline, so the
        calls run in parallel (``Config.workers``). Results keep the order
        of ``hits`` so the output stays deterministic.
        """
        workers = max(1, min(self.config.workers, len(hits)))

        if workers == 1:
            groups = [self._extract_from_hit(question, hit) for hit in hits]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                groups = list(
                    pool.map(
                        lambda hit: self._extract_from_hit(question, hit),
                        hits,
                    )
                )

        candidates = [item for group in groups for item in group]

        for number, item in enumerate(candidates, 1):
            item.evidence_id = f"C{number}"

        log(f"[EXTRACT] Candidates: {len(candidates)}", self.config.debug)

        return candidates

    def _verify_one(
        self,
        question: str,
        item: Evidence,
    ) -> tuple[Optional[bool], Optional[bool]]:
        """Return (supported, relevant); None means no usable verdict."""
        prompt = VERIFY_PROMPT.substitute(
            question=question,
            claim=item.claim,
            quote=item.quote,
        )
        verifier = getattr(self, "verify_llm", None) or self.llm

        try:
            result = parse_json_object(
                verifier.generate(
                    prompt,
                    system=VERIFY_SYSTEM,
                    json_mode=True,
                )
            )
        except ModelNotFoundError:
            raise  # a setup error: stop instead of dropping every candidate
        except Exception as exc:
            log(
                f"[VERIFY][WARN] {item.evidence_id}: {exc}",
                self.config.debug,
                logging.WARNING,
            )
            return None, None

        supported = result.get("supported")
        relevant = result.get("relevant")

        return (
            supported if isinstance(supported, bool) else None,
            relevant if isinstance(relevant, bool) else None,
        )

    def verify_evidence(
        self,
        question: str,
        candidates: list[Evidence],
    ) -> list[Evidence]:
        """Keep candidates the LLM judges both supported and relevant.

        Every candidate is judged in its own call (in parallel, order
        preserved). A missing verdict is logged as such, so it can be told
        apart from a genuine "no".
        """
        if not candidates:
            return []

        workers = max(1, min(self.config.workers, len(candidates)))

        if workers == 1:
            verdicts = [self._verify_one(question, item) for item in candidates]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                verdicts = list(
                    pool.map(
                        lambda item: self._verify_one(question, item),
                        candidates,
                    )
                )

        kept: list[Evidence] = []

        for item, (supported, relevant) in zip(candidates, verdicts):
            if supported is True and relevant is True:
                kept.append(item)
                continue

            def label(value: Optional[bool]) -> str:
                return "missing" if value is None else str(value)

            log(
                f"[VERIFY] drop {item.evidence_id} "
                f"supported={label(supported)} relevant={label(relevant)} :: "
                f"{item.quote[:80]!r}",
                self.config.debug,
            )

        log(f"[VERIFY] kept {len(kept)}/{len(candidates)}", self.config.debug)

        return kept

    def select_evidence(
        self,
        candidates: list[Evidence],
    ) -> list[Evidence]:
        def score(item: Evidence) -> float:
            return (
                item.directness * 2
                + item.retrieval_score * 0.25
                + min(len(item.quote), 500) / 500 * 0.15
            )

        selected: list[Evidence] = []
        seen_quotes: set[str] = set()

        for item in sorted(candidates, key=score, reverse=True):
            key = " ".join(tokenize(item.quote))

            if not key or key in seen_quotes:
                continue

            selected.append(item)
            seen_quotes.add(key)

            if len(selected) >= self.config.max_evidence:
                break

        for number, item in enumerate(selected, 1):
            item.evidence_id = f"E{number}"

        return selected
