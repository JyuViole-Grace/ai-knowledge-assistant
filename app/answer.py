"""Answer generation and citation validation."""

import json
import re
from string import Template

from .schemas import Evidence

ANSWER_SYSTEM = (
    "You are a careful RAG assistant. Do not invent facts or citations."
)

ANSWER_PROMPT = Template('''Question:
$question

Verified evidence:
$evidence

Answer rules:
- Use only facts supported by the evidence.
- Do not add external facts.
- Cite each important statement using [E1], [E2], or [E1, E2].
- Use only existing evidence IDs.
- Answer in the language of the question.
- Keep the answer concise.
''')


class AnswerMixin:
    @staticmethod
    def detect_language(question: str) -> str:
        lowered = question.lower()

        if re.search(r"[іїєґ]", lowered):
            return "uk"

        if re.search(r"[а-яё]", lowered):
            return "ru"

        return "en"

    def abstention(self, question: str) -> str:
        language = self.detect_language(question)

        if language == "uk":
            return (
                "У документах не знайдено достатньо перевірених "
                "доказів для надійної відповіді."
            )

        if language == "ru":
            return (
                "В проиндексированных документах не найдено "
                "достаточно проверенных доказательств для "
                "надёжного ответа."
            )

        return (
            "I could not find enough verified evidence in the "
            "indexed documents to answer reliably."
        )

    @staticmethod
    def validate_answer_citations(
        answer: str,
        evidence: list[Evidence],
    ) -> bool:
        """True if the answer cites at least one id and only existing ids.

        Supports both ``[E1]`` and ``[E1, E2, E3]`` formats.
        """
        valid_ids = {item.evidence_id for item in evidence}
        used_ids: set[str] = set()

        for group in re.findall(r"\[([^\]]+)\]", answer):
            used_ids.update(re.findall(r"\bE\d+\b", group))

        return bool(used_ids) and used_ids.issubset(valid_ids)

    def generate_answer(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> str:
        if not evidence:
            return self.abstention(question)

        # Only the verbatim quotes go to the model. The "claim" text is a
        # paraphrase written by the extraction step and may contain facts that
        # are not in the quote, so it must not feed the final answer.
        context = [
            {
                "id": item.evidence_id,
                "quote": item.quote,
                "source": self.doc_by_id[item.doc_id].title,
            }
            for item in evidence
        ]

        prompt = ANSWER_PROMPT.substitute(
            question=question,
            evidence=json.dumps(context, ensure_ascii=False),
        )

        answer = self.llm.generate(prompt, system=ANSWER_SYSTEM).strip()

        if not answer:
            return self.abstention(question)

        if not self.validate_answer_citations(answer, evidence):
            return self.abstention(question)

        return answer
