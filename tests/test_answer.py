from app.answer import AnswerMixin
from app.schemas import Evidence
from tests.helpers import FakeLLM, StubAssistant

TEXT = "Mercenary armies are useless and dangerous to any ruler."


def make_evidence(*ids):
    return [
        Evidence(
            evidence_id=evidence_id,
            claim="claim",
            quote=TEXT,
            doc_id="d1",
            chunk_id="d1:0",
            source_start=0,
            source_end=len(TEXT),
        )
        for evidence_id in ids
    ]


def test_validate_accepts_single_and_grouped_citations():
    evidence = make_evidence("E1", "E2", "E3")
    validate = AnswerMixin.validate_answer_citations

    assert validate("Because [E1].", evidence)
    assert validate("Because [E1, E2] and [E3].", evidence)


def test_validate_rejects_unknown_or_missing_citations():
    evidence = make_evidence("E1")
    validate = AnswerMixin.validate_answer_citations

    assert not validate("No citation at all.", evidence)
    assert not validate("Invented source [E7].", evidence)
    assert not validate("Mixed [E1, E2].", evidence)
    assert not validate("Citation-like [note] only.", evidence)


def test_abstention_follows_question_language():
    stub = StubAssistant(TEXT, FakeLLM(lambda prompt: ""))

    assert "enough verified evidence" in stub.abstention("Why?")
    assert "доказательств" in stub.abstention("Почему?")
    assert "доказів" in stub.abstention("Чому це так, їжак?")


def test_generate_answer_without_evidence_never_calls_llm():
    llm = FakeLLM(lambda prompt: "should not be used [E1]")
    stub = StubAssistant(TEXT, llm)

    answer = stub.generate_answer("Why?", [])

    assert "enough verified evidence" in answer
    assert llm.calls == []


def test_generate_answer_returns_valid_cited_answer():
    llm = FakeLLM(lambda prompt: "They are dangerous [E1].")
    stub = StubAssistant(TEXT, llm)

    answer = stub.generate_answer("Why?", make_evidence("E1"))

    assert answer == "They are dangerous [E1]."
    assert "book.txt" in llm.calls[0]  # source is passed to the model


def test_generate_answer_abstains_on_invalid_citation():
    stub = StubAssistant(TEXT, FakeLLM(lambda prompt: "Made up [E9]."))
    answer = stub.generate_answer("Why?", make_evidence("E1"))
    assert "enough verified evidence" in answer


def test_generate_answer_abstains_on_empty_llm_output():
    stub = StubAssistant(TEXT, FakeLLM(lambda prompt: "   "))
    answer = stub.generate_answer("Why?", make_evidence("E1"))
    assert "enough verified evidence" in answer


def test_answer_prompt_contains_quotes_but_never_the_unverified_claim():
    llm = FakeLLM(lambda prompt: "Dangerous [E1].")
    stub = StubAssistant(TEXT, llm)
    evidence = make_evidence("E1")
    evidence[0].claim = "SECRET-PARAPHRASED-CLAIM"

    stub.generate_answer("Why?", evidence)

    prompt = llm.calls[0]
    assert TEXT in prompt                     # the verbatim quote is given
    assert "SECRET-PARAPHRASED-CLAIM" not in prompt
