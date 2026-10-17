import pytest
import json
import random
import time

from app.llm import ModelNotFoundError
from app.schemas import Evidence
from tests.helpers import FakeLLM, StubAssistant, make_hit

GOOD = "Mercenary armies are useless and dangerous to any ruler who relies on them."
INTRO = "An unrelated introduction about gardens and weather in spring. "
OUTRO = " A closing remark about rivers and mountains in the far north."
TEXT = INTRO + GOOD + OUTRO


def relevant(items):
    return json.dumps({"relevant": True, "items": items})


def item(quote, claim="claim", directness=5):
    return {"claim": claim, "quote": quote, "directness": directness}


def full_hit(stub):
    return make_hit(stub.document, 0, len(TEXT))


def test_extract_keeps_exact_quote_and_returns_original_text():
    # Model returns a lower-cased, re-spaced version of the real sentence.
    sloppy = GOOD.lower().replace(" ", "  ")
    stub = StubAssistant(TEXT, FakeLLM(lambda p: relevant([item(sloppy)])))

    candidates = stub.extract_evidence("Why?", [full_hit(stub)])

    assert len(candidates) == 1
    assert candidates[0].quote == GOOD
    assert TEXT[candidates[0].source_start:candidates[0].source_end] == GOOD
    assert candidates[0].evidence_id == "C1"


def test_extract_drops_fabricated_quotes():
    fake = "Mercenary armies are wonderful and completely harmless to everyone."
    stub = StubAssistant(TEXT, FakeLLM(lambda p: relevant([item(fake)])))

    assert stub.extract_evidence("Why?", [full_hit(stub)]) == []


def test_extract_drops_quote_outside_the_retrieved_chunk():
    stub = StubAssistant(TEXT, FakeLLM(lambda p: relevant([item(GOOD)])))
    intro_only = make_hit(stub.document, 0, len(INTRO))

    assert stub.extract_evidence("Why?", [intro_only]) == []


def test_extract_ignores_irrelevant_and_malformed_responses():
    for response in [
        json.dumps({"relevant": False, "items": [item(GOOD)]}),
        "this is not json",
        json.dumps({"relevant": True, "items": "oops"}),
        json.dumps({"relevant": True, "items": [42, None]}),
        json.dumps({"relevant": True, "items": [item("")]}),
    ]:
        stub = StubAssistant(TEXT, FakeLLM(lambda p, r=response: r))
        assert stub.extract_evidence("Why?", [full_hit(stub)]) == []


def test_extract_survives_llm_errors():
    def boom(prompt):
        raise RuntimeError("Ollama connection failed")

    stub = StubAssistant(TEXT, FakeLLM(boom))

    assert stub.extract_evidence("Why?", [full_hit(stub)]) == []


def test_extract_clamps_bad_directness():
    stub = StubAssistant(
        TEXT,
        FakeLLM(lambda p: relevant([item(GOOD, directness="high")])),
    )
    candidates = stub.extract_evidence("Why?", [full_hit(stub)])
    assert candidates[0].directness == 3

    stub = StubAssistant(
        TEXT,
        FakeLLM(lambda p: relevant([item(GOOD, directness=99)])),
    )
    candidates = stub.extract_evidence("Why?", [full_hit(stub)])
    assert candidates[0].directness == 5


def test_parallel_extraction_preserves_hit_order():
    count = 8
    sentences = [
        f"Sentence number {i} says that mercenaries are dangerous for every prince."
        for i in range(count)
    ]
    text = " ".join(sentences)
    delays = list(range(count))
    random.Random(0).shuffle(delays)

    def responder(prompt):
        number = int(prompt.split("Sentence number ")[1].split(" ")[0])
        time.sleep(0.01 * delays[number])  # scramble completion order
        return relevant([item(sentences[number], claim=f"claim {number}")])

    stub = StubAssistant(text, FakeLLM(responder), workers=4)

    hits = []
    position = 0

    for ordinal, sentence in enumerate(sentences):
        start = text.index(sentence, position)
        hits.append(make_hit(stub.document, start, start + len(sentence), ordinal))
        position = start + len(sentence)

    candidates = stub.extract_evidence("Why?", hits)

    assert [c.claim for c in candidates] == [f"claim {i}" for i in range(count)]
    assert [c.evidence_id for c in candidates] == [f"C{i}" for i in range(1, count + 1)]


def evidence(quote, directness=3, score=0.0, evidence_id="C1"):
    return Evidence(
        evidence_id=evidence_id,
        claim="claim",
        quote=quote,
        doc_id="d1",
        chunk_id="d1:0",
        source_start=0,
        source_end=len(quote),
        directness=directness,
        retrieval_score=score,
    )


def test_select_evidence_dedupes_ranks_and_limits():
    stub = StubAssistant(TEXT, FakeLLM(lambda p: ""), max_evidence=2)
    candidates = [
        evidence("Quote A is long enough to be a quote.", directness=2, evidence_id="C1"),
        evidence("Quote B is long enough to be a quote.", directness=5, evidence_id="C2"),
        evidence("quote b IS long enough to be a quote!", directness=4, evidence_id="C3"),
        evidence("Quote D is long enough to be a quote.", directness=3, evidence_id="C4"),
    ]

    selected = stub.select_evidence(candidates)

    assert [item.evidence_id for item in selected] == ["E1", "E2"]
    assert selected[0].quote.startswith("Quote B")  # highest directness first
    assert selected[1].quote.startswith("Quote D")  # duplicate of B removed


def verdict_llm(verdicts):
    """FakeLLM judging by a marker found in the quote of each prompt."""

    def responder(prompt):
        for marker, (supported, relevant) in verdicts.items():
            if marker in prompt:
                return json.dumps({"supported": supported, "relevant": relevant})

        return "{}"

    return FakeLLM(responder)


def marked(*names):
    return [
        evidence(f"Quote {name} is long enough to count as a quote.", evidence_id=f"C{i}")
        for i, name in enumerate(names, 1)
    ]


def test_verify_keeps_only_supported_and_relevant():
    candidates = marked("ALPHA", "BETA", "GAMMA")
    llm = verdict_llm(
        {"ALPHA": (True, True), "BETA": (False, True), "GAMMA": (True, False)}
    )
    stub = StubAssistant(TEXT, llm)

    assert stub.verify_evidence("Why?", candidates) == [candidates[0]]
    assert len(llm.calls) == 3  # one short call per candidate


def test_verify_treats_a_missing_verdict_as_a_drop_not_a_crash():
    candidates = marked("ALPHA", "BETA")
    stub = StubAssistant(TEXT, verdict_llm({"ALPHA": (True, True)}))  # BETA -> "{}"

    assert stub.verify_evidence("Why?", candidates) == [candidates[0]]


def test_verify_rejects_non_boolean_verdicts():
    candidates = marked("ALPHA")
    stub = StubAssistant(TEXT, FakeLLM(lambda p: '{"supported": "yes", "relevant": "yes"}'))

    assert stub.verify_evidence("Why?", candidates) == []


def test_verify_failure_of_one_candidate_does_not_lose_the_others():
    candidates = marked("ALPHA", "BETA", "GAMMA")

    def responder(prompt):
        if "BETA" in prompt:
            raise RuntimeError("Ollama connection failed")

        return json.dumps({"supported": True, "relevant": True})

    stub = StubAssistant(TEXT, FakeLLM(responder))

    assert stub.verify_evidence("Why?", candidates) == [candidates[0], candidates[2]]


def test_verify_without_candidates_makes_no_call():
    llm = FakeLLM(lambda p: "{}")
    stub = StubAssistant(TEXT, llm)

    assert stub.verify_evidence("Why?", []) == []
    assert llm.calls == []


def test_verify_parallel_keeps_candidate_order():
    names = [f"N{i}" for i in range(8)]
    candidates = marked(*names)
    delays = list(range(8))
    random.Random(1).shuffle(delays)

    def responder(prompt):
        number = int(prompt.split("Quote N")[1].split(" ")[0])
        time.sleep(0.01 * delays[number])
        return json.dumps({"supported": number % 2 == 0, "relevant": True})

    stub = StubAssistant(TEXT, FakeLLM(responder), workers=4)

    kept = stub.verify_evidence("Why?", candidates)

    assert [c.evidence_id for c in kept] == ["C1", "C3", "C5", "C7"]


def test_verification_can_use_a_separate_model():
    candidates = marked("ALPHA")
    main = FakeLLM(lambda p: "{}")
    verifier = verdict_llm({"ALPHA": (True, True)})
    stub = StubAssistant(TEXT, main)
    stub.verify_llm = verifier

    assert stub.verify_evidence("Why?", candidates) == candidates
    assert len(verifier.calls) == 1
    assert main.calls == []


def missing_model(prompt):
    raise ModelNotFoundError("Ollama model 'x' was not found. Install it with: ollama pull x")


def test_missing_model_stops_extraction_instead_of_returning_no_evidence():
    stub = StubAssistant(TEXT, FakeLLM(missing_model))

    with pytest.raises(ModelNotFoundError):
        stub.extract_evidence("Why?", [full_hit(stub)])


def test_missing_verify_model_stops_verification_sequential_and_parallel():
    for workers in (1, 4):
        stub = StubAssistant(TEXT, FakeLLM(missing_model), workers=workers)

        with pytest.raises(ModelNotFoundError):
            stub.verify_evidence("Why?", marked("ALPHA", "BETA", "GAMMA"))
