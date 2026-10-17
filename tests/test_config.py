from app.config import (
    normalize_name,
    parse_json_object,
    source_matches,
    stable_id,
    tokenize,
)


def test_source_matches_ignores_case_extension_and_punctuation():
    assert source_matches("Annual Report.txt", "annual_report.TXT")
    assert source_matches("annual_report", "annual_report.pdf")


def test_source_matches_rejects_different_documents():
    assert not source_matches("budget.txt", "annual_report.pdf")
    assert not source_matches("", "anything.pdf")


def test_normalize_name_keeps_cyrillic():
    assert normalize_name("Государь v2.txt") == "государьv2"


def test_parse_json_object_variants():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"a": 1} Hope it helps') == {"a": 1}


def test_parse_json_object_rejects_garbage_and_non_objects():
    assert parse_json_object("not json at all") == {}
    assert parse_json_object("[1, 2, 3]") == {}
    assert parse_json_object("") == {}


def test_stable_id_is_deterministic_and_sized():
    assert stable_id("abc") == stable_id("abc")
    assert stable_id("abc") != stable_id("abd")
    assert len(stable_id("abc", 20)) == 20


def test_tokenize_handles_unicode():
    assert tokenize("Государь, Il Principe!") == ["государь", "il", "principe"]
