from app.ingestion import join_pdf_pages, make_chunks, normalize_pdf_text
from tests.helpers import make_document
import pytest


def test_joins_hyphenated_line_break():
    assert normalize_pdf_text("prin-\ncipe") == "principe"


def test_joins_hyphen_preceded_by_space():
    # Real pattern in the bundled Italian PDF: "in -\ntroduzione".
    assert normalize_pdf_text("in -\ntroduzione") == "introduzione"


def test_joins_word_split_by_a_page_number():
    # Real pattern in the bundled PDF: "ine -\n6\n\nsauribile".
    assert normalize_pdf_text("ine -\n6\n\nsauribile") == "inesauribile"


def test_keeps_dash_between_words():
    text = "Государь - это человек - сильный."
    assert normalize_pdf_text(text) == text


def test_keeps_digit_ranges():
    result = normalize_pdf_text("Machiavelli (1469-\n1527) visse")
    assert "1469" in result and "1527" in result
    assert "14691527" not in result


def test_single_newline_becomes_space_and_paragraphs_stay():
    result = normalize_pdf_text("one\ntwo\n\nthree")
    assert result == "one two\n\nthree"


def test_chunks_cover_text_and_match_offsets():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    document = make_document(text)

    chunks = make_chunks([document], chunk_size=300, overlap=50)

    assert len(chunks) > 1
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)

    for chunk in chunks:
        assert chunk.text == text[chunk.start:chunk.end]
        assert len(chunk.text) <= 300

    for previous, current in zip(chunks, chunks[1:]):
        assert current.start < previous.end  # overlap
        assert current.start > previous.start  # progress
        assert current.end > previous.end

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_short_text_is_one_chunk():
    document = make_document("Just one short sentence.")
    chunks = make_chunks([document], chunk_size=300, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == "Just one short sentence."


def test_invalid_chunk_parameters_raise():
    document = make_document("text")

    with pytest.raises(ValueError):
        make_chunks([document], chunk_size=50, overlap=0)

    with pytest.raises(ValueError):
        make_chunks([document], chunk_size=200, overlap=200)

    with pytest.raises(ValueError):
        make_chunks([document], chunk_size=200, overlap=-1)


def test_pipeline_on_a_synthetic_scanned_book():
    """Hyphenation, page numbers and chunking working together."""
    pages = [
        "The first page ends in the middle of a sen-\n1",
        "tence about mercenary armies. A new para-\n2",
        "graph starts here and ends with a full stop.\n3",
    ]

    text = normalize_pdf_text(join_pdf_pages(pages))

    assert "sentence about mercenary armies." in text
    assert "paragraph starts here" in text
    assert not any(char.isdigit() for char in text)

    book = " ".join([text] * 8)
    chunks = make_chunks([make_document(book)], chunk_size=100, overlap=20)

    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.text == book[chunk.start:chunk.end]
        assert book[chunk.start].isalnum()
