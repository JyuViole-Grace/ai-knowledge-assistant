from app.quotes import recover_source_quote

SOURCE = (
    "Intro line.\n"
    "Mercenary   armies\nare  USELESS and dangerous \u2014 very much so indeed.\n"
    "Outro line."
)


def test_exact_quote_returns_original_slice_and_offsets():
    text = "Alpha beta. Mercenary armies are useless and dangerous. Omega."
    quote = "Mercenary armies are useless and dangerous."

    recovered = recover_source_quote(text, quote, 20)

    assert recovered is not None
    exact, start, end = recovered
    assert exact == text[start:end]
    assert text[start:end].startswith("Mercenary armies")


def test_normalizes_whitespace_case_and_dashes():
    quote = "mercenary armies are useless and dangerous - very much so indeed."

    recovered = recover_source_quote(SOURCE, quote, 20)

    assert recovered is not None
    exact, start, end = recovered
    # The returned text is the ORIGINAL source text, not the model's version.
    assert exact == SOURCE[start:end]
    assert "\n" in exact and "USELESS" in exact


def test_guillemets_are_equivalent_to_straight_quotes():
    source = "\u00abЦель оправдывает средства\u00bb \u2013 таков закон политики."
    quote = '"Цель оправдывает средства" - таков закон политики.'

    recovered = recover_source_quote(source, quote, 20)

    assert recovered is not None
    assert recovered[0] in source


def test_rejects_too_short_quote():
    assert recover_source_quote(SOURCE, "Mercenary", 35) is None


def test_rejects_trailing_ellipsis():
    quote = "Mercenary armies are useless and dangerous..."
    assert recover_source_quote(SOURCE, quote, 20) is None
    assert recover_source_quote(SOURCE, quote[:-3] + "\u2026", 20) is None


def test_rejects_fabricated_quote():
    quote = "Mercenary armies are wonderful and completely safe forever."
    assert recover_source_quote(SOURCE, quote, 20) is None


def test_rejects_quote_starting_mid_word():
    quote = "ercenary armies are useless and dangerous"
    assert recover_source_quote(SOURCE, quote, 20) is None


def test_rejects_quote_ending_mid_word():
    quote = "Mercenary armies are useless and dange"
    assert recover_source_quote(SOURCE, quote, 20) is None
