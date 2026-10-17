import re
from typing import Optional

def canonicalize(text: str) -> tuple[str, list[int]]:
    output: list[str] = []
    mapping: list[int] = []
    pending_space = False

    for index, char in enumerate(text):
        if char.isspace():
            pending_space = bool(output)
            continue

        normalized = char.lower()
        normalized = normalized.replace("«", '"').replace("»", '"')
        normalized = normalized.replace("—", "-").replace("–", "-")

        if pending_space and output:
            output.append(" ")
            mapping.append(index)

        pending_space = False
        output.append(normalized)
        mapping.append(index)

    return "".join(output), mapping


def recover_source_quote(
        source: str,
        quote: str,
        min_chars: int,
) -> Optional[tuple[str, int, int]]:
    quote = quote.strip().strip('"“”„«»')

    if len(quote) < min_chars:
        return None

    if re.search(r"(?:\.\.\.|…)$", quote):
        return None

    source_normalized, mapping = canonicalize(source)
    quote_normalized, _ = canonicalize(quote)

    quote_normalized = quote_normalized.strip()
    position = source_normalized.find(quote_normalized)

    while position >= 0:
        end_position = position + len(quote_normalized) - 1

        if end_position < len(mapping):
            start = mapping[position]
            end = mapping[end_position] + 1

            if not (
                    start > 0
                    and source[start - 1].isalnum()
                    and source[start].isalnum()
            ) and not (
                    end < len(source)
                    and source[end - 1].isalnum()
                    and source[end].isalnum()
            ):
                return source[start:end], start, end

        position = source_normalized.find(
            quote_normalized,
            position + 1,
            )

    return None

