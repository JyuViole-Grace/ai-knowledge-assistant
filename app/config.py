"""Configuration, logging helpers and small shared utilities."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@dataclass
class Config:
    docs: str = "documents"
    ollama_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL
    # Optional stronger model for the support/relevance check (empty = same).
    verify_model: str = ""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    cache_dir: str = ".rag_cache"

    use_reranker: bool = True
    use_cache: bool = True
    # Retrieval toggles (used for ablation studies in retrieval_eval).
    use_dense: bool = True
    use_bm25: bool = True
    # Crude stemming for BM25: keep only the first N characters of every token
    # (0 = off). Helps inflected languages such as Russian or Italian.
    bm25_prefix: int = 0

    chunk_size: int = 1100
    chunk_overlap: int = 180
    retrieve_k: int = 18
    rerank_k: int = 10
    max_evidence: int = 5
    min_quote_chars: int = 35
    timeout: int = 180
    # Ollama context window. Its default is small and silently truncates
    # long prompts, so it is set explicitly.
    num_ctx: int = 8192
    # Parallel LLM calls during evidence extraction (one call per chunk).
    workers: int = 4
    debug: bool = False


logger = logging.getLogger("rag")


def setup_logging() -> None:
    """Send the ``rag`` logger to stdout once (idempotent)."""
    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log(
    message: str,
    enabled: bool = True,
    level: int = logging.INFO,
) -> None:
    """Log a message.

    INFO messages respect the ``enabled`` (debug) flag. WARNING and above
    are always emitted so real problems are never silently hidden.
    """
    if enabled or level >= logging.WARNING:
        logger.log(level, message)


def stable_id(value: str, length: int = 12) -> str:
    return hashlib.sha1(
        value.encode("utf-8", errors="replace")
    ).hexdigest()[:length]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def normalize_name(name: str) -> str:
    return re.sub(
        r"[^a-zа-яёіїєґ0-9]+",
        "",
        Path(name).stem.lower(),
    )


def source_matches(expected: str, actual: str) -> bool:
    expected_name = normalize_name(expected)
    actual_name = normalize_name(actual)

    return bool(
        expected_name
        and actual_name
        and (
            expected_name == actual_name
            or expected_name in actual_name
            or actual_name in expected_name
        )
    )


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return {}

    try:
        result = json.loads(match.group(0))
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}
