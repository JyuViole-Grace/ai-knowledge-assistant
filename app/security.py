"""Path sandboxing for the HTTP API."""

from __future__ import annotations

from pathlib import Path


def resolve_docs_path(requested: str, root: str | Path) -> Path:
    """Resolve ``requested`` and make sure it stays inside ``root``.

    Relative paths are interpreted relative to ``root``. Absolute paths,
    ``..`` segments and symlinks are all resolved first, so any attempt to
    escape the root is rejected.

    Raises:
        ValueError: the path is outside ``root``.
        FileNotFoundError: the path does not exist.
    """
    root_path = Path(root).expanduser().resolve()
    candidate = Path(requested or ".").expanduser()

    if not candidate.is_absolute():
        candidate = root_path / candidate

    candidate = candidate.resolve()

    if candidate != root_path and not candidate.is_relative_to(root_path):
        raise ValueError(
            "docs_path must be located inside the configured "
            "documents root."
        )

    if not candidate.exists():
        raise FileNotFoundError(f"Documents path not found: {requested!r}")

    return candidate
