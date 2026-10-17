import os
import tempfile
from pathlib import Path

import pytest

from app.security import resolve_docs_path


def make_tree(base):
    root = Path(base) / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("hello", encoding="utf-8")
    (Path(base) / "secret.txt").write_text("secret", encoding="utf-8")
    (Path(base) / "docs_evil").mkdir()
    return root


def test_allows_root_files_and_subdirectories():
    with tempfile.TemporaryDirectory() as base:
        root = make_tree(base)

        assert resolve_docs_path(".", root) == root.resolve()
        assert resolve_docs_path("", root) == root.resolve()
        assert resolve_docs_path("sub", root) == (root / "sub").resolve()
        assert resolve_docs_path("a.txt", root) == (root / "a.txt").resolve()
        assert resolve_docs_path(str(root / "a.txt"), root) == (root / "a.txt").resolve()


def test_blocks_parent_traversal():
    with tempfile.TemporaryDirectory() as base:
        root = make_tree(base)

        with pytest.raises(ValueError):
            resolve_docs_path("..", root)

        with pytest.raises(ValueError):
            resolve_docs_path("../secret.txt", root)

        with pytest.raises(ValueError):
            resolve_docs_path("sub/../../secret.txt", root)


def test_blocks_absolute_paths_outside_root():
    with tempfile.TemporaryDirectory() as base:
        root = make_tree(base)

        with pytest.raises(ValueError):
            resolve_docs_path(str(Path(base) / "secret.txt"), root)

        with pytest.raises(ValueError):
            resolve_docs_path("/etc/passwd", root)


def test_blocks_sibling_directory_with_common_prefix():
    # "docs_evil" starts with "docs": a naive startswith() check would pass.
    with tempfile.TemporaryDirectory() as base:
        root = make_tree(base)

        with pytest.raises(ValueError):
            resolve_docs_path(str(Path(base) / "docs_evil"), root)


def test_blocks_symlink_escaping_root():
    with tempfile.TemporaryDirectory() as base:
        root = make_tree(base)
        link = root / "link"

        try:
            os.symlink(Path(base) / "secret.txt", link)
        except (OSError, NotImplementedError):
            return  # symlinks not available on this platform

        with pytest.raises(ValueError):
            resolve_docs_path("link", root)


def test_missing_path_raises_file_not_found():
    with tempfile.TemporaryDirectory() as base:
        root = make_tree(base)

        with pytest.raises(FileNotFoundError):
            resolve_docs_path("nope.txt", root)
