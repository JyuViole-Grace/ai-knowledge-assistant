#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible launcher for AI Knowledge Assistant."""
from app.assistant import RAGAssistant  # noqa: F401  (re-exported)
from app.cli import main
from app.config import Config  # noqa: F401  (re-exported)
from app.evaluation import run_tests  # noqa: F401  (re-exported)

if __name__ == "__main__":
    raise SystemExit(main())
