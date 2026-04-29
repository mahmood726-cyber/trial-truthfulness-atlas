"""Test fixtures shared across tta tests."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def judge_cache_dir(tmp_path) -> Path:
    cache = tmp_path / "judge_cache"
    cache.mkdir()
    return cache
