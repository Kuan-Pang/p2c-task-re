"""Shared pytest fixtures. Area-specific helpers live in tests/<area>_helpers.py, not here."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
EXAMPLE_TASKS = REPO_ROOT / "examples" / "tasks"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
