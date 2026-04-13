"""Shared test fixtures and configuration.

Tests use the builtin embedding provider (no API key needed) and mock LLM calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aimemo.core.config import settings
from aimemo.core.embeddings import reset_provider
from aimemo.core.llm import reset_client


@pytest.fixture(autouse=True)
def _configure_test_settings():
    """Force builtin embeddings and reset singletons between tests."""
    original_provider = settings.embedding_provider
    original_dim = settings.embedding_dim
    settings.embedding_provider = "builtin"
    settings.embedding_dim = 128
    reset_provider()
    reset_client()
    yield
    settings.embedding_provider = original_provider
    settings.embedding_dim = original_dim
    reset_provider()
    reset_client()


@pytest.fixture
def mock_chat_completion():
    """Mock chat_completion to avoid real LLM calls."""
    with patch("aimemo.core.llm.chat_completion", new_callable=AsyncMock) as m:
        m.return_value = "Mocked LLM response for testing."
        yield m


@pytest.fixture
def mock_vision_describe():
    """Mock vision_describe to avoid real vision model calls."""
    with patch("aimemo.core.llm.vision_describe", new_callable=AsyncMock) as m:
        m.return_value = "A photograph showing a cat sitting on a windowsill."
        yield m


@pytest.fixture
def mock_analyze_memory():
    """Mock analyze_memory to avoid real LLM calls."""
    with patch("aimemo.core.llm.analyze_memory", new_callable=AsyncMock) as m:
        m.return_value = {
            "memory_type": "semantic",
            "importance": 0.8,
            "tags": ["python", "preference"],
        }
        yield m


@pytest.fixture
def mock_detect_contradiction():
    """Mock detect_contradiction to avoid real LLM calls."""
    with patch("aimemo.core.llm.detect_contradiction", new_callable=AsyncMock) as m:
        m.return_value = {
            "has_contradiction": True,
            "contradicted_indices": [0],
            "merged_fact": "Updated merged fact from LLM.",
        }
        yield m


@pytest.fixture
def mock_detect_no_contradiction():
    """Mock detect_contradiction returning no contradiction."""
    with patch("aimemo.core.llm.detect_contradiction", new_callable=AsyncMock) as m:
        m.return_value = {
            "has_contradiction": False,
            "contradicted_indices": [],
            "merged_fact": "",
        }
        yield m
