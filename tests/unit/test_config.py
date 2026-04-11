import os

import pytest
from src.config import Settings


def test_settings_initialization():
    """Test settings can be initialized."""
    settings = Settings()

    assert settings.app_version == "0.1.0"
    assert settings.debug is True
    assert settings.environment == "development"
    assert settings.service_name == "rag-api"


def test_settings_postgres_defaults():
    """Test PostgreSQL default configuration."""
    settings = Settings()

    assert "postgresql://" in settings.postgres_database_url
    assert settings.postgres_echo_sql is False
    assert settings.postgres_pool_size == 20
    assert settings.postgres_max_overflow == 0


def test_settings_opensearch_defaults():
    """Test OpenSearch default configuration."""
    settings = Settings()

    assert settings.opensearch.host == "http://localhost:9200"
    assert settings.opensearch.index_name == "arxiv-papers"


def test_settings_ollama_defaults():
    """Test Ollama default configuration."""
    settings = Settings()

    # In Docker environment, this should be ollama service host
    expected_host = "http://ollama:11434" if "OLLAMA_HOST" not in os.environ else settings.ollama_host
    assert settings.ollama_host in ["http://localhost:11434", "http://ollama:11434"]


def test_feishu_history_defaults():
    """Test Feishu conversation history defaults."""
    settings = Settings()

    assert settings.feishu.history_max_turns == 6
    assert settings.feishu.history_ttl_hours == 24
    assert settings.feishu.history_lock_timeout_seconds == 30
    assert settings.feishu.history_lock_ttl_seconds == 60
    assert settings.feishu.history_lock_poll_interval_seconds == 0.1
