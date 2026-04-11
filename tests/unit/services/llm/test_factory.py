from src.config import Settings
from src.services.llm.api_client import APICompatibleLLMClient
from src.services.llm.factory import make_llm_client
from src.services.ollama.client import OllamaClient


def test_make_llm_client_returns_ollama(monkeypatch):
    settings = Settings().model_copy(update={"llm": Settings().llm.model_copy(update={"provider": "ollama"})})
    monkeypatch.setattr("src.services.llm.factory.get_settings", lambda: settings)
    make_llm_client.cache_clear()

    client = make_llm_client()

    assert isinstance(client, OllamaClient)


def test_make_llm_client_returns_api_client(monkeypatch):
    base_settings = Settings()
    settings = base_settings.model_copy(
        update={
            "llm": base_settings.llm.model_copy(
                update={
                    "provider": "api",
                    "api": base_settings.llm.api.model_copy(
                        update={
                            "format": "openai_compatible",
                            "base_url": "https://example.com",
                            "api_key": "k",
                        }
                    ),
                }
            )
        }
    )
    monkeypatch.setattr("src.services.llm.factory.get_settings", lambda: settings)
    make_llm_client.cache_clear()

    client = make_llm_client()

    assert isinstance(client, APICompatibleLLMClient)
