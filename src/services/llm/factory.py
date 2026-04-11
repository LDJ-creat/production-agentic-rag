from functools import lru_cache

from src.config import get_settings
from src.services.llm.api_client import APICompatibleLLMClient
from src.services.llm.base import BaseLLMClient
from src.services.ollama.client import OllamaClient


@lru_cache(maxsize=1)
def make_llm_client() -> BaseLLMClient:
    """Create and return unified LLM client based on configured provider."""
    settings = get_settings()
    provider = settings.llm.provider

    if provider == "ollama":
        return OllamaClient(settings)

    return APICompatibleLLMClient(settings)
