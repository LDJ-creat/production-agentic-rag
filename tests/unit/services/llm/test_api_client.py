import pytest

from src.config import Settings
from src.services.llm.api_client import APICompatibleLLMClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_openai_compatible_generate(monkeypatch):
    settings = Settings()
    client = APICompatibleLLMClient(
        settings.model_copy(
            update={
                "llm": settings.llm.model_copy(
                    update={
                        "provider": "api",
                        "api": settings.llm.api.model_copy(
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
    )

    async def _fake_post(self, url, headers=None, json=None):
        return _FakeResponse(
            200,
            {
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    monkeypatch.setattr("httpx.AsyncClient.post", _fake_post)

    result = await client.generate(model="openai/gpt-4.1-mini", prompt="hi")

    assert result is not None
    assert result["response"] == "hello"
    assert result["usage_metadata"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_anthropic_compatible_generate(monkeypatch):
    settings = Settings()
    client = APICompatibleLLMClient(
        settings.model_copy(
            update={
                "llm": settings.llm.model_copy(
                    update={
                        "provider": "api",
                        "api": settings.llm.api.model_copy(
                            update={
                                "format": "anthropic",
                                "base_url": "https://example.com",
                                "api_key": "k",
                            }
                        ),
                    }
                )
            }
        )
    )

    async def _fake_post(self, url, headers=None, json=None):
        return _FakeResponse(
            200,
            {
                "content": [{"type": "text", "text": "anthropic text"}],
                "usage": {"input_tokens": 9, "output_tokens": 6},
            },
        )

    monkeypatch.setattr("httpx.AsyncClient.post", _fake_post)

    result = await client.generate(model="anthropic/claude-3-5-sonnet", prompt="hi")

    assert result is not None
    assert result["response"] == "anthropic text"
    assert result["usage_metadata"]["prompt_tokens"] == 9
    assert result["usage_metadata"]["completion_tokens"] == 6
