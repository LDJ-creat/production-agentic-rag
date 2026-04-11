import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List

import httpx
from src.config import Settings
from src.exceptions import LLMException
from src.services.llm.base import BaseLLMClient
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)


class APICompatibleLLMClient(BaseLLMClient):
    """LLM client for OpenAI-compatible and Anthropic-compatible APIs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_settings = settings.llm.api
        self.timeout = httpx.Timeout(float(self.settings.llm.timeout))
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

    @property
    def _is_openai_compatible(self) -> bool:
        return self.api_settings.format == "openai_compatible"

    def _build_url(self, path: str) -> str:
        return f"{self.api_settings.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _base_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._is_openai_compatible:
            headers["Authorization"] = f"Bearer {self.api_settings.api_key}"
        else:
            headers["x-api-key"] = self.api_settings.api_key
            headers["anthropic-version"] = self.api_settings.anthropic_version
        return headers

    async def health_check(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self._build_url("v1/models"),
                    headers=self._base_headers(),
                )

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "message": "API provider is reachable",
                }

            raise LLMException(f"API provider health check failed: {response.status_code}")
        except Exception as e:
            raise LLMException(f"API provider health check failed: {e}")

    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self._build_url("v1/models"),
                    headers=self._base_headers(),
                )

            if response.status_code != 200:
                raise LLMException(f"Failed to list models: {response.status_code}")

            data = response.json()
            return data.get("data", []) if isinstance(data, dict) else []
        except Exception as e:
            raise LLMException(f"Error listing models: {e}")

    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Dict[str, Any] | None:
        if self._is_openai_compatible:
            return await self._generate_openai_compatible(model=model, prompt=prompt, stream=stream, **kwargs)
        return await self._generate_anthropic_compatible(model=model, prompt=prompt, stream=stream, **kwargs)

    async def _generate_openai_compatible(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Dict[str, Any] | None:
        started = time.perf_counter()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self._build_url("v1/chat/completions"),
                headers=self._base_headers(),
                json=payload,
            )

        if response.status_code != 200:
            raise LLMException(f"OpenAI-compatible generation failed: {response.status_code} - {response.text}")

        result = response.json()
        text = ""
        choices = result.get("choices", [])
        if choices:
            message_content = choices[0].get("message", {}).get("content", "")
            if isinstance(message_content, str):
                text = message_content
            elif isinstance(message_content, list):
                text = "".join(
                    part.get("text", "")
                    for part in message_content
                    if isinstance(part, dict)
                )

        usage = result.get("usage", {})
        usage_metadata = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

        result["response"] = text
        result["usage_metadata"] = usage_metadata
        return result

    async def _generate_anthropic_compatible(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Dict[str, Any] | None:
        started = time.perf_counter()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.pop("max_tokens", self.api_settings.max_tokens),
            "stream": stream,
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self._build_url("v1/messages"),
                headers=self._base_headers(),
                json=payload,
            )

        if response.status_code != 200:
            raise LLMException(f"Anthropic-compatible generation failed: {response.status_code} - {response.text}")

        result = response.json()
        text = ""
        for part in result.get("content", []):
            if isinstance(part, dict) and part.get("type") == "text":
                text += part.get("text", "")

        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        usage_metadata = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

        result["response"] = text
        result["usage_metadata"] = usage_metadata
        return result

    async def generate_stream(self, model: str, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        if self._is_openai_compatible:
            async for chunk in self._generate_stream_openai_compatible(model=model, prompt=prompt, **kwargs):
                yield chunk
            return

        async for chunk in self._generate_stream_anthropic_compatible(model=model, prompt=prompt, **kwargs):
            yield chunk

    async def _generate_stream_openai_compatible(self, model: str, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                self._build_url("v1/chat/completions"),
                headers=self._base_headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise LLMException(f"OpenAI-compatible stream failed: {response.status_code} - {body.decode(errors='ignore')}")

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        yield {"done": True}
                        break

                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = payload.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    finish_reason = choices[0].get("finish_reason")

                    if token:
                        yield {"response": token, "done": False}
                    if finish_reason:
                        yield {"done": True}
                        break

    async def _generate_stream_anthropic_compatible(self, model: str, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.pop("max_tokens", self.api_settings.max_tokens),
            "stream": True,
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                self._build_url("v1/messages"),
                headers=self._base_headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise LLMException(f"Anthropic-compatible stream failed: {response.status_code} - {body.decode(errors='ignore')}")

                event_type = ""
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("event:"):
                        event_type = line.removeprefix("event:").strip()
                        continue

                    if not line.startswith("data:"):
                        continue

                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        yield {"done": True}
                        break

                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if event_type == "content_block_delta":
                        delta = payload.get("delta", {})
                        token = delta.get("text", "")
                        if token:
                            yield {"response": token, "done": False}

                    if event_type == "message_stop":
                        yield {"done": True}
                        break

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str,
        use_structured_output: bool = False,
    ) -> Dict[str, Any]:
        if use_structured_output:
            prompt_data = self.prompt_builder.create_structured_prompt(query, chunks)
            response = await self.generate(
                model=model,
                prompt=prompt_data["prompt"],
                temperature=0.7,
                top_p=0.9,
            )
            if response and response.get("response"):
                parsed_response = self.response_parser.parse_structured_response(response["response"])
                parsed_response["usage_metadata"] = response.get("usage_metadata", {})
                return parsed_response

        prompt = self.prompt_builder.create_rag_prompt(query, chunks)
        response = await self.generate(
            model=model,
            prompt=prompt,
            temperature=0.7,
            top_p=0.9,
        )

        if not response:
            raise LLMException("No response generated from API provider")

        answer_text = response.get("response", "")
        sources = []
        seen_urls = set()
        for chunk in chunks:
            arxiv_id = chunk.get("arxiv_id")
            if arxiv_id:
                arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                if pdf_url not in seen_urls:
                    sources.append(pdf_url)
                    seen_urls.add(pdf_url)

        citations = list(set(chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")))

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": "medium",
            "citations": citations[:5],
            "usage_metadata": response.get("usage_metadata", {}),
        }

    async def generate_rag_answer_stream(self, query: str, chunks: List[Dict[str, Any]], model: str):
        prompt = self.prompt_builder.create_rag_prompt(query, chunks)
        async for chunk in self.generate_stream(
            model=model,
            prompt=prompt,
            temperature=0.7,
            top_p=0.9,
        ):
            yield chunk

    def get_langchain_model(self, model: str, temperature: float = 0.0):
        if self._is_openai_compatible:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                api_key=self.api_settings.api_key,
                base_url=self.api_settings.base_url,
                timeout=float(self.settings.llm.timeout),
                temperature=temperature,
            )

        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            anthropic_api_key=self.api_settings.api_key,
            anthropic_api_url=self.api_settings.base_url,
            timeout=float(self.settings.llm.timeout),
            temperature=temperature,
        )

    def create_llm(self, model: str, temperature: float = 0.0):
        """Backward-compatible alias used in older tests."""
        return self.get_langchain_model(model=model, temperature=temperature)
