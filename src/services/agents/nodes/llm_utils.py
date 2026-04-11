import json
import logging
from typing import Type, TypeVar

from pydantic import BaseModel

from ..context import Context

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


def _extract_json_object(text: str) -> str:
    """Extract a JSON object from raw model text output."""
    if not text:
        return "{}"

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]

    return stripped


async def generate_text_with_fallback(
    runtime: Context,
    prompt: str,
    temperature: float = 0.0,
) -> str:
    """Generate plain text via unified LLM client."""
    response = await runtime.ollama_client.generate(
        model=runtime.model_name,
        prompt=prompt,
        temperature=temperature,
    )
    if not response:
        return ""
    return str(response.get("response", "")).strip()


async def generate_structured_with_fallback(
    runtime: Context,
    prompt: str,
    model_cls: Type[TModel],
    temperature: float = 0.0,
) -> TModel:
    """Generate structured output with LangChain first, then JSON fallback."""
    try:
        llm = runtime.ollama_client.get_langchain_model(
            model=runtime.model_name,
            temperature=temperature,
        )
        structured_llm = llm.with_structured_output(model_cls)
        return await structured_llm.ainvoke(prompt)
    except Exception as e:
        logger.warning(f"Structured LangChain generation unavailable, using JSON fallback: {e}")

    text = await generate_text_with_fallback(runtime, prompt, temperature=temperature)
    json_text = _extract_json_object(text)

    data = json.loads(json_text)
    return model_cls.model_validate(data)
