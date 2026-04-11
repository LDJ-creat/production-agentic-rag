from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from src.services.agents.context import Context
from src.services.agents.nodes.guardrail_node import ainvoke_guardrail_step


@pytest.mark.asyncio
async def test_guardrail_node_uses_get_langchain_model_for_api_provider():
    mock_structured_llm = Mock()
    mock_structured_llm.ainvoke = AsyncMock(return_value=Mock(score=80, reason="In scope"))

    mock_llm = Mock()
    mock_llm.with_structured_output = Mock(return_value=mock_structured_llm)

    mock_api_llm_client = Mock()
    mock_api_llm_client.get_langchain_model = Mock(return_value=mock_llm)

    context = Context(
        ollama_client=mock_api_llm_client,
        opensearch_client=Mock(),
        embeddings_client=Mock(),
        langfuse_tracer=None,
        trace=None,
        langfuse_enabled=False,
        model_name="openai/gpt-4.1-mini",
        temperature=0.0,
        top_k=3,
        max_retrieval_attempts=2,
        guardrail_threshold=60,
    )

    runtime = Mock(spec=Runtime)
    runtime.context = context

    result = await ainvoke_guardrail_step(
        state={"messages": [HumanMessage(content="What are transformers?")]},
        runtime=runtime,
    )

    mock_api_llm_client.get_langchain_model.assert_called_once_with(
        model="openai/gpt-4.1-mini",
        temperature=0.0,
    )
    assert result["guardrail_result"].score == 80
