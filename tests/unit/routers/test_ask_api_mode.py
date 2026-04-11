from unittest.mock import AsyncMock, Mock

import pytest

from src.routers.ask import ask_question
from src.schemas.api.ask import AskRequest


@pytest.mark.asyncio
async def test_ask_question_uses_unified_llm_client_for_api_mode():
    request = AskRequest(query="What is RAG?", model="openai/gpt-4.1-mini", top_k=2, use_hybrid=False)

    opensearch_client = Mock()
    opensearch_client.search_unified.return_value = {
        "hits": [
            {"arxiv_id": "2401.00001", "chunk_text": "RAG combines retrieval and generation."},
        ],
        "total": 1,
    }

    embeddings_service = Mock()
    embeddings_service.embed_query = AsyncMock(return_value=[0.1, 0.2])

    llm_client = Mock()
    llm_client.generate_rag_answer = AsyncMock(
        return_value={
            "answer": "RAG first retrieves documents, then generates answers based on them.",
            "sources": ["https://arxiv.org/pdf/2401.00001.pdf"],
            "confidence": "medium",
            "citations": ["2401.00001"],
            "usage_metadata": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    )

    langfuse_tracer = Mock()
    langfuse_tracer.client = None

    response = await ask_question(
        request=request,
        opensearch_client=opensearch_client,
        embeddings_service=embeddings_service,
        llm_client=llm_client,
        langfuse_tracer=langfuse_tracer,
        cache_client=None,
    )

    assert response.answer.startswith("RAG first retrieves")
    llm_client.generate_rag_answer.assert_awaited_once()
