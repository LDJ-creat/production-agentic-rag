from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.main import app
from src.services.agents.context import Context


@pytest.fixture(scope="session")
def anyio_backend() -> str:
	"""Async backend for testing."""
	return "asyncio"


@pytest.fixture
def mock_opensearch_client():
	client = Mock()
	client.search_unified = Mock(return_value={"hits": [], "total": 0})
	return client


@pytest.fixture
def mock_ollama_client():
	client = Mock()
	client.generate = AsyncMock(return_value={"response": "mocked answer", "usage_metadata": {}})
	client.generate_rag_answer = AsyncMock(
		return_value={
			"answer": "mocked rag answer",
			"sources": [],
			"confidence": "medium",
			"citations": [],
			"usage_metadata": {},
		}
	)
	client.generate_rag_answer_stream = AsyncMock()
	client.get_langchain_model = Mock(return_value=Mock())
	client.create_llm = Mock(return_value=Mock())
	client.health_check = AsyncMock(return_value={"status": "healthy", "message": "ok"})
	return client


@pytest.fixture
def mock_jina_embeddings_client():
	client = Mock()
	client.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
	return client


@pytest.fixture
def sample_human_message():
	return HumanMessage(content="What is machine learning?")


@pytest.fixture
def sample_ai_message():
	return AIMessage(content="Machine learning is a subset of AI.")


@pytest.fixture
def sample_tool_message():
	return ToolMessage(
		content="Transformers are neural network architectures using self-attention.",
		tool_call_id="call-1",
		name="retrieve_papers",
	)


@pytest.fixture
def test_context(mock_ollama_client, mock_opensearch_client, mock_jina_embeddings_client):
	return Context(
		ollama_client=mock_ollama_client,
		opensearch_client=mock_opensearch_client,
		embeddings_client=mock_jina_embeddings_client,
		langfuse_tracer=None,
		trace=None,
		langfuse_enabled=False,
		model_name="llama3.2:1b",
		temperature=0.0,
		top_k=3,
		max_retrieval_attempts=2,
		guardrail_threshold=60,
	)


@pytest_asyncio.fixture
async def client():
	"""HTTP client for API testing with mocked app services."""
	with (
		patch("src.db.interfaces.postgresql.PostgreSQLDatabase.startup") as mock_startup,
		patch("src.db.interfaces.postgresql.PostgreSQLDatabase.get_session") as mock_get_session,
		patch("src.services.opensearch.factory.make_opensearch_client") as mock_os,
		patch("src.services.arxiv.factory.make_arxiv_client") as mock_arxiv,
		patch("src.services.pdf_parser.factory.make_pdf_parser_service") as mock_pdf,
		patch("src.services.llm.factory.make_llm_client") as mock_llm,
		patch("src.services.cache.factory.make_cache_client") as mock_cache,
		patch("src.repositories.paper.PaperRepository.get_by_arxiv_id") as mock_get_by_id,
	):
		mock_startup.return_value = None

		mock_session = MagicMock()
		mock_get_session.return_value.__enter__.return_value = mock_session
		mock_get_session.return_value.__exit__.return_value = None
		mock_get_by_id.return_value = None

		mock_os_client = MagicMock()
		mock_os_client.health_check.return_value = True
		mock_os_client.setup_indices.return_value = {"hybrid_index": True}
		mock_os_client.client.count.return_value = {"count": 0}
		mock_os_client.search_unified.return_value = {"hits": [], "total": 0}

		mock_embed_client = MagicMock()
		mock_embed_client.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

		mock_llm_client = MagicMock()
		mock_llm_client.health_check = AsyncMock(return_value={"status": "healthy", "message": "ok"})
		mock_llm_client.generate_rag_answer = AsyncMock(
			return_value={"answer": "mock answer", "sources": [], "citations": [], "usage_metadata": {}}
		)
		mock_llm_client.generate_rag_answer_stream = AsyncMock()

		mock_os.return_value = mock_os_client
		mock_arxiv.return_value = AsyncMock()
		mock_pdf.return_value = AsyncMock()
		mock_llm.return_value = mock_llm_client
		mock_cache.return_value = None

		mock_langfuse_tracer = MagicMock()
		mock_langfuse_tracer.client = None
		mock_langfuse_tracer.flush = Mock()

		with (
			patch("src.main.make_opensearch_client", return_value=mock_os_client),
			patch("src.main.make_embeddings_service", return_value=mock_embed_client),
			patch("src.main.make_llm_client", return_value=mock_llm_client),
			patch("src.main.make_cache_client", return_value=None),
			patch("src.main.make_langfuse_tracer", return_value=mock_langfuse_tracer),
			patch("src.main.make_telegram_service", return_value=None),
			patch("src.main.make_feishu_service", return_value=None),
		):
			async with LifespanManager(app) as manager:
				async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as c:
					yield c
