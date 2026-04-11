from typing import Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.llm.base import BaseLLMClient
from src.services.opensearch.client import OpenSearchClient

from .agentic_rag import AgenticRAGService
from .config import GraphConfig


def make_agentic_rag_service(
    opensearch_client: OpenSearchClient,
    ollama_client: BaseLLMClient,
    embeddings_client: JinaEmbeddingsClient,
    langfuse_tracer: Optional[LangfuseTracer] = None,
    model: str = "llama3.2:1b",
    top_k: int = 3,
    use_hybrid: bool = True,
) -> AgenticRAGService:
    """
    Create AgenticRAGService with dependency injection.

    Args:
        opensearch_client: Client for document search
        ollama_client: Client for LLM generation
        embeddings_client: Client for embeddings
        langfuse_tracer: Optional Langfuse tracer for observability
        model: Default model name for LLM calls
        top_k: Number of documents to retrieve (default: 3)
        use_hybrid: Use hybrid search (default: True)

    Returns:
        Configured AgenticRAGService instance
    """
    # Create graph configuration with the provided parameters
    graph_config = GraphConfig(
        model=model,
        top_k=top_k,
        use_hybrid=use_hybrid,
    )

    return AgenticRAGService(
        opensearch_client=opensearch_client,
        ollama_client=ollama_client,
        embeddings_client=embeddings_client,
        langfuse_tracer=langfuse_tracer,
        graph_config=graph_config,
    )
