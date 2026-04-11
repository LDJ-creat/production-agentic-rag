from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List


class BaseLLMClient(ABC):
    """Unified interface for all LLM providers."""

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health status."""

    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from the provider."""

    @abstractmethod
    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Dict[str, Any] | None:
        """Generate text completion for a prompt."""

    @abstractmethod
    async def generate_stream(self, model: str, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Generate text completion stream for a prompt."""

    @abstractmethod
    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str,
        use_structured_output: bool = False,
    ) -> Dict[str, Any]:
        """Generate non-streaming answer for RAG flow."""

    @abstractmethod
    async def generate_rag_answer_stream(self, query: str, chunks: List[Dict[str, Any]], model: str):
        """Generate streaming answer for RAG flow."""

    @abstractmethod
    def get_langchain_model(self, model: str, temperature: float = 0.0):
        """Get provider-specific LangChain chat model."""
