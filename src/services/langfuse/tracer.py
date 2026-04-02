"""Simple, efficient Langfuse tracing utility for RAG pipeline."""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .client import LangfuseTracer


class RAGTracer:
    """Clean, purpose-built tracer for RAG operations."""

    def __init__(self, tracer: LangfuseTracer):
        self.tracer = tracer

    def _enabled(self) -> bool:
        return bool(getattr(self.tracer, "client", None))

    @contextmanager
    def trace_request(self, user_id: str, query: str):
        """Main request trace context manager."""
        if not self._enabled():
            yield None
            return

        trace = None
        yielded = False
        try:
            if hasattr(self.tracer, "trace_rag_request"):
                with self.tracer.trace_rag_request(
                    query=query, user_id=user_id, session_id=f"session_{user_id}", metadata={"simplified_tracing": True}
                ) as trace:
                    yielded = True
                    yield trace
                return

            if hasattr(self.tracer, "start_span"):
                with self.tracer.start_span(
                    name="rag_request",
                    input_data={"query": query, "user_id": user_id, "session_id": f"session_{user_id}"},
                ) as trace:
                    yielded = True
                    yield trace
                return

            yielded = True
            yield None
        except Exception:
            # If the request body already started, keep original exception semantics.
            if yielded:
                raise
            yielded = True
            yield None
        finally:
            if trace and hasattr(self.tracer, "flush"):
                try:
                    self.tracer.flush()
                except Exception:
                    pass

    @contextmanager
    def trace_embedding(self, trace, query: str):
        """Query embedding operation with timing."""
        if not self._enabled():
            yield None
            return

        start_time = time.time()
        span = None
        try:
            if hasattr(self.tracer, "create_span"):
                span = self.tracer.create_span(
                    trace=trace, name="query_embedding", input_data={"query": query, "query_length": len(query)}
                )
            elif hasattr(self.tracer, "start_span"):
                span = self.tracer.client.span(
                    name="query_embedding", input={"query": query, "query_length": len(query)}, metadata={}
                ) if getattr(self.tracer, "client", None) else None
        except Exception:
            span = None
        try:
            yield span
        finally:
            duration = time.time() - start_time
            if span:
                self.tracer.update_span(span=span, output={"embedding_duration_ms": round(duration * 1000, 2), "success": True})

    @contextmanager
    def trace_search(self, trace, query: str, top_k: int):
        """Search operation with timing."""
        if not self._enabled():
            yield None
            return

        span = None
        try:
            if hasattr(self.tracer, "create_span"):
                span = self.tracer.create_span(trace=trace, name="search_retrieval", input_data={"query": query, "top_k": top_k})
            elif hasattr(self.tracer, "start_span"):
                span = self.tracer.client.span(
                    name="search_retrieval", input={"query": query, "top_k": top_k}, metadata={}
                ) if getattr(self.tracer, "client", None) else None
        except Exception:
            span = None
        try:
            yield span
        finally:
            if span:
                pass

    def end_search(self, span, chunks: List[Dict], arxiv_ids: List[str], total_hits: int):
        """End search span with essential results."""
        if not self._enabled() or not span:
            return

        self.tracer.update_span(
            span=span,
            output={
                "chunks_returned": len(chunks),
                "unique_papers": len(set(arxiv_ids)),
                "total_hits": total_hits,
                "arxiv_ids": list(set(arxiv_ids)),
            },
        )

    @contextmanager
    def trace_prompt_construction(self, trace, chunks: List[Dict]):
        """Prompt building with timing."""
        if not self._enabled():
            yield None
            return

        span = None
        try:
            if hasattr(self.tracer, "create_span"):
                span = self.tracer.create_span(trace=trace, name="prompt_construction", input_data={"chunk_count": len(chunks)})
            elif hasattr(self.tracer, "start_span"):
                span = self.tracer.client.span(
                    name="prompt_construction", input={"chunk_count": len(chunks)}, metadata={}
                ) if getattr(self.tracer, "client", None) else None
        except Exception:
            span = None
        try:
            yield span
        finally:
            if span:
                pass

    def end_prompt(self, span, prompt: str):
        """End prompt span with final prompt."""
        if not self._enabled() or not span:
            return

        self.tracer.update_span(
            span=span,
            output={
                "prompt_length": len(prompt),
                # Don't duplicate the full prompt here since it's in llm_generation input
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            },
        )

    @contextmanager
    def trace_generation(self, trace, model: str, prompt: str):
        """LLM generation with timing."""
        if not self._enabled():
            yield None
            return

        span = None
        try:
            if hasattr(self.tracer, "create_span"):
                span = self.tracer.create_span(
                    trace=trace, name="llm_generation", input_data={"model": model, "prompt_length": len(prompt), "prompt": prompt}
                )
            elif hasattr(self.tracer, "start_generation"):
                with self.tracer.start_generation(
                    name="llm_generation", model=model, input_data={"prompt_length": len(prompt), "prompt": prompt}
                ) as gen:
                    yield gen
                    return
        except Exception:
            span = None
        try:
            yield span
        finally:
            if span:
                pass

    def end_generation(self, span, response: str, model: str):
        """End generation span with response."""
        if not self._enabled() or not span:
            return

        self.tracer.update_span(span=span, output={"response": response, "response_length": len(response), "model_used": model})

    def end_request(self, trace, response: str, total_duration: float):
        """End main request trace."""
        if not self._enabled() or not trace:
            return

        try:
            if hasattr(trace, "update"):
                trace.update(
                    output={"answer": response, "total_duration_seconds": round(total_duration, 3), "response_length": len(response)}
                )
            elif hasattr(self.tracer, "update_span"):
                self.tracer.update_span(
                    span=trace,
                    output={"answer": response, "total_duration_seconds": round(total_duration, 3), "response_length": len(response)},
                )
        except Exception:
            # Silently fail - don't break the request for tracing issues
            pass
