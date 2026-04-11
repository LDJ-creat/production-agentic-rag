import logging
import time
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.llm.base import BaseLLMClient
from src.services.opensearch.client import OpenSearchClient

from .config import GraphConfig
from .context import Context
from .nodes import (
    ainvoke_direct_response_step,
    ainvoke_evidence_check_step,
    ainvoke_generate_answer_step,
    ainvoke_grade_documents_step,
    ainvoke_intent_router_step,
    ainvoke_out_of_scope_step,
    ainvoke_retrieve_step,
    ainvoke_retrieval_planner_step,
    ainvoke_rewrite_query_step,
    continue_after_evidence_check,
    continue_after_intent_routing,
    continue_after_retrieve,
)
from .state import AgentState
from .tools import create_retriever_tool

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """Agentic RAG service 

    This implementation uses:
    - context_schema for dependency injection
    - Runtime[Context] for type-safe access in nodes
    - Direct client invocation (no pre-built runnables)
    - Lightweight nodes as pure functions
    """

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        ollama_client: BaseLLMClient,
        embeddings_client: JinaEmbeddingsClient,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        graph_config: Optional[GraphConfig] = None,
    ):
        """Initialize agentic RAG service.

        :param opensearch_client: Client for document search
        :param ollama_client: Client for LLM generation
        :param embeddings_client: Client for embeddings
        :param langfuse_tracer: Optional Langfuse tracer
        :param graph_config: Configuration for graph execution
        """
        self.opensearch = opensearch_client
        self.ollama = ollama_client
        self.embeddings = embeddings_client
        self.langfuse_tracer = langfuse_tracer
        self.graph_config = graph_config or GraphConfig()

        logger.info("Initializing AgenticRAGService with configuration:")
        logger.info(f"  Model: {self.graph_config.model}")
        logger.info(f"  Top-k: {self.graph_config.top_k}")
        logger.info(f"  Hybrid search: {self.graph_config.use_hybrid}")
        logger.info(f"  Max retrieval attempts: {self.graph_config.max_retrieval_attempts}")

        # Build graph once (no runnables needed!)
        self.graph = self._build_graph()
        logger.info("✓ AgenticRAGService initialized successfully")

    def _build_graph(self):
        """Build and compile the LangGraph workflow.

        Uses context_schema for type-safe dependency injection.
        Nodes are lightweight functions that receive Runtime[Context].

        :returns: Compiled graph ready for invocation
        """
        logger.info("Building LangGraph workflow with context_schema")

        # Create workflow with AgentState and Context schema
        workflow = StateGraph(AgentState, context_schema=Context)

        # Create tools (these still need to be created upfront for ToolNode)
        retriever_tool = create_retriever_tool(
            opensearch_client=self.opensearch,
            embeddings_client=self.embeddings,
            top_k=self.graph_config.top_k,
            use_hybrid=self.graph_config.use_hybrid,
        )
        tools = [retriever_tool]

        # Add nodes (just function references - no closures needed!)
        logger.info("Adding nodes to workflow graph")
        workflow.add_node("intent_router", ainvoke_intent_router_step)
        workflow.add_node("direct_response", ainvoke_direct_response_step)
        workflow.add_node("out_of_scope", ainvoke_out_of_scope_step)
        workflow.add_node("retrieval_planner", ainvoke_retrieval_planner_step)
        workflow.add_node("retrieve", ainvoke_retrieve_step)
        workflow.add_node("tool_retrieve", ToolNode(tools))
        workflow.add_node("grade_documents", ainvoke_grade_documents_step)
        workflow.add_node("evidence_check", ainvoke_evidence_check_step)
        workflow.add_node("rewrite_query", ainvoke_rewrite_query_step)
        workflow.add_node("generate_answer", ainvoke_generate_answer_step)

        # Add edges
        logger.info("Configuring graph edges and routing logic")

        # Start → intent routing
        workflow.add_edge(START, "intent_router")

        # Intent routing -> direct response, retrieval flow, or out-of-scope
        workflow.add_conditional_edges(
            "intent_router",
            continue_after_intent_routing,
            {
                "direct_response": "direct_response",
                "retrieve": "retrieval_planner",
                "out_of_scope": "out_of_scope",
            },
        )

        # Direct response -> END
        workflow.add_edge("direct_response", END)

        # Out of scope → END
        workflow.add_edge("out_of_scope", END)

        # Plan retrieval strategy before first retrieval call
        workflow.add_edge("retrieval_planner", "retrieve")

        # Retrieve node creates tool call
        workflow.add_conditional_edges(
            "retrieve",
            continue_after_retrieve,
            {
                "tool_retrieve": "tool_retrieve",
                "generate_answer": "generate_answer",
                "end": END,
            },
        )

        # After tool retrieval → grade documents
        workflow.add_edge("tool_retrieve", "grade_documents")

        # After grading -> evidence sufficiency check
        workflow.add_edge("grade_documents", "evidence_check")

        # After evidence check -> retrieve again, rewrite, or generate
        workflow.add_conditional_edges(
            "evidence_check",
            continue_after_evidence_check,
            {
                "retrieve": "retrieve",
                "rewrite_query": "rewrite_query",
                "generate_answer": "generate_answer",
            },
        )

        # After rewriting → try retrieve again
        workflow.add_edge("rewrite_query", "retrieve")

        # After answer generation → done
        workflow.add_edge("generate_answer", END)

        # Compile graph
        logger.info("Compiling LangGraph workflow")
        compiled_graph = workflow.compile()
        logger.info("✓ Graph compilation successful")

        return compiled_graph

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        model: Optional[str] = None,
    ) -> dict:
        """Ask a question using agentic RAG.

        :param query: User question
        :param user_id: User identifier for tracing
        :param model: Optional model override
        :returns: Dictionary with answer, sources, reasoning steps, and metadata
        :raises ValueError: If query is empty
        """
        model_to_use = model or self.graph_config.model

        logger.info("=" * 80)
        logger.info("Starting Agentic RAG Request")
        logger.info(f"Query: {query}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Model: {model_to_use}")
        logger.info("=" * 80)

        # Validate input
        if not query or len(query.strip()) == 0:
            logger.error("Empty query received")
            raise ValueError("Query cannot be empty")

        try:
            return await self._run_workflow(query, model_to_use, user_id, None, None)
        except Exception as e:
            logger.error(f"Error in Agentic RAG execution: {str(e)}")
            logger.exception("Full traceback:")
            raise

    async def _run_workflow(self, query: str, model_to_use: str, user_id: str, trace, callback_handler=None) -> dict:
        """Execute the workflow with the given trace context."""
        try:
            start_time = time.time()

            logger.info("Invoking LangGraph workflow")

            # State initialization
            state_input = {
                "messages": [HumanMessage(content=query)],
                "retrieval_attempts": 0,
                "guardrail_result": None,
                "routing_decision": None,
                "intent_route": None,
                "intent_reason": None,
                "retrieval_plan_reason": None,
                "planned_queries": [],
                "attempted_queries": [],
                "next_query_index": 0,
                "active_query": None,
                "evidence_reason": None,
                "followup_query": None,
                "sources": None,
                "relevant_sources": [],
                "relevant_tool_artefacts": None,
                "grading_results": [],
                "metadata": {},
                "original_query": None,
                "rewritten_query": None,
            }

            # Runtime context (dependencies)
            runtime_context = Context(
                ollama_client=self.ollama,
                opensearch_client=self.opensearch,
                embeddings_client=self.embeddings,
                langfuse_tracer=self.langfuse_tracer,
                trace=trace,
                langfuse_enabled=self.langfuse_tracer is not None and self.langfuse_tracer.client is not None,
                model_name=model_to_use,
                temperature=self.graph_config.temperature,
                top_k=self.graph_config.top_k,
                max_retrieval_attempts=self.graph_config.max_retrieval_attempts,
                guardrail_threshold=self.graph_config.guardrail_threshold,
            )

            # Create config with CallbackHandler if Langfuse is enabled (v3 SDK)
            config = {"thread_id": f"user_{user_id}_session_{int(time.time())}"}

            # CallbackHandler is intentionally disabled to avoid optional runtime deps.

            result = await self.graph.ainvoke(
                state_input,
                config=config,
                context=runtime_context,
            )

            execution_time = time.time() - start_time
            logger.info(f"✓ Graph execution completed in {execution_time:.2f}s")

            # Extract results
            answer = self._extract_answer(result)
            sources = self._extract_sources(result)
            retrieval_attempts = result.get("retrieval_attempts", 0)
            reasoning_steps = self._extract_reasoning_steps(result)

            # Update trace (cleanup handled by context manager)
            if trace and hasattr(trace, "update"):
                trace.update(
                    output={
                        "answer": answer,
                        "sources_count": len(sources),
                        "retrieval_attempts": retrieval_attempts,
                        "reasoning_steps": reasoning_steps,
                        "execution_time": execution_time,
                    }
                )
                if hasattr(trace, "end"):
                    trace.end()
                self.langfuse_tracer.flush()

            trace_id = self.langfuse_tracer.get_trace_id(trace) if self.langfuse_tracer else None

            logger.info("=" * 80)
            logger.info("Agentic RAG Request Completed Successfully")
            logger.info(f"Answer length: {len(answer)} characters")
            logger.info(f"Sources found: {len(sources)}")
            logger.info(f"Retrieval attempts: {retrieval_attempts}")
            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info("=" * 80)

            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "reasoning_steps": reasoning_steps,
                "retrieval_attempts": retrieval_attempts,
                "rewritten_query": result.get("rewritten_query"),
                "execution_time": execution_time,
                "guardrail_score": result.get("guardrail_result").score if result.get("guardrail_result") else None,
                "trace_id": trace_id,
                "intent_route": result.get("intent_route"),
                "out_of_scope": result.get("intent_route") == "out_of_scope",
                "max_retrieval_reached": retrieval_attempts >= self.graph_config.max_retrieval_attempts,
            }

        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            logger.exception("Full traceback:")

            # Update trace with error (cleanup handled by context manager)
            if trace and hasattr(trace, "update"):
                trace.update(output={"error": str(e)}, level="ERROR")
                if hasattr(trace, "end"):
                    trace.end()
                self.langfuse_tracer.flush()

            raise

    def _extract_answer(self, result: dict) -> str:
        """Extract final answer from graph result."""
        messages = result.get("messages", [])
        if not messages:
            return "No answer generated."

        final_message = messages[-1]
        return final_message.content if hasattr(final_message, "content") else str(final_message)

    def _extract_sources(self, result: dict) -> List[dict]:
        """Extract sources from graph result."""
        sources = []
        relevant_sources = result.get("relevant_sources", [])

        for source in relevant_sources:
            if hasattr(source, "to_dict"):
                sources.append(source.to_dict())
            elif isinstance(source, dict):
                sources.append(source)

        return sources

    def _extract_reasoning_steps(self, result: dict) -> List[str]:
        """Extract reasoning steps from graph result."""
        steps = []
        retrieval_attempts = result.get("retrieval_attempts", 0)
        guardrail_result = result.get("guardrail_result")
        grading_results = result.get("grading_results", [])

        if guardrail_result:
            steps.append(f"Validated query scope (score: {guardrail_result.score}/100)")

        if result.get("intent_route"):
            steps.append(f"Intent route: {result.get('intent_route')}")

        if result.get("retrieval_plan_reason"):
            steps.append("Planned retrieval strategy")

        if retrieval_attempts > 0:
            steps.append(f"Retrieved documents ({retrieval_attempts} attempt(s))")

        if grading_results:
            relevant_count = sum(1 for g in grading_results if g.is_relevant)
            steps.append(f"Graded documents ({relevant_count} relevant)")

        if result.get("rewritten_query"):
            steps.append("Rewritten query for better results")

        if result.get("evidence_reason"):
            steps.append("Evaluated evidence sufficiency")

        intent_route = result.get("intent_route")
        if intent_route == "direct_response":
            steps.append("Generated direct response")
        elif intent_route == "out_of_scope":
            steps.append("Returned out-of-scope guidance")
        else:
            steps.append("Generated answer from context")

        return steps

    def get_graph_visualization(self) -> bytes:
        """Get the LangGraph workflow visualization as PNG.

        This method generates a visual representation of the graph workflow
        using mermaid diagram format, then converts it to PNG.

        :returns: PNG image bytes
        :raises ImportError: If required dependencies (pygraphviz/graphviz) are not installed
        :raises Exception: If graph visualization generation fails

        Example:
            >>> service = AgenticRAGService(...)
            >>> png_bytes = service.get_graph_visualization()
            >>> with open("graph.png", "wb") as f:
            ...     f.write(png_bytes)
        """
        try:
            logger.info("Generating graph visualization as PNG")
            png_bytes = self.graph.get_graph().draw_mermaid_png()
            logger.info(f"✓ Generated PNG visualization ({len(png_bytes)} bytes)")
            return png_bytes
        except ImportError as e:
            logger.error(f"Failed to generate visualization - missing dependencies: {e}")
            logger.error("Install with: pip install pygraphviz or apt-get install graphviz")
            raise ImportError(
                "Graph visualization requires pygraphviz. "
                "Install with: pip install pygraphviz (requires graphviz system package)"
            ) from e
        except Exception as e:
            logger.error(f"Failed to generate graph visualization: {e}")
            raise

    def get_graph_mermaid(self) -> str:
        """Get the LangGraph workflow as a mermaid diagram string.

        This method generates the graph workflow representation in mermaid
        diagram syntax, which can be rendered in markdown or mermaid viewers.

        :returns: Mermaid diagram syntax as string

        Example:
            >>> service = AgenticRAGService(...)
            >>> mermaid = service.get_graph_mermaid()
            >>> print(mermaid)
            graph TD
                __start__ --> guardrail
                ...
        """
        try:
            logger.info("Generating graph as mermaid diagram")
            mermaid_str = self.graph.get_graph().draw_mermaid()
            logger.info(f"✓ Generated mermaid diagram ({len(mermaid_str)} characters)")
            return mermaid_str
        except Exception as e:
            logger.error(f"Failed to generate mermaid diagram: {e}")
            raise

    def get_graph_ascii(self) -> str:
        """Get ASCII representation of the graph.

        This method generates a simple ASCII art representation of the
        graph structure, useful for quick inspection in terminals.

        :returns: ASCII art representation of the graph

        Example:
            >>> service = AgenticRAGService(...)
            >>> print(service.get_graph_ascii())
        """
        try:
            logger.info("Generating ASCII graph representation")
            ascii_str = self.graph.get_graph().print_ascii()
            logger.info("✓ Generated ASCII graph representation")
            return ascii_str
        except Exception as e:
            logger.error(f"Failed to generate ASCII graph: {e}")
            raise
