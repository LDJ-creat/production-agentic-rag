import logging
import time
from typing import Dict, List

from langgraph.runtime import Runtime

from ..context import Context
from ..models import RetrievalPlanResult
from ..prompts import RETRIEVAL_PLANNER_PROMPT
from ..state import AgentState
from .llm_utils import generate_structured_with_fallback
from .utils import get_latest_query

logger = logging.getLogger(__name__)


def _normalize_query(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split())


def _sanitize_queries(queries: List[str]) -> List[str]:
    cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    return cleaned[:3]


def _filter_duplicate_queries(candidates: List[str], attempted: List[str]) -> List[str]:
    attempted_norm = {_normalize_query(q) for q in attempted if isinstance(q, str)}
    selected: List[str] = []
    selected_norm = set()

    for query in candidates:
        norm = _normalize_query(query)
        if not norm:
            continue
        if norm in attempted_norm or norm in selected_norm:
            continue
        selected.append(query)
        selected_norm.add(norm)

    return selected


async def ainvoke_retrieval_planner_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, object]:
    """Plan retrieval strategy including rewrite/decomposition."""
    logger.info("NODE: retrieval_planner")
    start_time = time.time()

    question = get_latest_query(state["messages"])
    attempted_queries = state.get("attempted_queries", []) or []

    planned_queries: List[str] = [question]
    reason = "No special planning required"
    rewritten_query = None

    try:
        attempted_text = "\n".join(f"- {q}" for q in attempted_queries) if attempted_queries else "- (none)"
        prompt = RETRIEVAL_PLANNER_PROMPT.format(question=question, attempted_queries=attempted_text)
        plan: RetrievalPlanResult = await generate_structured_with_fallback(
            runtime.context,
            prompt,
            RetrievalPlanResult,
            temperature=0.0,
        )

        if plan.should_decompose:
            sub_queries = _sanitize_queries(plan.sub_queries)
            if sub_queries:
                planned_queries = sub_queries
        elif plan.should_rewrite and plan.rewritten_query.strip():
            rewritten_query = plan.rewritten_query.strip()
            planned_queries = [rewritten_query]

        filtered_queries = _filter_duplicate_queries(planned_queries, attempted_queries)
        if filtered_queries:
            planned_queries = filtered_queries
        else:
            planned_queries = [question]

        reason = plan.reason or reason
    except Exception as e:
        logger.warning(f"Retrieval planning failed, using original query: {e}")

    logger.info(f"Planned {len(planned_queries)} retrieval query(ies)")
    logger.debug(f"Retrieval planner duration: {(time.time() - start_time):.2f}s")

    return {
        "original_query": state.get("original_query") or question,
        "rewritten_query": rewritten_query,
        "planned_queries": planned_queries,
        "next_query_index": 1,
        "active_query": planned_queries[0],
        "retrieval_plan_reason": reason,
    }
