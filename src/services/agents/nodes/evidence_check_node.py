import logging
import time
from typing import Dict

from langgraph.runtime import Runtime

from ..context import Context
from ..models import EvidenceCheckResult
from ..prompts import EVIDENCE_CHECK_PROMPT
from ..state import AgentState
from .llm_utils import generate_structured_with_fallback
from .utils import get_latest_context

logger = logging.getLogger(__name__)


def _normalize_query(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split())


def _is_duplicate_query(candidate: str, attempted: list[str]) -> bool:
    candidate_norm = _normalize_query(candidate)
    if not candidate_norm:
        return True
    attempted_norm = {_normalize_query(q) for q in attempted if isinstance(q, str)}
    return candidate_norm in attempted_norm


async def ainvoke_evidence_check_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str]:
    """Decide whether current evidence is sufficient or another retrieval is needed."""
    logger.info("NODE: evidence_check")
    start_time = time.time()

    question = state.get("original_query") or ""
    context = get_latest_context(state["messages"])
    planned_queries = state.get("planned_queries", []) or []
    attempted_queries = state.get("attempted_queries", []) or []
    next_query_index = state.get("next_query_index", 0)

    if not context.strip():
        if next_query_index < len(planned_queries):
            next_query = planned_queries[next_query_index]
            if _is_duplicate_query(next_query, attempted_queries):
                return {
                    "routing_decision": "rewrite_query",
                    "evidence_reason": "Next planned query duplicates previous retrieval; forcing rewrite",
                }
            return {
                "routing_decision": "retrieve",
                "active_query": next_query,
                "next_query_index": next_query_index + 1,
                "evidence_reason": "No context from previous retrieval; trying next planned query",
            }
        return {
            "routing_decision": "rewrite_query",
            "evidence_reason": "No context available; trying rewritten query",
        }

    need_more = False
    reason = "Current evidence appears sufficient"
    followup_query = ""

    try:
        attempted_text = "\n".join(f"- {q}" for q in attempted_queries) if attempted_queries else "- (none)"
        prompt = EVIDENCE_CHECK_PROMPT.format(question=question, attempted_queries=attempted_text, context=context)
        decision: EvidenceCheckResult = await generate_structured_with_fallback(
            runtime.context,
            prompt,
            EvidenceCheckResult,
            temperature=0.0,
        )
        need_more = decision.need_more_retrieval
        reason = decision.reason or reason
        followup_query = (decision.followup_query or "").strip()
    except Exception as e:
        logger.warning(f"Evidence check failed, defaulting to generate answer: {e}")

    if need_more:
        if next_query_index < len(planned_queries):
            next_query = planned_queries[next_query_index]
            if _is_duplicate_query(next_query, attempted_queries):
                logger.info("Evidence insufficient: next planned query duplicated, trying rewrite")
                return {
                    "routing_decision": "rewrite_query",
                    "evidence_reason": "Next planned query duplicated previous retrieval",
                    "followup_query": followup_query or None,
                }
            logger.info("Evidence insufficient: trying next planned query")
            return {
                "routing_decision": "retrieve",
                "active_query": next_query,
                "next_query_index": next_query_index + 1,
                "evidence_reason": reason,
                "followup_query": followup_query or None,
            }

        if followup_query and state.get("retrieval_attempts", 0) < runtime.context.max_retrieval_attempts:
            if _is_duplicate_query(followup_query, attempted_queries):
                logger.info("Evidence insufficient: follow-up query duplicated previous retrieval, generating answer")
                return {
                    "routing_decision": "generate_answer",
                    "evidence_reason": "Follow-up query duplicated previous retrieval",
                    "followup_query": followup_query,
                }
            logger.info("Evidence insufficient: using follow-up query")
            return {
                "routing_decision": "retrieve",
                "active_query": followup_query,
                "evidence_reason": reason,
                "followup_query": followup_query,
            }

        logger.info("Evidence insufficient but retrieval budget exhausted; generating answer")
        return {
            "routing_decision": "generate_answer",
            "evidence_reason": reason,
            "followup_query": followup_query or None,
        }

    logger.debug(f"Evidence check duration: {(time.time() - start_time):.2f}s")
    return {
        "routing_decision": "generate_answer",
        "evidence_reason": reason,
        "followup_query": followup_query or None,
    }


def continue_after_evidence_check(state: AgentState, runtime: Runtime[Context]) -> str:
    """Return next node based on evidence-check routing decision."""
    route = state.get("routing_decision") or "generate_answer"
    if route in {"retrieve", "rewrite_query", "generate_answer"}:
        return route
    return "generate_answer"
