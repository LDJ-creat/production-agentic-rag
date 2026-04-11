import logging
import time
from typing import Dict

from langgraph.runtime import Runtime

from ..context import Context
from ..models import IntentRouteResult
from ..prompts import INTENT_ROUTER_PROMPT
from ..state import AgentState
from .llm_utils import generate_structured_with_fallback
from .utils import get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_intent_router_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str]:
    """Route the request to direct response, retrieval, or out-of-scope."""
    logger.info("NODE: intent_router")
    start_time = time.time()

    question = get_latest_query(state["messages"])

    route = "retrieve"
    reason = "Defaulted to retrieval"

    try:
        prompt = INTENT_ROUTER_PROMPT.format(question=question)
        result: IntentRouteResult = await generate_structured_with_fallback(
            runtime.context,
            prompt,
            IntentRouteResult,
            temperature=0.0,
        )
        route = result.route
        reason = result.reason
    except Exception as e:
        logger.warning(f"Intent router failed, defaulting to retrieve: {e}")

    logger.info(f"Intent route={route}, reason={reason}")
    logger.debug(f"Intent routing duration: {(time.time() - start_time):.2f}s")

    return {
        "intent_route": route,
        "intent_reason": reason,
    }


def continue_after_intent_routing(state: AgentState, runtime: Runtime[Context]) -> str:
    """Return next node name based on intent route."""
    route = state.get("intent_route") or "retrieve"
    if route not in {"direct_response", "retrieve", "out_of_scope"}:
        return "retrieve"
    return route
