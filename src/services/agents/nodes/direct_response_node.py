import logging
import time
from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..prompts import DIRECT_CHAT_PROMPT
from ..state import AgentState
from .llm_utils import generate_text_with_fallback
from .utils import get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_direct_response_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Generate a direct conversational response without retrieval."""
    logger.info("NODE: direct_response")
    start_time = time.time()

    question = get_latest_query(state["messages"])
    try:
        prompt = DIRECT_CHAT_PROMPT.format(question=question)
        answer = await generate_text_with_fallback(runtime.context, prompt, temperature=0.3)
        if not answer:
            raise ValueError("empty direct response")
    except Exception as e:
        logger.warning(f"Direct response generation failed: {e}")
        answer = "你好，我在。你也可以问我和 CS/AI/ML arXiv 论文相关的问题，我可以帮你检索并总结。"

    logger.debug(f"Direct response duration: {(time.time() - start_time):.2f}s")
    return {"messages": [AIMessage(content=answer)]}
