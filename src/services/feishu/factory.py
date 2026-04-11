import logging
from typing import Optional

from src.config import get_settings
from src.services.feishu.bot import FeishuBot
from src.services.feishu.client import FeishuClient

logger = logging.getLogger(__name__)


def make_feishu_service(
    opensearch_client,
    embeddings_client,
    ollama_client,
    cache_client=None,
    langfuse_tracer=None,
) -> Optional[FeishuBot]:
    """Create Feishu bot service if configured."""
    settings = get_settings()
    feishu_settings = settings.feishu

    if not feishu_settings.enabled:
        logger.info("Feishu bot is disabled")
        return None

    if not feishu_settings.app_id or not feishu_settings.app_secret:
        logger.warning("Feishu app credentials are not configured")
        return None

    client = FeishuClient(app_id=feishu_settings.app_id, app_secret=feishu_settings.app_secret)
    bot = FeishuBot(
        client=client,
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        ollama_client=ollama_client,
        cache_client=cache_client,
        langfuse_tracer=langfuse_tracer,
        default_model=settings.llm.default_model,
        verification_token=feishu_settings.verification_token,
        encrypt_key=feishu_settings.encrypt_key,
        subscription_mode=feishu_settings.subscription_mode,
        app_id=feishu_settings.app_id,
        app_secret=feishu_settings.app_secret,
        history_max_turns=feishu_settings.history_max_turns,
        history_ttl_hours=feishu_settings.history_ttl_hours,
        history_lock_timeout_seconds=feishu_settings.history_lock_timeout_seconds,
        history_lock_ttl_seconds=feishu_settings.history_lock_ttl_seconds,
        history_lock_poll_interval_seconds=feishu_settings.history_lock_poll_interval_seconds,
    )
    return bot
