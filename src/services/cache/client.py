import hashlib
import asyncio
import json
import logging
import time
import uuid
from datetime import timedelta
from typing import Any, Optional

import redis
from src.config import RedisSettings
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)


class CacheClient:
    """Redis-based exact match cache for RAG queries."""

    def __init__(self, redis_client: redis.Redis, settings: RedisSettings):
        self.redis = redis_client
        self.settings = settings
        self.ttl = timedelta(hours=settings.ttl_hours)

    def _conversation_history_key(self, session_key: str) -> str:
        return f"feishu:conversation_history:{session_key}"

    def _conversation_lock_key(self, session_key: str) -> str:
        return f"feishu:conversation_lock:{session_key}"

    def _generate_cache_key(self, request: AskRequest) -> str:
        """Generate exact cache key based on request parameters."""
        key_data = {
            "query": request.query,
            "model": request.model,
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "categories": sorted(request.categories) if request.categories else [],
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"exact_cache:{key_hash}"

    async def find_cached_response(self, request: AskRequest) -> Optional[AskResponse]:
        """Find cached response for exact query match."""
        try:
            cache_key = self._generate_cache_key(request)

            # Simple Redis GET operation - O(1)
            cached_response = self.redis.get(cache_key)

            if cached_response:
                try:
                    response_data = json.loads(cached_response)
                    logger.info(f"Cache hit for exact query match")
                    return AskResponse(**response_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to deserialize cached response: {e}")
                    return None

            return None

        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None

    async def store_response(self, request: AskRequest, response: AskResponse) -> bool:
        """Store response for exact query matching."""
        try:
            cache_key = self._generate_cache_key(request)

            # Simple Redis SET operation with TTL
            success = self.redis.set(cache_key, response.model_dump_json(), ex=int(self.ttl.total_seconds()))

            if success:
                logger.info(f"Stored response in exact cache with key {cache_key[:16]}...")
                return True
            else:
                logger.warning(f"Failed to store response in cache")
                return False

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False

    async def get_recent_conversation_turns(self, session_key: str, max_turns: int) -> list[dict[str, Any]]:
        """Fetch the most recent conversation turns for a Feishu session."""
        try:
            if max_turns <= 0:
                return []

            raw_turns = self.redis.lrange(self._conversation_history_key(session_key), -max_turns, -1)
            turns: list[dict[str, Any]] = []

            for raw_turn in raw_turns:
                try:
                    turn = json.loads(raw_turn) if isinstance(raw_turn, str) else raw_turn
                    if isinstance(turn, dict):
                        turns.append(turn)
                except Exception as e:
                    logger.warning("Failed to deserialize Feishu conversation turn for %s: %s", session_key, e)

            return turns
        except Exception as e:
            logger.error("Error loading Feishu conversation history for %s: %s", session_key, e)
            return []

    async def append_conversation_turn(
        self,
        session_key: str,
        user_message: dict[str, Any],
        assistant_message: dict[str, Any],
        max_turns: int,
        ttl_hours: int,
    ) -> bool:
        """Append one user/assistant turn and trim to the latest N turns."""
        try:
            if max_turns <= 0:
                return True

            turn = {
                "turn_id": str(uuid.uuid4()),
                "created_at": time.time(),
                "user_message": user_message,
                "assistant_message": assistant_message,
            }
            key = self._conversation_history_key(session_key)
            self.redis.rpush(key, json.dumps(turn, ensure_ascii=False))
            self.redis.ltrim(key, -max_turns, -1)
            self.redis.expire(key, int(timedelta(hours=ttl_hours).total_seconds()))
            return True
        except Exception as e:
            logger.error("Error appending Feishu conversation turn for %s: %s", session_key, e)
            return False

    async def acquire_conversation_lock(
        self,
        session_key: str,
        timeout_seconds: int,
        lock_ttl_seconds: int,
        poll_interval_seconds: float = 0.1,
    ) -> Optional[str]:
        """Acquire a short-lived per-session lock to serialize read-process-write flows."""
        lock_key = self._conversation_lock_key(session_key)
        token = str(uuid.uuid4())
        deadline = time.monotonic() + max(0, timeout_seconds)

        while time.monotonic() < deadline:
            try:
                acquired = self.redis.set(lock_key, token, nx=True, ex=max(1, lock_ttl_seconds))
                if acquired:
                    return token
            except Exception as e:
                logger.error("Error acquiring Feishu conversation lock for %s: %s", session_key, e)
                return None

            await asyncio.sleep(max(0.01, poll_interval_seconds))

        return None

    async def release_conversation_lock(self, session_key: str, token: str) -> bool:
        """Release a per-session lock only if the token still matches."""
        lock_key = self._conversation_lock_key(session_key)
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        end
        return 0
        """

        try:
            self.redis.eval(script, 1, lock_key, token)
            return True
        except Exception as e:
            logger.warning("Failed to release Feishu conversation lock for %s: %s", session_key, e)
            return False
