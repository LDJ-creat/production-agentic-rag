import asyncio

import pytest

from src.config import RedisSettings
from src.services.cache.client import CacheClient


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.lists = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    def ltrim(self, key, start, end):
        items = self.lists.get(key, [])
        length = len(items)

        if start < 0:
            start = max(0, length + start)
        if end < 0:
            end = length + end

        if length == 0 or start >= length or end < start:
            self.lists[key] = []
            return True

        self.lists[key] = items[start : end + 1]
        return True

    def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        length = len(items)

        if start < 0:
            start = max(0, length + start)
        if end < 0:
            end = length + end

        if length == 0 or start >= length or end < start:
            return []

        return items[start : end + 1]

    def expire(self, key, seconds):
        return True

    def eval(self, script, numkeys, key, token):
        if self.values.get(key) == token:
            del self.values[key]
            return 1
        return 0


@pytest.mark.asyncio
async def test_conversation_history_keeps_latest_n_turns():
    cache = CacheClient(FakeRedis(), RedisSettings())

    await cache.append_conversation_turn(
        session_key="chat-1",
        user_message={"text": "q1"},
        assistant_message={"text": "a1"},
        max_turns=2,
        ttl_hours=24,
    )
    await cache.append_conversation_turn(
        session_key="chat-1",
        user_message={"text": "q2"},
        assistant_message={"text": "a2"},
        max_turns=2,
        ttl_hours=24,
    )
    await cache.append_conversation_turn(
        session_key="chat-1",
        user_message={"text": "q3"},
        assistant_message={"text": "a3"},
        max_turns=2,
        ttl_hours=24,
    )

    turns = await cache.get_recent_conversation_turns("chat-1", 2)

    assert [turn["user_message"]["text"] for turn in turns] == ["q2", "q3"]
    assert [turn["assistant_message"]["text"] for turn in turns] == ["a2", "a3"]


@pytest.mark.asyncio
async def test_conversation_lock_serializes_and_releases():
    fake_redis = FakeRedis()
    cache = CacheClient(fake_redis, RedisSettings())

    token = await cache.acquire_conversation_lock(
        session_key="chat-1",
        timeout_seconds=1,
        lock_ttl_seconds=10,
        poll_interval_seconds=0.01,
    )

    assert token is not None

    second_token = await cache.acquire_conversation_lock(
        session_key="chat-1",
        timeout_seconds=0.05,
        lock_ttl_seconds=10,
        poll_interval_seconds=0.01,
    )

    assert second_token is None

    released = await cache.release_conversation_lock("chat-1", token)
    assert released is True

    third_token = await cache.acquire_conversation_lock(
        session_key="chat-1",
        timeout_seconds=1,
        lock_ttl_seconds=10,
        poll_interval_seconds=0.01,
    )

    assert third_token is not None