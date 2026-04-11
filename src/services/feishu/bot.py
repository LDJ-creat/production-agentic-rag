import asyncio
import importlib.util
import json
import logging
import re
import threading
from typing import Any, Dict, Optional

from src.services.agents.factory import make_agentic_rag_service
from src.services.feishu.client import FeishuClient

logger = logging.getLogger(__name__)


class FeishuBot:
    """Feishu app bot for search and QA interactions."""
import asyncio
import importlib.util
import json
import logging
import re
import threading
import time
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage

from src.services.agents.factory import make_agentic_rag_service
from src.services.cache.client import CacheClient
from src.services.feishu.client import FeishuClient

logger = logging.getLogger(__name__)


class FeishuBot:
    """Feishu app bot for search and QA interactions."""

    def __init__(
        self,
        client: FeishuClient,
        opensearch_client,
        embeddings_client,
        ollama_client,
        cache_client=None,
        langfuse_tracer=None,
        default_model: str = "llama3.2:1b",
        verification_token: str = "",
        encrypt_key: str = "",
        subscription_mode: str = "long_connection",
        app_id: str = "",
        app_secret: str = "",
        history_max_turns: int = 6,
        history_ttl_hours: int = 24,
        history_lock_timeout_seconds: int = 30,
        history_lock_ttl_seconds: int = 60,
        history_lock_poll_interval_seconds: float = 0.1,
    ):
        self.client = client
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.ollama = ollama_client
        self.cache: CacheClient | None = cache_client
        self.default_model = default_model
        self.verification_token = verification_token
        self.encrypt_key = encrypt_key
        self.subscription_mode = subscription_mode
        self.app_id = app_id
        self.app_secret = app_secret
        self.history_max_turns = history_max_turns
        self.history_ttl_hours = history_ttl_hours
        self.history_lock_timeout_seconds = history_lock_timeout_seconds
        self.history_lock_ttl_seconds = history_lock_ttl_seconds
        self.history_lock_poll_interval_seconds = history_lock_poll_interval_seconds
        self._processed_message_ids: set[str] = set()
        self._processed_ids_max_size = 2000
        self._ws_client = None
        self._ws_thread: Optional[threading.Thread] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self.agentic_rag = make_agentic_rag_service(
            opensearch_client=self.opensearch,
            ollama_client=self.ollama,
            embeddings_client=self.embeddings,
            langfuse_tracer=langfuse_tracer,
            model=self.default_model,
            top_k=3,
            use_hybrid=True,
        )

    async def start(self) -> None:
        """Start Feishu bot in long_connection or webhook mode."""
        await self.client.get_tenant_access_token()
        self._main_loop = asyncio.get_running_loop()

        if self.subscription_mode == "long_connection":
            if importlib.util.find_spec("lark_oapi") is None:
                raise RuntimeError(
                    "Feishu long_connection mode requires package lark-oapi. "
                    "Please install dependencies and restart."
                )

            self._ws_thread = threading.Thread(target=self._run_ws_client, name="feishu-long-connection", daemon=True)
            self._ws_thread.start()
            logger.info("Feishu bot started in long_connection mode")
            return

        logger.info("Feishu bot started in webhook mode")

    def _run_ws_client(self) -> None:
        """Run SDK WS client in an isolated event loop within worker thread."""
        import importlib

        lark = importlib.import_module("lark_oapi")

        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            event_handler = (
                lark.EventDispatcherHandler.builder(self.encrypt_key or "", self.verification_token or "")
                .register_p2_im_message_receive_v1(self._on_long_connection_message)
                .build()
            )
            self._ws_client = lark.ws.Client(
                self.app_id,
                self.app_secret,
                event_handler=event_handler,
                log_level=lark.LogLevel.INFO,
            )
            logger.info("Feishu ws thread loop running=%s", loop.is_running())
            self._ws_client.start()
        except Exception as e:
            logger.error("Feishu long connection thread crashed: %s", e, exc_info=True)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def stop(self) -> None:
        """Stop Feishu bot if long connection client supports stopping."""
        if self._ws_client and hasattr(self._ws_client, "stop"):
            try:
                self._ws_client.stop()
            except Exception as e:
                logger.warning("Failed to stop Feishu long connection client cleanly: %s", e)
        logger.info("Feishu bot stopped")

    def _on_long_connection_message(self, data) -> None:
        """SDK callback for p2.im.message.receive_v1 in long_connection mode."""
        try:
            payload = self._build_payload_from_event_context(data)
            if self._main_loop and self._main_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self.handle_event(payload), self._main_loop)
                future.add_done_callback(self._log_event_future)
                return

            asyncio.run(self.handle_event(payload))
        except Exception as e:
            logger.error("Failed to process Feishu long connection event: %s", e, exc_info=True)

    def _log_event_future(self, future) -> None:
        try:
            _ = future.result()
        except Exception as e:
            logger.error("Failed to handle Feishu long connection message: %s", e, exc_info=True)

    def _build_payload_from_event_context(self, data) -> Dict[str, Any]:
        event_data = getattr(data, "event", None)
        message = getattr(event_data, "message", None)
        sender = getattr(event_data, "sender", None)
        sender_id = getattr(sender, "sender_id", None)

        sender_open_id = getattr(sender_id, "open_id", "") if sender_id else ""
        sender_user_id = getattr(sender_id, "user_id", "") if sender_id else ""
        sender_union_id = getattr(sender_id, "union_id", "") if sender_id else ""

        return {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "message_id": getattr(message, "message_id", ""),
                    "chat_id": getattr(message, "chat_id", ""),
                    "message_type": getattr(message, "message_type", ""),
                    "content": getattr(message, "content", "{}"),
                },
                "sender": {
                    "sender_id": {
                        "open_id": sender_open_id,
                        "user_id": sender_user_id,
                        "union_id": sender_union_id,
                    }
                },
            },
        }

    async def handle_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Feishu callback payloads."""
        if "challenge" in payload:
            if self.verification_token:
                token = payload.get("token")
                if token != self.verification_token:
                    logger.warning("Feishu URL verification token mismatch")
                    return {"code": 1, "msg": "invalid verification token"}
            return {"challenge": payload.get("challenge")}

        if self.verification_token and payload.get("token") and payload.get("token") != self.verification_token:
            logger.warning("Feishu callback token mismatch")
            return {"code": 1, "msg": "invalid callback token"}

        event_type = payload.get("header", {}).get("event_type")
        if event_type != "im.message.receive_v1":
            return {"code": 0, "msg": "ignored"}

        event = payload.get("event", {})
        message = event.get("message", {})
        sender = event.get("sender", {})

        message_id = message.get("message_id", "")
        if not message_id:
            return {"code": 0, "msg": "missing message id"}
        if message_id in self._processed_message_ids:
            return {"code": 0, "msg": "duplicate"}

        self._processed_message_ids.add(message_id)
        if len(self._processed_message_ids) > self._processed_ids_max_size:
            self._processed_message_ids.clear()

        await self._send_typing_ack(message_id)

        message_type = message.get("message_type")
        if message_type != "text":
            await self._respond_text(message_id=message_id, chat_id=message.get("chat_id", ""), text="目前仅支持文本消息。")
            return {"code": 0, "msg": "ok"}

        raw_content = message.get("content", "{}")
        try:
            content_dict = json.loads(raw_content)
        except Exception:
            content_dict = {}

        raw_text = content_dict.get("text", "").strip()
        user_text = re.sub(r"@_user_\d+\s*", "", raw_text).strip()
        if not user_text:
            await self._respond_text(message_id=message_id, chat_id=message.get("chat_id", ""), text="请发送问题或使用 /search 关键词")
            return {"code": 0, "msg": "ok"}

        sender_open_id = sender.get("sender_id", {}).get("open_id", "")
        chat_id = message.get("chat_id", "")
        session_key = self._get_conversation_session_key(chat_id=chat_id, sender_open_id=sender_open_id)

        lock_token = None
        history_messages: list[HumanMessage | AIMessage] = []
        if self.cache and session_key:
            lock_token = await self.cache.acquire_conversation_lock(
                session_key=session_key,
                timeout_seconds=self.history_lock_timeout_seconds,
                lock_ttl_seconds=self.history_lock_ttl_seconds,
                poll_interval_seconds=self.history_lock_poll_interval_seconds,
            )
            if not lock_token:
                logger.warning("Failed to acquire Feishu conversation lock for session=%s", session_key)
            else:
                history_turns = await self.cache.get_recent_conversation_turns(session_key, self.history_max_turns)
                history_messages = self._turns_to_messages(history_turns)

        try:
            if user_text.lower().startswith("/search"):
                query = user_text[7:].strip()
                response_text, should_store = await self._handle_search(query, message_id, chat_id)
            else:
                response_text, should_store = await self._handle_question(
                    user_text=user_text,
                    message_id=message_id,
                    chat_id=chat_id,
                    sender_open_id=sender_open_id,
                    history_messages=history_messages,
                )

            if should_store and self.cache and session_key and lock_token:
                await self._append_conversation_turn(session_key, message_id, user_text, response_text)

            return {"code": 0, "msg": "ok"}
        finally:
            if lock_token and self.cache and session_key:
                await self.cache.release_conversation_lock(session_key, lock_token)

    async def _handle_search(self, query: str, message_id: str, chat_id: str) -> tuple[str, bool]:
        if not query:
            response_text = "用法：/search 关键词"
            await self._respond_text(message_id=message_id, chat_id=chat_id, text=response_text)
            return response_text, False

        try:
            query_embedding = await self.embeddings.embed_query(query)
            results = self.opensearch.search_unified(
                query=query,
                query_embedding=query_embedding,
                size=10,
                use_hybrid=True,
            )
            hits = results.get("hits", [])

            seen_ids = set()
            unique_papers = []
            for hit in hits:
                arxiv_id = hit.get("arxiv_id", "")
                if arxiv_id and arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    unique_papers.append(hit)
                if len(unique_papers) >= 5:
                    break

            if not unique_papers:
                response_text = "未检索到相关论文，请尝试更换关键词。"
                await self._respond_text(message_id=message_id, chat_id=chat_id, text=response_text)
                return response_text, True

            lines = [f"检索到 {len(unique_papers)} 篇论文：", ""]
            for idx, hit in enumerate(unique_papers, 1):
                title = hit.get("title", "Untitled")
                arxiv_id = hit.get("arxiv_id", "")
                lines.append(f"{idx}. {title}")
                lines.append(f"https://arxiv.org/abs/{arxiv_id}")
                lines.append("")

            response_text = "\n".join(lines).strip()
            await self._respond_text(message_id=message_id, chat_id=chat_id, text=response_text)
            return response_text, True
        except Exception as e:
            logger.error("Feishu search failed: %s", e, exc_info=True)
            response_text = f"检索失败：{e}"
            await self._respond_text(message_id=message_id, chat_id=chat_id, text=response_text)
            return response_text, False

    async def _handle_question(
        self,
        user_text: str,
        message_id: str,
        chat_id: str,
        sender_open_id: str,
        history_messages: list[HumanMessage | AIMessage],
    ) -> tuple[str, bool]:
        try:
            result = await self.agentic_rag.ask(
                query=user_text,
                user_id=sender_open_id or "feishu_user",
                model=self.default_model,
                history_messages=history_messages,
            )

            intent_route = result.get("intent_route") if isinstance(result, dict) else None
            retrieval_attempts = result.get("retrieval_attempts", 0) if isinstance(result, dict) else 0
            out_of_scope = bool(result.get("out_of_scope", False)) if isinstance(result, dict) else False
            max_retrieval_reached = bool(result.get("max_retrieval_reached", False)) if isinstance(result, dict) else False
            logger.info(
                "Feishu agentic routing message_id=%s user=%s intent_route=%s retrieval_attempts=%s out_of_scope=%s max_retrieval_reached=%s",
                message_id,
                sender_open_id or "unknown",
                intent_route or "unknown",
                retrieval_attempts,
                out_of_scope,
                max_retrieval_reached,
            )

            answer = result.get("answer", "") if isinstance(result, dict) else ""
            sources = self._extract_sources(result.get("sources", []) if isinstance(result, dict) else [])

            formatted = self._format_answer(answer, sources)
            await self._respond_text(message_id=message_id, chat_id=chat_id, text=formatted)
            return formatted, True
        except Exception as e:
            logger.error("Feishu QA failed: %s", e, exc_info=True)
            response_text = f"处理失败：{e}"
            await self._respond_text(message_id=message_id, chat_id=chat_id, text=response_text)
            return response_text, False

    def _get_conversation_session_key(self, chat_id: str, sender_open_id: str) -> str:
        return chat_id or sender_open_id

    def _turns_to_messages(self, turns: list[dict[str, Any]]) -> list[HumanMessage | AIMessage]:
        messages: list[HumanMessage | AIMessage] = []

        for turn in turns:
            if not isinstance(turn, dict):
                continue

            user_message = turn.get("user_message", {})
            assistant_message = turn.get("assistant_message", {})

            user_text = str(user_message.get("text", "")).strip()
            assistant_text = str(assistant_message.get("text", "")).strip()

            if user_text:
                messages.append(HumanMessage(content=user_text))
            if assistant_text:
                messages.append(AIMessage(content=assistant_text))

        return messages

    async def _append_conversation_turn(self, session_key: str, message_id: str, user_text: str, assistant_text: str) -> None:
        if not self.cache:
            return

        await self.cache.append_conversation_turn(
            session_key=session_key,
            user_message={
                "message_id": message_id,
                "text": user_text,
                "timestamp": time.time(),
            },
            assistant_message={
                "text": assistant_text,
                "timestamp": time.time(),
            },
            max_turns=self.history_max_turns,
            ttl_hours=self.history_ttl_hours,
        )

    def _extract_sources(self, sources: list) -> list[str]:
        """Normalize agentic source objects to displayable URL strings."""
        normalized: list[str] = []
        seen: set[str] = set()

        for item in sources or []:
            url = ""
            if isinstance(item, str):
                url = item
            elif isinstance(item, dict):
                url = str(item.get("url") or item.get("source") or item.get("pdf_url") or "")

            if url and url not in seen:
                seen.add(url)
                normalized.append(url)

        return normalized

    async def _send_typing_ack(self, message_id: str) -> None:
        """Immediately reply on the original message to confirm request reception."""
        try:
            await self.client.reply_message(message_id=message_id, content={"text": "⌨️"})
        except Exception as e:
            logger.warning("Feishu typing ack failed for message_id=%s: %s", message_id, e)

    async def _respond_text(self, message_id: str, chat_id: str, text: str) -> None:
        """Reply to message first; fallback to chat send when reply endpoint rejects request."""
        try:
            await self.client.reply_message(message_id=message_id, content={"text": text})
            return
        except Exception as e:
            logger.warning("Feishu reply failed for message_id=%s, fallback to chat send: %s", message_id, e)

        if not chat_id:
            raise RuntimeError("Feishu fallback send failed: missing chat_id")

        await self.client.send_message(
            receive_id=chat_id,
            receive_id_type="chat_id",
            msg_type="text",
            content={"text": text},
        )

    def _format_answer(self, answer: str, sources: list[str]) -> str:
        text = f"回答：\n{answer.strip() if answer else '未生成有效回答。'}"
        if sources:
            text += "\n\n参考来源：\n" + "\n".join(f"{idx + 1}. {url}" for idx, url in enumerate(sources[:5]))
        if len(text) > 4000:
            text = text[:3990] + "..."
        return text
