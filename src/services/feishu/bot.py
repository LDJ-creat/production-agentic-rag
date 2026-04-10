import asyncio
import importlib.util
import json
import logging
import re
import threading
from typing import Any, Dict, Optional

from src.schemas.api.ask import AskRequest
from src.services.feishu.client import FeishuClient
from src.services.ollama.prompts import RAGPromptBuilder

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
        verification_token: str = "",
        encrypt_key: str = "",
        subscription_mode: str = "long_connection",
        app_id: str = "",
        app_secret: str = "",
    ):
        self.client = client
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.ollama = ollama_client
        self.cache = cache_client
        self.verification_token = verification_token
        self.encrypt_key = encrypt_key
        self.subscription_mode = subscription_mode
        self.app_id = app_id
        self.app_secret = app_secret
        self._processed_message_ids: set[str] = set()
        self._processed_ids_max_size = 2000
        self._ws_client = None
        self._ws_thread: Optional[threading.Thread] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

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
        import lark_oapi as lark

        # Avoid inheriting uvloop policy behavior; use an isolated standard loop in this thread.
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
            # SDK callback is sync. Dispatch coroutine onto the app event loop.
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

        payload = {
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
        return payload

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

        if user_text.lower().startswith("/search"):
            query = user_text[7:].strip()
            await self._handle_search(query, message_id, chat_id, sender_open_id)
            return {"code": 0, "msg": "ok"}

        await self._handle_question(user_text, message_id, chat_id, sender_open_id)
        return {"code": 0, "msg": "ok"}

    async def _handle_search(self, query: str, message_id: str, chat_id: str, sender_open_id: str) -> None:
        if not query:
            await self._respond_text(message_id=message_id, chat_id=chat_id, text="用法：/search 关键词")
            return

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
                await self._respond_text(message_id=message_id, chat_id=chat_id, text="未检索到相关论文，请尝试更换关键词。")
                return

            lines = [f"检索到 {len(unique_papers)} 篇论文：", ""]
            for idx, hit in enumerate(unique_papers, 1):
                title = hit.get("title", "Untitled")
                arxiv_id = hit.get("arxiv_id", "")
                lines.append(f"{idx}. {title}")
                lines.append(f"https://arxiv.org/abs/{arxiv_id}")
                lines.append("")

            await self._respond_text(message_id=message_id, chat_id=chat_id, text="\n".join(lines).strip())
        except Exception as e:
            logger.error("Feishu search failed: %s", e, exc_info=True)
            await self._respond_text(message_id=message_id, chat_id=chat_id, text=f"检索失败：{e}")

    async def _handle_question(self, query: str, message_id: str, chat_id: str, sender_open_id: str) -> None:
        ask_request = AskRequest(query=query, top_k=3, use_hybrid=True)

        try:
            if self.cache:
                try:
                    cached = await self.cache.find_cached_response(ask_request)
                    if cached:
                        await self._respond_text(
                            message_id=message_id,
                            chat_id=chat_id,
                            text=self._format_answer(cached.answer, cached.sources),
                        )
                        return
                except Exception as e:
                    logger.warning("Feishu cache lookup failed: %s", e)

            query_embedding = None
            if ask_request.use_hybrid:
                try:
                    query_embedding = await self.embeddings.embed_query(query)
                except Exception as e:
                    logger.warning("Feishu embedding generation failed, fallback BM25: %s", e)

            search_results = self.opensearch.search_unified(
                query=query,
                query_embedding=query_embedding,
                size=ask_request.top_k,
                use_hybrid=ask_request.use_hybrid and query_embedding is not None,
            )

            chunks = []
            sources = []
            seen_sources = set()
            for hit in search_results.get("hits", []):
                arxiv_id = hit.get("arxiv_id", "")
                chunks.append({"arxiv_id": arxiv_id, "chunk_text": hit.get("chunk_text", hit.get("abstract", ""))})
                if arxiv_id:
                    clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    src = f"https://arxiv.org/abs/{clean_id}"
                    if src not in seen_sources:
                        seen_sources.add(src)
                        sources.append(src)

            if not chunks:
                await self._respond_text(message_id=message_id, chat_id=chat_id, text="没有找到相关论文，请尝试换个问题。")
                return

            prompt = RAGPromptBuilder().create_rag_prompt(query=query, chunks=chunks)
            llm_response = await self.ollama.generate(model="llama3.2:1b", prompt=prompt, stream=False)
            answer = llm_response.get("response", "") if llm_response else ""
            formatted = self._format_answer(answer, sources)

            await self._respond_text(message_id=message_id, chat_id=chat_id, text=formatted)
        except Exception as e:
            logger.error("Feishu QA failed: %s", e, exc_info=True)
            await self._respond_text(message_id=message_id, chat_id=chat_id, text=f"处理失败：{e}")

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
