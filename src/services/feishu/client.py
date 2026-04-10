import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class FeishuClient:
    """Minimal async client for Feishu app bot APIs."""

    def __init__(self, app_id: str, app_secret: str, base_url: str = "https://open.feishu.cn/open-apis"):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = base_url.rstrip("/")
        self._tenant_access_token: Optional[str] = None
        self._tenant_token_expiry_epoch: float = 0.0

    async def _refresh_tenant_access_token(self) -> str:
        url = f"{self.base_url}/auth/v3/tenant_access_token/internal"
        payload = {"app_id": self.app_id, "app_secret": self.app_secret}

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        if data.get("code", -1) != 0:
            raise RuntimeError(f"Failed to get Feishu tenant access token: {data}")

        access_token = data.get("tenant_access_token")
        expires_in = int(data.get("expire", 0) or 0)
        if not access_token or expires_in <= 0:
            raise RuntimeError(f"Invalid token response from Feishu: {data}")

        self._tenant_access_token = access_token
        self._tenant_token_expiry_epoch = time.time() + max(0, expires_in - 120)
        return access_token

    async def get_tenant_access_token(self) -> str:
        if self._tenant_access_token and time.time() < self._tenant_token_expiry_epoch:
            return self._tenant_access_token
        return await self._refresh_tenant_access_token()

    async def send_message(
        self,
        receive_id: str,
        content: Dict[str, Any],
        msg_type: str = "text",
        receive_id_type: str = "chat_id",
        request_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        token = await self.get_tenant_access_token()
        url = f"{self.base_url}/im/v1/messages"

        body = {
            "receive_id": receive_id,
            "msg_type": msg_type,
            "content": json.dumps(content, ensure_ascii=False),
            "uuid": request_uuid or str(uuid.uuid4()),
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, params={"receive_id_type": receive_id_type}, json=body, headers=headers)
            data = self._parse_response(response, "send message")

        return data

    async def reply_message(
        self,
        message_id: str,
        content: Dict[str, Any],
        msg_type: str = "text",
        request_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        token = await self.get_tenant_access_token()
        url = f"{self.base_url}/im/v1/messages/{message_id}/reply"

        body = {
            "msg_type": msg_type,
            "content": json.dumps(content, ensure_ascii=False),
            "uuid": request_uuid or str(uuid.uuid4()),
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=body, headers=headers)
            data = self._parse_response(response, "reply message")

        return data

    def _parse_response(self, response: httpx.Response, action: str) -> Dict[str, Any]:
        try:
            data = response.json()
        except Exception:
            data = {"raw": response.text}

        if response.status_code >= 400:
            raise RuntimeError(
                f"Feishu {action} HTTP {response.status_code}: {data}"
            )

        if data.get("code", -1) != 0:
            raise RuntimeError(f"Feishu {action} failed: {data}")

        return data
