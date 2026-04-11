import json
import os
import requests

base = "https://open.feishu.cn/open-apis"
app_id = os.getenv("FEISHU__APP_ID")
app_secret = os.getenv("FEISHU__APP_SECRET")

resp = requests.post(
    f"{base}/auth/v3/tenant_access_token/internal",
    json={"app_id": app_id, "app_secret": app_secret},
    timeout=20,
)
print("token_status", resp.status_code)
print("token_body", resp.text[:300])
resp.raise_for_status()
token = resp.json().get("tenant_access_token")

headers = {"Authorization": f"Bearer {token}"}
chats = requests.get(
    f"{base}/im/v1/chats",
    headers=headers,
    params={"page_size": 50},
    timeout=20,
)
print("chats_status", chats.status_code)
print(chats.text)
